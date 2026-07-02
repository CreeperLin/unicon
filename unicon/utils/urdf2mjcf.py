"""Functional version of URDF to MJCF converter using only functions."""

import os
import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, List

import mujoco

import io
import re
from xml.dom import minidom


def create_mjcf_elements_config() -> dict:
    """Create default MJCF elements configuration."""
    return dict(
        compiler_attribs={
            "angle": "radian",
            "meshdir": "meshes",
            "eulerseq": "zyx",
            "autolimits": "true",
        },
        option_attribs={
            "iterations": "50",
            "timestep": "0.001",
            "solver": "PGS",
            "gravity": "0 0 -9.81",
        },
        joint_defaults={
            "limited": "true",
            "damping": "0.01",
            "armature": "0.01",
            "frictionloss": "0.01",
        },
        geom_defaults={
            "condim": "4",
            "contype": "1",
            "conaffinity": "15",
            "friction": "0.9 0.2 0.2",
            "solref": "0.001 2",
            "density": "0",
            "group": "0",
        }
    )


def preprocess_urdf(config: dict, temp_urdf_path: Path) -> None:
    """Preprocess URDF file for MuJoCo conversion."""
    urdf_path = config['urdf_path']
    try:
        urdf_tree = ET.parse(temp_urdf_path)
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse URDF file: {e}") from e
    try:
        robot = urdf_tree.getroot()

        if robot.tag != 'robot':
            raise ValueError(f"Invalid URDF root element: expected 'robot', got '{robot.tag}'")

        # Find or create mujoco element
        mj = robot.find("mujoco")
        if mj is None:
            mj = ET.SubElement(robot, "mujoco", attrib={})

        # Find or create compiler element
        compiler = mj.find('compiler')
        if compiler is None:
            compiler = ET.SubElement(mj, "compiler", attrib={})

        # Set compiler attributes
        compiler.attrib.pop('meshdir', None)
        compiler.attrib['discardvisual'] = 'true' if config['discardvisual'] is True else 'false'
        compiler.attrib['fusestatic'] = 'false'
        compiler.attrib['strippath'] = 'false'

        package_dirs = config['package_dirs']
        for mesh in urdf_tree.iter("mesh"):
            filename = mesh.attrib.get("filename")
            if filename is None:
                continue
            if filename.startswith("package://"):
                filename = filename[len("package://"):]
                package, filename = filename.split("/", 1)
                package_path = Path(package_dirs.get(package, urdf_path.parent))
                path = (package_path / filename).resolve()
            elif filename.startswith('/'):
                path = Path(filename).resolve()
            else:
                path = (urdf_path.parent / filename).resolve()
            mesh.attrib["filename"] = path.resolve().as_posix()
            if config['verbose']:
                print(f"Updated mesh filename: {filename} -> {path}")
        urdf_tree.write(temp_urdf_path)
        if config['dump']:
            urdf_tree.write('test.urdf')
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess URDF: {e}") from e


# XML building functions
def save_xml(path: Union[str, Path, io.StringIO], tree: Union[ET.ElementTree, ET.Element]) -> None:
    """Save XML tree with pretty formatting."""
    if isinstance(tree, ET.ElementTree):
        tree = tree.getroot()
    xmlstr = minidom.parseString(ET.tostring(tree)).toprettyxml(indent="  ")
    xmlstr = re.sub(r"\n\s*\n", "\n", xmlstr)

    # Add newlines between second-level nodes
    root = ET.fromstring(xmlstr)
    for child in root[:-1]:
        child.tail = "\n\n  "
    xmlstr = ET.tostring(root, encoding="unicode")

    if isinstance(path, io.StringIO):
        path.write(xmlstr)
    else:
        with open(path, "w") as f:
            f.write(xmlstr)


def replace_or_insert_element(
        root: ET.Element, tag: str, new_element: ET.Element, insert_at_start: bool = False) -> None:
    """Replace existing element or insert new one."""
    existing = root.find(tag)
    if existing is not None:
        root.remove(existing)

    if insert_at_start:
        root.insert(0, new_element)
    else:
        root.append(new_element)


def build_compiler_element(elements_config: dict) -> ET.Element:
    """Build compiler element."""
    return ET.Element("compiler", attrib=elements_config['compiler_attribs'])


def build_option_element(elements_config: dict) -> ET.Element:
    """Build option element."""
    return ET.Element("option", attrib=elements_config['option_attribs'])


def build_default_element(elements_config: dict) -> ET.Element:
    """Build default element with sub-elements."""
    default = ET.Element("default")

    # Add default sub-elements
    ET.SubElement(default, "joint", attrib=elements_config['joint_defaults'])
    ET.SubElement(default, "geom", attrib=elements_config['geom_defaults'])
    ET.SubElement(default, "motor", attrib={"ctrllimited": "true"})
    ET.SubElement(default, "equality", attrib={"solref": "0.001 2"})

    return default


def add_compiler(root: ET.Element, config: dict = None) -> None:
    """Add compiler element to root."""
    elements_config = config['elements_config']

    element = build_compiler_element(elements_config)
    replace_or_insert_element(root, "compiler", element, insert_at_start=True)
    root.find("compiler").attrib['fusestatic'] = 'true' if config['fusestatic'] is True else 'false'


def add_default(root: ET.Element, config: dict = None) -> None:
    """Add default element to root."""
    elements_config = config['elements_config']

    element = build_default_element(elements_config)
    replace_or_insert_element(root, "default", element, insert_at_start=True)


def add_option(root: ET.Element, config: dict = None) -> None:
    """Add option element to root."""
    elements_config = config['elements_config']

    element = build_option_element(elements_config)
    replace_or_insert_element(root, "option", element, insert_at_start=True)


def add_assets(root: ET.Element) -> None:
    """Add asset elements including textures and materials."""
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add texture
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "name": "texplane",
            "type": "2d",
            "builtin": "checker",
            "rgb1": ".0 .0 .0",
            "rgb2": ".8 .8 .8",
            "width": "100",
            "height": "100",
        })

    # Add materials
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "matplane",
            "reflectance": "0.",
            "texture": "texplane",
            "texrepeat": "1 1",
            "texuniform": "true",
        })


def get_max_foot_distance(root: ET.Element) -> float:
    """Calculate maximum distance from origin to lowest geometric point.

    This is used to determine appropriate initial height for the robot.

    Args:
        root: The root element of the MJCF file.

    Returns:
        Maximum distance to the lowest point in the model.
    """

    def recursive_search(element: ET.Element, current_z: float = 0) -> float:
        max_distance = 0.0
        for child in element:
            if child.tag == "body":
                body_pos = child.get("pos")
                if body_pos:
                    body_z = float(body_pos.split()[2])
                else:
                    body_z = 0
                max_distance = max(max_distance, recursive_search(child, current_z + body_z))
            elif child.tag == "geom":
                geom_pos = child.get("pos")
                if geom_pos:
                    geom_z = float(geom_pos.split()[2])
                    max_distance = max(max_distance, -(current_z + geom_z))
        return max_distance

    worldbody = root.find("worldbody")
    if worldbody is None:
        return 0.0
    return recursive_search(worldbody)


def add_root_body(root: ET.Element, fix_base_link: bool = False, imu_site_name: str = 'imu') -> None:
    """Add root body with optional freejoint and IMU site."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Calculate the initial height
    foot_distance = get_max_foot_distance(root)
    epsilon = 0.5 if fix_base_link else 0.01
    initial_height = foot_distance + epsilon

    # Create a root body
    root_body = ET.Element(
        "body",
        attrib={
            "name": "__root__",
            "pos": f"0 0 {initial_height}",
            "quat": "1 0 0 0",
        },
    )

    # Add a freejoint
    if not fix_base_link:
        ET.SubElement(
            root_body,
            "freejoint",
            attrib={"name": "__root_joint__"},
        )

    # Add imu site
    ET.SubElement(
        root_body,
        "site",
        attrib={
            "name": imu_site_name,
            "size": "0.01",
            "pos": "0 0 0",
        },
    )

    # Move existing bodies and geoms under base_body
    elements_to_move = list(worldbody)
    for elem in elements_to_move:
        if elem.tag in {"body", "geom"}:
            worldbody.remove(elem)
            root_body.append(elem)
    worldbody.append(root_body)


def add_worldbody_elements(root: ET.Element) -> None:
    """Add ground plane and lighting to the worldbody.

    Args:
        root: The root element of the MJCF file.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    # Add ground plane
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "pos": "0 0 0",
                "size": "100 100 0.001",
                "quat": "1 0 0 0",
                "material": "matplane",
                "condim": "3",
                "conaffinity": "15",
            },
        ),
    )

    # Add lights
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.6 0.6 0.6",
                "specular": "0.2 0.2 0.2",
                "pos": "0 0 4",
                "dir": "0 0 -1",
            },
        ),
    )
    worldbody.insert(
        0,
        ET.Element(
            "light",
            attrib={
                "directional": "true",
                "diffuse": "0.4 0.4 0.4",
                "specular": "0.1 0.1 0.1",
                "pos": "0 0 5.0",
                "dir": "0 0 -1",
                "castshadow": "false",
            },
        ),
    )


def add_actuators(root: ET.Element, no_frc_limit: bool = False, actuator_type: str = 'motor') -> None:
    """Add actuator elements for each joint."""
    actuator_element = ET.Element("actuator")

    # For each joint, add a motor actuator
    for joint in root.iter("joint"):
        joint_name = joint.attrib.get("name")
        if joint_name is None:
            continue
        joint_range = joint.attrib.get("range")
        if joint_range is None:
            joint.attrib['range'] = "-2147483648 +2147483648"

        # Get joint limits if present
        limit_element = joint.find("limit")
        lower_limit = limit_element.get("lower") if limit_element is not None else None
        upper_limit = limit_element.get("upper") if limit_element is not None else None

        if actuator_type == 'motor':
            if no_frc_limit:
                ctrlrange = "-10000 10000"
            elif lower_limit is not None and upper_limit is not None:
                ctrlrange = f"{lower_limit} {upper_limit}"
            else:
                actuatorfrcrange = joint.attrib.get("actuatorfrcrange")
                ctrlrange = actuatorfrcrange if actuatorfrcrange is not None else "-1 1"

            ET.SubElement(
                actuator_element,
                "motor",
                attrib={
                    "name": joint_name,
                    "joint": joint_name,
                    "ctrllimited": "true",
                    "ctrlrange": ctrlrange,
                    "gear": "1",
                },
            )
        elif actuator_type == 'position':
            if joint_range is None:
                joint_range = ' '.join(map(str, [-3.14, 3.14]))
            ctrlrange = joint_range

            ET.SubElement(
                actuator_element,
                "position",
                attrib={
                    "name": joint_name,
                    "joint": joint_name,
                    "ctrllimited": "true",
                    "ctrlrange": ctrlrange,
                },
            )

    replace_or_insert_element(root, "actuator", actuator_element)


def add_sensors(root: ET.Element, imu_site_name: str = 'imu') -> None:
    """Add sensor elements for actuators and IMU."""
    sensor_element = ET.Element("sensor")

    # For each actuator, add sensors
    actuators = root.find("actuator")
    if actuators is not None:
        for actuator in actuators.iter("motor"):
            actuator_name = actuator.attrib.get("name")
            if actuator_name is None:
                continue

            # Add actuatorpos sensor
            ET.SubElement(
                sensor_element,
                "actuatorpos",
                attrib={
                    "name": f"{actuator_name}_p",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorvel sensor
            ET.SubElement(
                sensor_element,
                "actuatorvel",
                attrib={
                    "name": f"{actuator_name}_v",
                    "actuator": actuator_name,
                },
            )

            # Add actuatorfrc sensor
            ET.SubElement(
                sensor_element,
                "actuatorfrc",
                attrib={
                    "name": f"{actuator_name}_f",
                    "actuator": actuator_name,
                    "noise": "0.001",
                },
            )

    # Add additional sensors
    imu_site = None
    for site in root.iter("site"):
        if site.attrib.get("name") == imu_site_name:
            imu_site = site
            break

    if imu_site is not None:
        # Add IMU sensors
        sensors_to_add = [
            ("framequat", {
                "name": "orientation",
                "objtype": "site",
                "objname": imu_site_name
            }),
            ("gyro", {
                "name": "angular-velocity",
                "site": imu_site_name,
                "cutoff": "34.9"
            }),
            ("accelerometer", {
                "name": "accelerometer",
                "site": imu_site_name
            }),
            ("velocimeter", {
                "name": "velocimeter",
                "site": imu_site_name
            }),
        ]

        for sensor_type, attribs in sensors_to_add:
            ET.SubElement(sensor_element, sensor_type, attrib=attribs)

    replace_or_insert_element(root, "sensor", sensor_element)


def add_cameras(root: ET.Element, distance: float = 3.0, height_offset: float = 0.5) -> None:
    """Add fixed and tracking cameras to the worldbody.

    Args:
        root: The root element of the MJCF file.
        distance: Distance of cameras from the model.
        height_offset: Additional height offset for camera positioning.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    foot_distance = get_max_foot_distance(root)
    camera_height = foot_distance + height_offset

    # Add cameras
    cameras = [
        ("fixed", {
            "pos": f"0 {-distance} {camera_height}",
            "xyaxes": "1 0 0 0 0 1"
        }),
        ("track", {
            "mode": "trackcom",
            "pos": f"0 {-distance} {camera_height}",
            "xyaxes": "1 0 0 0 0 1"
        }),
    ]

    for name, attribs in cameras:
        ET.SubElement(worldbody, "camera", attrib={"name": name, **attribs})


def calculate_dof_from_xml(root: ET.Element) -> int:
    """Calculate the total degrees of freedom (DOF) of a robot model.

    Counts DOFs based on joint types:
    - Free joints contribute 7 DOFs (3 translational + 4 quaternion)
    - Regular joints with range attributes contribute 1 DOF each

    Args:
        root: Root element of the MJCF XML.

    Returns:
        Total number of degrees of freedom (qpos size).

    Raises:
        RuntimeError: If unable to parse joint information.
    """
    try:
        dof = 0

        # Count free joints
        for _ in root.iter("freejoint"):
            dof += 7

        # Count joints with a 'range' attribute
        for joint in root.iter("joint"):
            if "range" in joint.attrib:
                dof += 1

        return dof
    except Exception as e:
        raise RuntimeError(f"Failed to calculate DOF: {e}") from e


def add_default_position(root: ET.Element, default_position: List[float]) -> None:
    """Add a keyframe to the root element with the default start position.

    Args:
        root: The root element of the MJCF file.
        default_position: The default positions of the robot.

    Raises:
        ValueError: If default position length doesn't match DOF.
    """
    try:
        num_dof = calculate_dof_from_xml(root)
        if len(default_position) != num_dof:
            raise ValueError(f"Default position must have {num_dof} values, got {len(default_position)}.")

        # Add the keyframe with the default position
        keyframe = ET.Element("keyframe")
        key = ET.SubElement(keyframe, "key")
        key.set("name", "default")
        key.set("qpos", " ".join(map(str, default_position)))
        root.append(keyframe)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Failed to add default position: {e}") from e


def handle_force_limits(root: ET.Element) -> None:
    """Handle force limit settings."""
    for joint in root.iter("joint"):
        if "actuatorfrcrange" in joint.attrib:
            del joint.attrib["actuatorfrcrange"]


def handle_collision_settings(root: ET.Element) -> None:
    """Configure collision settings."""
    for geom in root.iter("geom"):
        if geom.attrib.get('contype') == '0' and geom.attrib.get('conaffinity') == '0':
            continue
        geom.attrib.update({"contype": "1", "conaffinity": "0", "density": "0", "group": "1"})


def handle_cylinder_conversion(root: ET.Element) -> None:
    """Convert cylinders to boxes if requested."""
    for geom in root.iter("geom"):
        if geom.get("type") == 'cylinder':
            geom.set("type", "box")
            old_size = geom.get("size")
            if old_size:
                r, h = map(float, old_size.split(' '))
                new_size = f"{r} {r} {h}"
                geom.set("size", new_size)


def postprocess_mjcf(root: ET.Element, config: dict) -> None:
    """Apply all post-processing steps to the MJCF."""
    if config['no_frc_limit']:
        handle_force_limits(root)
    if config['no_collision_mesh']:
        handle_collision_settings(root)
    if config['cylinder2box']:
        handle_cylinder_conversion(root)

    # Add all required elements
    add_default(root, config)
    add_compiler(root, config)
    add_option(root, config)
    add_assets(root)
    add_cameras(root, distance=config['camera_distance'], height_offset=config['camera_height_offset'])
    add_root_body(root, fix_base_link=config['fix_base_link'], imu_site_name=config['imu_site_name'])
    add_worldbody_elements(root)
    add_actuators(root, no_frc_limit=config['no_frc_limit'], actuator_type=config['actuator_type'])

    if config['use_sensor']:
        add_sensors(root, imu_site_name=config['imu_site_name'])

    # add_visual_geom_logic(root)

    if config['default_position'] is not None:
        add_default_position(root, config['default_position'])


default_config = dict(
    urdf_path=None,
    mjcf_path=None,
    no_collision_mesh=False,
    copy_meshes=False,
    camera_distance=3.0,
    camera_height_offset=0.5,
    no_frc_limit=False,
    default_position=None,
    fix_base_link=False,
    cylinder2box=False,
    use_sensor=True,
    actuator_type='position',
    verbose=True,
    dump=False,
    imu_site_name='imu',
    package_dirs={},
    fusestatic=True,
    # discardvisual=True,
    discardvisual=False,
)


# Main conversion function
def urdf2mjcf(
    **kwds,
) -> None:
    urdf_path = kwds.pop('urdf_path')
    mjcf_path = kwds.pop('mjcf_path')
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else urdf_path.with_suffix(".xml")
    config = {**default_config, 'urdf_path': urdf_path, 'mjcf_path': mjcf_path, **kwds}
    elements_config = create_mjcf_elements_config()
    config['elements_config'] = elements_config
    package_dirs = config['package_dirs']
    p = urdf_path.parent
    while p != p.parent:
        if (p / 'package.xml').exists():
            package_name = p.name
            package_dirs[package_name] = p
        p = p.parent
    print('package_dirs', package_dirs)

    # Ensure output directory exists
    config['mjcf_path'].parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        try:
            temp_urdf_path = temp_dir_path / config['urdf_path'].name
            temp_urdf_path.write_bytes(config['urdf_path'].read_bytes())
        except Exception as e:
            raise RuntimeError(f"Failed to set up temporary environment: {e}") from e

        if config['verbose']:
            print(f'temp_urdf_path: {temp_urdf_path}')
            os.system(f'find {temp_dir_path.resolve()}')

        # Preprocess URDF
        preprocess_urdf(config, temp_urdf_path)

        # Convert URDF to MJCF using MuJoCo
        temp_mjcf_path = temp_dir_path / config['mjcf_path'].name
        try:
            model = mujoco.MjModel.from_xml_path(temp_urdf_path.as_posix())
            mujoco.mj_saveLastXML(temp_mjcf_path.as_posix(), model)
        except Exception as e:
            raise RuntimeError(f"MuJoCo conversion failed: {e}") from e

        if config['dump']:
            mujoco.mj_saveLastXML('/tmp/dump.xml', model)

        # Post-process MJCF
        mjcf_tree = ET.parse(temp_mjcf_path)
        root = mjcf_tree.getroot()
        postprocess_mjcf(root, config)

        # Write final MJCF file
        save_xml(config['mjcf_path'], mjcf_tree)


def main() -> None:
    """Command line interface for URDF to MJCF conversion."""
    parser = argparse.ArgumentParser(description="Convert a URDF file to an MJCF file.")
    parser.add_argument("urdf_path", type=str, help="The path to the URDF file.")
    parser.add_argument("--no-collision-mesh", action="store_true", help="Do not include collision meshes.")
    parser.add_argument("--output", type=str, help="The path to the output MJCF file.")
    parser.add_argument("--copy-meshes", action="store_true", help="Copy mesh files to the output MJCF directory.")
    parser.add_argument("--camera-distance", type=float, default=3.0, help="Camera distance from the robot.")
    parser.add_argument("--camera-height-offset", type=float, default=0.5, help="Camera height offset.")
    parser.add_argument("--no-frc-limit", action="store_true", help="Do not include force limit for the actuators.")
    parser.add_argument("--default-position", type=str, help="Default position for the robot.")
    args = parser.parse_args()

    urdf2mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        no_collision_mesh=args.no_collision_mesh,
        copy_meshes=args.copy_meshes,
        camera_distance=args.camera_distance,
        camera_height_offset=args.camera_height_offset,
        no_frc_limit=args.no_frc_limit,
        default_position=None if args.default_position is None else list(map(float, args.default_position.split())),
    )


if __name__ == "__main__":
    main()
