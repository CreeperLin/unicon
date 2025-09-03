# URDF class from yourdfpy (https://github.com/clemense/yourdfpy), simplified

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

from unicon.utils import compose_mat_np

_logger = logging.getLogger(__name__)


def _str2float(s):
    return float(s) if s is not None else None


def load_xml_lxml(fname_or_file):
    from lxml import etree
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(fname_or_file, parser=parser)
        xml_root = tree.getroot()
    except Exception as e:
        _logger.error(e)
        _logger.error('Using different parsing approach.')
        events = ('start', 'end', 'start-ns', 'end-ns')
        xml = etree.iterparse(fname_or_file, recover=True, events=events)
        # Iterate through all XML elements
        for action, elem in xml:
            # Skip comments and processing instructions,
            # because they do not have names
            if not (isinstance(elem, etree._Comment) or isinstance(elem, etree._ProcessingInstruction)):
                # Remove a namespace URI in the element's name
                # elem.tag = etree.QName(elem).localname
                if action == 'end' and ':' in elem.tag:
                    elem.getparent().remove(elem)
        xml_root = xml.root
    # Remove comments
    etree.strip_tags(xml_root, etree.Comment)
    etree.cleanup_namespaces(xml_root)
    return xml_root


def load_xml_xml(fname_or_file):
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(fname_or_file)
        xml_root = tree.getroot()
    except ET.ParseError as e:
        _logger.error(e)
        _logger.error('Using different parsing approach.')

        xml_root = None
        for event, elem in ET.iterparse(fname_or_file, events=('start', 'end')):
            if event == 'end':
                if ':' in elem.tag:
                    parent = elem.getparent() if hasattr(elem, 'getparent') else None
                    if parent is not None:
                        parent.remove(elem)
            if xml_root is None:
                xml_root = elem
    return xml_root


@dataclass(eq=False)
class TransmissionJoint:
    name: str
    hardware_interfaces: List[str] = field(default_factory=list)


@dataclass(eq=False)
class Actuator:
    name: str
    mechanical_reduction: Optional[float] = None
    # The follwing is only valid for ROS Indigo and prior versions
    hardware_interfaces: List[str] = field(default_factory=list)


@dataclass(eq=False)
class Transmission:
    name: str
    type: Optional[str] = None
    joints: List[TransmissionJoint] = field(default_factory=list)
    actuators: List[Actuator] = field(default_factory=list)


@dataclass
class Calibration:
    rising: Optional[float] = None
    falling: Optional[float] = None


@dataclass
class Mimic:
    joint: str
    multiplier: Optional[float] = None
    offset: Optional[float] = None


@dataclass
class SafetyController:
    soft_lower_limit: Optional[float] = None
    soft_upper_limit: Optional[float] = None
    k_position: Optional[float] = None
    k_velocity: Optional[float] = None


@dataclass
class Sphere:
    radius: float


@dataclass
class Cylinder:
    radius: float
    length: float


@dataclass(eq=False)
class Box:
    size: np.ndarray


@dataclass(eq=False)
class Mesh:
    filename: str
    scale: Optional[Union[float, np.ndarray]] = None


@dataclass
class Geometry:
    box: Optional[Box] = None
    cylinder: Optional[Cylinder] = None
    sphere: Optional[Sphere] = None
    mesh: Optional[Mesh] = None


@dataclass(eq=False)
class Color:
    rgba: np.ndarray


@dataclass
class Texture:
    filename: str


@dataclass
class Material:
    name: Optional[str] = None
    color: Optional[Color] = None
    texture: Optional[Texture] = None


@dataclass(eq=False)
class Visual:
    name: Optional[str] = None
    origin: Optional[np.ndarray] = None
    geometry: Optional[Geometry] = None  # That's not really optional according to ROS
    material: Optional[Material] = None


@dataclass(eq=False)
class Collision:
    name: str
    origin: Optional[np.ndarray] = None
    geometry: Geometry = None


@dataclass(eq=False)
class Inertial:
    origin: Optional[np.ndarray] = None
    mass: Optional[float] = None
    inertia: Optional[np.ndarray] = None


@dataclass(eq=False)
class Link:
    name: str
    inertial: Optional[Inertial] = None
    visuals: List[Visual] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)


@dataclass
class Dynamics:
    damping: Optional[float] = None
    friction: Optional[float] = None
    armature: Optional[float] = None


@dataclass
class Limit:
    effort: Optional[float] = None
    velocity: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass(eq=False)
class Joint:
    name: str
    type: str = None
    parent: str = None
    child: str = None
    origin: np.ndarray = None
    axis: np.ndarray = None
    dynamics: Optional[Dynamics] = None
    limit: Optional[Limit] = None
    mimic: Optional[Mimic] = None
    calibration: Optional[Calibration] = None
    safety_controller: Optional[SafetyController] = None


@dataclass(eq=False)
class Robot:
    name: str
    links: List[Link] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    transmission: List[str] = field(default_factory=list)
    gazebo: List[str] = field(default_factory=list)


class URDF:

    def __init__(
        self,
        robot: Robot = None,
        **kwds,
    ):
        self.robot = robot

    @property
    def joint_names(self):
        return [j.name for j in self.robot.joints]

    @property
    def actuated_joints(self):
        return self._actuated_joints

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices

    @property
    def actuated_joint_indices(self):
        return self._actuated_joint_indices

    @property
    def actuated_joint_names(self):
        return [j.name for j in self._actuated_joints]

    @property
    def num_actuated_joints(self):
        return len(self.actuated_joints)

    @property
    def num_dofs(self):
        total_num_dofs = 0
        for j in self._actuated_joints:
            if j.type in ['revolute', 'prismatic', 'continuous']:
                total_num_dofs += 1
            elif j.type == 'floating':
                total_num_dofs += 6
            elif j.type == 'planar':
                total_num_dofs += 2
        return total_num_dofs

    @property
    def base_link(self):
        return self._base_link

    @staticmethod
    def load(fname_or_file, **kwargs):
        if isinstance(fname_or_file, str):
            if not os.path.isfile(fname_or_file):
                raise ValueError('{} is not a file'.format(fname_or_file))

            if 'mesh_dir' not in kwargs:
                kwargs['mesh_dir'] = os.path.dirname(fname_or_file)

        try:
            xml_root = load_xml_lxml(fname_or_file)
        except Exception as e:
            _logger.error(e)
            xml_root = load_xml_xml(fname_or_file)

        return URDF(robot=URDF._parse_robot(xml_element=xml_root), **kwargs)

    def _determine_base_link(self):
        link_names = [k.name for k in self.robot.links]

        for j in self.robot.joints:
            link_names.remove(j.child)

        if len(link_names) == 0:
            # raise Error?
            return None

        return link_names[0]

    def _parse_mimic(xml_element):
        if xml_element is None:
            return None

        return Mimic(
            joint=xml_element.get('joint'),
            multiplier=_str2float(xml_element.get('multiplier', 1.0)),
            offset=_str2float(xml_element.get('offset', 0.0)),
        )

    def _parse_safety_controller(xml_element):
        if xml_element is None:
            return None

        return SafetyController(
            soft_lower_limit=_str2float(xml_element.get('soft_lower_limit')),
            soft_upper_limit=_str2float(xml_element.get('soft_upper_limit')),
            k_position=_str2float(xml_element.get('k_position')),
            k_velocity=_str2float(xml_element.get('k_velocity')),
        )

    def _parse_transmission_joint(xml_element):
        if xml_element is None:
            return None

        transmission_joint = TransmissionJoint(name=xml_element.get('name'))

        for h in xml_element.findall('hardware_interface'):
            transmission_joint.hardware_interfaces.append(h.text)

        return transmission_joint

    def _parse_actuator(xml_element):
        if xml_element is None:
            return None

        actuator = Actuator(name=xml_element.get('name'))
        if xml_element.find('mechanicalReduction'):
            actuator.mechanical_reduction = float(xml_element.find('mechanicalReduction').text)

        for h in xml_element.findall('hardwareInterface'):
            actuator.hardware_interfaces.append(h.text)

        return actuator

    def _parse_transmission(xml_element):
        if xml_element is None:
            return None

        transmission = Transmission(name=xml_element.get('name'))

        for j in xml_element.findall('joint'):
            transmission.joints.append(URDF._parse_transmission_joint(j))
        for a in xml_element.findall('actuator'):
            transmission.actuators.append(URDF._parse_actuator(a))

        return transmission

    def _parse_calibration(xml_element):
        if xml_element is None:
            return None

        return Calibration(
            rising=_str2float(xml_element.get('rising')),
            falling=_str2float(xml_element.get('falling')),
        )

    def _parse_box(xml_element):
        # In case the element uses comma as a separator
        size = xml_element.attrib['size'].replace(',', ' ').split()
        return Box(size=np.array(size, dtype=np.float64))

    def _parse_cylinder(xml_element):
        return Cylinder(
            radius=float(xml_element.attrib['radius']),
            length=float(xml_element.attrib['length']),
        )

    def _parse_sphere(xml_element):
        return Sphere(radius=float(xml_element.attrib['radius']))

    def _parse_scale(xml_element):
        if 'scale' in xml_element.attrib:
            # In case the element uses comma as a separator
            s = xml_element.get('scale').replace(',', ' ').split()
            if len(s) == 0:
                return None
            elif len(s) == 1:
                return float(s[0])
            else:
                return np.array(list(map(float, s)))
        return None

    def _parse_mesh(xml_element):
        return Mesh(filename=xml_element.get('filename'), scale=URDF._parse_scale(xml_element))

    def _parse_geometry(xml_element):
        geometry = Geometry()
        if xml_element[0].tag == 'box':
            geometry.box = URDF._parse_box(xml_element[0])
        elif xml_element[0].tag == 'cylinder':
            geometry.cylinder = URDF._parse_cylinder(xml_element[0])
        elif xml_element[0].tag == 'sphere':
            geometry.sphere = URDF._parse_sphere(xml_element[0])
        elif xml_element[0].tag == 'mesh':
            geometry.mesh = URDF._parse_mesh(xml_element[0])
        else:
            raise ValueError(f'Unknown tag: {xml_element[0].tag}')

        return geometry

    def _parse_origin(xml_element):
        if xml_element is None:
            return None

        xyz = xml_element.get('xyz', default='0 0 0')
        rpy = xml_element.get('rpy', default='0 0 0')

        mat = compose_mat_np(
            rpy=np.array(list(map(float, rpy.split()))),
            xyz=list(map(float, xyz.split())),
        )
        return mat

    def _parse_color(xml_element):
        if xml_element is None:
            return None

        rgba = xml_element.get('rgba', default='1 1 1 1')

        return Color(rgba=np.array(list(map(float, rgba.split()))))

    def _parse_texture(xml_element):
        if xml_element is None:
            return None

        # TODO: use texture filename handler
        return Texture(filename=xml_element.get('filename', default=None))

    def _parse_material(xml_element):
        if xml_element is None:
            return None

        material = Material(name=xml_element.get('name'))
        material.color = URDF._parse_color(xml_element.find('color'))
        material.texture = URDF._parse_texture(xml_element.find('texture'))

        return material

    def _parse_visual(xml_element):
        visual = Visual(name=xml_element.get('name'))

        visual.geometry = URDF._parse_geometry(xml_element.find('geometry'))
        visual.origin = URDF._parse_origin(xml_element.find('origin'))
        visual.material = URDF._parse_material(xml_element.find('material'))

        return visual

    def _parse_collision(xml_element):
        collision = Collision(name=xml_element.get('name'))

        collision.geometry = URDF._parse_geometry(xml_element.find('geometry'))
        collision.origin = URDF._parse_origin(xml_element.find('origin'))

        return collision

    def _parse_inertia(xml_element):
        if xml_element is None:
            return None

        x = xml_element

        return np.array(
            [
                [
                    x.get('ixx', default=1.0),
                    x.get('ixy', default=0.0),
                    x.get('ixz', default=0.0),
                ],
                [
                    x.get('ixy', default=0.0),
                    x.get('iyy', default=1.0),
                    x.get('iyz', default=0.0),
                ],
                [
                    x.get('ixz', default=0.0),
                    x.get('iyz', default=0.0),
                    x.get('izz', default=1.0),
                ],
            ],
            dtype=np.float64,
        )

    def _parse_mass(xml_element):
        if xml_element is None:
            return None

        return _str2float(xml_element.get('value', default=0.0))

    def _parse_inertial(xml_element):
        if xml_element is None:
            return None

        inertial = Inertial()
        inertial.origin = URDF._parse_origin(xml_element.find('origin'))
        inertial.inertia = URDF._parse_inertia(xml_element.find('inertia'))
        inertial.mass = URDF._parse_mass(xml_element.find('mass'))

        return inertial

    def _parse_link(xml_element):
        link = Link(name=xml_element.attrib['name'])

        link.inertial = URDF._parse_inertial(xml_element.find('inertial'))

        for v in xml_element.findall('visual'):
            link.visuals.append(URDF._parse_visual(v))

        for c in xml_element.findall('collision'):
            link.collisions.append(URDF._parse_collision(c))

        return link

    def _parse_axis(xml_element):
        if xml_element is None:
            return np.array([1.0, 0, 0])

        xyz = xml_element.get('xyz', '1 0 0')
        return np.array(list(map(float, xyz.split())))

    def _parse_limit(xml_element):
        if xml_element is None:
            return None

        return Limit(
            effort=_str2float(xml_element.get('effort', default=None)),
            velocity=_str2float(xml_element.get('velocity', default=None)),
            lower=_str2float(xml_element.get('lower', default=None)),
            upper=_str2float(xml_element.get('upper', default=None)),
        )

    def _parse_dynamics(xml_element):
        if xml_element is None:
            return None

        dynamics = Dynamics()
        dynamics.damping = xml_element.get('damping', default=None)
        dynamics.friction = xml_element.get('friction', default=None)
        dynamics.armature = xml_element.get('armature', default=None)

        return dynamics

    def _parse_joint(xml_element):
        joint = Joint(name=xml_element.attrib['name'])

        joint.type = xml_element.get('type', default=None)
        joint.parent = xml_element.find('parent').get('link')
        joint.child = xml_element.find('child').get('link')
        joint.origin = URDF._parse_origin(xml_element.find('origin'))
        joint.axis = URDF._parse_axis(xml_element.find('axis'))
        joint.limit = URDF._parse_limit(xml_element.find('limit'))
        joint.dynamics = URDF._parse_dynamics(xml_element.find('dynamics'))
        joint.mimic = URDF._parse_mimic(xml_element.find('mimic'))
        joint.calibration = URDF._parse_calibration(xml_element.find('calibration'))
        joint.safety_controller = URDF._parse_safety_controller(xml_element.find('safety_controller'))

        return joint

    @staticmethod
    def _parse_robot(xml_element):
        robot = Robot(name=xml_element.attrib['name'])

        for k in xml_element.findall('link'):
            robot.links.append(URDF._parse_link(k))
        for j in xml_element.findall('joint'):
            robot.joints.append(URDF._parse_joint(j))
        for m in xml_element.findall('material'):
            robot.materials.append(URDF._parse_material(m))
        return robot


def urdf2dict(urdf=None, urdf_path=None):

    def export_axis(axis):
        if axis is None:
            return {}
        return dict(xyz=axis.tolist())

    def export_box(box):
        attrib = {'size': box.size.tolist()}
        return attrib

    def export_cylinder(cylinder):
        attrib = {'radius': cylinder.radius, 'length': cylinder.length}
        return attrib

    def export_sphere(sphere):
        return dict(radius=sphere.radius)

    def export_mesh(mesh):
        attrib = {} if mesh.scale is None else {'scale': mesh.scale}
        return dict(**attrib, filename=mesh.filename)
        # return export_scale(d, mesh.scale)

    def export_geometry(geometry):
        attrib = {}
        if geometry is None:
            return attrib
        if geometry.box is not None:
            attrib['box'] = export_box(geometry.box)
        elif geometry.cylinder is not None:
            attrib['cylinder'] = export_cylinder(geometry.cylinder)
        elif geometry.sphere is not None:
            attrib['sphere'] = export_sphere(geometry.sphere)
        elif geometry.mesh is not None:
            attrib['mesh'] = export_mesh(geometry.mesh)
        return attrib

    def export_origin(origin):
        if origin is None:
            return {}
        return {
            'mat': origin,
        }

    def export_visual(visual):
        attrib = {'name': visual.name} if visual.name is not None else {}
        geometry = export_geometry(visual.geometry)
        origin = export_origin(visual.origin)
        material = export_material(visual.material)
        return dict(**attrib, geometry=geometry, origin=origin, material=material)

    def export_collision(collision):
        attrib = {'name': collision.name} if collision.name is not None else {}
        geometry = export_geometry(collision.geometry)
        origin = export_origin(collision.origin)
        return dict(**attrib, geometry=geometry, origin=origin)

    def export_inertia(inertia):
        if inertia is None:
            return {}
        attrib = {
            'ixx': inertia[0, 0].item(),
            'ixy': inertia[0, 1].item(),
            'ixz': inertia[0, 2].item(),
            'iyy': inertia[1, 1].item(),
            'iyz': inertia[1, 2].item(),
            'izz': inertia[2, 2].item(),
        }
        return attrib

    def export_mass(mass):
        if mass is None:
            return {}
        attrib = {
            'value': mass,
        }
        return attrib

    def export_inertial(inertial):
        if inertial is None:
            return {}
        origin = export_origin(inertial.origin)
        mass = export_mass(inertial.mass)
        inertia = export_inertia(inertial.inertia)
        return dict(origin=origin, mass=mass, inertia=inertia)

    def export_link(link):
        inertial = export_inertial(link.inertial)
        visuals = [export_visual(visual) for visual in link.visuals]
        collisions = [export_collision(collision) for collision in link.collisions]
        return dict(name=link.name, inertial=inertial, visuals=visuals, collisions=collisions)

    def export_limit(limit):
        attrib = {}
        if limit is None:
            return attrib
        if limit.effort is not None:
            attrib['effort'] = limit.effort
        if limit.velocity is not None:
            attrib['velocity'] = limit.velocity
        if limit.lower is not None:
            attrib['lower'] = limit.lower
        if limit.upper is not None:
            attrib['upper'] = limit.upper
        return attrib

    def export_dynamics(dynamics):
        attrib = {}
        if dynamics is None:
            return attrib
        if dynamics.damping is not None:
            attrib['damping'] = dynamics.damping
        if dynamics.friction is not None:
            attrib['friction'] = dynamics.friction
        return attrib

    def export_joint(joint):
        attrib = {
            'name': joint.name,
            'type': joint.type,
            'parent': joint.parent,
            'link': joint.child,
        }
        origin = export_origin(joint.origin)
        axis = export_axis(joint.axis)
        limit = export_limit(joint.limit)
        dynamics = export_dynamics(joint.dynamics)
        return dict(**attrib, origin=origin, axis=axis, limit=limit, dynamics=dynamics)

    def export_color(color):
        if color is None:
            return {}
        return dict(rgba=color.rgba.tolist())

    def export_texture(texture):
        if texture is None:
            return {}
        return dict(filename=texture.filename)

    def export_material(material):
        if material is None:
            return {}
        attrib = {'name': material.name} if material.name is not None else {}
        color = export_color(material.color)
        texture = export_texture(material.texture)
        return dict(**attrib, color=color, texture=texture)

    # from yourdfpy import URDF
    urdf = URDF.load(urdf_path) if urdf is None else urdf
    robot = urdf.robot
    attrib = {'name': robot.name}
    links = [export_link(link) for link in robot.links]
    joints = [export_joint(joint) for joint in robot.joints]
    materials = [export_material(material) for material in robot.materials]
    return dict(**attrib, links=links, joints=joints, materials=materials)
