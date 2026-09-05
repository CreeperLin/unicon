"""ARX LIFT2 whole-body hardware adapter for unicon.

Drives the complete robot -- both arms, the lift lead-screw, pan/tilt head,
and the 3-wheel omni chassis -- from a single set of joint arrays.

Each subsystem lives on its own CAN bus:

    can1  -> left arm       (<arm_sdk>.InterfacesPy, direct binding)
    can3  -> right arm      (<arm_sdk>.InterfacesPy, direct binding)
    can5  -> chassis        (arx_lift_python.LiftHeadControlLoop)
                              - lift lead-screw (joint_lift)
                              - head yaw / pitch (joint_h_1, joint_h_2)
                              - 3 omni wheels    (joint_wheel_1..3)

The arm driver is the **underlying** ``arx_<model>_python.InterfacesPy`` C++
binding (e.g. ``arx_r5_python`` for the default R5 arms, ``arx_x5_python`` for
X5, ``arx_x7_python`` for X7S) used directly -- the ``bimanual.SingleArm``
Python wrapper is bypassed. The arm model is selected by ``arm_model``; the
underlying pybind11 API is identical across models, so the binding is used
interchangeably. Its
``set_joint_positions`` takes a 7-element vector where index 6 is the gripper
in hardware units [0, 5] -- no separate ``set_catch`` / ``set_gripper_pos``
call is needed. The URDF carries two symmetric finger joints per gripper
(``left_catch_joint1`` range (-0.0445, 0), ``left_catch_joint2`` range
(0, 0.0445), both 0 = closed); the adapter collapses those into the single
hardware gripper value and expands it back on read.

**CAN bring-up is fully automatic.** When ``auto_can_setup=True`` (default),
the adapter installs the vendor udev rules (idempotently) and spawns
``slcand`` for each of can1/can3/can5 at init, replacing the manual
``sudo ./arx_canX.sh &`` pre-launch step. Requires root or passwordless
sudo. Disable by passing ``auto_can_setup=False`` (then you must bring up
the buses yourself, exactly as before).

Joint layout is derived from the URDF at call time via ``_load_layout``.
``states_q_ctrl`` length must match the URDF's actuated DOF count.

Wheel joints in ``states_qd_ctrl`` are **angular velocities** (rad/s);
the chassis driver only supports velocity control via ``set_wheel_vel``
(which always takes 4 args -- the 4th is unused on LIFT2's 3-wheel hardware
and gets padded with 0.0). All other joints are positions in radians (lift
is exposed in metres and converted to motor revolutions internally, factor
1/41.54).

**Chassis base velocity** can alternatively be commanded as a 6-vector
``states_cmd_vel = [vx, vy, vz, wx, wy, wz]`` (linear x/y/z + angular
roll/pitch/yaw, m/s and rad/s). This routes to ``set_chassis_cmd`` instead
of ``set_wheel_vel``. The omni base only realizes planar motion, so indices
[2,3,4] (vz, roll, pitch) are silently dropped (one-shot warning if
non-zero); indices [0,1,5] map to ``(vx, vy, wz/yaw)``. Per vendor
convention (lift_controller.cpp), the two paths are mutually exclusive --
when ``states_cmd_vel`` is provided, the wheel command is zeroed, and vice
versa. The ``chassis_mode`` flag (1=enabled, 2=idle) is always carried via
``set_chassis_cmd`` regardless of which path is active.

Building requirements:
  * arx_lift_python pybind11 module built against the ROS2 tree's
    libarx_lift_src.so + headers (see lift_control_loop.cpp build comment).
  * arx_<model>_python module built via ``bimanual/build.sh`` from the
    matching arm SDK (R5: ARX/R5/py/ARX_R5_python, X5: ARX/ARX_X5/py/
    arx_x5_python, X7S: ARX/X7s/x7s/py/ARX_x7_python). Set arm_sdk_root
    if it isn't on the default search path.
"""
import os
import re
import time
from typing import Optional

import numpy as np

from unicon.utils import find, get_ctx, list2slice, validate_sudo, cmd
from unicon.utils.numpy import mat2quat, quat2mat

# --------------------------------------------------------------------------- #
# Module-level defaults -- previously lived in unicon/defs/lift2.py.          #
# Override any of them at the cb_lift2_recv_send_close(...) call site.        #
# --------------------------------------------------------------------------- #

# Hardware constants the URDF cannot express.
_DEFAULT_NUM_WHEELS     = 3
# _DEFAULT_LIFT_M_PER_REV = 1.0 / 41.54
_DEFAULT_LIFT_REV_PER_M = 22.056826744186047

# URDF catch_joint1/2 are symmetric ±0.0445; collapsed to one hardware value
# (signed: 0 = closed, magnitude scales with openness, sign = direction).
_DEFAULT_GRIPPER_URDF_RANGE = 0.0445

# Pybind11 module name by arm model. All three SDKs expose the same C++ API
# (InterfacesPy ctor, arx_x, set_joint_positions, set_arm_status, set_catch,
# get_joint_positions / get_joint_velocities / get_joint_currents); only the
# import name differs.
_ARM_MODULE_BY_MODEL = {
    'r5':  'arx_r5_python',
    'x5':  'arx_x5_python',
    'x7s': 'arx_x7_python',
}

# URDF filename by (arm_model, arm_type). Each arm SDK ships its URDFs in
# bimanual/script/; the int selects the hardware variant (left/right/master).
# Mirrors the if/elif chain in each SDK's bimanual/script/single_arm.py.
_ARM_URDF_BY_MODEL_AND_TYPE = {
    'r5':  {0: 'X5liteaa0.urdf', 1: 'R5_master.urdf'},
    'x5':  {0: 'x5.urdf',        1: 'x5_master.urdf', 2: 'x5_2025.urdf'},
    'x7s': {0: 'X7Sleft1.urdf',  1: 'x7sRIGHT.urdf'},
}

# Signed "fully open" hardware gripper value by (arm_model, arm_type).
# x5_2025 closes in the negative direction; everything else is +5.
_ARM_GRIPPER_HW_OPEN = {
    ('r5',  0): 5.0,
    ('r5',  1): 5.0,
    ('x5',  0): 5.0,
    ('x5',  1): 5.0,                     # x5_master: assumed same as x5
    ('x5',  2): -3.14,                   # x5_2025: negative direction
    ('x7s', 0): 5.0,
    ('x7s', 1): 5.0,
}

# Joint routing (first match wins).
_DEFAULT_JOINT_CATEGORIES = [
    ('left_arm',      ['left_arm']),
    ('right_arm',     ['right_arm']),
    ('left_gripper',  ['left_catch']),
    ('right_gripper', ['right_catch']),
    ('lift',          ['joint_lift']),
    ('head_yaw',      ['joint_h_1', 'head_yaw']),
    ('head_pitch',    ['joint_h_2', 'head_pitch']),
    ('wheel',         ['wheel']),
]


# --------------------------------------------------------------------------- #
# URDF-driven joint layout                                                     #
# --------------------------------------------------------------------------- #
def _categorize_joint(name: str, categories) -> Optional[str]:
    """Return the first category whose substring pattern matches ``name``."""
    for category, patterns in categories:
        if any(p in name for p in patterns):
            return category
    return None


def _gripper_urdf_to_hw(catch_vals, hw_open, urdf_range=_DEFAULT_GRIPPER_URDF_RANGE):
    # mag = float(np.mean(np.abs(np.asarray(catch_vals, dtype=float))))
    mag = np.abs(catch_vals[0])
    return mag / urdf_range * hw_open


def _gripper_hw_to_urdf(hw_value, hw_open, urdf_range=_DEFAULT_GRIPPER_URDF_RANGE):
    mag = abs(float(hw_value)) / abs(hw_open) * urdf_range
    return mag


def _mat_to_pose_xyzwxyz(T):
    """Convert a 4x4 homogeneous transform to the vendor EEF pose format.

    Returns ``[x, y, z, w, qx, qy, qz]`` -- position followed by a w-first
    quaternion -- matching what ``InterfacesPy.set_ee_pose`` expects.
    """
    qx, qy, qz, qw = mat2quat(np.asarray(T, dtype=float))
    return [*T[:3, 3].tolist(), float(qw), float(qx), float(qy), float(qz)]


def _load_layout(
    joint_categories = None,
    num_wheels: int = _DEFAULT_NUM_WHEELS,
) -> dict:
    """Build the index layout from the URDF.

    Prefers ``get_ctx()['robot_def']['DOF_NAMES']`` (already loaded by the
    runner); falls back to calling ``parse_urdf`` directly if no context is
    set or the context doesn't carry DOF_NAMES.

    Returns a dict with keys:
        dof_names          : list[str]   -- full DOF_NAMES from URDF
        num_dofs           : int
        left_arm_slice     : slice | None
        right_arm_slice    : slice | None
        left_gripper_slice : slice | None   (optional -- may be None)
        right_gripper_slice: slice | None   (optional -- may be None)
        lift_idx           : int | None
        head_yaw_idx       : int | None
        head_pitch_idx     : int | None
        wheel_slice        : slice | None
        categories         : dict[name -> category]
    """
    if joint_categories is None:
        joint_categories = _DEFAULT_JOINT_CATEGORIES

    dof_names = None
    ctx = get_ctx()
    robot_def = ctx.get('robot_def') if ctx else None
    dof_names = robot_def.get('DOF_NAMES')

    # Categorize every joint name, building per-subsystem index lists.
    layout = {
        'dof_names': list(dof_names),
        'num_dofs':  len(dof_names),
        'categories': {},
        '_left_arm_idx':      [],
        '_right_arm_idx':     [],
        '_left_gripper_idx':  [],
        '_right_gripper_idx': [],
        '_wheel_idx':         [],
        'lift_idx':           None,
        'head_yaw_idx':       None,
        'head_pitch_idx':     None,
    }
    for i, name in enumerate(dof_names):
        cat = _categorize_joint(name, joint_categories)
        if cat is None:
            raise ValueError(
                f'joint {name!r} at index {i} does not match any category '
                f'in joint_categories={joint_categories!r}. '
                f'Add a substring pattern for this joint.')
        layout['categories'][name] = cat
        if   cat == 'left_arm':      layout['_left_arm_idx'].append(i)
        elif cat == 'right_arm':     layout['_right_arm_idx'].append(i)
        elif cat == 'left_gripper':  layout['_left_gripper_idx'].append(i)
        elif cat == 'right_gripper': layout['_right_gripper_idx'].append(i)
        elif cat == 'wheel':         layout['_wheel_idx'].append(i)
        elif cat == 'lift':
            if layout['lift_idx'] is not None:
                raise ValueError(
                    f'multiple lift joints found: {dof_names[layout["lift"]]} '
                    f'and {name}; tighten joint_categories')
            layout['lift_idx'] = i
        elif cat == 'head_yaw':
            if layout['head_yaw_idx'] is not None:
                raise ValueError(f'multiple head_yaw joints: {name!r}')
            layout['head_yaw_idx'] = i
        elif cat == 'head_pitch':
            if layout['head_pitch_idx'] is not None:
                raise ValueError(f'multiple head_pitch joints: {name!r}')
            layout['head_pitch_idx'] = i

    # Convert index lists to slices (requires contiguity within each arm).
    _left_arm_idx = layout.pop('_left_arm_idx')
    _left_arm_idx = sorted(_left_arm_idx, key=lambda x: dof_names[x])
    _right_arm_idx = layout.pop('_right_arm_idx')
    _right_arm_idx = sorted(_right_arm_idx, key=lambda x: dof_names[x])
    layout['left_arm_slice']      = list2slice(_left_arm_idx)
    layout['right_arm_slice']     = list2slice(_right_arm_idx)
    layout['left_gripper_slice']  = list2slice(layout.pop('_left_gripper_idx'))
    layout['right_gripper_slice'] = list2slice(layout.pop('_right_gripper_idx'))
    layout['wheel_slice']         = list2slice(layout.pop('_wheel_idx'))

    # Validate expected subsystems are present (arms/lift/head/wheels required,
    # grippers optional -- some URDFs may not have them).
    missing = []
    if layout['left_arm_slice']  is None: missing.append('left_arm')
    if layout['right_arm_slice'] is None: missing.append('right_arm')
    if layout['lift_idx']        is None: missing.append('lift')
    if layout['head_yaw_idx']    is None: missing.append('head_yaw')
    if layout['head_pitch_idx']  is None: missing.append('head_pitch')
    if missing:
        print(f'URDF is missing expected joint categories: {missing}. '
        f'Checked patterns: {joint_categories!r}')
    n_wheels_found = (layout['wheel_slice'].stop - layout['wheel_slice'].start
                      if layout['wheel_slice'] else 0)
    if n_wheels_found != num_wheels:
        raise ValueError(
            f'expected {num_wheels} wheel joints in URDF, found '
            f'{n_wheels_found}: '
            f'{[n for n,c in layout["categories"].items() if c=="wheel"]}')

    def _slice_len(s):
        return (s.stop - s.start) if s is not None else 0
    print(f'[lift2] URDF layout: {layout["num_dofs"]} DOFs, '
          f'left_arm={layout["left_arm_slice"]} '
          f'({_slice_len(layout["left_arm_slice"])}dof) '
          f'+ left_gripper={layout["left_gripper_slice"]} '
          f'({_slice_len(layout["left_gripper_slice"])}dof), '
          f'right_arm={layout["right_arm_slice"]} '
          f'({_slice_len(layout["right_arm_slice"])}dof) '
          f'+ right_gripper={layout["right_gripper_slice"]} '
          f'({_slice_len(layout["right_gripper_slice"])}dof), '
          f'lift={layout["lift_idx"]}, '
          f'head_yaw={layout["head_yaw_idx"]}, '
          f'head_pitch={layout["head_pitch_idx"]}, '
          f'wheels={layout["wheel_slice"]}')
    return layout


# --------------------------------------------------------------------------- #
# CAN bus bring-up (replaces vendor's ARX_CAN/arx_can/*.sh scripts)           #
# --------------------------------------------------------------------------- #
# Idempotent helpers that install the udev rules and spawn slcand for each
# bus. Skipped if the interface is already up. sudo is required -- callers
# without passwordless sudo will see a clear error.

def _can_interface_is_up(iface: str) -> bool:
    """True iff interface `iface` exists and its flags include UP."""
    try:
        out = cmd('ip -br link show', [iface], capture_output=True).stdout.strip()
        if not out:
            return False
        # Output looks like: "can5             CAN          UNKNOWN,<UP,LOWER_UP>  ..."
        # or just "can5             CAN          DOWN        mtu 16 ..."
        parts = out.split()
        return len(parts) >= 3 and 'UP' in parts[2]
    except Exception:
        return False


def _bring_up_can(can_device: str, can_interface: str,
                  speed_idx: int = 8) -> str:
    """Ensure `can_interface` (e.g. can5) is bound to `can_device` (/dev/arxcan5).

    Spawns slcand with the same flags the vendor's bash scripts use
    (``-o -f -s8``: open immediately, fork to background, bitrate index 8).

    Idempotent: if the interface is already UP, does nothing.
    """
    if _can_interface_is_up(can_interface):
        return can_interface

    if not os.path.exists(can_device):
        raise FileNotFoundError(
            f'{can_device} does not exist; check the udev rules (expected '
            f'symlink for USB CAN dongle with ARX vendor id 16d0:117e)')

    validate_sudo(interactive=True)

    # Kill any stale slcand bound to this device before restarting.
    cmd('sudo pkill -9 -f', [rf'slcand.*{can_device}'])
    time.sleep(0.3)

    # Spawn slcand. -f makes it daemonise (matching vendor behaviour).
    cmd(f'sudo slcand -o -f -s{speed_idx}', [can_device, can_interface])
    time.sleep(0.5)

    # Bring the interface up (modern equivalent of `ifconfig canN up`).
    cmd('sudo ip link set', [can_interface], 'up')

    if not _can_interface_is_up(can_interface):
        raise RuntimeError(
            f'{can_interface} failed to come up after slcand + ip link set up; '
            f'check `dmesg | tail` and that {can_device} is the right dongle')
    print(f'[lift2] {can_interface} up (via {can_device}, slcand -s{speed_idx})')
    return can_interface


def _bring_up_all_can_buses(buses, speed_idx: int = 8):
    """Bring up a list of (device, interface) pairs.

    Returns the list of interfaces successfully brought up. Raises on hard
    failure (interface won't come up).
    """
    for device, iface in buses:
        _bring_up_can(device, iface, speed_idx=speed_idx)


def _teardown_can_interface(iface: str):
    """Take an interface down and kill its slcand. Best-effort."""
    if not _can_interface_is_up(iface):
        return
    try:
        cmd('sudo ip link set', [iface], 'down')
        cmd('sudo pkill -9 -f', [rf'slcand.*{iface}'])
        print(f'[lift2] {iface} taken down')
    except Exception as exc:
        print(f'[lift2] teardown of {iface} failed: {exc}')


# --------------------------------------------------------------------------- #
# SDK discovery                                                                #
# --------------------------------------------------------------------------- #

def _resolve_arm_sdk(arm_sdk_root: Optional[str], model: str) -> str:
    """Locate the bimanual arm SDK root (parent of bimanual/)."""
    if arm_sdk_root is not None:
        arm_sdk_root = os.path.expanduser(arm_sdk_root)
        if os.path.isdir(arm_sdk_root):
            return arm_sdk_root
        raise FileNotFoundError(f'arm_sdk_root not found: {arm_sdk_root}')
    # Default search patterns for the various arm SDK layouts shipped by ARX.
    patterns_by_model = {
        'x5':  ['arx_x5_python', 'arx_l5pro_python'],
        'x7s': ['arx_x7_python', 'arx_x7s_python'],
        'r5':  ['arx_r5_python', 'ARX_R5_python'],
    }
    for pat in patterns_by_model.get(model.lower(), [f'arx_{model}_python']):
        # Look for the bimanual package directory itself.
        hits = find('~', name='bimanual', path=f'*{pat}*', quit=False)
        if hits:
            # Climb to the parent of bimanual/ -- that's the importable root.
            p = os.path.abspath(hits[0])
            root = os.path.abspath(os.path.join(p, '..'))
            print(f'found ARX-{model.upper()} arm sdk via "{pat}" at {root}')
            return root
    raise FileNotFoundError(
        f'ARX {model.upper()} arm sdk not found, set arm_sdk_root explicitly')


def _mute_arm_sdk_banner(so_path, backup=True):
    import shutil
    with open(so_path, 'rb') as f:
          data = bytearray(f.read())
    banner = 'ARX方舟无限'.encode('utf-8')
    bnr = data.find(banner)
    if bnr < 0:
        nulled_marker = b'\x00' + banner[1:]
        if data.find(nulled_marker) >= 0:
            print('already patched')
        else:
            print('banner string not found')
        return

    data[bnr] = 0x00

    # Find the unique LEA rsi, [rip+disp32] referencing the banner.
    lea = None
    pos = data.find(b'\x48\x8d\x35')
    while pos >= 0:
        disp = int.from_bytes(data[pos+3 : pos+7], 'little', signed=True)
        if (pos + 7 + disp) == bnr:
            lea = pos
            break
        pos = data.find(b'\x48\x8d\x35', pos + 1)
    if lea is None:
        print('no LEA references the banner')
        return
    # --- Pattern A: R5 / X7S — JNE short at LEA-2 ---
    if data[lea - 2] == 0x75:
        patch_off = lea - 2
        old, new = 0x75, 0xeb
        data[patch_off] = new
        print(f'pattern A (R5/X7S): flipped JNE -> JMP at 0x{patch_off:x}')
        patched = True

    # --- Pattern B: X5 — JBE near (0f 86) somewhere in LEA-200..LEA-10 ---
    search_start = max(0, lea - 200)
    cerr_load = data.rfind(b'\x4c\x8b\x3d', search_start, lea)
    if cerr_load >= 0:
        # The JBE should jump to cerr_load. Scan backward for 0f 86 <disp32>
        # where disp32 resolves to cerr_load.
        for candidate in range(search_start, lea - 10):
            if data[candidate] == 0x0f and data[candidate + 1] == 0x86:
                disp = int.from_bytes(data[candidate+2 : candidate+6],
                                      'little', signed=True)
                target = (candidate + 6) + disp
                if target == cerr_load:
                    # Patch: NOP the 6-byte JBE.
                    for i in range(6):
                        data[candidate + i] = 0x90
                    print(f'pattern B (X5): NOPed JBE at 0x{candidate:x} (6 bytes)')
                    patched = True

    if not patched:
        print(f'unrecognized pattern around banner LEA at 0x{lea:x}')
        return

    if backup and not os.path.exists(so_path + '.bak'):
        shutil.copy2(so_path, so_path + '.bak')
    with open(so_path, 'wb') as f:
        f.write(data)
    print(f'patched {so_path} to mute the banner')


def cb_lift2_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_q,
    states_qd,
    states_quat=None,
    states_q_tau=None,
    states_qd_ctrl=None,                # wheel velocity commands live here
    states_cmd_vel=None,                # chassis base velocity cmd [vx,vy,vz,wx,wy,wz]
    states_lin_acc=None,
    states_x_ctrl=None,
    states_x=None,
    # SDK roots
    sdk_root: Optional[str] = None,        # ARX_LIFT_python (chassis)
    arm_sdk_root: Optional[str] = None,    # bimanual arm SDK
    arm_model: str = 'r5',                 # 'r5' (default on LIFT2), 'x5', 'x7s'
    arm_type: int = 0,                     # hardware variant int (per-model:
                                           #   r5:  0=X5liteaa0, 1=R5_master
                                           #   x5:  0=x5, 1=x5_master, 2=x5_2025
                                           #   x7s: 0=left, 1=right)
    # URDF / joint routing (override module defaults if needed)
    joint_categories: Optional[list] = None,    # falls back to _DEFAULT_JOINT_CATEGORIES
    num_wheels: int = _DEFAULT_NUM_WHEELS,
    lift_rev_per_m: float = _DEFAULT_LIFT_REV_PER_M,
    # CAN buses
    can_left: str = 'can1',
    can_right: str = 'can3',
    can_chassis: str = 'can5',
    # CAN bring-up (replaces vendor's ARX_CAN/arx_can/*.sh scripts)
    auto_can_setup: bool = True,
    can_speed_idx: int = 8,                     # slcand -s8 (matches vendor)
    can_device_template: str = '/dev/arxcan{}', # {iface_num} -> device path
    teardown_can_on_close: bool = False,        # leave CAN up between runs by default
    # Driver config
    robot_type: str = 'LIFTS',             # 'LIFT', 'LIFTS' (=LIFT2) or 'X7S'
    chassis_mode: int = 1,                 # 1=chassis vel, 2=idle, 3=wheel vel
    # Safety
    safe_mode_on_close: bool = True,
    use_protect_on_close: bool = True,
    arm_max_vel: float = 500.,
    arm_max_acc: float = 2000.,
    use_builtin_ik: bool = False,
    use_builtin_fk: bool = False,
    inv_x_reset: bool = False,
    gravity_comp: bool = False,
    kp=None,
    kd=None,
):
    """Return (cb_recv, cb_send, cb_close) driving the full LIFT2 body.

    Joint layout (DOF_NAMES, slices, indices) is derived from the URDF at
    call time via ``_load_layout``. The caller's ``states_q_ctrl`` length
    must equal the URDF's actuated DOF count -- if it doesn't, the error
    message includes both numbers so you can adjust either side.

    When ``auto_can_setup=True`` (default), the adapter also installs the
    vendor udev rules (idempotently) and spawns ``slcand`` for each of
    ``can_left`` / ``can_right`` / ``can_chassis`` before constructing the
    driver objects. This replaces the manual ``sudo ./arx_canX.sh &``
    pre-launch step. Requires root or passwordless sudo.
    """
    from unicon.utils import find_import_ext, expect, get_ctx
    from unicon.utils.numpy import mat4_eye, mat2rpy
    ctx = get_ctx()
    robot_def = ctx.get('robot_def')
    # ---- bring up CAN buses (replaces vendor's ARX_CAN/*.sh scripts) --------
    if auto_can_setup:
        # Extract the numeric suffix from each iface name to map to
        # /dev/arxcanN. e.g. 'can5' -> '/dev/arxcan5'.
        def _dev_for(iface_name):
            m = re.search(r'(\d+)$', iface_name)
            if not m:
                raise ValueError(f'cannot derive device from {iface_name!r}; '
                                 f'pass can_device_template=... explicitly')
            return can_device_template.format(m.group(1))

        _bring_up_all_can_buses(
            buses=[
                (_dev_for(can_left),   can_left),
                (_dev_for(can_right),  can_right),
                (_dev_for(can_chassis), can_chassis),
            ],
            speed_idx=can_speed_idx,
        )

    # ---- resolve SDKs -------------------------------------------------------
    sdk_root = '~' if sdk_root is None else os.path.expanduser(sdk_root)
    try:
        arx = find_import_ext('arx_lift_python', root=sdk_root)
    except Exception as exc:
        raise ImportError(
            f'failed to import arx_lift_python from {sdk_root}; '
            f'add <sdk_root>/lift_controller/api/arx_lift_src to LD_LIBRARY_PATH') from exc

    arm_sdk_root = _resolve_arm_sdk(arm_sdk_root, arm_model)
    print(f'using ARX-{arm_model.upper()} arm sdk at {arm_sdk_root}, '
          f'can_ports={can_left}/{can_right}, type={arm_type}')
    lib_dir = os.path.join(arm_sdk_root, 'bimanual', 'api')
    so_built = os.path.join(lib_dir, f'arx_{arm_model}_src', f'libarx_{arm_model}_src.so')
    _mute_arm_sdk_banner(so_built)

    try:
        # Use the underlying pybind11 binding directly -- the SingleArm
        # wrapper adds nothing we need (its set_gripper_pos / set_catch_pos
        # map to the separate set_catch call, but the underlying
        # set_joint_positions accepts 7 values where index 6 IS the gripper
        # per the user spec). The module name differs per arm model but the
        # C++ API is identical (InterfacesPy ctor, arx_x, set_joint_positions,
        # set_arm_status, set_catch, get_joint_positions/velocities/currents).
        arm_module_name = _ARM_MODULE_BY_MODEL.get(arm_model.lower())
        if arm_module_name is None:
            raise ValueError(
                f'unsupported arm_model {arm_model!r}; supported models: '
                f'{sorted(_ARM_MODULE_BY_MODEL)}')
        # arx_arm = importlib.import_module(arm_module_name)
        arx_arm = find_import_ext(arm_module_name, root=lib_dir)
    except Exception as exc:
        raise ImportError(
            f'failed to import {arm_module_name} from '
            f'{arm_sdk_root}/bimanual/api; build the SDK first (run build.sh '
            f'in bimanual/), and ensure {lib_dir} is on LD_LIBRARY_PATH'
        ) from exc

    # Resolve the arm URDF filename by (arm_model, arm_type). Mirrors the
    # if/elif chain in each SDK's bimanual/script/single_arm.py.
    arm_urdf_by_type = _ARM_URDF_BY_MODEL_AND_TYPE.get(arm_model.lower(), {})
    arm_urdf_name = arm_urdf_by_type.get(arm_type)
    if arm_urdf_name is None:
        raise ValueError(
            f'no URDF mapping for arm_model={arm_model!r}, arm_type={arm_type}; '
            f'known mappings: {_ARM_URDF_BY_MODEL_AND_TYPE!r}')
    arm_urdf_path = os.path.join(arm_sdk_root, 'bimanual', 'script', arm_urdf_name)
    if not os.path.isfile(arm_urdf_path):
        raise FileNotFoundError(
            f'arm URDF not found at {arm_urdf_path}; check arm_sdk_root '
            f'({arm_sdk_root}), arm_model ({arm_model}) and arm_type ({arm_type})')

    # ---- validate DOF layout (URDF-driven) ---------------------------------
    layout = _load_layout(
        joint_categories=joint_categories,
        num_wheels=num_wheels,
    )
    n_q = len(states_q_ctrl) if states_q_ctrl is not None else 0
    if n_q != layout['num_dofs']:
        raise ValueError(
            f'states_q_ctrl has length {n_q}, but the URDF defines '
            f'{layout["num_dofs"]} actuated DOFs ({layout["dof_names"]}). '
            f'Either update the URDF or adjust the state array size.')
    left_arm_slice       = layout['left_arm_slice']
    right_arm_slice      = layout['right_arm_slice']
    left_gripper_slice   = layout['left_gripper_slice']
    right_gripper_slice  = layout['right_gripper_slice']
    lift_idx             = layout['lift_idx']
    head_yaw_idx         = layout['head_yaw_idx']
    head_pitch_idx       = layout['head_pitch_idx']
    wheel_slice          = layout['wheel_slice']
    n_arms               = abs(left_arm_slice.stop - left_arm_slice.start)

    # Last commanded gripper values (hardware units, [0, 5]). Echoed back into
    # states_q[gripper_slice] in cb_recv only if the SDK doesn't return a
    # gripper position; the underlying InterfacesPy.get_joint_positions()
    # returns 7 values per arm including the gripper, so the echo is just a
    # fallback.

    lift_m_per_rev = 1 / lift_rev_per_m

    # Cache gripper mapping params (model+type-aware, immutable during session).
    hw_open = _ARM_GRIPPER_HW_OPEN.get((arm_model.lower(), arm_type), 5.0)

    # ---- instantiate driver objects ---------------------------------------
    _rt = robot_type.upper()
    robot_type_enum = {
        'LIFT':  arx.RobotType.LIFT,
        # 'LIFTS': arx.RobotType.LIFTS,
        'LIFTS': arx.RobotType.X7S,
        'X7S':   arx.RobotType.X7S,
    }.get(_rt, arx.RobotType.LIFT)
    if _rt not in ('LIFT', 'LIFTS', 'X7S'):
        print(f'warning: unknown robot_type {robot_type!r}, defaulting to LIFT')

    expect(chassis_mode != 1 or states_cmd_vel is not None)
    expect(chassis_mode != 3 or states_qd_ctrl is not None)
    expect(use_builtin_ik is False or states_x_ctrl is not None)

    print(f'[lift2] chassis driver: can_port={can_chassis}, robot_type={robot_type_enum}')
    chassis = arx.LiftHeadControlLoop(can_chassis, robot_type_enum)
    print(f'[lift2] arm driver: InterfacesPy({arm_urdf_path!r}, {can_left}, {can_right}, {arm_type})')
    _DEFAULT_ARM_ARX_X = (arm_max_vel, arm_max_acc, 10.0)
    left_arm = arx_arm.InterfacesPy(arm_urdf_path, can_left, arm_type)
    left_arm.arx_x(*_DEFAULT_ARM_ARX_X)
    right_arm = arx_arm.InterfacesPy(arm_urdf_path, can_right, arm_type)
    right_arm.arx_x(*_DEFAULT_ARM_ARX_X)

    print('resetting chassis')
    chassis.set_waist_pos(0.0)
    chassis.set_wheel_vel(0.0, 0.0, 0.0, 0.0)
    chassis.set_chassis_cmd(0.0, 0.0, 0.0, 2)  # idle mode
    chassis.update()
    chassis.write()
    time.sleep(1)
    chassis.set_chassis_cmd(0.0, 0.0, 0.0, chassis_mode)
    chassis.set_height(0.0)
    chassis.set_head_yaw(0.0)
    chassis.set_head_pitch(0.0)
    chassis.update()
    chassis.write()
    chassis.get_height()
    # for _ in range(50):
    #     chassis.update(); chassis.write()

    ARM_MODE_FREE = 0
    ARM_MODE_HOME = 1
    ARM_MODE_PROTECT = 2
    ARM_MODE_GRAVITY_COMP = 3
    ARM_MODE_EEF_CONTROL = 4
    ARM_MODE_POSITION_CONTROL = 5

    print('resetting arms')
    left_arm.set_ee_pose([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # zero pose
    right_arm.set_ee_pose([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # zero pose
    left_arm.set_joint_positions([0.0] * 7)
    right_arm.set_joint_positions([0.0] * 7)
    if gravity_comp:
        left_arm.set_arm_status(ARM_MODE_GRAVITY_COMP)
        right_arm.set_arm_status(ARM_MODE_GRAVITY_COMP)
    elif use_builtin_ik:
        left_arm.set_arm_status(ARM_MODE_EEF_CONTROL)
        right_arm.set_arm_status(ARM_MODE_EEF_CONTROL)
    else:
        left_arm.set_arm_status(ARM_MODE_POSITION_CONTROL)
        right_arm.set_arm_status(ARM_MODE_POSITION_CONTROL)

    left_arm.get_joint_positions()
    right_arm.get_joint_positions()

    states_qd[:] = 0.0

    # Resolve EEF link indices for use_builtin_ik. x_ctrl_eef_inds is set by
    # run.py from x_ctrl_eef_names (config); convention is [left_ee, right_ee].
    x_ctrl_eef_inds = ctx.get('x_ctrl_eef_inds')
    x_idx_left_ee, x_idx_right_ee = None, None
    if use_builtin_ik:
        expect(x_ctrl_eef_inds is not None and len(x_ctrl_eef_inds) == 2)
        print(f'[lift2] use_builtin_ik: left_ee=link{x_ctrl_eef_inds[0]}, right_ee=link{x_ctrl_eef_inds[1]}')
        LINK_NAMES = robot_def['LINK_NAMES']
        inds = x_ctrl_eef_inds
        inds = inds if any(x in LINK_NAMES[inds[0]].lower() for x in ['left', 'l_']) else inds[::-1]
        x_idx_left_ee, x_idx_right_ee = inds
        x_reset = ctx.get('x_reset')
        x_reset_left, x_reset_right = [mat4_eye() for _ in range(2)]
        if x_reset is not None:
            x_reset_left = x_reset[x_idx_left_ee]
            x_reset_right = x_reset[x_idx_right_ee]
        print('x_reset_left', x_reset_left[:3, 3], mat2rpy(x_reset_left[:3, :3]))
        print('x_reset_right', x_reset_right[:3, 3], mat2rpy(x_reset_right[:3, :3]))
        x_reset_left_inv = np.linalg.inv(x_reset_left)
        x_reset_right_inv = np.linalg.inv(x_reset_right)
        if inv_x_reset is False:
            expect(np.all(np.isclose(x_reset_left, mat4_eye())))
            expect(np.all(np.isclose(x_reset_right, mat4_eye())))

    # ----------------------------------------------------------------------- #
    def cb_recv():
        # ---- chassis: refresh buffer, read lift/head/wheels/IMU -----------
        try:
            chassis.read()
        except Exception as exc:
            print('recv_read', f'chassis.read() failed: {exc}')

        if states_q is not None:
            try:
                if lift_idx is not None:
                    h_rev = chassis.get_height()
                    states_q[lift_idx] = h_rev * lift_m_per_rev
                if head_yaw_idx is not None:
                    states_q[head_yaw_idx]   = chassis.get_head_yaw()
                if head_pitch_idx is not None:
                    states_q[head_pitch_idx] = chassis.get_head_pitch()
                # Wheel positions are not tracked -- leave the slice at zero.
                # Velocities go into states_qd below.
            except Exception as exc:
                print('recv_chassis', f'failed to read chassis state: {exc}')

        # Wheel velocity feedback -> states_qd (not states_q).
        # Driver always returns 4 wheel vels; slice to URDF-defined count.
        if states_qd is not None and wheel_slice is not None:
            try:
                wheels_all = chassis.get_wheel_vel()
                states_qd[wheel_slice] = wheels_all[:num_wheels]
            except Exception as exc:
                print('recv_wheel', f'failed to read wheel vel: {exc}')

        # arms + gripper: get_joint_positions returns 7 values; index 6 is
        # the gripper in hardware units. Split arm/gripper and expand the
        # gripper to URDF catch_joint1/2 via _gripper_hw_to_urdf().
        if states_q is not None:
            try:
                lq = left_arm.get_joint_positions()
                if lq is not None and len(lq) > n_arms:
                    states_q[left_arm_slice] = lq[:n_arms]
                    if left_gripper_slice is not None:
                        v = _gripper_hw_to_urdf(lq[n_arms], hw_open)
                        states_q[left_gripper_slice] = v
            except Exception as exc:
                print('recv_larm', f'failed to read left arm: {exc}')
            try:
                rq = right_arm.get_joint_positions()
                if rq is not None and len(rq) > n_arms:
                    states_q[right_arm_slice] = rq[:n_arms]
                    if right_gripper_slice is not None:
                        v = _gripper_hw_to_urdf(rq[n_arms], hw_open)
                        states_q[right_gripper_slice] = v
            except Exception as exc:
                print('recv_rarm', f'failed to read right arm: {exc}')

        # Velocities: arms + gripper from the arm SDK; wheels from chassis
        # driver (written above). Lift/head have no velocity sensor -- zero.
        if states_qd is not None:
            try:
                lqd = left_arm.get_joint_velocities()  # 7-vector
                if lqd is not None and len(lqd) >= n_arms:
                    states_qd[left_arm_slice] = lqd[:n_arms]
            except Exception:
                pass
            try:
                rqd = right_arm.get_joint_velocities()
                if rqd is not None and len(rqd) >= n_arms:
                    states_qd[right_arm_slice] = rqd[:n_arms]
            except Exception:
                pass

        # Torques / currents: arms only (chassis doesn't expose them).
        # get_joint_currents() returns 7 values; drop the gripper current.
        if states_q_tau is not None:
            try:
                lc = left_arm.get_joint_currents()
                if lc is not None and len(lc) >= n_arms:
                    states_q_tau[left_arm_slice] = lc[:n_arms]
            except Exception:
                pass
            try:
                rc = right_arm.get_joint_currents()
                if rc is not None and len(rc) >= n_arms:
                    states_q_tau[right_arm_slice] = rc[:n_arms]
            except Exception:
                pass

        # IMU -> rpy, ang_vel, lin_acc (chassis driver supplies these).
        if states_rpy is not None:
            try:
                states_rpy[:] = chassis.get_orientation()
            except Exception as exc:
                print('recv_rpy', f'failed to read IMU rpy: {exc}')
        if states_ang_vel is not None:
            try:
                states_ang_vel[:] = chassis.get_angular_vel()
            except Exception as exc:
                print('recv_ang', f'failed to read IMU ang vel: {exc}')
        if states_lin_acc is not None:
            try:
                states_lin_acc[:] = chassis.get_accel()
            except Exception as exc:
                print('recv_acc', f'failed to read IMU accel: {exc}')

        if states_x is not None and use_builtin_fk:
            try:
                lT = left_arm.get_ee_pose()
                rT = right_arm.get_ee_pose()
                quats = np.stack([lT[3:7], rT[3:7]], axis=0)
                mats = quat2mat(quats, w_first=True)
                states_x[x_idx_left_ee, :3, 3] = lT[:3]
                states_x[x_idx_left_ee, :3, :3] = mats[0]
                states_x[x_idx_right_ee, :3, 3] = rT[:3]
                states_x[x_idx_right_ee, :3, :3] = mats[1]
            except Exception as exc:
                print('recv_fk', f'failed to read FK: {exc}')

    # ----------------------------------------------------------------------- #
    def cb_send():
        try:
            # ---- arms: build 7-element vector [6 arm joints + 1 gripper] -
            # The underlying InterfacesPy.set_joint_positions takes 7 values;
            # index 6 is the gripper in hardware units [0, 5]. We collapse the
            # URDF's symmetric catch_joint1/2 (±0.0445) into that one value.
            l_joints = list(states_q_ctrl[left_arm_slice])
            r_joints = list(states_q_ctrl[right_arm_slice])

            if left_gripper_slice is not None:
                lg = _gripper_urdf_to_hw(states_q_ctrl[left_gripper_slice], hw_open)
                l_joints.append(lg)
            if right_gripper_slice is not None:
                rg = _gripper_urdf_to_hw(states_q_ctrl[right_gripper_slice], hw_open)
                r_joints.append(rg)

            # ---- dispatch arms: builtin IK (Cartesian) or joint-space ----
            # When use_builtin_ik=True and states_x_ctrl carries a non-zero
            # EEF target (T[3,3] != 0) for a given arm, route to the SDK's
            # internal IK via set_ee_pose + set_arm_status(4=END_CONTROL).
            # Otherwise fall back to set_joint_positions + set_arm_status
            # (5=POSITION_CONTROL). The gripper is part of the 7-element
            # joint vector in position mode, but must be sent separately via
            # set_catch in Cartesian mode.
            if use_builtin_ik:
                T_l = states_x_ctrl[x_ctrl_eef_inds[0]]
                T_r = states_x_ctrl[x_ctrl_eef_inds[1]]
                if inv_x_reset:
                    T_l = x_reset_left_inv @ T_l
                    T_r = x_reset_right_inv @ T_r
                if T_l[3, 3] != 0.0:
                    left_arm.set_ee_pose(_mat_to_pose_xyzwxyz(T_l))
                if T_r[3, 3] != 0.0:
                    right_arm.set_ee_pose(_mat_to_pose_xyzwxyz(T_r))
                if left_gripper_slice is not None:
                    left_arm.set_catch(l_joints[-1])
                if right_gripper_slice is not None:
                    right_arm.set_catch(r_joints[-1])
            else:
                left_arm.set_joint_positions(l_joints)
                right_arm.set_joint_positions(r_joints)

            # ---- chassis: lift / head / wheels ---------------------------
            if lift_idx is not None:
                lift_rev = float(states_q_ctrl[lift_idx]) * lift_rev_per_m
                chassis.set_height(lift_rev)
            if head_yaw_idx is not None:
                chassis.set_head_yaw(float(states_q_ctrl[head_yaw_idx]))
            if head_pitch_idx is not None:
                chassis.set_head_pitch(float(states_q_ctrl[head_pitch_idx]))
            # ---- chassis base velocity: two mutually exclusive paths -----
            # Per vendor lift_controller.cpp, setChassisCmd(vx,vy,wz,mode)
            # and setWheelVel(w1..w4) are both packed into the same CAN
            # frame every tick, but the firmware honors one based on mode.
            # Vendor convention is to zero whichever path is not actively
            # driven, so we never send two non-zero sources simultaneously.
            #
            # Path A -- states_cmd_vel (6-vector: lin x/y/z + ang r/p/y):
            #   The omni chassis only realizes planar motion, so we use
            #   indices [0,1,5] = (vx, vy, wz/yaw) and silently drop
            #   [2,3,4] (vz, roll, pitch) -- one-shot warning if non-zero.
            #
            # Path B -- states_qd_ctrl[wheel_slice] (per-wheel angular vel):
            #   Direct wheel command; set_chassis_cmd carries only the mode.
            if chassis_mode == 1:
                cv = list(states_cmd_vel)
                vx, vy, wz = float(cv[0]), float(cv[1]), float(cv[5])
                # print('set_chassis_cmd', vx, vy, wz, chassis_mode)
                chassis.set_chassis_cmd(vx, vy, wz, chassis_mode)
            elif chassis_mode == 3:
                # Per-wheel velocity command.
                ws = list(states_qd_ctrl[wheel_slice])
                chassis.set_wheel_vel(*ws, 0.0)

            chassis.update()
            chassis.write()
        except Exception as exc:
            print('send', f'failed to tick drivers: {exc}')

    # ----------------------------------------------------------------------- #
    def cb_close():
        if safe_mode_on_close:
            chassis.set_chassis_cmd(0.0, 0.0, 0.0, 2)   # idle
            chassis.set_wheel_vel(0.0, 0.0, 0.0, 0.0)
            for _ in range(50):                          # flush ~100 ms
                chassis.update(); chassis.write()
            # last_q_left = states_q[left_arm_slice]
            # last_q_right = states_q[right_arm_slice]
            # for i in range(50):
            #     left_arm.set_joint_positions(last_q_left * (i / 50.0))
            #     right_arm.set_joint_positions(last_q_right * (i / 50.0))
            #     left_arm.set_arm_status(ARM_MODE_POSITION_CONTROL)
            #     right_arm.set_arm_status(ARM_MODE_POSITION_CONTROL)
            #     time.sleep(0.02)
            left_arm.set_arm_status(ARM_MODE_HOME)
            time.sleep(1.0)
            right_arm.set_arm_status(ARM_MODE_HOME)
            time.sleep(1.0)
        if use_protect_on_close:
            chassis.protect()
            left_arm.set_arm_status(ARM_MODE_PROTECT)
            right_arm.set_arm_status(ARM_MODE_PROTECT)
            time.sleep(1.0)
            left_arm.set_arm_status(ARM_MODE_FREE)
            right_arm.set_arm_status(ARM_MODE_FREE)
        if teardown_can_on_close:
            # Tear down interfaces in reverse so the chassis (which
            # often is the active master for the body) goes last.
            for iface in (can_chassis, can_right, can_left):
                _teardown_can_interface(iface)

    if gravity_comp:
        cb_send = lambda: None  # no-op send if gravity compensation is on

    return cb_recv, cb_send, cb_close
