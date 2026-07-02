import socket
import time
import json

import numpy
from enum import Enum
from math import *
import struct

from fi_fsa_logger import logger

# if add to Robot-RCS project, uncomment this line!!!
# from fi_rcs_config import gl_rcs_config
from fi_fsa_config import gl_rcs_config


class FSAFunctionResult:
    SUCCESS = 0
    FAIL = -1
    RUNNING = 1
    PREPARE = 2
    EXECUTE = 3
    NOT_EXECUTE = 4
    TIMEOUT = 5


class FSAFlagState:
    CLEAR = 0
    SET = 1


class FSAActuatorType:
    TYPE_DEFAULT = 0x00000001


class FSAActuatorDirection:
    DIRECTION_NORMAL = 1
    DIRECTION_REVERSE = -1


class FSAActuatorReductionRatio:
    REDUCTION_RATIO_7 = 7
    REDUCTION_RATIO_30 = 30
    REDUCTION_RATIO_36 = 36
    REDUCTION_RATIO_50 = 50
    REDUCTION_RATIO_80 = 80
    REDUCTION_RATIO_100 = 100
    REDUCTION_RATIO_120 = 120


class FSAControlWord:
    NONE = 0
    CALIBRATE_PREPARE = 0xA0
    CALIBRATE_ADC = 0xA1
    CALIBRATE_MOTOR = 0xA2
    CALIBRATE_ENCODER = 0xA3
    CALIBRATE_DIRECTION = 0xA4
    CALIBRATE_OFFSET = 0xA5
    OPEN_LOOP_CONTORL = 0x04
    SERVO_OFF = 0x06
    SERVO_ON = 0x0F
    MOTION_ABS = 0x103F
    CLEAR_FAULT = 0x86


class FSAModeOfOperation:
    NONE = 0
    CURRENT_CLOSE_LOOP_CONTROL = 4
    VELOCITY_CONTROL = 3
    POSITION_CONTROL = 1
    TRAPEZOIDAL_CONTROL = 5


class FSAState:
    IDLE = 0
    INIT = 1
    NORMAL = 2
    FAULT = 3
    ENCODER_CAIL = 4


class FSAErrorCode(Enum):
    ERROR_NONE = 0x0
    ERROR_ADC_CAL_FAULT = 0x01
    ERROR_CAN_COM_TIMEOUT = 0x02
    ERROR_OVER_CURRENT = 0x04
    ERROR_OVER_VBUS = 0x08

    ERROR_UNDER_VBUS = 0x10
    ERROR_OVER_TEMP_TRIP = 0x20
    ERROR_OVER_TEMP_MOSFET = 0x40
    ERROR_OVER_PHASE_A_CURRENT = 0x80

    ERROR_OVER_PHASE_B_CURRENT = 0x100
    ERROR_OVER_PHASE_C_CURRENT = 0x200
    ERROR_OVER_HARD_PHASE_CURRENT = 0x400
    ERROR_OPD_FAULT = 0x800

    ERROR_ENCODER_NOT_CALIBRATED = 0x1000
    ERROR_ENCODER_LOSS = 0x2000
    ERROR_FLASH_ERROR = 0x4000
    ERROR_MOTOR_STALL = 0x8000

    ERROR_POSITION_LIMIT_ERROR = 0x10000
    ERROR_ENCODER_REVERSAL = 0x20000
    ERROR_MOTOR_TYPE_NULL = 0x40000
    ERROR_HARDWARE_TYPE_NULL = 0x80000


class FSAInputMode:
    INPUT_MODE_INACTIVE = 0
    INPUT_MODE_PASSTHROUGH = 1
    INPUT_MODE_VEL_RAMP = 2
    INPUT_MODE_POS_FILTER = 3
    INPUT_MODE_TRAP_TRAJ = 5
    INPUT_MODE_TORQUE_RAMP = 6


class FSAMotorType:
    TYPE_MOTOR_NULL = 0
    TYPE_JIOALONG = 1
    FSA130_20V0 = 2
    FSA100_15V0 = 3
    FSA80_10V0 = 4
    FSA60_08V0 = 5
    FSA36_08V0 = 6
    FSA25_08V0 = 7
    FSA36_10V0 = 8


class FSAHardwareType:
    TYPE_HARDWARE_NULL = 0
    TYPE_ODrive_V36 = 1
    TYPE_H95V104 = 2
    TYPE_H66V104 = 3
    TYPE_H46V104 = 4
    TYPE_H30V303 = 5


class FSAMotorDirection:
    ABC = 1
    ACB = -1


class FSAMotorVBUS:
    VBUS_DEFAULT = 48
    VBUS_36V = 36
    VBUS_48V = 48


class FSAMotorPolePairs:
    POLE_PAIRS_7 = 7
    POLE_PAIRS_10 = 10
    POLE_PAIRS_21 = 21


class FSAMotorMaxSpeed:
    MAX_SPEED_1000 = 1000  # rpm
    MAX_SPEED_2000 = 2000  # rpm
    MAX_SPEED_3000 = 3000  # rpm


class FSAMotorMaxAcceleration:
    MAX_ACCELERATION_6000 = 6000  # rpm/s
    MAX_ACCELERATION_10000 = 10000  # rpm/s
    MAX_ACCELERATION_30000 = 30000  # rpm/s
    MAX_ACCELERATION_60000 = 60000  # rpm/s


class FSAMotorVIBCADCRatio:
    _ADC_PRECISION = 4096.0
    _ADC_SHUNT_RESISTANCE_2mR = 0.002
    _ADC_SHUNT_RESISTANCE_05mR = 0.0005
    _ADC_REF_V = 3.3
    _ADC_AMPLIFIER = 20
    VIBC_ADC_RATIO_2mR = (_ADC_REF_V / _ADC_PRECISION / _ADC_SHUNT_RESISTANCE_2mR / _ADC_AMPLIFIER)
    VIBC_ADC_RATIO_05mR = (_ADC_REF_V / _ADC_PRECISION / _ADC_SHUNT_RESISTANCE_05mR / _ADC_AMPLIFIER)


class FSAMotorVBUSADCRatio:
    _ADC_PRECISION = 4096.0
    _UP_SHUNT_RESISTANCE = 18000
    _DOWN_SHUNT_RESISTANCE = 1000
    _ADC_REF_V = 3.3
    VBUS_ADC_RATIO = (
        _ADC_REF_V / _ADC_PRECISION * (_UP_SHUNT_RESISTANCE + _DOWN_SHUNT_RESISTANCE) / _DOWN_SHUNT_RESISTANCE)


class FSAEncoderDirection:
    DIRECTION_CW = -1
    DIRECTION_CCW = 1


class FSAEncoderResolution:
    RESOLUTION_4000 = 4000  # 4000 pulse/rev


default_fsa_timeout = 0.2
default_fsa_port_ctrl = 2333
default_fsa_port_comm = 2334
default_fsa_port_pt = 10000
default_fsa_network = "192.168.137.255"

fsa_timeout = 0
fsa_port_ctrl = 0
fsa_port_comm = 0
fsa_port_pt = 0
fsa_network = ""
fsa_debug = True

if gl_rcs_config.parameters != {}:  # 如果配置文件不为空
    if gl_rcs_config.parameters["hardware"]["use_fsa"] is True:
        fsa_timeout = gl_rcs_config.parameters["fsa"]["timeout"]
        fsa_port_ctrl = gl_rcs_config.parameters["fsa"]["port_ctrl"]
        fsa_port_comm = gl_rcs_config.parameters["fsa"]["port_comm"]
        fsa_port_pt = gl_rcs_config.parameters["fsa"]["port_pt"]
        fsa_network = gl_rcs_config.parameters["fsa"]["network"]
        fsa_debug = gl_rcs_config.parameters["fsa"]["debug"]
    else:
        fsa_timeout = default_fsa_timeout
        fsa_port_ctrl = default_fsa_port_ctrl
        fsa_port_comm = default_fsa_port_comm
        fsa_port_pt = default_fsa_port_pt
        fsa_network = default_fsa_network
else:
    fsa_timeout = default_fsa_timeout
    fsa_port_ctrl = default_fsa_port_ctrl
    fsa_port_comm = default_fsa_port_comm
    fsa_port_pt = default_fsa_port_pt
    fsa_network = default_fsa_network

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(fsa_timeout)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

logger.print_trace("FSA start listening for broadcast...")

# ---------------------------------------------------------------------------------------------------------------------


def init(server_ip):
    return FSAFunctionResult.SUCCESS


def init_group(server_ips):
    return FSAFunctionResult.SUCCESS


# ---------------------------------------------------------------------------------------------------------------------
# Control Parameters of FSA


# fsa Get root attributes
# Parameters: including device IP
# Get all basic attributes of fsa, including serial number, bus voltage, motor temperature, inverter temperature, version number
def get_root(server_ip):
    data = {"method": "GET", "reqTarget": "/", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root() except")


# fsa enable
# Parameters: including device IP and motor number
# Each fsa can control two motors, M0 and M1
def set_enable(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/control_word",
        "property": "",
        "control_word": FSAControlWord.SERVO_ON,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.enable() except")
        return None


# fsa Disable
# Parameters: including device IP and motor number
# Each fsa can control two motors, M0 and M1
def set_disable(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/control_word",
        "property": "",
        "control_word": FSAControlWord.SERVO_OFF,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.disable() except")
        return None


# CALIBRATE_ENCODER = 0xA3


# fsa Calibrate Encoder
# Parameters: including device IP and motor number
# Auto rotate clockwise once ,then change direction turn around
def set_calibrate_encoder(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/control_word",
        "property": "",
        "control_word": FSAControlWord.CALIBRATE_ENCODER,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_calibrate_motor() except")
        return None


# fsa Clear Fault
# Parameters: including device IP and motor number
# Clear Fault
def clear_fault(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/control_word",
        "property": "",
        "control_word": FSAControlWord.CLEAR_FAULT,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_clear_fault() except")
        return None


# fsa Get current status
# Parameters: including device IP
# Get fsa Get current status
def get_state(server_ip):
    data = {"method": "GET", "reqTarget": "/state", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return json_obj.get("state")

        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_state() except")
        return None


# fsa set control mode
# Parameters: including server ip，motor number
# no return code
def set_mode_of_operation(server_ip, mode_of_operation):
    data = {
        "method": "SET",
        "reqTarget": "/mode_of_operation",
        "mode_of_operation": mode_of_operation,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip + " : Recv Data Error !")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.control_mode() except")
        return None


def get_home_offset(server_ip):
    data = {"method": "GET", "reqTarget": "/home_offset", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


# fsa reset linear count
# Parameters: including server ip，motor number
# no return code
def set_home_offset(server_ip, home_offset):
    data = {
        "method": "SET",
        "reqTarget": "/home_offset",
        "home_offset": home_offset,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_linear_count() except")
        return None


def set_home_position(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/home_position",
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_linear_count() except")
        return None


# fsa Get Root Config property
# Parameters: including device IP
# Get fsa bus voltage over-voltage and under-voltage protection threshold
def get_pid_param(server_ip):
    data = {"method": "GET", "reqTarget": "/pid_param", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


# fsa set Root Config properties
# Parameter: The protection threshold of bus voltage overvoltage and undervoltage
# Return success or failure
def set_pid_param(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/pid_param",
        "property": "",
        "control_position_kp": dict["control_position_kp"],
        "control_velocity_kp": dict["control_velocity_kp"],
        "control_velocity_ki": dict["control_velocity_ki"],
        "control_current_kp": dict["control_current_kp"],
        "control_current_ki": dict["control_current_ki"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_pid_param() except")
        return None


def clear_pid_param(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/pid_param",
        "property": "clear",
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.clear_pid_param() except")
        return None


# fsa set Root Config properties
# Parameter: The protection threshold of bus voltage overvoltage and undervoltage
# Return success or failure
def set_pid_param_imm(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/pid_param_imm",
        "property": "",
        "control_position_kp_imm": dict["control_position_kp_imm"],
        "control_velocity_kp_imm": dict["control_velocity_kp_imm"],
        "control_velocity_ki_imm": dict["control_velocity_ki_imm"],
        "control_current_kp_imm": dict["control_current_kp_imm"],
        "control_current_ki_imm": dict["control_current_ki_imm"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_pid_param_imm() except")
        return None


# fsa Get Root Config property
# Parameters: including device IP
# Get fsa bus voltage over-voltage and under-voltage protection threshold
def get_control_param(server_ip):
    data = {"method": "GET", "reqTarget": "/control_param", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_control_param() except")
        return None


# fsa set Control Config properties
# Parameter: Set Motor Max Speed ,acceleration and current
# Return success or failure
def set_control_param(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/control_param",
        "property": "",
        "motor_max_speed": dict["motor_max_speed"],
        "motor_max_acceleration": dict["motor_max_acceleration"],
        "motor_max_current": dict["motor_max_current"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_control_param() except")
        return None


# fsa get Control Config properties
# Parameters: including device IP
# Get fsa bus voltage over-voltage and under-voltage protection threshold
def get_control_param_imm(server_ip):
    data = {"method": "GET", "reqTarget": "/control_param_imm", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


# fsa set Control Config properties
# Parameter: Set Motor Max Speed ,acceleration and current
# Return success or failure
def set_control_param_imm(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/control_param_imm",
        "property": "",
        "motor_max_speed_imm": dict["motor_max_speed_imm"],
        "motor_max_acceleration_imm": dict["motor_max_acceleration_imm"],
        "motor_max_current_imm": dict["motor_max_current_imm"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_control_param() except")
        return None


def get_flag_of_operation(server_ip):
    data = {"method": "GET", "reqTarget": "/flag_of_operation", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


def set_flag_of_operation(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/flag_of_operation",
        "property": "",
        "flag_do_use_store_actuator_param": dict["flag_do_use_store_actuator_param"],
        "flag_do_use_store_motor_param": dict["flag_do_use_store_motor_param"],
        "flag_do_use_store_encoder_param": dict["flag_do_use_store_encoder_param"],
        "flag_do_use_store_pid_param": dict["flag_do_use_store_pid_param"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_flag_of_operation() except")
        return None


def clear_flag_of_operation(server_ip):
    data = {
        "method": "SET",
        "reqTarget": "/flag_of_operation",
        "property": "clear",
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.clear_flag_of_operation() except")
        return None


# fsa Get Root Config property
# Parameters: including device IP
# Get fsa bus voltage over-voltage and under-voltage protection threshold
def get_config(server_ip):
    data = {"method": "GET", "reqTarget": "/config", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


# fsa set Root Config properties
# Parameter: The protection threshold of bus voltage overvoltage and undervoltage
# Return success or failure
def set_config(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/config",
        "property": "",
        "actuator_type": dict["actuator_type"],
        "actuator_direction": dict["actuator_direction"],
        "actuator_reduction_ratio": dict["actuator_reduction_ratio"],
        "motor_type": dict["motor_type"],
        "motor_hardware_type": dict["motor_hardware_type"],
        "motor_vbus": dict["motor_vbus"],
        "motor_direction": dict["motor_direction"],
        "motor_pole_pairs": dict["motor_pole_pairs"],
        "motor_max_speed": dict["motor_max_speed"],
        "motor_max_acceleration": dict["motor_max_acceleration"],
        "motor_max_current": dict["motor_max_current"],
        "encoder_direction": dict["encoder_direction"],
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_config() except")
        return None


# fsa save configuration
# Parameters: including device IP
def save_config(server_ip):
    data = {"method": "SET", "reqTarget": "/config", "property": "save"}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.save_config() except")
        return None


# fsa clear configuration
# Parameters: including device IP
def erase_config(server_ip):
    data = {"method": "SET", "reqTarget": "/config", "property": "erase"}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.erase_config() except")
        return None


# fsa restart the motor drive
# Parameters: including device IP
def reboot(server_ip):
    data = {"method": "SET", "reqTarget": "/reboot", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.reboot_motor_drive() except")
        return None


# fsa Get error code
# Parameters: including server IP
def get_error_code(server_ip):
    data = {"method": "GET", "reqTarget": "/error_code", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            temp = json_obj.get("error_code")
            count = 0
            for i in FSAErrorCode:
                temp_error_code = (temp << 1 >> count) & 0x01
                if temp_error_code == 1:
                    logger.print_trace("Now Error Type = ", i.name)
                count = count + 1
            return json_obj.get("error_code")
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_error() except")
        return None


# fsa Get actuator position, velocity, current
# Parameters: including server ip，motor number
# Return position, speed, current in tuple
def get_pvc(server_ip):
    data = {
        "method": "GET",
        "reqTarget": "/measured",
        "position": True,
        "velocity": True,
        "current": True,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return FSAFunctionResult.FAIL

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return FSAFunctionResult.TIMEOUT

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_pvc() except")
        return FSAFunctionResult.FAIL


def get_pvcc(server_ip):
    data = {
        "method": "GET",
        "reqTarget": "/measured",
        "position": True,
        "velocity": True,
        "current": True,
        "current_id": True,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current"), json_obj.get(
                "current_id")
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return FSAFunctionResult.FAIL

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return FSAFunctionResult.TIMEOUT

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_pvc() except")
        return FSAFunctionResult.FAIL


def get_pvcccc(server_ip):
    data = {
        "method": "GET",
        "reqTarget": "/measured",
        "position": True,
        "velocity": True,
        "current": True,
        "current_id": True,
        "phase_current_ib": True,
        "phase_current_ic": True,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current"), \
                json_obj.get("current_id"), json_obj.get("phase_current_ib"), json_obj.get("phase_current_ic")
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return FSAFunctionResult.FAIL

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return FSAFunctionResult.TIMEOUT

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_pvc() except")
        return FSAFunctionResult.FAIL


# fsa position control
# parameter: server IP["xxx.xxx.xxx.xxx"], position[deg], velocity feedforward[deg/s], current feedforward[A]
# return position, velocity, current
def set_position_control(server_ip, position, velocity_ff=0.0, current_ff=0.0):
    data = {
        "method": "SET",
        "reqTarget": "/position_control",
        "reply_enable": True,
        "position": position,
        "velocity_ff": velocity_ff,
        "current_ff": current_ff,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        json_obj = json.loads(data.decode("utf-8"))

        if fsa_debug is True:
            logger.print_trace(
                server_ip + " : " + "Position = %.2f, Velocity = %.0f, Current = %.4f \n" %
                (json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_position() except")
        return None


# fsa velocity control
# parameter: server IP["xxx.xxx.xxx.xxx"], velocity[deg/s], current feedforward[A]
# return position, velocity, current
def set_velocity_control(server_ip, velocity, current_ff=0.0):
    data = {
        "method": "SET",
        "reqTarget": "/velocity_control",
        "reply_enable": True,
        "velocity": velocity,
        "current_ff": current_ff,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        json_obj = json.loads(data.decode("utf-8"))

        if fsa_debug is True:
            logger.print_trace(
                server_ip + " : " + "Position = %.2f, Velocity = %.0f, Current = %.4f \n" %
                (json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_velocity() except")
        return None


# fsa current control
# parameter: server IP["xxx.xxx.xxx.xxx"], current[A]
# return position, velocity, current
def set_current_control(server_ip, current):
    data = {
        "method": "SET",
        "reqTarget": "/current_control",
        "reply_enable": True,
        "current": current,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        json_obj = json.loads(data.decode("utf-8"))

        if fsa_debug is True:
            logger.print_trace(
                server_ip + " : " + "Position = %.2f, Velocity = %.0f, Current = %.4f \n" %
                (json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("current")
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_current() except")
        return None


# fsa torque control
# parameter: server IP["xxx.xxx.xxx.xxx"], torque[Nm]
# return position, velocity, torque
def set_torque_control(server_ip, torque):
    data = {
        "method": "SET",
        "reqTarget": "/torque_control",
        "reply_enable": True,
        "torque": torque,
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    try:
        data, address = s.recvfrom(1024)

        json_obj = json.loads(data.decode("utf-8"))

        if fsa_debug is True:
            logger.print_trace(
                server_ip + " : " + "Position = %.2f, Velocity = %.0f, Torque = %.4f \n" %
                (json_obj.get("position"), json_obj.get("velocity"), json_obj.get("torque")))

        if json_obj.get("status") == "OK":
            return json_obj.get("position"), json_obj.get("velocity"), json_obj.get("torque")
        else:
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_torque_control() except")
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Control Parameters of FSA Group


def enable_group(server_ips):
    time.sleep(1)

    for i in range(len(server_ips)):
        server_ip = server_ips[i]

        data = {
            "method": "SET",
            "reqTarget": "/control_word",
            "property": FSAControlWord.SERVO_ON[0],
        }

        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address

            if response.get(recv_ip) is not None:
                response.get(recv_ip).update({"data": data})
            else:
                logger.print_trace_warning("fsa.enable_group() receive wrong ip address ", (recv_ip, recv_port))

                # 接收到错误的 ip，就再接收一次；如果仍然接收错误，则放弃处理，执行下一次的接收
                data, address = s.recvfrom(1024)
                recv_ip, recv_port = address

                if response.get(recv_ip) is not None:
                    response.get(recv_ip).update({"data": data})
                else:
                    logger.print_trace_warning("fsa.enable_group() receive wrong ip address ", (recv_ip, recv_port))
                    continue

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(recv_ip, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.enable_group() Didn't receive enough data! [Timeout]")
            continue

        except e:
            logger.print_trace_warning("fsa.enable_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace("response = ", response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})

            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_warning("fsa.enable_group() except")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace("func_result = ", func_result)

    return func_result


def disable_group(server_ips):
    for i in range(len(server_ips)):
        server_ip = server_ips[i]

        data = {
            "method": "SET",
            "reqTarget": "/control_word",
            "property": FSAControlWord.SERVO_OFF,
        }
        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address

            if response.get(recv_ip) is not None:
                response.get(recv_ip).update({"data": data})
            else:
                logger.print_trace_warning("fsa.disable_group() receive wrong ip address ", (recv_ip, recv_port))

                # 接收到错误的 ip，就再接收一次；如果仍然接收错误，则放弃处理，执行下一次的接收
                data, address = s.recvfrom(1024)
                recv_ip, recv_port = address

                if response.get(recv_ip) is not None:
                    response.get(recv_ip).update({"data": data})
                else:
                    logger.print_trace_warning("fsa.disable_group() receive wrong ip address ", (recv_ip, recv_port))
                    continue

            if fsa_debug is True:
                try:
                    logger.print_trace("Received from {}:{}".format(recv_ip, data.decode("utf-8")))
                except:
                    logger.print_trace("Received from {} data none".format(recv_ip))
                    pass

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.disable_group() Didn't receive enough data! [Timeout]")
            continue

        except e:
            logger.print_trace_warning("fsa.disable_group() except", e)
            continue

    if fsa_debug is True:
        logger.print_trace("response = ", response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            if data is None:
                logger.print_trace_error(server_ip, " receive value is None!")
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                continue

            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})

            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        # except e:
        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            # logger.print_trace_warning("fsa.disable_group() except", e)
            logger.print_trace_warning("fsa.disable_group() except")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace("func_result = ", func_result)

    return func_result


def get_state_group(server_ips):
    for i in range(len(server_ips)):
        server_ip = server_ips[i]

        data = {
            "method": "GET",
            "reqTarget": "/state",
        }

        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(recv_ip, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.get_state_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.get_state_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})

            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            logger.print_trace_warning("fsa.get_state_group() except")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace(func_result)

    return func_result


def get_error_group(server_ips, motor_numbers=None):
    if motor_numbers is None:
        motor_numbers = numpy.ones_like(server_ips).astype(int).tolist()

    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        motor_number = motor_numbers[i]

        data = {
            "method": "GET",
            "reqTarget": "/m0/error",
        }
        if motor_number == 0:
            data["reqTarget"] = "/m0/error"
        elif motor_number == 1:
            data["reqTarget"] = "/m1/error"
        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.get_error_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.get_error_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})
                response.get(server_ip).update({"axis": json_obj.get("axis")})
                response.get(server_ip).update({"motor": json_obj.get("motor")})
                response.get(server_ip).update({"encoder": json_obj.get("encoder")})
            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_warning(server_ip + " fi_fsa.get_error_group() except")
            continue

    axis = []
    motors = []
    encoders = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        axis.append(response.get(server_ip).get("axis"))
        motors.append(response.get(server_ip).get("motor"))
        encoders.append(response.get(server_ip).get("encoder"))

    if fsa_debug is True:
        logger.print_trace(axis)
        logger.print_trace(motors)
        logger.print_trace(encoders)
    logger.print_trace("axis=", axis)
    logger.print_trace("motor=", motors)
    logger.print_trace("encoder=", encoders)

    return axis, motors, encoders


def clear_error_group(server_ips, motor_numbers=None):
    if motor_numbers is None:
        motor_numbers = numpy.ones_like(server_ips).astype(int).tolist()

    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        motor_number = motor_numbers[i]

        data = {"method": "SET", "reqTarget": "/m0/error", "clear_error": True}
        if motor_number == 0:
            data["reqTarget"] = "/m0/error"
        elif motor_number == 1:
            data["reqTarget"] = "/m1/error"
        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(recv_ip, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.clear_error_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.clear_error_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})
            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_warning(server_ip + " fi_fsa.clear_error_group() except")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace(func_result)

    return func_result


def get_pvc_group(server_ips, motor_numbers=None):
    if motor_numbers is None:
        motor_numbers = numpy.ones_like(server_ips).astype(int).tolist()

    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        motor_number = motor_numbers[i]

        data = {
            "method": "GET",
            "reqTarget": "/m0/CVP",
        }
        if motor_number == 0:
            data["reqTarget"] = "/m0/CVP"
        elif motor_number == 1:
            data["reqTarget"] = "/m1/CVP"
        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(recv_ip, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error(response)
            logger.print_trace_error("fsa.get_pvc_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.get_pvc_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})
                response.get(server_ip).update({"position": json_obj.get("position")})
                response.get(server_ip).update({"velocity": json_obj.get("velocity")})
                response.get(server_ip).update({"current": json_obj.get("current")})
            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_warning(server_ip + " fi_fsa.get_pvc_group() except")
            continue

    positions = []
    velocitys = []
    currents = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        positions.append(response.get(server_ip).get("position"))
        velocitys.append(response.get(server_ip).get("velocity"))
        currents.append(response.get(server_ip).get("current"))

    if fsa_debug is True:
        logger.print_trace(positions)
        logger.print_trace(velocitys)
        logger.print_trace(currents)

    return positions, velocitys, currents


def set_mode_of_operation_group(input_modes, server_ips):
    for i in range(len(server_ips)):
        input_modeaa = input_modes[i]
        server_ip = server_ips[i]

        set_mode_of_operation(input_modeaa, server_ip)


# fsa reset linear count
# Parameters: including server ip，motor number
# no return code
def set_home_offset_group(server_ips, home_offsets=None):
    if home_offsets is None:
        set_linear_counts = numpy.zeros_like(server_ips).tolist()

    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        home_offset = home_offsets[i]

        data = {"method": "SET", "reqTarget": "/home_offset", "home_offset": home_offset}

        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.set_linear_count_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.set_linear_count_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})
            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_error(server_ip, " receive data decode error!")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace(func_result)

    return func_result


# ---------------------------------------------------------------------------------------------------------------------
# Communication Parameters of FSA


# fsa Get root attributes
# Parameters: including device IP
# Get all basic attributes of fsa, including serial number, bus voltage, motor temperature, inverter temperature, version number
def get_comm_root(server_ip):
    data = {
        "method": "GET",
        "reqTarget": "/",
        "property": "",
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        return json_obj
    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root() except")


# fsa Get Root Config property
# Parameters: including device IP
# Get fsa bus voltage over-voltage and under-voltage protection threshold
def get_comm_config(server_ip):
    data = {
        "method": "GET",
        "reqTarget": "/config",
        "property": "",
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_root_config() except")
        return None


# fsa set Root Config properties
# Parameter: The protection threshold of bus voltage overvoltage and undervoltage
# Return success or failure
def set_comm_config(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/config",
        "property": "",
        "name": dict.get("name"),
        "DHCP_enable": dict.get("DHCP_enable"),
        "SSID": dict.get("SSID"),
        "password": dict.get("password"),
        "static_IP": dict.get("static_IP"),
        "gateway": dict.get("gateway"),
        "subnet_mask": dict.get("subnet_mask"),
        "dns_1": dict.get("dns_1"),
        "dns_2": dict.get("dns_2"),
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.set_comm_config() except")
        return None


# fsa save configuration
# Parameters: including device IP
def save_comm_config(server_ip):
    data = {"method": "SET", "reqTarget": "/config", "property": "save"}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.save_config() except")
        return None


# fsa restart
# Parameters: including device IP
def reboot_comm(server_ip):
    data = {"method": "SET", "reqTarget": "/reboot", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return FSAFunctionResult.SUCCESS
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return None

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return None

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.reboot() except")
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Communication Parameters of FSA Group


# fsa restart
# Parameters: including device IP
def reboot_comm_group(server_ips):
    for i in range(len(server_ips)):
        server_ip = server_ips[i]

        data = {"method": "SET", "reqTarget": "/reboot", "property": ""}

        json_str = json.dumps(data)

        if fsa_debug is True:
            logger.print_trace("Send JSON Obj:", json_str)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))

    response = {}
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        response.update({server_ip: {}})

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len(server_ips)):
        try:
            data, address = s.recvfrom(1024)
            recv_ip, recv_port = address
            response.get(recv_ip).update({"data": data})

            if fsa_debug is True:
                logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        except socket.timeout:  # fail after 1 second of no activity
            logger.print_trace_error("fsa.reboot_group() Didn't receive enough data! [Timeout]")
            continue

        except:
            logger.print_trace_warning("fsa.reboot_group() except")
            continue

    if fsa_debug is True:
        logger.print_trace(response)

    for i in range(len((server_ips))):
        server_ip = server_ips[i]
        data = response.get(server_ip).get("data")

        try:
            json_obj = json.loads(data.decode("utf-8"))

            if json_obj.get("status") == "OK":
                response.get(server_ip).update({"return": FSAFunctionResult.SUCCESS})
            else:
                response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
                logger.print_trace_error(server_ip, " receive status is not OK!")
                continue

        except:
            response.get(server_ip).update({"return": FSAFunctionResult.FAIL})
            logger.print_trace_warning(server_ip + " fi_fsa.reboot_group() except")
            continue

    func_result = []
    for i in range(len(server_ips)):
        server_ip = server_ips[i]
        func_result.append(response.get(server_ip).get("return"))

    if fsa_debug is True:
        logger.print_trace(func_result)

    return func_result


# ---------------------------------------------------------------------------------------------------------------------
# Communication Parameters of FSE


# get abs encoder position
# Parameters: including server ip
# Return position in tuple
def get_abs_encoder_angle(server_ip):
    data = {"method": "GET", "reqTarget": "/measured", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode("utf-8"))

        if json_obj.get("status") == "OK":
            return json_obj.get("angle")
        else:
            logger.print_trace_error(server_ip, " receive status is not OK!")
            return FSAFunctionResult.FAIL

    except socket.timeout:  # fail after 1 second of no activity
        logger.print_trace_error(server_ip + " : Didn't receive anymore data! [Timeout]")
        return FSAFunctionResult.TIMEOUT

    except:
        logger.print_trace_warning(server_ip + " fi_fsa.get_abs_encoder_angle() except")
        return FSAFunctionResult.FAIL


# ---------------------------------------------------------------------------------------------------------------------
# Broadcast Query FSA in LAN


# 广播查询局域网下的全部
# 参数：无
# 返回 成功 失败 超时
def broadcast_func():
    logger.print_trace("FSA start listening for broadcast...")

    found_server = False
    address_list = []

    s.sendto("Is any fourier smart server here?".encode("utf-8"), (fsa_network, fsa_port_comm))
    print("\n")

    while True:
        try:
            data, address = s.recvfrom(1024)
            address_list.append(address[0])
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))
            # json_obj = json.loads(data.decode("utf-8"))
            found_server = True

        except socket.timeout:  # fail after 1 second of no activity
            if found_server:
                print("\n")
                print("found servers")
                print(address_list)
                print("lookup Finished! \n")
                time.sleep(2)
                return address_list
            else:
                logger.print_trace_error("Do not have any server! [Timeout] \n")
                return False


# 广播查询局域网下的全部 filter_type = "Actuator" or "AbsEncoder" or "CtrlBox"
def broadcast_func_with_filter(filter_type=None):
    logger.print_trace("FSA start listening for broadcast...")

    found_server = False
    address_list = []

    s.sendto("Is any fourier smart server here?".encode("utf-8"), (fsa_network, fsa_port_comm))
    print("\n")

    while True:
        try:
            data, address = s.recvfrom(1024)
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

            # 如果没有过滤条件，直接返回所有的IP
            # 如果有过滤条件，只返回符合条件的IP
            if filter_type is None:
                address_list.append(address[0])
                found_server = True
                continue
            else:
                pass

            json_obj = json.loads(data.decode("utf-8"))
            if "type" in json_obj:
                if json_obj["type"] == filter_type:
                    address_list.append(address[0])
                    found_server = True

        except socket.timeout:  # fail after 1 second of no activity
            if found_server:
                print("\n")
                print("found servers")
                print(address_list)
                print("lookup Finished! \n")
                time.sleep(2)
                return address_list
            else:
                logger.print_trace_error("Do not have any server! [Timeout] \n")
                return False


def ota(server_ip):
    data = {"method": "SET", "reqTarget": "/ota", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_test(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_test", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_devel(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_devel", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_cloud(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_cloud", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_driver(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_driver", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_driver_test(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_driver_test", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_driver_devel(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_driver_devel", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def ota_driver_cloud(server_ip):
    data = {"method": "SET", "reqTarget": "/ota_driver_cloud", "property": ""}

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")


def encrypt(server_ip, dict):
    data = {
        "method": "SET",
        "reqTarget": "/encrypt",
        "property": "",
        "username": dict["username"],
        "password": dict["password"]
    }

    json_str = json.dumps(data)

    if fsa_debug is True:
        logger.print_trace("Send JSON Obj:", json_str)

    s.sendto(str.encode(json_str), (server_ip, fsa_port_comm))
    try:
        data, address = s.recvfrom(1024)

        if fsa_debug is True:
            logger.print_trace("Received from {}:{}".format(address, data.decode("utf-8")))

        json_obj = json.loads(data.decode('utf-8'))

    except socket.timeout:  # fail after 1 second of no activity
        print("Didn't receive anymore data! [Timeout]")
