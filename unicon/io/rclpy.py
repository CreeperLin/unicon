import traceback
import threading
import time
from functools import partial

# ROS 2 imports
import rclpy
import rclpy.node
import sensor_msgs.msg


def msg_to_dict(msg):
    return {k: getattr(msg, k) for k in msg.get_fields_and_field_types()}


def th_ros2_pub_cb(
    msg_type,
    topic,
    is_running=None,
    cb_init=None,
    cb_tick=None,
    node_name=None,
    qos=10,
    init=False,
    args=None,
    intv=0.1,
    stamp=True,
):
    node_name = ('node_' + topic.replace('/', '_').replace('-', '_')) if node_name is None else node_name
    cb_tick = (lambda x: None) if cb_tick is None else cb_tick
    is_running = (lambda: True) if is_running is None else is_running
    if init or not rclpy.ok() or not rclpy.get_default_context().ok():
        rclpy.init(args=args)
    try:
        node = rclpy.create_node(node_name)
        pub = node.create_publisher(msg_type, topic, qos)
        msg = msg_type()
        if cb_init is not None:
            cb_init(msg, node)
        t0 = time.perf_counter()
        while is_running() and rclpy.ok():
            if stamp:
                msg.header.stamp = node.get_clock().now().to_msg()
            ret = cb_tick(msg, node)
            if ret is True:
                break
            if ret is not False:
                pub.publish(msg)
            t0 += intv
            t_s = t0 - time.perf_counter()
            if t_s > 0:
                time.sleep(t_s)
            else:
                print('pub timeout', t_s)
    except Exception:
        traceback.print_exc()
    print('ros2 pub exit', node_name)


def th_ros2_sub_cb(
    msg_type,
    topic,
    is_running=None,
    cb_recv=None,
    node_name=None,
    qos=10,
    init=False,
    args=None,
    intv=0.1,
    to_dict=True,
):

    def on_recv(msg):
        if to_dict:
            msg = msg_to_dict(msg)
        cb_recv(msg)

    node_name = ('node_' + topic.replace('/', '_').replace('-', '_')) if node_name is None else node_name
    is_running = (lambda: True) if is_running is None else is_running
    if init or not rclpy.ok() or not rclpy.get_default_context().ok():
        rclpy.init(args=args)
    try:
        node = rclpy.create_node(node_name)
        sub = node.create_subscription(msg_type, topic, on_recv, qos)
        sub
        t0 = time.perf_counter()
        if intv is None:
            rclpy.spin(node)
            return
        while is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=intv * 0.9)
            t0 += intv
            t_s = t0 - time.perf_counter()
            if t_s > 0:
                time.sleep(t_s)
            else:
                print('sub timeout', t_s)
    except Exception:
        traceback.print_exc()
    print('ros2 sub exit', node_name)


def cb_rclpy_send(
    pub_dt=0.1,
    pub_topic='/default',
    keys=None,
    msg_keys=None,
    msg_type=None,
    **states,
):
    msg_type = sensor_msgs.msg.JointState if msg_type is None else msg_type

    def cb_pub_tick(msg, node):
        for k, mk in zip(keys, msg_keys):
            setattr(msg, mk, states[k].tolist())

    running = True
    th_pub_fn = partial(
        th_ros2_pub_cb,
        topic=pub_topic,
        msg_type=msg_type,
        is_running=lambda: running,
        cb_tick=cb_pub_tick,
        intv=pub_dt,
    )

    th_pub = threading.Thread(target=th_pub_fn)
    th_pub.start()

    def close_fn():
        nonlocal running
        running = False
        th_pub.join()
        rclpy.try_shutdown()

    return None, close_fn


def cb_rclpy_recv(
    sub_dt=0.1,
    topic='/default',
    keys=None,
    msg_keys=None,
    msg_type=None,
    **states,
):
    msg_type = sensor_msgs.msg.JointState if msg_type is None else msg_type

    def cb_recv(msg):
        for k, mk in zip(keys, msg_keys):
            m = msg.get(mk)
            if m is None:
                continue
            states[k][:] = m

    running = True
    th_sub_fn = partial(
        th_ros2_sub_cb,
        topic=topic,
        msg_type=msg_type,
        is_running=lambda: running,
        cb_recv=cb_recv,
        intv=sub_dt,
    )
    th_sub = threading.Thread(target=th_sub_fn)
    th_sub.start()

    def close_fn():
        nonlocal running
        running = False
        th_sub.join()
        rclpy.try_shutdown()

    return None, close_fn


if __name__ == '__main__':
    rclpy.init()
