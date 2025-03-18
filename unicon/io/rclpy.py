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


def run_fn(
    cb_init=None,
    cb_recv=None,
    cb_send=None,
    pub_dt=0.1,
    sub_dt=0.1,
    sub_topic='/isaac_joint_commands',
    pub_topic='/isaac_joint_states',
    verbose=False,
):
    _ctx = None
    if cb_init is not None:
        _ctx = cb_init()

    def set_msg(msg, ctx):
        send_name = ctx.get('name')
        send_position = ctx.get('position')
        send_velocity = ctx.get('velocity')
        send_effort = ctx.get('effort')
        if send_name is None or send_position is None or send_velocity is None or send_effort is None:
            print('invalid send msg', ctx)
            return False
        msg.name = send_name
        msg.position = send_position
        msg.velocity = send_velocity
        msg.effort = send_effort

    def cb_pub_init(msg, node):
        if _ctx is not None:
            set_msg(msg, _ctx)

    def cb_pub_tick(msg, node):
        if cb_send is None:
            return
        ctx = cb_send()
        if ctx is None:
            return False
        if set_msg(msg, ctx) is False:
            return False
        if verbose:
            print('send', msg.header.stamp)

    running = True
    th_pub_fn = partial(
        th_ros2_pub_cb,
        topic=pub_topic,
        msg_type=sensor_msgs.msg.JointState,
        is_running=lambda: running,
        cb_init=cb_pub_init,
        cb_tick=cb_pub_tick,
        intv=pub_dt,
    )
    th_sub_fn = partial(
        th_ros2_sub_cb,
        topic=sub_topic,
        msg_type=sensor_msgs.msg.JointState,
        is_running=lambda: running,
        cb_recv=cb_recv,
        intv=sub_dt,
    )
    th_sub = threading.Thread(target=th_sub_fn)
    th_pub = threading.Thread(target=th_pub_fn)

    th_sub.start()
    th_pub.start()

    def close_fn():
        nonlocal running
        running = False
        cb_recv(True)
        th_sub.join()
        th_pub.join()
        rclpy.try_shutdown()

    return close_fn


if __name__ == '__main__':
    rclpy.init()
