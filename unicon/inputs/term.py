def cb_input_term(
    states_input,
    verbose=False,
    # verbose=True,
    input_keys=None,
    step=0.05,
    max_hold=5,
    release_gamma=0.98,
):
    import termios
    import fcntl
    import sys
    import os
    import numpy as np
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys
    nabs_inds = [i for i, k in enumerate(input_keys) if not k.startswith('ABS')]
    abs_inds = [i for i, k in enumerate(input_keys) if k.startswith('ABS')]
    mapping = {
        'd': 'ABS_X+',
        'a': 'ABS_X-',
        'w': 'ABS_Y-',
        's': 'ABS_Y+',
        'l': 'ABS_Z+',
        'j': 'ABS_Z-',
        'i': 'ABS_RZ-',
        'k': 'ABS_RZ+',
        '1': 'BTN_TL',
        '0': 'BTN_TR',
        '5': 'BTN_A',
        '6': 'BTN_B',
        't': 'BTN_X',
        'y': 'BTN_Y',
    }
    fd = sys.stdin.fileno()

    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)

    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    s = np.zeros(len(states_input))
    hold = 0
    while True:
        c = sys.stdin.read(1)
        if not c:
            break
        print('term read', ord(c))

    class cb_cls():

        def __call__(self):
            nonlocal hold
            if hold == 0:
                s[nabs_inds] = 0
                # states_input[:] = s
                mx = np.max(np.abs(s))
                if mx > 0.005:
                    # s[:] = s * release_gamma
                    s[abs_inds] = s[abs_inds] * release_gamma
                    states_input[:] = s
                    if verbose:
                        print('term release', mx, np.round(s, decimals=2))
                else:
                    s[:] = 0
                    states_input[:] = 0
            else:
                hold -= 1
            c = 0
            try:
                c = sys.stdin.read(1)
                if c == '\x1b':
                    c = sys.stdin.read(2)[-1]
            except IOError:
                pass
            v = None if not c else mapping.get(c)
            if v is None:
                return
            if verbose:
                print('term input', c)
            z = v[-1]
            k = v
            d = None
            if z == '+' or z == '-':
                k = v[:-1]
                d = 1 if z == '+' else -1
            if k not in input_keys:
                return
            idx = input_keys.index(k)
            s[nabs_inds] = 0
            if d is None:
                val = 1
            else:
                val = min(1, max(-1, s[idx] + d * step))
            s[idx] = val
            states_input[:] = s
            hold = max_hold
            if verbose:
                print('states_input', states_input.tolist())

        def __del__(self):
            if termios and termios.tcsetattr:
                termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            if fcntl and fcntl.fcntl:
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
            os.system('stty sane')

    return cb_cls()


if __name__ == '__main__':
    import numpy as np
    import time
    cb = cb_input_term(np.zeros(8), verbose=True)
    for i in range(1000):
        cb()
        time.sleep(0.02)
