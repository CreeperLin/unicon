def cb_input_pygame(
    states_input,
    device=0,
    verbose=True,
    input_keys=None,
):
    import pygame
    pygame.init()
    pygame.joystick.init()
    js = pygame.joystick.Joystick(device)
    js.init()

    num_buttons = js.get_numbuttons()
    num_axes = js.get_numaxes()
    num_hats = js.get_numhats()
    print(num_buttons, num_axes, num_hats)

    def cb():
        axes = [js.get_axis(i) for i in range(num_axes)]
        btns = [js.get_button(i) for i in range(num_buttons)]
        hats = [js.get_hat(i) for i in range(num_hats)]
        print(axes, btns, hats)

    return cb


if __name__ == '__main__':
    import numpy as np
    states_input = np.zeros(7)
    cb = cb_input_pygame(states_input)
    while True:
        cb()
