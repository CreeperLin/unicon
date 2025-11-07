import os
import argparse
import numpy as np
import pygame
from unicon.states import states_get, states_init

# 可编辑参数
CIRCLE_RADIUS = 1.0
Z_CLIP = 0.2  # target坐标clip范围（米）
TARGET_STEP = 0.05 # 每次按键移动步长（米）
WINDOW_SIZE = (600, 400)
FONT_SIZE = 24

COLOR_BG = (30, 30, 30)
COLOR_ON = (0, 200, 0)
COLOR_OFF = (200, 0, 0)
COLOR_TARGET = (255, 255, 0)
COLOR_TEXT = (200, 200, 200)

KEY_MAP = {
    pygame.K_w: np.array([TARGET_STEP, 0, 0]),   # x+ 前
    pygame.K_s: np.array([-TARGET_STEP, 0, 0]),  # x- 后
    pygame.K_a: np.array([0, TARGET_STEP, 0]),   # y+ 左
    pygame.K_d: np.array([0, -TARGET_STEP, 0]),  # y- 右
    pygame.K_o: np.array([0, 0, TARGET_STEP]),   # z+ 上
    pygame.K_p: np.array([0, 0, -TARGET_STEP]),  # z- 下
}

MODE_NONE = 0
MODE_LEFT = 1
MODE_RIGHT = 2
MODE_BOTH = 3


def clip_target(target, circle_radius=CIRCLE_RADIUS, z_clip=Z_CLIP):
    dist = np.linalg.norm(target[0:2])
    if dist > circle_radius:
        theta = np.arctan2(target[1], target[0])
        target[0:2] = circle_radius * np.array([np.cos(theta), np.sin(theta)])
    target[2] = np.clip(target[2], -z_clip, z_clip)
    return target

def draw_status(screen, font, reach_mask, left_target, right_target, mode):
    # 灯状态
    pygame.draw.circle(screen, COLOR_ON if reach_mask[0] else COLOR_OFF, (80, 60), 30)
    pygame.draw.circle(screen, COLOR_ON if reach_mask[1] else COLOR_OFF, (260, 60), 30)
    screen.blit(font.render('Reach-L', True, COLOR_TEXT), (40, 100))
    screen.blit(font.render('Reach-R', True, COLOR_TEXT), (220, 100))
    # target显示
    lt_str = ', '.join([f'{v:.3f}' for v in left_target[:3]])
    rt_str = ', '.join([f'{v:.3f}' for v in right_target[:3]])
    screen.blit(font.render(f'Left Target: [{lt_str}]', True, COLOR_TARGET), (40, 160))
    screen.blit(font.render(f'Right Target: [{rt_str}]', True, COLOR_TARGET), (40, 200))
    # 当前模式
    mode_str = {MODE_NONE: 'Normal', MODE_LEFT: 'Edit Left', MODE_RIGHT: 'Edit Right', MODE_BOTH: 'Edit Both'}[mode]
    screen.blit(font.render(f'Mode: {mode_str}', True, COLOR_TEXT), (40, 250))
    screen.blit(font.render('WASD: x/y, O/P: z', True, COLOR_TEXT), (40, 280))
    screen.blit(font.render('1/2 edit mask', True, COLOR_TEXT), (40, 310))
    screen.blit(font.render('Ctrl+L/R/M/0 switch edit', True, COLOR_TEXT), (40, 340))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ut', '--unit_test', action='store_true', help='unit_test mode')
    parser.add_argument('-hdls', '--headless', action='store_true', help='run in headless mode')
    args = parser.parse_args()
    headless = args.headless

    if not args.unit_test:
        states_init(use_shm=True, load=True, reuse=True)
    else:
        print('Unit test mode: states not initialized')

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Manual Target Control')
    font = pygame.font.SysFont('Arial', FONT_SIZE)
    clock = pygame.time.Clock()

    mode = MODE_NONE
    running = True

    _local_reach_mask = np.zeros(2)
    _local_left_target = np.zeros(7)
    _local_right_target = np.zeros(7)
    while running:
        screen.fill(COLOR_BG)
        # 读取共享内存
        if args.unit_test:
            reach_mask = _local_reach_mask
            left_target = _local_left_target
            right_target = _local_right_target
        else:
            reach_mask = states_get('reach_mask')
            left_target = states_get('left_target_real_time')
            right_target = states_get('right_target_real_time')
        draw_status(screen, font, reach_mask, left_target, right_target, mode)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # mask切换
                if event.key == pygame.K_1:
                    reach_mask[0] = not reach_mask[0]
                elif event.key == pygame.K_2:
                    reach_mask[1] = not reach_mask[1]
                # 进入编辑模式
                elif event.key == pygame.K_l and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    mode = MODE_LEFT
                elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    mode = MODE_RIGHT
                elif event.key == pygame.K_m and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    mode = MODE_BOTH
                # 编辑target
                elif mode in [MODE_LEFT, MODE_RIGHT, MODE_BOTH]:
                    if event.key == pygame.K_0 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        mode = MODE_NONE
                    elif event.key in KEY_MAP:
                        delta = KEY_MAP[event.key]
                        if mode == MODE_LEFT:
                            left_target[:3] += delta
                            left_target = clip_target(left_target)
                        elif mode == MODE_RIGHT:
                            right_target[:3] += delta
                            right_target = clip_target(right_target)
                        elif mode == MODE_BOTH:
                            left_target[:3] += delta
                            right_target[:3] += delta
                            left_target = clip_target(left_target)
                            right_target = clip_target(right_target)
                        if np.sum(np.abs(left_target[3:])) < 1e-6:
                            left_target[3:] = np.array([0,0,0,1])
                        if np.sum(np.abs(right_target[3:])) < 1e-6:
                            right_target[3:] = np.array([0,0,0,1])
                # 退出
                elif event.key == pygame.K_q:
                    running = False
        clock.tick(30)
    pygame.quit()

if __name__ == '__main__':
    main()
