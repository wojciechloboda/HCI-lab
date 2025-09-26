#!/usr/bin/env python3

import math
import random
import argparse
import time
import datetime
import os

try:
    import pygame
except Exception:
    pygame = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Screen PPI, needed for cm to pixel calculations (per device)
PPI = 96 


def index_of_difficulty(D, W):
    return math.log2(D / W + 1.0)

def save_csv(df, out_path):
    df.to_csv(out_path, index=False)

def pick_valid_target(D, W, screen_w, screen_h, start_x, start_y):
    while True:
        angle = random.uniform(0, 2 * math.pi)
        tx = int(start_x + D * math.cos(angle))
        ty = int(start_y + D * math.sin(angle))

        if W <= tx <= screen_w - W and W <= ty <= screen_h - W:
            return (tx, ty)    

def cm_to_px(cm):
    return int(cm * PPI / 2.54)            

def run_trials(conditions, trials_per_condition):
    if pygame is None:
        print("Pygame not available. Install pygame to use interactive mode: pip install pygame")
        return
    pygame.init()

    # use fullscreen by default
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    trials = []

    waiting_for_click = False
    trial_idx = 0
    start_time = None
    D = None
    W = None

    # create the list of indices, make every contiion apear as meany times as specified by the parameter and ....
    condition_indices = []
    for i in range(len(conditions)):
        condition_indices.extend([i] * trials_per_condition)
    random.shuffle(condition_indices)   
    
    conditions_px = [(cm_to_px(D), cm_to_px(W)) for D, W in conditions]

    running = False
    quit = False
    while not quit:
        while not running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit = True
                    else:
                        running = True
                screen.fill((240,240,240))
                txt = font.render(f"Press any key to start. Then click the circle. Esc to quit.", True, (0,0,0))
                screen.blit(txt, (10,10))
                pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if waiting_for_click and target_pos is not None:
                    mx,my = event.pos
                    dx = mx - target_pos[0]
                    dy = my - target_pos[1]
                    dist = math.hypot(dx,dy)

                    end_time = time.time()
                    mt_ms = (end_time - start_time) * 1000.0

                    if dist <= target_radius:
                        error = 0   # hit
                    else:
                        error = 1   # miss

                    trials.append({
                        "trial": trial_idx,
                        "D": D,
                        "W": W,
                        "ID": index_of_difficulty(D, W),
                        "MT": mt_ms,
                        "error": error
                    })            

                    trial_idx += 1
                    waiting_for_click = False   

        if not waiting_for_click:
            if trial_idx >= len(condition_indices): 
                quit = True
                continue

            start_x, start_y = pygame.mouse.get_pos()
            D, W = conditions_px[condition_indices[trial_idx]]
            target_pos = pick_valid_target(D, W, screen_w, screen_h, start_x, start_y)
            target_radius = W // 2

            start_time = time.time()
            waiting_for_click = True

        screen.fill((240,240,240))
        if target_pos is not None:
            pygame.draw.circle(screen, (60,60,60), (int(target_pos[0]), int(target_pos[1])), int(target_radius))

        txt = font.render(f"Trial {trial_idx}/{len(condition_indices)}. Click the circle. Esc to quit.", True, (0,0,0))
        screen.blit(txt, (10,10))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return pd.DataFrame(trials)

def get_filepath(dirname):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M')
    filename = timestamp + ".csv"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return os.path.join(dirname, filename)

def main():
    parser = argparse.ArgumentParser(description="Fitts' Law demo")
    parser.add_argument("--out", type=str, default="results", help="CSV output foldername")
    args = parser.parse_args()

    # (D, W) in cm
    conditions = [
        (10, 1),
        (5, 2),
        (20, 1),
        (15, 2),
    ]

    # How many times we repreat each condition
    trials_per_condition = 2

    df = run_trials(conditions, trials_per_condition)

    if df is None or df.empty:
        print("No trials recorded.")
        return

    filepath = get_filepath(args.out)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    main()
