#!/usr/bin/env python3
"""
fitts_demo.py

Interactive Fitts' Law demo (requires pygame) and a --simulate mode that runs
a quick simulated experiment and outputs results for analysis.

Usage:
  Interactive (requires pygame): python fitts_demo.py
  Simulate (no GUI needed):     python fitts_demo.py --simulate

The interactive mode shows circular targets; click them as they appear. The script logs
Movement Time (MT), Distance (D), Width (W) and computes Index of Difficulty (ID).
After the session ends it fits MT = a + b * ID and shows a plot and saves CSV.
"""

import sys
import math
import csv
import random
import argparse
import time

try:
    import pygame
except Exception:
    pygame = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def index_of_difficulty(D, W):
    # D and W must be positive. Fitts' original formula: ID = log2( D / W + 1 )
    return math.log2(D / W + 1.0)

def run_simulation(trials=40, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    # Choose a variety of distances and widths
    widths = np.array([20, 40, 80, 160])  # pixels
    distances = np.array([50, 100, 200, 400])  # pixels
    rows = []
    # underlying true parameters (simulate human behavior)
    a_true = 150.0  # ms (intercept)
    b_true = 100.0  # ms/bit (slope)
    for i in range(trials):
        W = float(random.choice(widths))
        D = float(random.choice(distances))
        ID = index_of_difficulty(D, W)
        # Simulated MT (ms) with some noise
        mt = a_true + b_true * ID + np.random.normal(scale=40.0)
        rows.append({"trial": i+1, "D": D, "W": W, "ID": ID, "MT": mt})
    return pd.DataFrame(rows)

def fit_linear(df):
    # Fit MT = a + b * ID using numpy.polyfit
    x = df["ID"].values
    y = df["MT"].values
    b, a = np.polyfit(x, y, 1)  # returns slope, intercept for x->y; we'll reorder
    # compute R^2
    yhat = a + b * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"a_est": float(a), "b_est": float(b), "r2": float(r2)}

def save_csv(df, out_path):
    df.to_csv(out_path, index=False)

def show_plot(df, fit_result, title="Fitts' Law (MT vs ID)"):
    import matplotlib.pyplot as plt
    x = df["ID"].values
    y = df["MT"].values
    # Single plot only (no explicit color)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = fit_result["a_est"] + fit_result["b_est"] * xs
    ax.plot(xs, ys)
    ax.set_xlabel("Index of Difficulty (bits)")
    ax.set_ylabel("Movement Time (ms)")
    ax.set_title(f"{title}\nMT = {fit_result['a_est']:.1f} + {fit_result['b_est']:.1f} * ID  (R²={fit_result['r2']:.3f})")
    fig.tight_layout()
    plt.show()

def interactive_mode(num_trials=30):
    if pygame is None:
        print("Pygame not available. Install pygame to use interactive mode: pip install pygame")
        return
    pygame.init()
    screen_w, screen_h = 800, 600
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Fitts' Law Demo - click the targets")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    trials = []
    # starting position (center)
    current_pos = (screen_w//2, screen_h//2)
    target_radius = 30
    target_pos = None

    def spawn_target():
        # Choose random edge position for more movement
        margin = 50
        x = random.randint(margin, screen_w - margin)
        y = random.randint(margin, screen_h - margin)
        return (x,y)

    waiting_for_click = False
    trial_idx = 0
    start_time = None
    D = None
    W = target_radius * 2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if waiting_for_click and target_pos is not None:
                    mx,my = event.pos
                    dx = mx - target_pos[0]
                    dy = my - target_pos[1]
                    dist = math.hypot(dx,dy)
                    # consider hit if within radius
                    if dist <= target_radius:
                        end_time = time.time()
                        mt_ms = (end_time - start_time) * 1000.0
                        trials.append({"trial": trial_idx+1, "D": D, "W": W, "ID": index_of_difficulty(D, W), "MT": mt_ms})
                        waiting_for_click = False

        if not waiting_for_click:
            if trial_idx >= num_trials:
                running = False
                continue
            # spawn next
            target_pos = spawn_target()
            # compute D as distance from center of screen (simulate starting point)
            start_x, start_y = current_pos
            D = math.hypot(target_pos[0] - start_x, target_pos[1] - start_y)
            start_time = time.time()
            waiting_for_click = True
            trial_idx += 1

        screen.fill((240,240,240))
        # draw target
        if target_pos is not None:
            pygame.draw.circle(screen, (60,60,60), (int(target_pos[0]), int(target_pos[1])), int(target_radius))
        # instructions / status
        txt = font.render(f"Trial {trial_idx}/{num_trials}. Click the circle. Esc to quit.", True, (0,0,0))
        screen.blit(txt, (10,10))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    # convert to DataFrame
    import pandas as pd
    return pd.DataFrame(trials)

def main():
    parser = argparse.ArgumentParser(description="Fitts' Law demo (interactive and simulate modes)")
    parser.add_argument("--simulate", action="store_true", help="Run a simulated experiment (no GUI)")
    parser.add_argument("--trials", type=int, default=40, help="Number of trials for simulate or interactive mode")
    parser.add_argument("--out", type=str, default="fitts_demo_results.csv", help="CSV output filename")
    args = parser.parse_args()

    if args.simulate:
        df = run_simulation(trials=args.trials)
    else:
        df = interactive_mode(num_trials=args.trials)
        if df is None or df.empty:
            print("No trials recorded.")
            return

    fit = fit_linear(df)
    print("Fit result:", fit)
    df.to_csv(args.out, index=False)
    try:
        show_plot(df, fit)
    except Exception as e:
        print("Plotting failed:", e)
    print("Saved results to", args.out)

if __name__ == "__main__":
    main()
