"""
demo_lib.py: shared helpers for the BrickGPT demo (generation with logging, rendering).

- generate(): runs BrickGPT on one prompt and returns the bricks plus a log of what the
  inference loop did (bricks, rejections by reason, physics rollbacks, stability checks,
  seconds). It wraps the official BrickGPT class; nothing in the model is changed.
- render_png() / render_gif(): matplotlib 3D renders from the brick list (one box per
  brick, real LEGO proportions: a plate unit is 8 mm wide and a brick 9.6 mm tall).
  Used instead of the official Blender/LDraw renderer, which needs bpy + the LDraw library.
- plotly_figure(): interactive 3D view for the Gradio app.

Imported by run_demo.py and app.py; not run directly.
"""

import time
from collections import Counter

import matplotlib

matplotlib.use('Agg')  # headless: write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import transformers
from brickgpt.data import BrickStructure
from brickgpt.models import BrickGPT, BrickGPTConfig
from brickgpt.stability_analysis import stability_analysis

Z_SCALE = 9.6 / 8.0  # brick height / stud pitch, so renders have true proportions

# Classic LEGO colours (red, blue, yellow, green, orange, light grey, dark grey, white).
PALETTE = ['#C91A09', '#0055BF', '#F2CD37', '#237841', '#FE8A18', '#A0A5A9', '#6C6E68', '#F4F4F4']


def load_model(rollback: bool = True, physics: bool = True, max_regenerations: int = 100,
               device: str = 'auto') -> BrickGPT:
    """Loads Llama-3.2-1B-Instruct + the BrickGPT LoRA weights (downloaded on first use)."""
    cfg = BrickGPTConfig(max_regenerations=max_regenerations if rollback else 0,
                         use_gurobi=physics, device=device)
    return BrickGPT(cfg)


def generate(model: BrickGPT, prompt: str, seed: int = 42, verbose: bool = True) -> dict:
    """Generates one structure and logs the inference loop (rollbacks, stability checks)."""
    log = {'rollbacks': [], 'n_stability_checks': 0, 'stability_time_s': 0.0}

    # Count the physics checks and their time by wrapping the solver call.
    orig_score = stability_analysis.stability_score

    def timed_score(*a, **k):
        t = time.time()
        out = orig_score(*a, **k)
        log['n_stability_checks'] += 1
        log['stability_time_s'] += time.time() - t
        return out

    # Record every physics-informed rollback: how many bricks were kept of how many.
    orig_remove = model._remove_all_bricks_after_first_unstable_brick

    def logged_remove(bricks):
        kept = orig_remove(bricks)
        log['rollbacks'].append({'had': len(bricks), 'kept': len(kept)})
        if verbose:
            print(f'  [physics] structure with {len(bricks)} bricks is unstable '
                  f'-> roll back to the first {len(kept)} stable bricks and regenerate', flush=True)
        return kept

    import brickgpt.data.brick_structure as bs_mod
    bs_mod.stability_score = timed_score
    model._remove_all_bricks_after_first_unstable_brick = logged_remove
    try:
        transformers.set_seed(seed)
        t0 = time.time()
        out = model(prompt)
        seconds = time.time() - t0
    finally:
        bs_mod.stability_score = orig_score
        model._remove_all_bricks_after_first_unstable_brick = orig_remove

    bricks: BrickStructure = out['bricks']
    reasons: Counter = out['rejection_reasons']
    # Final verdict with the full physics model, whatever mode generated it.
    final_stable = bool(bricks.is_stable()) if len(bricks) else False
    return {
        'prompt': prompt, 'seed': seed, 'seconds': round(seconds, 2),
        'n_bricks': len(bricks),
        'n_rejections': int(sum(reasons.values())),
        'rejection_reasons': dict(reasons),
        'n_regenerations': int(out['n_regenerations'] or 0),
        'rollbacks': log['rollbacks'],
        'n_stability_checks': log['n_stability_checks'],
        'stability_time_s': round(log['stability_time_s'], 3),
        'final_stable_physics': final_stable,
        'final_connected': bool(bricks.is_connected()) if len(bricks) else False,
        'bricks': bricks,
    }


def _boxes(bricks: BrickStructure):
    """(x0, y0, z0, dx, dy, dz) for every brick, in stud units with true height."""
    out = []
    for b in bricks.bricks:
        h, w = b.h, b.w
        out.append((b.x, b.y, b.z * Z_SCALE, h, w, Z_SCALE))
    return out


def _cuboid_faces(x, y, z, dx, dy, dz):
    p = np.array([[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
                  [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]])
    return [p[[0, 1, 2, 3]], p[[4, 5, 6, 7]], p[[0, 1, 5, 4]],
            p[[2, 3, 7, 6]], p[[1, 2, 6, 5]], p[[0, 3, 7, 4]]]


def _draw(ax, bricks: BrickStructure, n_show: int | None = None, highlight_last: bool = False,
          scores: np.ndarray | None = None):
    boxes = _boxes(bricks)
    n_show = len(boxes) if n_show is None else n_show
    for i, (x, y, z, dx, dy, dz) in enumerate(boxes[:n_show]):
        if scores is not None:
            b = bricks.bricks[i]
            s = float(scores[b.slice].max())
            color = plt.cm.RdYlGn_r(min(s, 1.0))
        else:
            color = PALETTE[i % len(PALETTE)]
        edge = 'black' if not (highlight_last and i == n_show - 1) else '#00E5FF'
        lw = 0.4 if edge == 'black' else 1.6
        ax.add_collection3d(Poly3DCollection(_cuboid_faces(x, y, z, dx, dy, dz), facecolors=color,
                                             edgecolors=edge, linewidths=lw, alpha=1.0))
        # Studs as small squares on top, so it reads as LEGO even in a still image.
        for sx in range(int(dx)):
            for sy in range(int(dy)):
                cx, cy = x + sx + 0.5, y + sy + 0.5
                top = z + dz
                ax.add_collection3d(Poly3DCollection(
                    _cuboid_faces(cx - 0.3, cy - 0.3, top, 0.6, 0.6, 0.2),
                    facecolors=color, edgecolors='black', linewidths=0.15))
    # Fixed view box from the whole structure, so GIF frames do not jump.
    if boxes:
        a = np.array(boxes)
        lo = a[:, :3].min(0)
        lo[2] = 0
        hi = np.maximum((a[:, :3] + a[:, 3:]).max(0), lo + 1)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(0, hi[2] + 0.2)
        # Equal units on all axes, box fitted to the model (less white space).
        ax.set_box_aspect(tuple(hi - lo + np.array([0, 0, 0.2])), zoom=0.9)
    else:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()


def render_png(bricks: BrickStructure, path: str, title: str | None = None,
               scores: np.ndarray | None = None, size: float = 5.0):
    fig = plt.figure(figsize=(size, size), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    _draw(ax, bricks, scores=scores)
    if title:
        fig.suptitle(title, fontsize=10, y=0.95)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    plt.close(fig)


def render_gif(bricks: BrickStructure, path: str, title: str | None = None, max_frames: int = 60,
               fps: int = 6):
    """Brick-by-brick build animation (the order in which the model generated them)."""
    import imageio.v2 as imageio
    n = len(bricks)
    if n == 0:
        return
    steps = sorted(set(np.linspace(1, n, min(n, max_frames)).round().astype(int)))
    frames = []
    for k in steps:
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        _draw(ax, bricks, n_show=k, highlight_last=True)
        fig.suptitle(f'{title + ": " if title else ""}brick {k}/{n}', fontsize=9, y=0.95)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    frames += [frames[-1]] * fps * 2  # hold the finished model for 2 s
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def plotly_figure(bricks: BrickStructure, title: str = ''):
    """Interactive 3D model (rotate/zoom in the browser) for the Gradio app."""
    import plotly.graph_objects as go
    fig = go.Figure()
    # Unit-cube triangulation (12 triangles) reused for every brick.
    I = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
    J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 4, 6, 7]
    for i, (x, y, z, dx, dy, dz) in enumerate(_boxes(bricks)):
        vx = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
        vy = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
        vz = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
        b = bricks.bricks[i]
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, color=PALETTE[i % len(PALETTE)],
                                flatshading=True, opacity=1.0, hoverinfo='text',
                                hovertext=f'#{i + 1}: {b.to_txt().strip()}', showscale=False))
        # Brick outlines
        ex, ey, ez = [], [], []
        for a, c in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            ex += [vx[a], vx[c], None]
            ey += [vy[a], vy[c], None]
            ez += [vz[a], vz[c], None]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=2),
                                   hoverinfo='skip', showlegend=False))
    fig.update_layout(title=title, showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False,
                                 zaxis_visible=False))
    return fig
