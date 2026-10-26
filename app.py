"""
app.py: local web demo of BrickGPT (text -> stable brick structure), for the live demo.

Type a description, pick a seed, press Generate. The page shows a rotatable 3D model, the
build animation, the inference log (rejections, physics rollbacks, time) and the brick
list, and saves every run under results/runs/ like run_demo.py does.

Run: ./demo.sh app   (or, venv active: python app.py)   then open http://127.0.0.1:7860
Options: --share (temporary public gradio.live link), --port 7860
"""

import argparse
import json

import gradio as gr

import demo_lib
import run_demo

MODEL = None  # loaded once at start-up (~5 GB of weights in fp32 on the GPU)


def generate(prompt: str, seed: float, rollback: bool):
    prompt = (prompt or '').strip()
    if not prompt:
        raise gr.Error('Type a description first.')
    mode = 'full' if rollback else 'no_rollback'
    MODEL.max_regenerations = 100 if rollback else 0
    MODEL.use_gurobi = True  # physics model (HiGHS backend)
    r = run_demo.run_one(MODEL, mode, prompt, int(seed), render=True)
    d = run_demo.OUT / 'runs' / r['name']
    bricks_txt = (d / 'bricks.txt').read_text()
    from brickgpt.data import BrickStructure
    fig = demo_lib.plotly_figure(BrickStructure.from_txt(bricks_txt))
    rb = ', '.join(f"{x['had']}→{x['kept']}" for x in r['rollbacks']) or 'none'
    log = (f"**{r['n_bricks']} bricks** in **{r['seconds']:.1f} s** on {r['gpu']}  \n"
           f"Rejected bricks during sampling: {r['n_rejections']} {r['rejection_reasons'] or ''}  \n"
           f"Physics rollbacks: {r['n_regenerations']} (bricks had→kept: {rb})  \n"
           f"Stability analyses: {r['n_stability_checks']} ({r['stability_time_s']:.2f} s)  \n"
           f"Final structure physically stable: **{r['final_stable_physics']}**")
    gif = d / 'build.gif'
    stab = d / 'render_stability.png'
    return fig, str(gif) if gif.exists() else None, str(stab) if stab.exists() else None, log, bricks_txt


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    args = ap.parse_args()

    print('Loading BrickGPT ...', flush=True)
    MODEL = demo_lib.load_model(rollback=True, physics=True)
    print(f'Model on {MODEL.device}', flush=True)

    with gr.Blocks(title='BrickGPT demo') as ui:
        gr.Markdown('## BrickGPT: text → physically stable brick structure  \n'
                    'Llama-3.2-1B + BrickGPT LoRA · rejection sampling · physics-informed rollback')
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label='Describe an object', lines=2,
                                    value=run_demo.DEMO_PROMPTS[0])
                seed = gr.Number(label='Seed', value=42, precision=0)
                rollback = gr.Checkbox(label='Physics-informed rollback', value=True)
                btn = gr.Button('Generate', variant='primary')
                # Each example sets prompt + the seed used for the reference runs (bench: seed 7).
                gr.Examples([[p, 7 if 'bench' in p else 42] for p in run_demo.DEMO_PROMPTS],
                            inputs=[prompt, seed])
                log = gr.Markdown()
            with gr.Column(scale=2):
                plot = gr.Plot(label='3D model (drag to rotate)')
                with gr.Row():
                    gif = gr.Image(label='Build order', type='filepath')
                    stab = gr.Image(label='Stress score (green = safe, red = fails)', type='filepath')
                bricks = gr.Textbox(label='Model output (bricks.txt)', lines=8)
        btn.click(generate, [prompt, seed, rollback], [plot, gif, stab, log, bricks])
    ui.queue().launch(server_name='127.0.0.1', server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
