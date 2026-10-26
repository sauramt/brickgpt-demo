"""
make_gallery.py: builds results/gallery.html, a single offline page with every generated
structure (build GIF, stress render, numbers), as the fallback for the live demo.

Input : reference_runs/runs/*/metrics.json (committed runs from the RTX 5060 laptop) and
        results/runs/*/metrics.json (your own runs), plus the PNG/GIF next to each
Output: results/gallery.html (open it in any browser; relative links, no internet needed)
Run   : python make_gallery.py      (or ./demo.sh gallery, which also opens it)
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def main():
    runs = []
    for source, label in ((ROOT / 'reference_runs', 'reference run'), (OUT, 'your run')):
        for m in sorted(source.glob('runs/*/metrics.json')):
            r = json.loads(m.read_text())
            # Links are relative to results/, where the page is written.
            r['dir'] = '../' + m.parent.relative_to(ROOT).as_posix()
            r['source'] = label
            runs.append(r)
    OUT.mkdir(exist_ok=True)
    # Group the two modes of the same prompt+seed side by side.
    runs.sort(key=lambda r: (r['source'], r['prompt'], r['seed'], r['mode'] != 'full'))

    cards = []
    for r in runs:
        rb = ', '.join(f"{x['had']}→{x['kept']}" for x in r['rollbacks']) or 'none'
        ok = 'stable' if r['final_stable_physics'] else 'UNSTABLE'
        mode = {'full': 'BrickGPT (rollback on)', 'no_rollback': 'No rollback',
                'connectivity': 'Connectivity rollback'}.get(r['mode'], r['mode'])
        imgs = ''.join(f'<img src="{r["dir"]}/{f}" alt="{f}">' for f in ('build.gif', 'render_stability.png')
                       if (OUT / r['dir'] / f).resolve().exists())
        cards.append(f'''
<div class="card {'bad' if not r['final_stable_physics'] else ''}">
  <div class="mode">{html.escape(mode)} · seed {r['seed']} · {r['source']} · {html.escape(r.get('gpu', ''))}</div>
  <div class="prompt">“{html.escape(r['prompt'])}”</div>
  <div class="imgs">{imgs}</div>
  <div class="nums"><b>{r['n_bricks']}</b> bricks · <b>{r['seconds']:.0f} s</b> ·
  {r['n_rejections']} rejected · rollbacks: {r['n_regenerations']} ({rb}) ·
  physics: <b class="{ok}">{ok}</b></div>
</div>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>BrickGPT demo runs</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}} .card.bad{{border-color:#c91a09}}
.mode{{font-size:12px;color:#666}} .prompt{{font-weight:600;margin:4px 0}}
.imgs{{display:flex;gap:6px}} .imgs img{{width:50%;height:auto;border:1px solid #eee}}
.nums{{font-size:13px;margin-top:6px}} .stable{{color:#237841}} .UNSTABLE{{color:#c91a09}}
</style></head><body>
<h1>BrickGPT: generated structures ({len(runs)} runs)</h1>
<p>Left: build order (GIF). Right: stress score from the force-equilibrium analysis
(green = safe, red = connection at its limit or failing).</p>
<div class="grid">{''.join(cards)}</div></body></html>'''
    (OUT / 'gallery.html').write_text(page, encoding='utf-8')
    print(f'wrote {OUT / "gallery.html"} ({len(runs)} runs)')


if __name__ == '__main__':
    main()
