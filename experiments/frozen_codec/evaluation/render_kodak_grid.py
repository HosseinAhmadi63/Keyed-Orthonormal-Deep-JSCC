from pathlib import Path
import os
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
ROOT = Path(os.environ.get('JSCC_ASSETS', str(Path(__file__).resolve().parents[1]))) / 'evaluation'
RESULTS = ROOT / 'results/kodak'
document = json.loads((RESULTS / 'kodak_per_image.json').read_text())
rows = document['images']
indices = [0, 4, 8, 12, 16, 20]
models = [('ground_truth', 'Native source'), ('bob', 'Bob: correct key'), ('eve_no_nonce_transfer', 'Eve: archived no-nonce'), ('eve_bob_style_transfer', 'Eve: archived Bob-style'), ('mean_training_image', 'CIFAR training-image prior')]
fontpath = 'DejaVuSans.ttf'
regular = ImageFont.truetype(fontpath, 30)
small = ImageFont.truetype(fontpath, 25)
title = ImageFont.truetype(fontpath, 40)
cellw, cellh, labelw, top = (400, 330, 330, 130)
canvas = Image.new('RGB', (labelw + len(indices) * cellw, len(models) * cellh + top), 'white')
draw = ImageDraw.Draw(canvas)
draw.text((labelw, 15), 'Native Kodak images through independent 32 × 32 packets', font=title, fill='black')
draw.text((labelw, 72), 'Bob: 12 dB; Eve: 20 dB; HKDF-SHA256/AES-256-CTR parameter generation', font=regular, fill='black')
for row, (model, label) in enumerate(models):
    draw.text((15, top + row * cellh + 125), label, font=small, fill='black')
    for col, index in enumerate(indices):
        record = rows[index]
        name = record['name']
        path = ROOT / 'data/kodak' / name if model == 'ground_truth' else RESULTS / 'reconstructions' / model / name
        image = Image.open(path).convert('RGB')
        image = ImageOps.contain(image, (cellw - 18, cellh - 62), Image.Resampling.LANCZOS)
        x = labelw + col * cellw + (cellw - image.width) // 2
        y = top + row * cellh + (cellh - 62 - image.height) // 2
        canvas.paste(image, (x, y))
        if model == 'ground_truth':
            text = name
        else:
            metric = record['models'][model]
            text = f"{metric['psnr']:.2f} dB; SSIM {metric['ssim']:.3f}"
        draw.text((labelw + col * cellw + 15, top + (row + 1) * cellh - 50), text, font=small, fill='black')
canvas.save(RESULTS / 'kodak_deterministic_examples.png')
canvas.save(RESULTS / 'kodak_deterministic_examples.pdf', resolution=180)
print('Saved predefined 6-image Kodak contact sheet.', flush=True)
