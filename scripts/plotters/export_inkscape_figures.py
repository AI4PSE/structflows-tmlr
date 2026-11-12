"""Export the hand-drawn diagrams from the Inkscape canvas images/pgm-structure.svg as vector PDFs.

The canvas holds several diagrams side by side. Each is cropped by giving a copy of the SVG a
viewBox around its region (coordinates in px, as reported by `inkscape --query-all`), then
exported with text converted to paths so the PDF does not depend on installed fonts.

Run: python scripts/plotters/export_inkscape_figures.py   (needs inkscape >= 1.0 on PATH)
"""
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.normpath(os.path.join(HERE, '..', '..', 'images'))
CANVAS = os.path.join(IMAGES, 'pgm-structure.svg')

# output name -> (x0, y0, x1, y1) in px on the canvas
REGIONS = {
    'collider-pgm-dual': (390, 957, 1306, 1452),         # Figure 1, the pipeline
    'collider-counterfactual': (-463, -801, 840, -302),  # counterfactual inference via flow inversion
    'collider-pgm-structured': (312, 57, 762, 373),      # static vs dynamic conditioning
}
MM_PER_PX = 25.4 / 96  # the canvas's user unit is mm


def crop(svg, box):
    x0, y0, x1, y1 = (v * MM_PER_PX for v in box)
    w, h = x1 - x0, y1 - y0
    svg = re.sub(r'viewBox="[^"]*"', f'viewBox="{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}"', svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', rf'\1 width="{w:.3f}mm"', svg, count=1, flags=re.S)
    return re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', rf'\1 height="{h:.3f}mm"', svg, count=1, flags=re.S)


def main():
    svg = open(CANVAS).read()
    with tempfile.TemporaryDirectory() as tmp:
        for name, box in REGIONS.items():
            src = os.path.join(tmp, f'{name}.svg')
            open(src, 'w').write(crop(svg, box))
            out = os.path.join(IMAGES, f'{name}.pdf')
            subprocess.run(['inkscape', src, '--export-type=pdf', '--export-text-to-path',
                            f'--export-filename={out}'], check=True, capture_output=True)
            print('saved', out)


if __name__ == '__main__':
    main()
