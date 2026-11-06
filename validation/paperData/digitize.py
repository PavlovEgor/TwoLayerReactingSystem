#!/usr/bin/env python3
"""Digitize the raster figures of Bratsun et al. 2023 into CSV files.

The figures are extracted from the PDF in ../../doc with pdfimages. The axes
were calibrated by hand from the tick labels (pixel coordinates below); the
curves and the symbols are then found by colour.

Usage: digitize.py [path/to/paper.pdf]
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# Index of the images in `pdfimages -f 9 -l 12`
FIG2A, FIG2B, FIG3A, FIG3B, FIG4A, FIG4B, FIG4C = 0, 1, 2, 3, 4, 5, 6


def load_figures(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdfimages", "-png", "-f", "9", "-l", "12", str(pdf), f"{tmp}/f"],
            check=True,
        )
        return [
            np.asarray(Image.open(p).convert("RGB")).astype(float)
            for p in sorted(Path(tmp).glob("f-*.png"))
        ]


def write_csv(name, header, rows, note=""):
    with (HERE / name).open("w", newline="") as f:
        if note:
            f.write(f"# {note}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


# Colour masks
def is_red(i):
    return (i[..., 0] > 190) & (i[..., 1] < 90) & (i[..., 2] < 90)


def is_green(i):
    return (i[..., 1] > 190) & (i[..., 0] < 120) & (i[..., 2] < 120)


def is_blue(i):
    return (i[..., 2] > 190) & (i[..., 0] < 100) & (i[..., 1] < 130)


def is_brown(i):
    return (
        (i[..., 0] > 120)
        & (i[..., 0] < 200)
        & (i[..., 1] < 100)
        & (i[..., 2] < 100)
        & (i[..., 1] > 30)
    )


def is_dark(i, level=95):
    return i.max(axis=-1) < level


def trace(mask):
    """Mean row of the mask pixels for each column."""
    rows = []
    for x in range(mask.shape[1]):
        ys = np.nonzero(mask[:, x])[0]
        if len(ys):
            rows.append((x, ys.mean()))
    return np.array(rows)


def fig2a(img):
    # z = (x - 1)/68.2, rho = 0.8 - (y - 1)/545
    legend = np.zeros(img.shape[:2], bool)
    legend[:165, 295:] = True
    # Arrow and dashed line of omega are grey, excluded by the dark threshold
    curves = {
        "rho": is_dark(img, 80) & ~legend,
        "A": is_red(img) & ~legend,
        "B": is_green(img) & ~legend,
        "C": is_blue(img) & ~legend,
        "R": is_brown(img) & ~legend,
    }
    rows = []
    for name, mask in curves.items():
        mask[431:, :] = False  # Base line
        for x, y in trace(mask):
            rows.append(
                (name, round((x - 1) / 68.2, 4), round(0.8 - (y - 1) / 545, 4))
            )
    write_csv(
        "fig2a_baseState_t5.csv",
        ["curve", "z", "value"],
        rows,
        "Fig 2a: gamma=0.4, t*=5, omega marked at z=2.33. "
        "rho is shifted in the figure: see README",
    )


def fig2b(img):
    # log10(t) = (x - 1)/82 - 4, log10(v) = (31 - y)/100
    legend = np.zeros(img.shape[:2], bool)
    legend[:120, :180] = True
    curves = {
        "omega": is_red(img) & ~legend,
        "dVw_over_Vrz": is_blue(img) & ~legend,
        "dVw": is_green(img) & ~legend,
    }
    rows = []
    for name, mask in curves.items():
        for x, y in trace(mask):
            rows.append(
                (
                    name,
                    float(f"{10 ** ((x - 1) / 82 - 4):.5g}"),
                    float(f"{10 ** ((31 - y) / 100):.5g}"),
                )
            )
    write_csv(
        "fig2b_waterRelease.csv",
        ["curve", "t", "value"],
        rows,
        "Fig 2b: gamma=0.4, base state",
    )


def fig3a(img):
    # A0 = (x - 0.5)/66.75, B0 = (394 - y)/65.25
    # Seeds were read from the figure, centres are refined by colour
    seeds = {
        "DL": [(208, 101), (209, 133), (279, 57), (134, 232)],
        "DCC": [
            (348, 7), (348, 55), (349, 79), (421, 53), (421, 76), (209, 162),
            (209, 181), (210, 201), (210, 217), (210, 232), (280, 133),
            (280, 162), (280, 181), (348, 133), (134, 277),
        ],
        "SW": [
            (349, 156), (421, 133), (421, 185), (282, 231), (349, 232),
            (421, 231), (208, 263), (280, 264), (208, 278), (349, 279),
            (208, 293), (280, 293), (134, 314), (208, 314), (349, 315),
            (422, 316), (134, 340), (208, 341), (349, 340), (134, 355),
            (208, 357), (134, 370),
        ],
        "RT": [
            (349, 356), (349, 367), (349, 384), (421, 356), (421, 366),
            (208, 373), (208, 383), (134, 384),
        ],
    }
    masks = {"DL": is_blue, "DCC": is_red, "SW": is_dark, "RT": is_green}
    marked = {(134, 232), (280, 133), (134, 355), (208, 383)}

    rows = []
    for regime, points in seeds.items():
        mask = masks[regime](img)
        for sx, sy in points:
            win = 9
            y0, x0 = max(sy - win, 0), max(sx - win, 0)
            ys, xs = np.nonzero(mask[y0 : sy + win + 1, x0 : sx + win + 1])
            if len(xs) < 4:
                raise RuntimeError(f"No symbol at {sx, sy}")
            x, y = xs.mean() + x0, ys.mean() + y0
            rows.append(
                (
                    regime,
                    round((x - 0.5) / 66.75, 2),
                    round((394 - y) / 65.25, 2),
                    int((sx, sy) in marked),
                )
            )
    write_csv(
        "fig3a_stabilityMap_experiment.csv",
        ["regime", "A0_molL", "B0_molL", "shownInFig5"],
        rows,
        "Fig 3a: open symbols, experiment. Boundaries: gamma=1.2, 0.76, 0.15",
    )


def fig3b(img):
    # (name, gamma, Krho, yTop, yBottom, value at y1, y1, value at y2, y2,
    #  excluded label box x0, x1, y0, y1)
    panels = [
        ("DL", 2.0, 0.7, 1, 73, 5.0, 0, 3.8, 62, (8, 50, 6, 34)),
        ("DCC", 1.0, 0.9, 79, 150, 2.4, 88, 2.2, 130, (8, 67, 86, 115)),
        ("DCC-SW", 0.76, 1.0, 158, 230, 2.3, 161, 2.1, 216, None),
        ("SW", 0.4, 1.3, 237, 308, 1.3, 237, 0.9, 300, (8, 52, 273, 304)),
        ("RT", 0.1, 2.8, 316, 387, 0.7, 315, 0.3, 372, (12, 54, 351, 382)),
    ]
    rows = []
    for name, gamma, krho, top, bottom, v1, y1, v2, y2, box in panels:
        mask = np.zeros(img.shape[:2], bool)
        mask[top : bottom + 1] = is_dark(img, 110)[top : bottom + 1]
        if box:
            x0, x1, b0, b1 = box
            mask[b0 : b1 + 1, x0 : x1 + 1] = False
        for x, y in trace(mask):
            value = v1 + (v2 - v1) * (y - y1) / (y2 - y1)
            rows.append(
                (name, gamma, krho, round((x - 0.5) / 44.4, 3), round(value, 4))
            )
    write_csv(
        "fig3b_densityProfiles.csv",
        ["regime", "gamma", "Krho_paper", "z", "rho"],
        rows,
        "Fig 3b: base state density, eq (2.8), t* not stated in the paper. "
        "The DCC-SW panel axis does not match gamma*RB, see README",
    )


def fig4(images):
    """Front depth in units of h for the panels of Fig 4."""
    # Sub-panel x ranges, interface row, pixels per h (283 px for L = 40)
    panel_x = [(1, 283), (301, 584), (603, 886)]
    interface_row = 132
    scale = 283 / 40

    # A <= threshold is violet in the colour map of the figure
    def coloured(i):
        violet = (i[..., 0] < 175) & (i[..., 1] < 50) & (i[..., 2] > 200)
        grey = (i.max(-1) - i.min(-1)) < 60
        return ~(violet | grey)

    sets = [
        ("DCC", 1.0, images[FIG4A], (5, 10, 20)),
        ("SW", 0.4, images[FIG4B], (1, 5, 10)),
        ("RT", 0.1, images[FIG4C], (0.1, 0.5, 1)),
    ]
    rows = []
    for regime, gamma, img, times in sets:
        for (x0, x1), t in zip(panel_x, times):
            mask = coloured(img[interface_row + 3 : 412, x0 + 2 : x1 - 1])
            # Remove the isolated pixels of JPEG noise around the labels
            neigh = (
                mask[:-2, 1:-1].astype(int) + mask[2:, 1:-1] + mask[1:-1, :-2]
                + mask[1:-1, 2:]
            )
            clean = np.zeros_like(mask)
            clean[1:-1, 1:-1] = mask[1:-1, 1:-1] & (neigh >= 3)
            depth = np.array(
                [
                    (np.nonzero(c)[0].max() + 3) / scale if c.any() else 0.0
                    for c in clean.T
                ]
            )
            profile = depth - depth.mean()
            spectrum = np.abs(np.fft.rfft(profile))[1:]
            cells = int(np.argmax(spectrum) + 1)
            rows.append(
                (
                    regime, gamma, t,
                    round(float(depth.mean()), 2),
                    round(float(depth.std()), 2),
                    round(float(np.median(depth)), 2),
                    round(float(depth.max()), 2),
                    cells,
                )
            )
    write_csv(
        "fig4_frontDepth.csv",
        ["regime", "gamma", "t", "meanDepth", "stdDepth", "medianDepth",
         "maxDepth", "dominantCells"],
        rows,
        "Fig 4: depth of the coloured region below the interface, units of h",
    )


def main():
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pdf is None:
        pdf = next((HERE.parent.parent / "doc").glob("*.pdf"))
    images = load_figures(pdf)
    fig2a(images[FIG2A])
    fig2b(images[FIG2B])
    fig3a(images[FIG3A])
    fig3b(images[FIG3B])
    fig4(images)


if __name__ == "__main__":
    main()
