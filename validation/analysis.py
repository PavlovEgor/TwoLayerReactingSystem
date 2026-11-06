#!/usr/bin/env python3
"""Compare the results of the study with the data digitized from the paper."""

import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from foamlib import FoamCase
from matplotlib.colors import BoundaryNorm
from PIL import Image

import baseState1D

HERE = Path(__file__).resolve().parent
PAPER = HERE / "paperData"

# Fixed order of the categorical colours, ink for the paper data
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "font.size": 10,
        "legend.frameon": False,
    }
)

# Colour levels of Fig 4 of the paper
FIG4_LEVELS = [0, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.125, 0.25, 0.5, 1.0]

# A/A0 at the edge of the violet colour, the same as digitize.py detects
FRONT_THRESHOLD = 0.0025

H = 1.2e-3  # gap width, m
UNIT_U = 3.3445e-9 / H  # velocity unit DA/h, m/s


def read_paper(name):
    with (PAPER / name).open() as f:
        return list(csv.DictReader(line for line in f if line[0] != "#"))


def paper_curve(rows, key, value, x, y):
    pts = sorted(
        (float(r[x]), float(r[y])) for r in rows if r[key] == value
    )
    return np.array(pts)


class Result:
    """Fields and history of a finished case on its structured mesh."""

    def __init__(self, spec, path, tau):
        self.spec, self.path, self.tau = spec, Path(path), tau
        self.case = FoamCase(path)
        self.A0 = spec.A0 * 1000
        self.times = [t for t in self.case if t.time > 0]

    def field(self, time_dir, name):
        s = self.spec
        data = np.asarray(time_dir[name].internal_field, dtype=float)
        return data.reshape(s.ny, s.nx, *data.shape[1:])

    def nearest(self, t_star):
        return min(self.times, key=lambda t: abs(t.time / self.tau - t_star))

    def history(self):
        rows = [
            np.loadtxt(f, ndmin=2)
            for f in sorted(self.path.glob(
                "postProcessing/twoLayerReactingSystem/*/history.dat"))
        ]
        return np.vstack(rows)

    def front(self, time_dir):
        """Depth of the acid region below the interface for each column."""
        s = self.spec
        A = self.field(time_dir, "A")[: s.ny // 2] / self.A0
        dz = 80.0 / s.ny
        below = A > FRONT_THRESHOLD
        # Row 0 is the bottom of the lower layer
        deepest = np.array(
            [np.nonzero(c)[0].min() if c.any() else s.ny // 2 for c in below.T]
        )
        return (s.ny // 2 - deepest) * dz

    def metrics(self):
        rows = []
        for t in self.times:
            depth = self.front(t)
            profile = depth - depth.mean()
            cells = 0
            if len(depth) > 4 and depth.std() > 0:
                cells = int(np.argmax(np.abs(np.fft.rfft(profile))[1:]) + 1)
            U = self.field(t, "U")[: self.spec.ny // 2]
            rows.append(
                {
                    "t": t.time / self.tau,
                    "meanDepth": depth.mean(),
                    "stdDepth": depth.std(),
                    "medianDepth": np.median(depth),
                    "maxDepth": depth.max(),
                    "dominantCells": cells,
                    "maxUlower": np.abs(U).max() / UNIT_U,
                }
            )
        return rows

    def regime(self):
        """Regime from the final state of the lower layer.

        DL: flat front. DCC: the front stays near the diffusive one.
        SW: convective front with an aligned line of the tips. RT: not aligned.
        """
        last = self.metrics()[-1]
        s = self.spec
        if last["t"] < 1.0 and s.gamma > 0.15:
            return "too early"
        base = baseState1D.solve(s.gamma, min(last["t"], 10.0), A0=s.A0,
                                 Vm=s.Vm * 1000, n=500, dt_max=0.01)
        diffusive = base.z[base.A > FRONT_THRESHOLD].max()
        roughness = last["stdDepth"] / max(last["meanDepth"], 1e-9)
        if last["maxUlower"] < 5.0 or last["stdDepth"] < 0.1:
            return "DL"
        if last["meanDepth"] < 1.5 * diffusive:
            return "DCC"
        return "RT" if roughness > 0.25 else "SW"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: f"{v:.4g}" if isinstance(v, float) else v
                 for k, v in row.items()}
            )


def plot_base_state(results, out, summary):
    bases = [r for r in results if r.spec.group == "base"]
    if not bases:
        return

    fig2a = read_paper("fig2a_baseState_t5.csv")
    fig2b = read_paper("fig2b_waterRelease.csv")
    fig3b = read_paper("fig3b_densityProfiles.csv")

    lines = ["## Base state", "",
             "| gamma | t* | Krho OpenFOAM | Krho 1D | Krho paper |"
             " rms rho vs Fig 3b | rms rho vs 1D |",
             "|---|---|---|---|---|---|---|"]

    fig3, axes3 = plt.subplots(len(bases), 1, figsize=(6, 1.6 * len(bases) + 1),
                               sharex=True, squeeze=False, layout="constrained")

    for ax, r in zip(axes3[:, 0], sorted(bases, key=lambda b: -b.spec.gamma)):
        s = r.spec
        last = r.times[-1]
        t_star = last.time / r.tau
        ref = baseState1D.solve(s.gamma, t_star, Vm=s.Vm * 1000)

        ny = s.ny // 2
        dz = 80.0 / s.ny
        z = (np.arange(ny) + 0.5) * dz
        A = r.field(last, "A")[:ny, 0][::-1] / r.A0
        B = r.field(last, "B")[:ny, 0][::-1] / r.A0
        C = r.field(last, "C")[:ny, 0][::-1] / r.A0
        hist = r.history()
        W = hist[-1, 1]
        rho = A + baseState1D.RB * B + baseState1D.RC * W * C
        # Linear extrapolation from the cell centres to the interface
        krho = (1.5 * rho[0] - 0.5 * rho[1]) / (baseState1D.RB * s.gamma)

        pts = np.array([(float(p["z"]), float(p["rho"])) for p in fig3b
                        if float(p["gamma"]) == s.gamma])
        rms_paper = rms_ref = float("nan")
        krho_paper = "-"
        mask = z < 9
        rms_ref = np.sqrt(np.mean((rho[mask] - np.interp(z[mask], ref.z, ref.rho))**2))
        if len(pts):
            pts = pts[np.argsort(pts[:, 0])]
            rms_paper = np.sqrt(np.mean((np.interp(pts[:, 0], z, rho) - pts[:, 1])**2))
            krho_paper = next(p["Krho_paper"] for p in fig3b
                              if float(p["gamma"]) == s.gamma)
            ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.2, ls="--",
                    label="paper, Fig 3b")
        ax.plot(z[mask], rho[mask], color=BLUE, label="OpenFOAM")
        ax.plot(ref.z[ref.z < 9], ref.rho[ref.z < 9], color=ORANGE, lw=1,
                label="1D check")
        ax.set_ylabel(rf"$\rho$, $\gamma$={s.gamma:g}")
        lines.append(
            f"| {s.gamma:g} | {t_star:.1f} | {krho:.3f} | {ref.krho:.3f} |"
            f" {krho_paper} | {rms_paper:.3f} | {rms_ref:.4f} |"
        )

        if s.gamma == 0.4:
            plot_fig2(r, z, A, B, C, s, fig2a, fig2b, hist, out)

    axes3[0, 0].legend(ncols=3, loc="lower right")
    axes3[-1, 0].set_xlabel("z / h")
    fig3.suptitle("Base state density, Fig 3b (t* is not stated in the paper)")
    fig3.savefig(out / "fig3b_density.png", dpi=150)
    plt.close(fig3)

    summary.extend(lines + [""])


def plot_fig2(r, z, A, B, C, s, fig2a, fig2b, hist, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    Da = 1000.0
    t_star = r.times[-1].time / r.tau

    colours = {"A": ORANGE, "B": AQUA, "C": BLUE}
    for name, f in (("A", A), ("B", B), ("C", C)):
        ax.plot(z, f, color=colours[name], label=f"{name}, OpenFOAM")
        pts = paper_curve(fig2a, "curve", name, "z", "value")
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1, ls="--",
                label="paper, Fig 2a" if name == "A" else None)
    ax.plot(z, Da * A * B, color=YELLOW, label="R, OpenFOAM")
    pts = paper_curve(fig2a, "curve", "R", "z", "value")
    ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1, ls="--")
    ax.set(xlim=(0, 6), ylim=(0, 0.5), xlabel="z / h",
           title=rf"Profiles at t*={t_star:.1f}, $\gamma$=0.4")
    ax.legend()

    t = hist[:, 0] / r.tau
    bx.loglog(t, hist[:, 2] / H, color=ORANGE, label=r"$\omega$, OpenFOAM")
    bx.loglog(t, hist[:, 3], color=BLUE, label=r"$\Delta V_w/V_{rz}$, OpenFOAM")
    bx.loglog(t, hist[:, 3] * hist[:, 2] / H, color=AQUA,
              label=r"$\Delta V_w$, OpenFOAM")
    for i, name in enumerate(("omega", "dVw_over_Vrz", "dVw")):
        pts = paper_curve(fig2b, "curve", name, "t", "value")
        bx.loglog(pts[:, 0], pts[:, 1], color=INK, lw=1, ls="--",
                  label="paper, Fig 2b" if i == 0 else None)
    bx.set(xlim=(1e-3, 6), ylim=(1e-4, 3), xlabel="t*",
           title="Water release (the paper's curves are 2 times lower)")
    bx.legend()
    fig.savefig(out / "fig2_baseState.png", dpi=150)
    plt.close(fig)


def paper_panels(pdf_dir):
    """The panels of Fig 4 from the PDF, if it is available."""
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return {}
    panels = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-png", "-f", "11", "-l", "11",
                        str(pdfs[0]), f"{tmp}/f"], check=True)
        files = sorted(Path(tmp).glob("f-*.png"))
        for regime, f in zip(("DCC", "SW", "RT"), files):
            image = np.asarray(Image.open(f).convert("RGB"))
            panels[regime] = [image[:, a:b] for a, b in
                              ((1, 283), (301, 584), (603, 886))]
    return panels


def plot_fig4(results, out, summary):
    runs = [r for r in results if r.spec.group == "fig4"]
    if not runs:
        return

    reference = read_paper("fig4_frontDepth.csv")
    panels = paper_panels(HERE.parent / "doc")
    norm = BoundaryNorm(FIG4_LEVELS, 256)

    lines = ["## Nonlinear regimes (Fig 4), depth of the acid front in h", "",
             "| case | t* | mean, paper | mean | std, paper | std | max, paper | max |",
             "|---|---|---|---|---|---|---|---|"]

    for r in runs:
        s = r.spec
        regime = s.name.split("_")[0]
        times = [float(p["t"]) for p in reference if p["regime"] == regime]
        rows = {round(m["t"], 2): m for m in r.metrics()}
        available = [t for t in times if t <= s.t_end * 1.001] or [s.t_end]
        with_paper = bool(panels.get(regime))

        fig, axes = plt.subplots(
            1 + with_paper, len(available), figsize=(3 * len(available), 4 + 4 * with_paper),
            squeeze=False, layout="constrained",
        )
        for i, t_star in enumerate(available):
            t_dir = r.nearest(t_star)
            A = r.field(t_dir, "A") / r.A0
            ax = axes[0, i]
            ax.contourf(A, levels=FIG4_LEVELS, norm=norm, cmap="rainbow",
                        extent=(0, 40, -40, 40), extend="max")
            ax.set(ylim=(-40, 18.6), aspect=1, xticks=[], yticks=[],
                   title=f"this work, t*={t_dir.time / r.tau:.3g}")
            ax.grid(False)
            if with_paper and i < len(panels[regime]):
                paper_ax = axes[1, i]
                paper_ax.imshow(panels[regime][i])
                paper_ax.set(xticks=[], yticks=[], title=f"paper, t*={t_star:g}")
                paper_ax.grid(False)
            metric = rows.get(round(t_dir.time / r.tau, 2))
            ref = next((p for p in reference if p["regime"] == regime
                        and float(p["t"]) == t_star), None)
            if metric and ref and abs(t_dir.time / r.tau - t_star) < 0.05 * t_star:
                lines.append(
                    f"| {s.name} | {t_star:g} | {ref['meanDepth']} |"
                    f" {metric['meanDepth']:.2f} | {ref['stdDepth']} |"
                    f" {metric['stdDepth']:.2f} | {ref['maxDepth']} |"
                    f" {metric['maxDepth']:.2f} |"
                )
        fig.suptitle(rf"{s.name}: acid A/A0, $\gamma$={s.gamma:g}")
        fig.savefig(out / f"fig4_{s.name}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    colours = {"DCC": BLUE, "SW": ORANGE, "RT": AQUA}
    for r in runs:
        regime = r.spec.name.split("_")[0]
        m = r.metrics()
        ax.plot([x["t"] for x in m], [x["meanDepth"] for x in m],
                color=colours.get(regime, YELLOW), marker="o", ms=4, label=r.spec.name)
    for regime, colour in colours.items():
        pts = [(float(p["t"]), float(p["meanDepth"])) for p in reference
               if p["regime"] == regime]
        ax.plot(*zip(*pts), ls="none", marker="s", ms=9, mfc="none",
                mec=colour, mew=1.5, label=f"{regime}, paper Fig 4")
    ax.set(xscale="log", yscale="log", xlabel="t*", ylabel="mean front depth / h",
           title="Depth of the acid front")
    ax.legend(fontsize=8)
    fig.savefig(out / "fig4_frontDepth.png", dpi=150)
    plt.close(fig)

    summary.extend(lines + [""])


def plot_water(results, out, summary):
    runs = [r for r in results if r.spec.group == "water"
            or (r.spec.group == "fig4" and r.spec.name.startswith("SW"))]
    if len(runs) < 2:
        return

    lines = ["## Water release at gamma = 0.4", "",
             "| case | t* | mean depth | std | std/mean | W |",
             "|---|---|---|---|---|---|"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for r, colour in zip(runs, (BLUE, ORANGE, AQUA, YELLOW)):
        m = r.metrics()
        t = [x["t"] for x in m]
        ax.plot(t, [x["meanDepth"] for x in m], color=colour, marker="o",
                ms=4, label=r.spec.name)
        bx.plot(t, [x["stdDepth"] / max(x["meanDepth"], 1e-9) for x in m],
                color=colour, marker="o", ms=4, label=r.spec.name)
        last = m[-1]
        lines.append(
            f"| {r.spec.name} | {last['t']:.1f} | {last['meanDepth']:.2f} |"
            f" {last['stdDepth']:.2f} |"
            f" {last['stdDepth'] / max(last['meanDepth'], 1e-9):.3f} |"
            f" {r.history()[-1, 1]:.4f} |"
        )
    ax.set(xlabel="t*", ylabel="mean front depth / h", title="Front depth")
    bx.set(xlabel="t*", ylabel="std / mean of the depth",
           title="Roughness of the line of fingertips")
    ax.legend(fontsize=8)
    fig.savefig(out / "water_release.png", dpi=150)
    plt.close(fig)
    summary.extend(lines + [""])


def plot_map(results, out, summary):
    runs = [r for r in results if r.spec.group in ("sweep", "exp", "fig4")
            and r.spec.seed == 1234]
    experiment = read_paper("fig3a_stabilityMap_experiment.csv")

    styles = {"DL": ("^", BLUE), "DCC": ("s", ORANGE), "SW": ("o", INK),
              "RT": ("D", AQUA)}
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    for regime, (marker, colour) in styles.items():
        pts = [(float(p["A0_molL"]), float(p["B0_molL"])) for p in experiment
               if p["regime"] == regime]
        ax.plot(*zip(*pts), ls="none", marker=marker, ms=8, mfc="none",
                mec=colour, label=f"{regime}, experiment")
    for slope in (1.2, 0.76, 0.15):
        ax.plot([0, 6.75], [0, 6.75 * slope], color=MUTED, lw=0.8)

    lines = ["## Regimes", "",
             "| case | A0, M | gamma | regime | expected from the paper |",
             "|---|---|---|---|---|"]
    expected = lambda g: ("DL" if g > 1.2 else "DCC" if g > 0.76
                          else "SW" if g > 0.15 else "RT")
    for r in runs:
        s = r.spec
        regime = r.regime()
        if regime in styles:
            marker, colour = styles[regime]
            ax.plot(s.A0, s.A0 * s.gamma, ls="none", marker=marker, ms=9,
                    color=colour, mec=SURFACE, mew=1.5)
        lines.append(f"| {s.name} | {s.A0:g} | {s.gamma:g} | {regime} |"
                     f" {expected(s.gamma)} |")
    ax.plot([], [], ls="none", marker="o", color=MUTED, ms=9,
            label="this work (filled)")
    ax.set(xlim=(0, 6.75), ylim=(0, 6), xlabel=r"$A_0$, mol/L",
           ylabel=r"$B_0$, mol/L", title="Stability map, Fig 3a")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig3a_map.png", dpi=150)
    plt.close(fig)
    summary.extend(lines + [""])


def report(cases, out, profile, tau):
    out.mkdir(parents=True, exist_ok=True)

    results = []
    missing = []
    for spec, path in cases:
        if (Path(path) / ".done").exists():
            results.append(Result(spec, path, tau))
        else:
            missing.append(spec.name)

    summary = [f"# Validation summary: profile `{profile}`", ""]
    if missing:
        summary += ["Not finished: " + ", ".join(missing), ""]

    metrics = []
    for r in results:
        for row in r.metrics():
            metrics.append({"case": r.spec.name, "seed": r.spec.seed, **row})
    write_csv(out / "metrics.csv", metrics)

    plot_base_state(results, out, summary)
    plot_fig4(results, out, summary)
    plot_water(results, out, summary)
    plot_map(results, out, summary)

    (out / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"Results in {out}")
    print("\n".join(summary))
