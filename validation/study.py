#!/usr/bin/env python3
"""Parametric study of Bratsun et al. 2023 with foamlib.

Clones the tutorial for each set of parameters, runs the cases, and calls the
post-processing (analysis.py) to compare the results with the paper.

Usage:
    study.py --profile {smoke,short,full} [--cpus N] [--cpus-per-case N]
             [--slurm] [--only NAME_SUBSTRING] [--post-only]
"""

import argparse
import asyncio
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from foamlib import AsyncFoamCase, AsyncSlurmFoamCase

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "tutorials/twoLayerReactingFoam/pentanolWaterHCl-NaOH"

# Time unit h^2/DA of the template, s
TAU = (1.2e-3) ** 2 / 3.3445e-9

VM_FORMULA = 4.5e-5  # eq (2.14), m3/mol
VM_FIG2B = 2.26e-5  # the value consistent with the curves of Fig 2b


@dataclass(frozen=True)
class Spec:
    """Parameters of a single case. Concentrations in mol/L, times in t*."""

    name: str
    group: str
    A0: float = 1.0
    gamma: float = 0.4
    nx: int = 200
    ny: int = 400
    t_end: float = 10.0
    write: float = 1.0
    water: bool = True
    Vm: float = VM_FORMULA
    amplitude: float = 1e-3
    seed: int = 1234
    width: float = 0.048  # m
    max_dt: float = 0.5  # s
    serial: bool = False


def base_state(gamma, ny, t_end):
    """Quiescent one-dimensional state: Fig 2 and Fig 3b."""
    return Spec(
        f"base_g{gamma:g}", "base", gamma=gamma, nx=1, ny=ny, t_end=t_end,
        write=t_end / 5, amplitude=0, width=0.0024, max_dt=0.1, serial=True,
    )


def nonlinear(regime, gamma, t_end, write, nx, ny, **kwargs):
    """Numerical point of the map at A0 = 1 M: Fig 4."""
    return Spec(f"{regime}_g{gamma:g}_{nx}x{ny}", "fig4", gamma=gamma,
                nx=nx, ny=ny, t_end=t_end, write=write, **kwargs)


def fig4_set(nx, ny):
    return [
        nonlinear("DCC", 1.0, 20, 5, nx, ny),
        nonlinear("SW", 0.4, 10, 1, nx, ny),
        nonlinear("RT", 0.1, 1, 0.1, nx, ny),
    ]


def gamma_sweep(nx, ny, gammas):
    return [
        Spec(f"sweep_g{g:g}_{nx}x{ny}", "sweep", gamma=g, nx=nx, ny=ny,
             t_end=2.0 if g <= 0.15 else 10.0,
             write=0.2 if g <= 0.15 else 1.0)
        for g in gammas
    ]


def experiment_set(nx, ny):
    """Experimental patterns of Fig 5, A0 and the final time of the caption."""
    points = [
        ("DL", 2.0, 1.25, 2230, 260),
        ("DCC", 4.2, 0.95, 960, 320),
        ("SW", 2.0, 0.40, 1040, 70),
        ("RT", 3.1, 0.065, 320, 50),
    ]
    return [
        Spec(f"exp_{r}_A{a:g}_g{g:g}_{nx}x{ny}", "exp", A0=a, gamma=g, nx=nx,
             ny=ny, t_end=t / TAU, write=w / TAU, max_dt=0.2)
        for r, a, g, t, w in points
    ]


def water_set(nx, ny, t_end=10.0):
    """The key mechanism of the paper: SW with and without water release."""
    return [
        Spec(f"SW_noWater_{nx}x{ny}", "water", gamma=0.4, nx=nx, ny=ny,
             t_end=t_end, water=False),
        Spec(f"SW_VmFig2b_{nx}x{ny}", "water", gamma=0.4, nx=nx, ny=ny,
             t_end=t_end, Vm=VM_FIG2B),
    ]


def profile_cases(profile):
    if profile == "smoke":
        return [
            base_state(0.4, 400, 1.0),
            Spec("SW_g0.4_50x100", "fig4", gamma=0.4, nx=50, ny=100,
                 t_end=0.5, write=0.1),
        ]
    if profile == "short":
        return (
            [base_state(g, 800, 5.0) for g in (2.0, 1.0, 0.76, 0.4, 0.1)]
            + fig4_set(100, 200)
            + water_set(100, 200, t_end=5.0)
        )
    if profile == "full":
        gammas = (2.0, 1.25, 1.0, 0.9, 0.76, 0.6, 0.25, 0.15, 0.05)
        return (
            [base_state(g, 800, 5.0) for g in (2.0, 1.0, 0.76, 0.4, 0.1)]
            + fig4_set(200, 400)
            + fig4_set(100, 200)
            + water_set(200, 400)
            + gamma_sweep(200, 400, gammas)
            + experiment_set(200, 400)
            + [
                nonlinear("SW", 0.4, 10, 1, 200, 400, seed=s)
                for s in (1, 2)
            ]
        )
    raise ValueError(profile)


def case_dir(run_root, spec):
    seed = "" if spec.seed == 1234 else f"_seed{spec.seed}"
    return run_root / f"{spec.name}{seed}"


async def run_case(spec, run_root, cpus, case_class):
    dst = case_dir(run_root, spec)
    done = dst / ".done"
    if done.exists():
        return dst

    cpus = 1 if spec.serial else cpus
    resume = (dst / "processor0").is_dir()

    if resume:
        case = case_class(dst)
    else:
        if dst.exists():
            shutil.rmtree(dst)
        case = await case_class(TEMPLATE).clone(dst)

        mesh = case.block_mesh_dict
        mesh["nx"] = spec.nx
        mesh["ny"] = spec.ny
        mesh["L"] = spec.width

        props = case.file("constant/twoLayerProperties")
        props["A0"] = spec.A0 * 1000
        props["B0"] = spec.gamma * spec.A0 * 1000
        props["molarVolume"] = spec.Vm
        props["waterRelease"] = spec.water
        props["perturbation"]["amplitude"] = spec.amplitude
        props["perturbation"]["seed"] = spec.seed

        control = case.control_dict
        control["endTime"] = spec.t_end * TAU
        control["writeInterval"] = spec.write * TAU
        control["maxDeltaT"] = spec.max_dt

        case.decompose_par_dict["numberOfSubdomains"] = cpus

        await case.restore_0_dir()
        await case.block_mesh()
        await case.run(["setTwoLayerFields"], parallel=False)

        if cpus > 1:
            await case.decompose_par()

    await case.run(["twoLayerReactingFoam"], parallel=cpus > 1, cpus=cpus)

    if cpus > 1:
        await case.reconstruct_par()
        for proc in dst.glob("processor*"):
            shutil.rmtree(proc)

    done.write_text(str(asdict(spec)) + "\n")
    return dst


async def run_all(specs, run_root, cpus_per_case, case_class):
    run_root.mkdir(parents=True, exist_ok=True)

    async def one(spec):
        await run_case(spec, run_root, cpus_per_case, case_class)
        print(f"finished: {spec.name} (seed {spec.seed})", flush=True)

    results = await asyncio.gather(
        *(one(s) for s in specs), return_exceptions=True
    )
    failed = [
        (s.name, r) for s, r in zip(specs, results) if isinstance(r, Exception)
    ]
    for name, error in failed:
        print(f"FAILED: {name}: {error}", flush=True)
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("smoke", "short", "full"), default="smoke"
    )
    parser.add_argument(
        "--cpus", type=int, default=os.cpu_count(), help="Total number of CPUs"
    )
    parser.add_argument(
        "--cpus-per-case", type=int, default=None,
        help="CPUs of a single case (default: 4 for smoke, 8 otherwise)",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit the runs with foamlib AsyncSlurmFoamCase",
    )
    parser.add_argument("--only", default="", help="Filter by the case name")
    parser.add_argument(
        "--post-only", action="store_true", help="Only run the post-processing"
    )
    args = parser.parse_args()

    run_root = HERE / "run" / args.profile
    results = HERE / "results" / args.profile

    specs = [s for s in profile_cases(args.profile) if args.only in s.name]
    per_case = args.cpus_per_case or (4 if args.profile == "smoke" else 8)
    per_case = min(per_case, args.cpus)

    ok = True
    if not args.post_only:
        case_class = AsyncSlurmFoamCase if args.slurm else AsyncFoamCase
        case_class.max_cpus = args.cpus
        ok = asyncio.run(run_all(specs, run_root, per_case, case_class))

    import analysis

    analysis.report(
        [(s, case_dir(run_root, s)) for s in specs], results, args.profile, TAU
    )
    if not ok:
        raise SystemExit("Some of the cases failed")


if __name__ == "__main__":
    main()
