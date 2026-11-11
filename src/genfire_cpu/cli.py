"""Command line entry point: ``genfire-cpu``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import io
from .config import ReconstructionConfig, RESOLUTION_MODES
from .fft import RealFFT3D, default_threads, wisdom_dir
from .reconstruct import reconstruct


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="genfire-cpu",
        description="CPU-parallel GENFIRE reconstruction for large 3D grids.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a reconstruction")
    r.add_argument("projections", help="projection stack (.mrc/.npy/.mat)")
    r.add_argument("angles", help="Euler angles in degrees (.txt/.npy/.mat)")
    r.add_argument("-o", "--output", default="genfire_rec.mrc")
    r.add_argument("--support", default=None)
    r.add_argument("--initial-object", default=None)
    r.add_argument("--pixel-size", type=float, required=True,
                   help="projection pixel size in angstroms, e.g. 0.3")
    r.add_argument("--oversampling", type=float, default=2.0)
    r.add_argument("-n", "--iterations", type=int, default=100)
    r.add_argument("-t", "--threads", type=int, default=None,
                   help="FFTW worker threads (default: all %d logical cores)"
                        % default_threads())
    r.add_argument("--precision", choices=["single", "double"], default="single")
    r.add_argument("--planner", default="FFTW_MEASURE",
                   choices=["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT"])
    r.add_argument("--out-of-place", action="store_true",
                   help="do not alias the volume and its spectrum (uses ~1.5x memory)")
    r.add_argument("--interpolation-cutoff", type=float, default=0.7)
    r.add_argument("--resolution-mode", default="none", choices=sorted(RESOLUTION_MODES))
    r.add_argument("--no-positivity", action="store_true")
    r.add_argument("--no-support", action="store_true")
    r.add_argument("--support-shape", choices=["box", "sphere"], default="box",
                   help="shape of the default support (ignored with --support)")
    r.add_argument("--soft-edge", type=float, default=0.0, metavar="PX",
                   help="taper the support over this many voxels; removes the "
                        "step at the support wall that otherwise rings across "
                        "the whole Fourier volume")
    r.add_argument("--no-prune", action="store_true",
                   help="do not skip the FFT passes over the support's zeros")
    r.add_argument("--rfree", action="store_true", help="track R-free")
    r.add_argument("--max-gigabytes", type=float, default=None)
    r.add_argument("--dry-run", action="store_true",
                   help="print the plan and memory estimate, then stop")
    r.add_argument("-q", "--quiet", action="store_true")

    b = sub.add_parser("bench-fft", help="time the 3D FFT at a given size")
    b.add_argument("size", type=int, nargs="?", default=512)
    b.add_argument("-t", "--threads", type=int, action="append", default=None)
    b.add_argument("--precision", choices=["single", "double"], default="single")
    b.add_argument("--planner", default="FFTW_MEASURE")
    b.add_argument("--repeats", type=int, default=3)

    sub.add_parser("info", help="report the environment genfire-cpu will use")
    return p


def _config_from_args(a) -> ReconstructionConfig:
    return ReconstructionConfig(
        pixel_size=a.pixel_size,
        oversampling=a.oversampling,
        num_iterations=a.iterations,
        threads=a.threads,
        dtype=np.float32 if a.precision == "single" else np.float64,
        planner=a.planner,
        in_place_fft=not a.out_of_place,
        interpolation_cutoff=a.interpolation_cutoff,
        resolution_mode=a.resolution_mode,
        positivity=not a.no_positivity,
        use_support=not a.no_support,
        support_shape=a.support_shape,
        support_soft_edge=a.soft_edge,
        prune_fft=not a.no_prune,
        calculate_rfree=a.rfree,
        max_gigabytes=a.max_gigabytes,
        verbose=not a.quiet,
    )


def cmd_run(a) -> int:
    cfg = _config_from_args(a)
    projections = io.read_projections(a.projections, dtype=cfg.dtype)
    angles = io.read_angles(a.angles)
    if angles.shape[0] != projections.shape[2]:
        raise SystemExit("got %d projections but %d angles"
                         % (projections.shape[2], angles.shape[0]))

    if a.dry_run:
        print(cfg.describe(projections.shape[0],
                           num_projections=projections.shape[2]))
        print("projections          : %d" % projections.shape[2])
        return 0

    support = io.read_volume(a.support) if a.support else None
    initial = io.read_volume(a.initial_object) if a.initial_object else None

    result = reconstruct(projections, angles, cfg,
                         support=support, initial_object=initial)
    io.save_results(a.output, result, pixel_size=cfg.pixel_size)
    if not a.quiet:
        print("wrote %s" % a.output)
    return 0


def cmd_bench_fft(a) -> int:
    import time

    dtype = np.float32 if a.precision == "single" else np.float64
    threads = a.threads or [1, max(1, default_threads() // 2), default_threads()]
    n = a.size
    print("%d^3  %s  planner=%s" % (n, np.dtype(dtype).name, a.planner))
    for th in threads:
        plan = RealFFT3D((n, n, n), dtype=dtype, threads=th, planner=a.planner)
        rng = np.random.default_rng(0)
        # generate in the working dtype; a float64 draw at 1024^3 is an 8.6 GB
        # temporary on top of the buffer we are trying to measure
        plan.real[...] = rng.standard_normal((n, n, n), dtype=dtype)
        fwd = []
        bwd = []
        for _ in range(a.repeats):
            t = time.perf_counter(); plan.forward(); fwd.append(time.perf_counter() - t)
            t = time.perf_counter(); plan.backward(); bwd.append(time.perf_counter() - t)
        print("  threads=%-3d  forward %6.3f s  backward %6.3f s  "
              "round trip %6.3f s  (plan %.1f s, %.2f GB)"
              % (th, min(fwd), min(bwd), min(fwd) + min(bwd),
                 plan.plan_seconds, plan.nbytes / 2 ** 30))
        del plan
    return 0


def cmd_info(a) -> int:
    from . import __version__
    from .fft import HAVE_PYFFTW

    print("genfire-cpu %s" % __version__)
    print("numpy       %s" % np.__version__)
    print("FFT backend %s" % ("pyfftw" if HAVE_PYFFTW else "scipy.fft (install "
                              "pyfftw for a large speedup)"))
    print("threads     %d (override with GENFIRE_NUM_THREADS)" % default_threads())
    print("wisdom      %s" % wisdom_dir())
    try:
        import psutil

        vm = psutil.virtual_memory()
        print("memory      %.1f GB total, %.1f GB available"
              % (vm.total / 2 ** 30, vm.available / 2 ** 30))
    except ImportError:
        pass
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "bench-fft": cmd_bench_fft, "info": cmd_info}[
        args.command](args)


if __name__ == "__main__":
    sys.exit(main())
