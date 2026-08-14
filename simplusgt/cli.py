"""Command line entry point for the Python SimplusGT pipeline."""

from __future__ import annotations

import argparse

from .analysis import stability_report
from .export import export_dashboard_json, export_greybox_json
from .greybox import FrequencyGrid, GreyboxConfig, GreyboxLayerSelection, run_greybox
from .pipeline import run_case
from .plotting import plot_case_fundamentals, plot_greybox_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simplusgt")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run a SimplusGT JSON or Excel case")
    run_parser.add_argument("case")
    run_parser.add_argument("--format", default=None, help="Input format override: json/xlsx/xlsm")
    export_parser = sub.add_parser("export", help="Export dashboard-ready modal results JSON")
    export_parser.add_argument("case")
    export_parser.add_argument("--output", "-o", required=True)
    export_parser.add_argument("--top-states", type=int, default=12)
    export_parser.add_argument("--max-matrix-size", type=int, default=160)
    greybox_parser = sub.add_parser("greybox", help="Run standalone greybox impedance analysis")
    greybox_parser.add_argument("case", nargs="?", help="Case JSON or Excel file")
    greybox_parser.add_argument("--config", help="Greybox config JSON file")
    greybox_parser.add_argument("--output", "-o")
    greybox_parser.add_argument("--layers", default=None, help="Comma-separated layers, all, or admittance-only")
    greybox_parser.add_argument("--modes", default=None, help="Comma-separated mode indices")
    greybox_parser.add_argument("--apparatus", default=None, help="Comma-separated apparatus indices")
    greybox_parser.add_argument("--layer3-apparatus", default=None, help="Comma-separated apparatus indices for Layer 3")
    greybox_parser.add_argument("--sensitivity-lines", default=None, help="Comma-separated line indices for sensitivity Layer 3")
    greybox_parser.add_argument("--freq-min", type=float, default=None)
    greybox_parser.add_argument("--freq-max", type=float, default=None)
    greybox_parser.add_argument("--freq-count", type=int, default=None)
    plot_parser = sub.add_parser("plot", help="Create MATLAB-style fundamental / greybox plots")
    plot_parser.add_argument("case")
    plot_parser.add_argument("--output-dir", "-o", default="Results/plots")
    plot_parser.add_argument("--show", action="store_true", help="Display figures interactively")
    plot_parser.add_argument("--greybox", action="store_true", help="Also plot greybox Ysys/Zsys and Layer 1/2")
    plot_parser.add_argument("--layers", default="app-l1,app-l2,sens-l12", help="Greybox layers when --greybox is set")
    plot_parser.add_argument("--modes", default="auto", help="Greybox mode indices, or 'auto' for oscillatory modes")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_case(args.case, args.format)
        report = stability_report(result.eigenvalues)
        print(f"buses: {result.netlists.buses.shape[0]}")
        print(f"lines: {result.netlists.lines.shape[0]}")
        print(f"states: {result.whole_system_dss.nx}")
        print(f"stable: {report.stable}")
        if result.warnings:
            print("warnings:")
            for warning in result.warnings:
                print(f"- {warning}")
    elif args.command == "export":
        output = export_dashboard_json(args.case, args.output, top_states=args.top_states, max_matrix_size=args.max_matrix_size)
        print(f"exported: {output}")
    elif args.command == "greybox":
        overrides = {}
        if args.layers is not None:
            overrides["layers"] = GreyboxLayerSelection.from_names(args.layers)
        if args.modes is not None:
            overrides["modes"] = _parse_ints(args.modes)
        if args.apparatus is not None:
            overrides["apparatus"] = _parse_ints(args.apparatus)
        if args.layer3_apparatus is not None:
            overrides["layer3_apparatus"] = _parse_ints(args.layer3_apparatus)
        if args.sensitivity_lines is not None:
            overrides["sensitivity_lines"] = _parse_ints(args.sensitivity_lines)
        if args.freq_min is not None or args.freq_max is not None or args.freq_count is not None:
            overrides["frequency_grid"] = FrequencyGrid(
                min_hz=args.freq_min if args.freq_min is not None else 0.1,
                max_hz=args.freq_max if args.freq_max is not None else 1000.0,
                count=args.freq_count if args.freq_count is not None else 80,
            )
        output = export_greybox_json(args.case, args.output, config_path=args.config, **overrides)
        print(f"exported: {output}")
    elif args.command == "plot":
        result = run_case(args.case)
        saved = plot_case_fundamentals(result, output_dir=args.output_dir, show=args.show)
        print(f"fundamental plots: {args.output_dir}")
        for key, path in saved.items():
            if path is not None:
                print(f"- {key}: {path}")
        if args.greybox:
            modes = (
                _auto_modes(result.eigenvalues)
                if str(args.modes).strip().lower() == "auto"
                else _parse_ints(args.modes)
            )
            greybox = run_greybox(
                GreyboxConfig(
                    case_path=args.case,
                    layers=GreyboxLayerSelection.from_names(args.layers),
                    modes=modes,
                )
            )
            saved_gb = plot_greybox_summary(greybox, output_dir=args.output_dir, show=args.show)
            for key, path in saved_gb.items():
                if path is not None:
                    print(f"- {key}: {path}")
        if result.warnings:
            print("warnings:")
            for warning in result.warnings:
                print(f"- {warning}")
    return 0


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _auto_modes(eigenvalues, *, count: int = 3) -> tuple[int, ...]:
    """Pick oscillatory modes with largest |imag| for greybox Layer 1/2 plots."""

    import numpy as np

    values = np.asarray(eigenvalues, dtype=complex).ravel()
    scored = [
        (idx, abs(float(np.imag(value))))
        for idx, value in enumerate(values)
        if np.isfinite(value) and abs(float(np.imag(value))) > 1e-6
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored:
        return tuple(idx for idx, _ in scored[:count])
    finite = [idx for idx, value in enumerate(values) if np.isfinite(value)]
    return tuple(finite[:count]) if finite else (0,)


if __name__ == "__main__":
    raise SystemExit(main())
