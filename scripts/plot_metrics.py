from __future__ import annotations

import argparse
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from fetchreach_il.config import ARTIFACT_DIR, LEGACY_PLOT_DATA_DIR, PLOTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics CSV files for FetchReach experiments.")
    parser.add_argument("paths", nargs="*", default=[str(LEGACY_PLOT_DATA_DIR), str(ARTIFACT_DIR / "metrics")], help="CSV files or directories to scan.")
    parser.add_argument("--output-dir", default=str(PLOTS_DIR), help="Directory where the plots will be saved.")
    return parser.parse_args()


def collect_csv_files(paths: list[str]) -> list[Path]:
    csv_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            csv_files.extend(sorted(path.glob("*.csv")))
        elif path.suffix.lower() == ".csv" and path.exists():
            csv_files.append(path)
    return csv_files


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")

    from fetchreach_il.plotting import plot_metrics_file
    csv_files = collect_csv_files(args.paths)
    if not csv_files:
        raise FileNotFoundError("No CSV metrics files were found in the provided paths.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        try:
            output_path = plot_metrics_file(csv_file, output_dir=output_dir)
            print(f"Plotted {csv_file} -> {output_path}")
        except ValueError as error:
            print(f"Skipping {csv_file}: {error}")


if __name__ == "__main__":
    main()
