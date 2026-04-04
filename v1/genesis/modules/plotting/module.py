from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "genesis-mpl-config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genesis.models import FigureResult, FigureSpec


class PlottingModule:
    def __init__(self, output_root: Union[str, Path]):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def generate_figure(self, spec: FigureSpec) -> FigureResult:
        stem = self._slugify(spec.title)
        figure_dir = self.output_root / stem
        figure_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = figure_dir / f"{stem}.pdf"
        png_path = figure_dir / f"{stem}.png"
        data = self._load_data(spec.data_source)
        figure_type = spec.figure_type.lower()
        self._apply_style(spec.style)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x_values = data.get("x") or list(range(len(data.get("y", []))))
        y_values = data.get("y") or []

        if figure_type == "line":
            self._validate_xy(x_values, y_values)
            ax.plot(x_values, y_values, linewidth=2.0, marker="o")
        elif figure_type == "scatter":
            self._validate_xy(x_values, y_values)
            ax.scatter(x_values, y_values, s=36)
        elif figure_type == "histogram":
            values = y_values or x_values
            if not values:
                raise ValueError("histogram data is empty")
            ax.hist(values, bins=min(12, max(3, len(values))))
        elif figure_type == "heatmap":
            matrix = np.asarray(data.get("matrix") or data.get("values") or [[0.0]], dtype=float)
            if matrix.ndim != 2:
                raise ValueError("heatmap data must be two-dimensional")
            image = ax.imshow(matrix, aspect="auto", cmap="viridis")
            fig.colorbar(image, ax=ax)
        else:
            self._validate_xy(x_values, y_values)
            ax.plot(x_values, y_values, linewidth=2.0, marker="o")

        if spec.axis_labels:
            ax.set_xlabel(spec.axis_labels[0] if len(spec.axis_labels) > 0 else "")
            ax.set_ylabel(spec.axis_labels[1] if len(spec.axis_labels) > 1 else "")
        ax.set_title(spec.title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        metadata = {
            "title": spec.title,
            "figure_type": figure_type,
            "axis_labels": spec.axis_labels,
            "style": spec.style,
            "data_points": len(y_values) if y_values else len(x_values),
            "data_source": str(spec.data_source) if isinstance(spec.data_source, str) else "inline",
            "pdf_path": str(pdf_path),
            "png_path": str(png_path),
        }
        (figure_dir / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return FigureResult(pdf_path=str(pdf_path), png_path=str(png_path), metadata=metadata)

    def _load_data(self, source: Union[str, list[float], dict[str, Any]]) -> dict[str, Any]:
        if isinstance(source, dict):
            return source
        if isinstance(source, list):
            return {"y": source}
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix == ".tsv" else ","
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter=delimiter))
            if not rows:
                return {"x": [], "y": []}
            numeric_columns = {
                key: [float(row[key]) for row in rows if row.get(key) not in {"", None}]
                for key in rows[0].keys()
            }
            keys = list(numeric_columns)
            if len(keys) >= 2:
                return {"x": numeric_columns[keys[0]], "y": numeric_columns[keys[1]]}
            return {"y": numeric_columns[keys[0]]}
        values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {"y": values}

    def _slugify(self, value: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
        return slug or "figure"

    def _apply_style(self, style: str) -> None:
        if style == "publication":
            plt.style.use("seaborn-v0_8-whitegrid")
        else:
            plt.style.use("default")

    def _validate_xy(self, x_values: list[float], y_values: list[float]) -> None:
        if not y_values:
            raise ValueError("figure data is empty")
        if len(x_values) != len(y_values):
            raise ValueError("x and y series must be the same length")
