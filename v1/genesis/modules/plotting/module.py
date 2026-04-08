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

        skip_generic_formatting = False
        
        if figure_type == "line":
            self._validate_xy(x_values, y_values)
            ax.plot(x_values, y_values, linewidth=1.5, marker="o", markersize=4)
            ax.grid(alpha=0.3)
        elif figure_type == "scatter":
            self._validate_xy(x_values, y_values)
            ax.scatter(x_values, y_values, s=24, alpha=0.5, edgecolors="none")
            ax.grid(alpha=0.3)
        elif figure_type == "lightcurve":
            self._validate_xy(x_values, y_values)
            # Light curves typically demand high-density scatter with small markers (flux vs time)
            ax.scatter(x_values, y_values, s=6, alpha=0.8, color='black', marker='.')
            ax.grid(alpha=0.15)
        elif figure_type == "histogram":
            values = y_values or x_values
            if not values:
                raise ValueError("histogram data is empty")
            ax.hist(values, bins=min(20, max(5, len(values))), alpha=0.75, edgecolor='white')
            ax.grid(alpha=0.3)
        elif figure_type == "heatmap":
            matrix = np.asarray(data.get("matrix") or data.get("values") or [[0.0]], dtype=float)
            if matrix.ndim != 2:
                raise ValueError("heatmap data must be two-dimensional")
            image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
            fig.colorbar(image, ax=ax)
            ax.grid(False) # CRITICAL: Do not plot grid over heatmap cells!
            for spine in ax.spines.values():
                spine.set_visible(False)
        elif figure_type == "corner":
            # Astrophysics MCMC Corner plots showing multi-dimensional distributions
            matrix = np.asarray(data.get("matrix") or [], dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] < 2:
                raise ValueError("corner data must be at least 2 dimensions")
            
            plt.close(fig) # Abandon the single axis Figure
            num_vars = matrix.shape[1]
            fig, axes = plt.subplots(num_vars, num_vars, figsize=(min(8, 2 * num_vars), min(8, 2 * num_vars)))
            
            for i in range(num_vars):
                for j in range(num_vars):
                    ax_i = axes[i, j]
                    if i < j:
                        ax_i.axis('off')
                    elif i == j:
                        # Diagonal is a 1D histogram
                        ax_i.hist(matrix[:, i], bins=15, histtype="step", color='black')
                        if i < num_vars - 1:
                            ax_i.set_xticklabels([])
                        ax_i.set_yticks([]) # Hide y-axis scale for 1D distributions
                    else:
                        # Off diagonal is a 2D scatter posterior
                        ax_i.scatter(matrix[:, j], matrix[:, i], s=3, alpha=0.2, color='black')
                        if i < num_vars - 1:
                            ax_i.set_xticklabels([])
                        if j > 0:
                            ax_i.set_yticklabels([])
                            
                    # Add labels to outer edges
                    if i == num_vars - 1 and spec.axis_labels and j < len(spec.axis_labels):
                        ax_i.set_xlabel(spec.axis_labels[j])
                    if j == 0 and spec.axis_labels and i < len(spec.axis_labels) and i > 0:
                        ax_i.set_ylabel(spec.axis_labels[i])
            skip_generic_formatting = True
        else:
            plt.close(fig)
            raise ValueError(f"unsupported figure_type: {figure_type}")

        if not skip_generic_formatting:
            if spec.axis_labels:
                ax.set_xlabel(spec.axis_labels[0] if len(spec.axis_labels) > 0 else "")
                ax.set_ylabel(spec.axis_labels[1] if len(spec.axis_labels) > 1 else "")
            ax.set_title(spec.title)
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
            
        # Enforce a consistent, professional academic font globally across all models
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm", # Computer Modern for math
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9
        })

    def _validate_xy(self, x_values: list[float], y_values: list[float]) -> None:
        if not y_values:
            raise ValueError("figure data is empty")
        if len(x_values) != len(y_values):
            raise ValueError("x and y series must be the same length")
