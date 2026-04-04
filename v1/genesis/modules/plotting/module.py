from __future__ import annotations

from pathlib import Path
from typing import Union

from genesis.models import FigureResult, FigureSpec


class PlottingModule:
    def __init__(self, output_root: Union[str, Path]):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def generate_figure(self, spec: FigureSpec) -> FigureResult:
        stem = spec.title.lower().replace(" ", "_") or "figure"
        pdf_path = self.output_root / f"{stem}.pdf"
        png_path = self.output_root / f"{stem}.png"
        payload = (
            f"title={spec.title}\nfigure_type={spec.figure_type}\n"
            f"axis_labels={spec.axis_labels}\nstyle={spec.style}\n"
        )
        pdf_path.write_text(payload, encoding="utf-8")
        png_path.write_text(payload, encoding="utf-8")
        return FigureResult(pdf_path=str(pdf_path), png_path=str(png_path), metadata={"title": spec.title})
