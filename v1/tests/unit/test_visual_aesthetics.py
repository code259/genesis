import json
import math
import textwrap
from pathlib import Path

from genesis.models import FigureSpec
from genesis.modules.plotting.module import PlottingModule
from genesis.paper.synthesizer import PaperSynthesizer

OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_visual_outputs"
SHARED_FIGURES_DIR = OUTPUT_DIR / "paper" / "test_formatting_project" / "outputs" / "paper" / "figures"

def test_figure_production_line_chart():
    """Produce a line chart to confirm aesthetic styling."""
    plotter = PlottingModule(SHARED_FIGURES_DIR)
    
    # Generate some nice-looking trajectory data (e.g. learning curve)
    data_x = list(range(1, 101))
    data_y = [2.0 * math.exp(-0.05 * x) + 0.1 * math.sin(x) + 0.2 for x in data_x]
    
    spec = FigureSpec(
        title="Training Loss Trajectory",
        figure_type="line",
        data_source={"x": data_x, "y": data_y},
        axis_labels=["Epochs", "Loss"],
        style="publication"
    )
    
    result = plotter.generate_figure(spec)
    
    assert Path(result.pdf_path).exists()
    assert Path(result.png_path).exists()
    print(f"\n[Visual Check] Line chart (publication style) saved to: {result.png_path}")


def test_figure_production_scatter_plot():
    """Produce a scatter plot to confirm aesthetic styling."""
    plotter = PlottingModule(SHARED_FIGURES_DIR)
    
    # Generate clustered scatter data
    import random
    random.seed(42)
    data_x = [random.gauss(0, 1) for _ in range(50)] + [random.gauss(4, 1.5) for _ in range(50)]
    data_y = [random.gauss(0, 1) for _ in range(50)] + [random.gauss(4, 1.5) for _ in range(50)]
    
    spec = FigureSpec(
        title="Cluster Analysis",
        figure_type="scatter",
        data_source={"x": data_x, "y": data_y},
        axis_labels=["Feature 1", "Feature 2"],
        style="publication"
    )
    
    result = plotter.generate_figure(spec)
    
    assert Path(result.pdf_path).exists()
    assert Path(result.png_path).exists()
    print(f"\n[Visual Check] Scatter plot saved to: {result.png_path}")


def test_figure_production_heatmap():
    """Produce a heatmap to confirm aesthetic styling."""
    plotter = PlottingModule(SHARED_FIGURES_DIR)
    
    # Generate a correlation-like matrix
    matrix = []
    for i in range(10):
        row = []
        for j in range(10):
            val = math.exp(-0.1 * ((i - 5)**2 + (j - 5)**2))
            row.append(val)
        matrix.append(row)
        
    spec = FigureSpec(
        title="Attention Weights Heatmap",
        figure_type="heatmap",
        data_source={"matrix": matrix},
        axis_labels=["Query", "Key"],
        style="publication"
    )
    
    result = plotter.generate_figure(spec)
    
    assert Path(result.pdf_path).exists()
    assert Path(result.png_path).exists()
    print(f"\n[Visual Check] Heatmap saved to: {result.png_path}")

def test_figure_production_lightcurve():
    """Produce an astrophysics light curve to confirm aesthetic styling."""
    plotter = PlottingModule(SHARED_FIGURES_DIR)
    
    # Generate mock transit data
    import random
    random.seed(111)
    # Background flux with noise
    data_x = []
    data_y = []
    for i in range(300):
        t = i * 0.1
        data_x.append(t)
        # Transit dip between 10 and 15
        if 10 < t < 15:
            flux = 0.98 + random.gauss(0, 0.005)
        else:
            flux = 1.0 + random.gauss(0, 0.005)
        data_y.append(flux)
        
    spec = FigureSpec(
        title="Exoplanet Transit Light Curve",
        figure_type="lightcurve",
        data_source={"x": data_x, "y": data_y},
        axis_labels=["Time [BJD - 2454833]", "Relative Flux"],
        style="publication"
    )
    
    result = plotter.generate_figure(spec)
    print(f"\n[Visual Check] Light Curve saved to: {result.png_path}")


def test_figure_production_corner_plot():
    """Produce an astrophysics MCMC corner plot."""
    plotter = PlottingModule(SHARED_FIGURES_DIR)
    
    # Generate mock 3D posterior samples (Mass, Radius, Temperature)
    import random
    random.seed(42)
    matrix = []
    for _ in range(800):
        # correlated normal draws
        m = random.gauss(1.0, 0.1)
        r = m * 0.8 + random.gauss(0, 0.05) + 0.2
        t = (r ** 2) * 5000 + random.gauss(0, 100)
        matrix.append([m, r, t])
        
    spec = FigureSpec(
        title="MCMC Corner Plot",
        figure_type="corner",
        data_source={"matrix": matrix},
        axis_labels=[r"Mass [$M_\odot$]", r"Radius [$R_\odot$]", "Teff [K]"],
        style="publication"
    )
    
    result = plotter.generate_figure(spec)
    print(f"\n[Visual Check] Corner Plot saved to: {result.png_path}")


def test_paper_synthesis_formatting():
    """
    Produce a mock research paper output (main.tex) to verify traditional formatting
    and aesthetics.
    """
    # Create a mock project structure that PaperSynthesizer expects
    project_id = "test_formatting_project"
    project_dir = OUTPUT_DIR / "paper" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    (project_dir / "spec.json").write_text(json.dumps({
        "research_question": "Can aesthetics and formatting be verified dynamically?",
        "domain": "Computer Science"
    }))
    
    (project_dir / "knowledge").mkdir(exist_ok=True)
    (project_dir / "knowledge" / "domain_context.md").write_text("We are exploring visually pleasing scientific outputs.")
    (project_dir / "knowledge" / "citations_cache.json").write_text("{}")
    
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    run_1_dir = runs_dir / "run_1"
    run_1_dir.mkdir(exist_ok=True)
    
    # Generate mock 3D posterior samples for the paper specifically
    import random
    random.seed(99)
    synth_matrix = []
    for _ in range(400):
        m = random.gauss(1.1, 0.05)
        r = m * 0.8 + random.gauss(0, 0.02) + 0.1
        t = (r ** 2) * 5100 + random.gauss(0, 50)
        synth_matrix.append([m, r, t])
        
    (run_1_dir / "result.json").write_text(json.dumps({
        "task_id": "test_aesthetic_generation",
        "primary_metric": 0.95,
        "summary": "Generated a highly aesthetic paper output.",
        "citations": [{"title": "Aesthetics in AI", "doi": "10.1234/test", "year": 2026}],
        "selected_experiment": {
            "trajectory": [0.1, 0.4, 0.7, 0.9, 0.95],
            "corner_matrix": synth_matrix
        }
    }))
    (run_1_dir / "verification_report.json").write_text(json.dumps({
        "passed": True
    }))
    
    # Run the synthesizer
    synth = PaperSynthesizer(OUTPUT_DIR / "paper")
    report = synth.synthesize(
        project_id=project_id,
        final=True,
        completion_reason="Testing full aesthetic pipeline.",
        project_status="complete"
    )
    
    assert "latex_path" in report
    tex_path = Path(report["latex_path"])
    assert tex_path.exists()
    
    # Basic validation of the LaTeX traditional formatting structure
    tex_content = tex_path.read_text()
    assert "\\documentclass" in tex_content
    assert "\\section{Introduction}" in tex_content
    assert "\\section{Methods}" in tex_content
    assert "\\section{Results}" in tex_content
    
    print(f"\n[Visual Check] LaTeX paper syntax saved to: {tex_path}")
    print(f"You can review the full research paper PDF compile by using pdflatex on this file:\n  cd {tex_path.parent} && pdflatex main.tex\n")
    
    # Produce an easy preview image for IDE visualization avoiding binary PDF renders
    # Sips only extracts the first page. We write a small Swift script leveraging PDFKit to save every page!
    import subprocess
    pdf_path = tex_path.parent / "main.pdf"
    jpg_path = tex_path.parent / "main_paper_preview.jpg"
    
    swift_script = tex_path.parent / "pdf2img.swift"
    swift_code = r"""import Foundation
import AppKit
import PDFKit

let args = ProcessInfo.processInfo.arguments
if args.count < 3 { exit(1) }

let inputPath = args[1]
let outputBase = args[2]

guard let pdfDocument = PDFDocument(url: URL(fileURLWithPath: inputPath)) else { exit(1) }

let baseURL = URL(fileURLWithPath: outputBase).deletingPathExtension().path

for i in 0..<pdfDocument.pageCount {
    guard let page = pdfDocument.page(at: i) else { continue }
    let pageRect = page.bounds(for: .mediaBox)
    
    let dpiScale: CGFloat = 300.0 / 72.0
    let scaledSize = NSSize(width: pageRect.size.width * dpiScale, height: pageRect.size.height * dpiScale)
    
    let image = NSImage(size: scaledSize)
    image.lockFocus()
    
    if let context = NSGraphicsContext.current?.cgContext {
        context.setFillColor(NSColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: scaledSize.width, height: scaledSize.height))
        context.scaleBy(x: dpiScale, y: dpiScale)
        page.draw(with: .mediaBox, to: context)
    }
    
    image.unlockFocus()
    
    if let tiffData = image.tiffRepresentation,
       let bitmap = NSBitmapImageRep(data: tiffData),
       let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9]) {
        let outPath = "\(baseURL)_page_\(i + 1).jpg"
        try? jpegData.write(to: URL(fileURLWithPath: outPath))
        print("Generated \(outPath)")
    }
}
"""
    
    if pdf_path.exists():
        try:
            swift_script.write_text(swift_code)
            subprocess.run(["swift", str(swift_script), str(pdf_path), str(jpg_path)], capture_output=True)
            print(f">>> Paper Snapshot Images successfully exported locally based on page count!")
        except Exception:
            pass

if __name__ == "__main__":
    import sys
    print("Running aesthetic test cases...")
    
    # Run the setup manually
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Call the tests
    try:
        test_figure_production_line_chart()
        test_figure_production_scatter_plot()
        test_figure_production_heatmap()
        test_figure_production_lightcurve()
        test_figure_production_corner_plot()
        
        
        # Test paper synthesis
        test_paper_synthesis_formatting()
            
            
        print("\nAll aesthetic test cases passed. Outputs are available in the 'test_visual_outputs' folder for manual inspection.")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
