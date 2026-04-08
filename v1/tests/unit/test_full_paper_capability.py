import json
import math
import random
import subprocess
from pathlib import Path
from unittest.mock import patch

from genesis.paper.synthesizer import PaperSynthesizer

OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_visual_outputs" / "full_paper_demo"

def test_run_full_paper_demo():
    print("Initializing full scientific paper emulation...")
    project_id = "exoplanet_discovery_demo"
    project_dir = OUTPUT_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    (project_dir / "spec.json").write_text(json.dumps({
        "research_question": "Confirmation and parameterization of a potentially habitable rocky exoplanet transit signal.",
        "domain": "Astrophysics"
    }))
    
    (project_dir / "knowledge").mkdir(exist_ok=True)
    domain_knowledge = "Exoplanet transit photometry combined with MCMC posterior modeling allows exact radii extraction."
    (project_dir / "knowledge" / "domain_context.md").write_text(domain_knowledge)
    
    citations = {
        "10.1038/nature12345": {"title": "A rocky planet in the habitable zone", "year": "2024", "authors": [{"name": "J Kepler"}]},
        "10.1038/nature54321": {"title": "Advanced Markov Chain Monte Carlo methods in Astrophysics", "year": "2025", "authors": [{"name": "D Foreman Mackey"}]},
    }
    (project_dir / "knowledge" / "citations_cache.json").write_text(json.dumps(citations))
    
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_dir = runs_dir / "run_opt"
    run_dir.mkdir(exist_ok=True)
    
    # Generate convincing realistic MCMC distribution (mass, radius, teff)
    random.seed(42)
    mcmc_samples = []
    for _ in range(800):
        m = random.gauss(1.05, 0.08)
        r = m * 0.95 + random.gauss(0, 0.03)
        t = (r ** 2) * 4500 + random.gauss(0, 150)
        mcmc_samples.append([m, r, t])
        
    # Generate realistic Light curve with a transit dip
    light_curve = []
    for i in range(200):
        val = 1.0 + random.gauss(0, 0.001)
        if 80 < i < 110:
            val -= 0.015 # transit depth
        light_curve.append(val)
        
    (run_dir / "result.json").write_text(json.dumps({
        "task_id": "mcmc_fitting",
        "primary_metric": 0.998,
        "summary": "Fitted transit parameters safely.",
        "citations": [
             {"doi": "10.1038/nature12345", "title": "A rocky planet in the habitable zone", "year": "2024", "authors": [{"name": "J Kepler"}]},
             {"doi": "10.1038/nature54321", "title": "Advanced Markov Chain Monte Carlo methods in Astrophysics", "year": "2025", "authors": [{"name": "D Foreman Mackey"}]}
        ],
        "selected_experiment": {
            "trajectory": light_curve,
            "corner_matrix": mcmc_samples
        }
    }))
    (run_dir / "verification_report.json").write_text(json.dumps({"passed": True}))
    
    synth = PaperSynthesizer(OUTPUT_DIR)
    
    # We will mock the LLM text generation so we can feed highly realistic multi-paragraph scientific writing
    # without needing to burn real LLM inference API calls just to test the visual formatter!
    def mock_generate_section(section_name, base_text, context):
        if section_name == "abstract":
            return r"We present the confirmation of Kepler-186f, a rocky exoplanet orbiting within the habitable zone of a M-dwarf star. Using high-precision photometric data, we model the transit light curve utilizing Markov Chain Monte Carlo (MCMC) pipelines. Our findings indicate a planetary radius of $R_p = 1.11 \pm 0.05 R_\oplus$ and a tight correlation between planetary mass and effective temperature, suggesting a substantial iron core. The high statistical convergence (Primary Metric = 0.998) demonstrates the capacity of automated autonomous research systems to independently isolate transit parameters without human loop intervention."
        if section_name == "introduction":
            return r"The detection of Earth-sized exoplanets within the habitable zones of main-sequence stars is a paramount objective of modern astrophysics \cite{j_kepler_2024_a_rocky_planet_in_the_habitable_zone}. The challenge lies heavily in the robust statistical verification of minimal photometric transit dips amidst pervasive stellar noise. While human-engineered pipelines are capable of fitting theoretical limbs to empirical flux data, autonomous scientific evaluators provide an un-biased, exhaustive exploration of the parametric space. In this paper, we utilize the Genesis v1 engine to synthetically ingest raw stellar flux data, build an analytical model, and natively resolve the system's orbital and physical parameters."
        if section_name == "methods":
            return r"Photometric data was acquired representing an aggregated folded sequence of orbital periods. The system utilized a uniform prior distribution spanning realistic physical constraints for an M-dwarf host. To isolate the transit depth and ingress/egress durations, we implemented a non-linear optimization pipeline followed seamlessly by an Affine-Invariant Ensemble MCMC sampler \cite{d_foreman_mackey_2025_advanced_markov_chain_monte_carlo_methods_in_astrophysics}. The sampler utilized 800 walkers integrated through 2000 steps, aggressively burning the first 500. This methodology ensures that any detected local minimum is safely bounded within the global maximum likelihood."
        if section_name == "results":
            return r"The primary pipeline output converged effectively, recognizing a distinct 1.5\% drop in relative stellar flux indicative of a planetary transit. We find that the trajectory of the transit geometry is highly stable across subsequent analytical epochs."
        if section_name == "discussion":
            return r"The derived parameters constrain the planet entirely within the theoretical habitable zone, making it a prime candidate for future spectroscopic atmospheric analysis. The MCMC correlation matrices emphasize that while radius is tightly constrained by the transit geometry, the mass retains a larger variance intrinsically linked to the undefined density prior."
        return base_text

    print("Generating LaTeX and Figures via PaperSynthesizer...")
    
    # Patch the synthesizer so it thinks it is using the LLM runtime
    synth.runtime = True 
    with patch.object(synth, '_generate_section', side_effect=mock_generate_section):
        report = synth.synthesize(
            project_id=project_id,
            final=True,
            completion_reason="Completed full analysis.",
            project_status="complete"
        )
        
    tex_path = Path(report["latex_path"])
    print(f"Paper perfectly saved to {tex_path}")
    
    # Convert PDF to JPEG
    pdf_path = tex_path.parent / "main.pdf"
    if pdf_path.exists():
        print("Compiling native macOS Swift script for multi-page export...")
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
        swift_script.write_text(swift_code)
        jpg_path = tex_path.parent / "main_paper_preview.jpg"
        print(f"Running multi-page split via Swift API...")
        subprocess.run(["swift", str(swift_script), str(pdf_path), str(jpg_path)], capture_output=True)
        print("Done! You can inspect the final multi-page images locally!")

if __name__ == "__main__":
    test_run_full_paper_demo()
