import Foundation
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
