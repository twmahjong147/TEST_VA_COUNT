import SwiftUI
import CoreML

struct ContentView: View {
    @State private var resultText: String = "No result yet"
    @State private var image: UIImage = UIImage(named: "sample_placeholder") ?? UIImage()
    @State private var overlayImage: UIImage? = nil
    @State private var confidenceThreshold: Double = 0.12

    var body: some View {
        VStack(spacing: 20) {
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 320, maxHeight: 320)
                if let overlay = overlayImage {
                    Image(uiImage: overlay)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: 320, maxHeight: 320)
                }
            }

            Button("Run Inference") {
                Task.detached {
                    await runInference()
                }
            }

            HStack {
                Text(String(format: "Threshold: %.2f", confidenceThreshold))
                Slider(value: $confidenceThreshold, in: 0.0...1.0, step: 0.01)
            }

            Text(resultText)
                .padding()
        }
        .padding()
        .onAppear(perform: loadSampleImage)
    }

    func loadSampleImage() {
        // Load sample image from asset catalog named "Sample"
        // Try asset catalog first
        if let ui = UIImage(named: "sample4") {
            image = ui
            print("[ContentView] loadSampleImage -> loaded from asset named 'sample4'")
            return
        }

        // Try loading from bundle resource (common if image was added as a file resource)
        if let url = Bundle.main.url(forResource: "sample4", withExtension: "JPG") ?? Bundle.main.url(forResource: "sample4", withExtension: "jpg") {
            if let data = try? Data(contentsOf: url), let ui2 = UIImage(data: data) {
                image = ui2
                print("[ContentView] loadSampleImage -> loaded from bundle resource: \(url.path)")
                return
            }
        }

        // Simulator-only developer path fallback: useful when running from Xcode without bundling the sample image.
        // #if targetEnvironment(simulator)
        // let devPath = "/Users/alex/data/work/TEST_VA_COUNT/demo/AICounter/AICounter/sample4.JPG"
        // if FileManager.default.fileExists(atPath: devPath), let ui3 = UIImage(contentsOfFile: devPath) {
        //     image = ui3
        //     print("[ContentView] loadSampleImage -> loaded from simulator devPath: \(devPath)")
        //     return
        // }
        // #endif

        print("[ContentView] loadSampleImage -> no sample image found; using blank placeholder")
    }

    func runInference() async {
        do {
            guard let modelURL = Bundle.main.url(forResource: "checkpoint_FSC", withExtension: "mlmodelc") ?? Bundle.main.url(forResource: "checkpoint_FSC", withExtension: "mlpackage") else {
                resultText = "Model not found in bundle. Add checkpoint_FSC.mlpackage to project."
                return
            }

            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)

            // Prepare image input (resize to 384x384)
            let resized = MLHelpers.resizeImage(image, targetSize: CGSize(width: 384, height: 384))
            let imageArray = try MLHelpers.imageToMLMultiArray(resized)

            // Prepare exemplar boxes: avoid using extreme corners (they produce bias/false positives).
            // Use the image center and two nearby regions (not at the edges).
            let w = Int(resized.size.width)
            let h = Int(resized.size.height)
            let cx = w/2
            let cy = h/2
            let cropSize = 64
            let margin = max(32, cropSize) // avoid edges
            let boxCenter = (x1: cx - cropSize/2, y1: cy - cropSize/2, x2: cx + cropSize/2, y2: cy + cropSize/2)
            let boxLeft = (x1: max(margin, cx - w/6 - cropSize/2), y1: max(margin, cy - h/6 - cropSize/2), x2: min(w - margin, max(margin, cx - w/6 - cropSize/2) + cropSize), y2: min(h - margin, max(margin, cy - h/6 - cropSize/2) + cropSize))
            let boxRight = (x1: min(w - margin - cropSize, cx + w/6 - cropSize/2), y1: min(h - margin - cropSize, cy + h/6 - cropSize/2), x2: min(w - margin, min(w - margin - cropSize, cx + w/6 - cropSize/2) + cropSize), y2: min(h - margin, min(h - margin - cropSize, cy + h/6 - cropSize/2) + cropSize))
            let boxes = [boxCenter, boxLeft, boxRight]

            let boxesArray = try MLHelpers.cropBoxesAsMLMultiArray(from: resized, boxes: boxes, cropSize: 64, shotNum: 3)

            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageArray, "boxes": boxesArray])

            let out = try await mlModel.prediction(from: input)

            if let outArr = out.featureValue(for: out.featureNames.first ?? "output")?.multiArrayValue {
                // sum up density as predicted count (divide by 60 like in PyTorch inference)
                var s: Float = 0
                let count = outArr.count
                for i in 0..<count {
                    s += outArr[i].floatValue
                }
                let predCount = s / 60.0

                // convert to 2D density map, detect peaks, convert to boxes and draw overlay
                let density2d = MLHelpers.multiArrayTo2D(outArr)
                // Diagnostic: also consider a transposed interpretation of the density map
                func transpose(_ a: [[Float]]) -> [[Float]] {
                    let h = a.count
                    let w = a.first?.count ?? 0
                    var t = Array(repeating: Array(repeating: Float(0), count: h), count: w)
                    for y in 0..<h {
                        for x in 0..<w {
                            t[x][y] = a[y][x]
                        }
                    }
                    return t
                }

                var peaks = MLHelpers.detectPeaks(density: density2d, minDistance: 10, relThresh: 0.12)
                // Filter out peaks that lie on the extreme image border (likely false positives)
                let borderMargin = 4
                peaks = peaks.filter { p in
                    let wD = density2d.first?.count ?? 0
                    let hD = density2d.count
                    if p.x <= borderMargin || p.y <= borderMargin { return false }
                    if p.x >= wD - 1 - borderMargin || p.y >= hD - 1 - borderMargin { return false }
                    return true
                }
                let transposed = transpose(density2d)
                let peaksTransposed = MLHelpers.detectPeaks(density: transposed, minDistance: 10, relThresh: 0.12)
                print("[ContentView] peaks(original)=\(peaks.count), peaks(transposed)=\(peaksTransposed.count)")
                let boxSize = 48
                let boxesWithConf = MLHelpers.peaksToFixedBoxes(peaks: peaks, boxSize: boxSize, imgShape: (h: density2d.count, w: density2d.first?.count ?? 0))
                let boxesWithConfTransposed = MLHelpers.peaksToFixedBoxes(peaks: peaksTransposed, boxSize: boxSize, imgShape: (h: transposed.count, w: transposed.first?.count ?? 0))

                // Scale boxes from density-map coordinates to resized image pixel coordinates
                let densityH = density2d.count
                let densityW = density2d.first?.count ?? 1
                let resizedImg = MLHelpers.resizeImage(image, targetSize: CGSize(width: 384, height: 384))
                let scaleX = resizedImg.size.width / CGFloat(max(1, densityW))
                let scaleY = resizedImg.size.height / CGFloat(max(1, densityH))

                var pixelBoxes: [(CGRect, Float)] = []
                for (rect, conf) in boxesWithConf {
                    let r = CGRect(x: rect.origin.x * scaleX,
                                   y: rect.origin.y * scaleY,
                                   width: rect.size.width * scaleX,
                                   height: rect.size.height * scaleY)
                    pixelBoxes.append((r, conf))
                }

                print("[ContentView] density size=\(densityW)x\(densityH), scale=\(scaleX)x\(scaleY), boxes=\(pixelBoxes.count)")

                // Diagnostic: also compute pixel boxes for transposed mapping (swap axes when mapping)
                var pixelBoxesTransposed: [(CGRect, Float)] = []
                let transposedDensityH = transposed.count
                let transposedDensityW = transposed.first?.count ?? 1
                // When density is transposed, width/height are swapped relative to the original
                let scaleX_T = resizedImg.size.width / CGFloat(max(1, transposedDensityW))
                let scaleY_T = resizedImg.size.height / CGFloat(max(1, transposedDensityH))
                for (rect, conf) in boxesWithConfTransposed {
                    // rect is in transposed density coordinates; map using transposed scales
                    let r = CGRect(x: rect.origin.x * scaleX_T,
                                   y: rect.origin.y * scaleY_T,
                                   width: rect.size.width * scaleX_T,
                                   height: rect.size.height * scaleY_T)
                    pixelBoxesTransposed.append((r, conf))
                }
                print("[ContentView] diagnostic: originalBoxes=\(pixelBoxes.count) transposedBoxes=\(pixelBoxesTransposed.count)")

                // filter by confidence threshold, then draw (adjust color/thickness/fontSize as desired)
                let filtered = pixelBoxes.filter { Double($0.1) >= confidenceThreshold }
                // Diagnostic toggle: show transposed overlay if needed
                let debugShowTransposed = true
                let overlay: UIImage
                if debugShowTransposed {
                    let filteredT = pixelBoxesTransposed.filter { Double($0.1) >= confidenceThreshold }
                    overlay = MLHelpers.drawBoxesOnImage(resizedImg, boxes: filteredT, color: .systemRed, thickness: 3.0, showConfidence: true, fontSize: 14.0)
                } else {
                    overlay = MLHelpers.drawBoxesOnImage(resizedImg, boxes: filtered, color: .systemGreen, thickness: 3.0, showConfidence: true, fontSize: 14.0)
                }

                DispatchQueue.main.async {
                    resultText = String(format: "Predicted count: %.2f", predCount)
                    overlayImage = overlay
                }
            } else {
                DispatchQueue.main.async {
                    resultText = "No valid output tensor"
                }
            }

        } catch {
            DispatchQueue.main.async {
                resultText = "Error: \(error.localizedDescription)"
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
