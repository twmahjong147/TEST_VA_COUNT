import Foundation
import UIKit
import CoreML

struct MLHelpers {
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    static func resizeImage(_ image: UIImage, targetSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return newImage
    }

    static func imageToMLMultiArray(_ image: UIImage) throws -> MLMultiArray {
        // image expected in RGB, size 384x384
        // Ensure we are working with a rendered bitmap at the expected size (preserve color conversion)
        let targetSize = CGSize(width: 384, height: 384)
        let resizedImage = image.resized(to: targetSize)
        guard let cg = resizedImage.cgImage else { throw NSError(domain: "ML", code: -1, userInfo: nil) }
        let width = cg.width
        let height = cg.height

        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)].map({ NSNumber(value: $0.intValue) }), dataType: .float32)

        // extract pixel data
        // Use source image color space when available to avoid color conversion issues (Display P3 etc.)
        let colorSpace = cg.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        var raw = [UInt8](repeating: 0, count: Int(height * bytesPerRow))
        // Use iOS' native bitmap layout: BGRA in little-endian memory (byteOrder32Little + premultipliedFirst)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let ctx = CGContext(data: &raw, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo.rawValue) else {
            fatalError("Unable to create CGContext")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Fill MLMultiArray in (1, C, H, W) with normalized floats
        for y in 0..<height {
            for x in 0..<width {
                let pxIndex = y * bytesPerRow + x * 4
                // Memory layout is BGRA (byteOrder32Little + premultipliedFirst) on iOS.
                let b = Float(raw[pxIndex + 0]) / 255.0
                let g = Float(raw[pxIndex + 1]) / 255.0
                let r = Float(raw[pxIndex + 2]) / 255.0

                let rn = (r - mean[0]) / std[0]
                let gn = (g - mean[1]) / std[1]
                let bn = (b - mean[2]) / std[2]

                let h = y
                let w = x
                // array order: [1, C, H, W]
                array[[0, 0 as NSNumber, NSNumber(value: h), NSNumber(value: w)]] = NSNumber(value: rn)
                array[[0, 1 as NSNumber, NSNumber(value: h), NSNumber(value: w)]] = NSNumber(value: gn)
                array[[0, 2 as NSNumber, NSNumber(value: h), NSNumber(value: w)]] = NSNumber(value: bn)
            }
        }

        // Diagnostic: sample corner & center pixels and compute per-channel means
        func meanAndCorners() {
            var sumR: Double = 0
            var sumG: Double = 0
            var sumB: Double = 0
            let total = Double(width * height)
            for y in 0..<height {
                for x in 0..<width {
                    let idx = y * bytesPerRow + x * 4
                    // raw layout is BGRA -> idx+0 = B, +1 = G, +2 = R
                    sumB += Double(raw[idx + 0])
                    sumG += Double(raw[idx + 1])
                    sumR += Double(raw[idx + 2])
                }
            }
            let meanR = sumR / total / 255.0
            let meanG = sumG / total / 255.0
            let meanB = sumB / total / 255.0

            let corners = [ (x:0,y:0), (x:width-1,y:0), (x:0,y:height-1), (x:width-1,y:height-1) ]
            var cornerStrs: [String] = []
            for c in corners {
                let idx = c.y * bytesPerRow + c.x * 4
                if idx + 2 < raw.count {
                    // report as R,G,B using BGRA layout
                    let b = Float(raw[idx + 0]) / 255.0
                    let g = Float(raw[idx + 1]) / 255.0
                    let r = Float(raw[idx + 2]) / 255.0
                    cornerStrs.append(String(format: "(%d,%d): R=%.3f,G=%.3f,B=%.3f", c.x, c.y, r, g, b))
                }
            }
            let meanStr = String(format: "%.3f,%.3f,%.3f", meanR, meanG, meanB)
            let cornerJoined = cornerStrs.joined(separator: ", ")
            print("[MLHelpers] input mean R,G,B=\(meanStr) corners=\(cornerJoined)")
        }
        meanAndCorners()
        return array
    }

    static func cropBoxesAsMLMultiArray(from image: UIImage, boxes: [(x1:Int,y1:Int,x2:Int,y2:Int)], cropSize: Int = 64, shotNum: Int = 3) throws -> MLMultiArray {
        // Create array shape [1, shotNum, 3, cropSize, cropSize]
        // But Core ML input was exported as [1,3,3,64,64] in our conversion; we'll produce that
        let shape: [NSNumber] = [1, NSNumber(value: shotNum), 3, NSNumber(value: cropSize), NSNumber(value: cropSize)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)

        for s in 0..<shotNum {
            let box: (x1:Int,y1:Int,x2:Int,y2:Int)
            if s < boxes.count {
                box = boxes[s]
            } else {
                let centerX = Int(image.size.width / 2)
                let centerY = Int(image.size.height / 2)
                let x1 = max(0, centerX - cropSize/2)
                let y1 = max(0, centerY - cropSize/2)
                let x2 = min(Int(image.size.width), centerX + cropSize/2)
                let y2 = min(Int(image.size.height), centerY + cropSize/2)
                box = (x1: x1, y1: y1, x2: x2, y2: y2)
            }
            let cropRect = CGRect(x: CGFloat(box.x1), y: CGFloat(box.y1), width: CGFloat(box.x2 - box.x1), height: CGFloat(box.y2 - box.y1))
            guard let cg = image.cgImage else { continue }
            guard let cropped = cg.cropping(to: cropRect) else { continue }
            let uiCrop = UIImage(cgImage: cropped).resized(to: CGSize(width: cropSize, height: cropSize))

            // get pixel data
            guard let cg2 = uiCrop.cgImage else { continue }
            let width = cg2.width
            let height = cg2.height
            let bytesPerRow = width * 4
            var raw = [UInt8](repeating: 0, count: Int(height * bytesPerRow))
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
            guard let ctx = CGContext(data: &raw, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo.rawValue) else {
                fatalError("Unable to create CGContext for crop")
            }
            ctx.draw(cg2, in: CGRect(x: 0, y: 0, width: width, height: height))

            for y in 0..<height {
                for x in 0..<width {
                    let pxIndex = y * bytesPerRow + x * 4
                    let b = Float(raw[pxIndex + 0]) / 255.0
                    let g = Float(raw[pxIndex + 1]) / 255.0
                    let r = Float(raw[pxIndex + 2]) / 255.0

                    let idxR = [0, NSNumber(value: s), 0, NSNumber(value: y), NSNumber(value: x)] as [NSNumber]
                    let idxG = [0, NSNumber(value: s), 1, NSNumber(value: y), NSNumber(value: x)] as [NSNumber]
                    let idxB = [0, NSNumber(value: s), 2, NSNumber(value: y), NSNumber(value: x)] as [NSNumber]

                    arr[idxR] = NSNumber(value: (r - mean[0]) / std[0])
                    arr[idxG] = NSNumber(value: (g - mean[1]) / std[1])
                    arr[idxB] = NSNumber(value: (b - mean[2]) / std[2])
                }
            }
        }
        return arr
    }

    static func multiArrayTo2D(_ arr: MLMultiArray) -> [[Float]] {
        // Robust conversion of MLMultiArray to 2D Float array.
        // We treat the last two dimensions as H, W (this covers [H,W], [1,H,W], [1,1,H,W], etc.).
        let shape = arr.shape.map { $0.intValue }
        let dim = shape.count
        guard dim >= 2 else {
            // fallback to single value
            let v = arr[0].floatValue
            return [[v]]
        }

        let h = shape[dim - 2]
        let w = shape[dim - 1]

        var out = Array(repeating: Array(repeating: Float(0), count: w), count: h)

        // Build an indices array of zeros length 'dim' and reuse it.
        var indices = Array(repeating: 0, count: dim)
        for y in 0..<h {
            indices[dim - 2] = y
            for x in 0..<w {
                indices[dim - 1] = x
                // access via multi-dimensional subscript to avoid assuming flat layout
                let nums = indices.map { NSNumber(value: $0) }
                let val = arr[nums].floatValue
                out[y][x] = val
            }
            indices[dim - 2] = 0
        }

        // Log basic stats for debugging
        var minV: Float = Float.greatestFiniteMagnitude
        var maxV: Float = -Float.greatestFiniteMagnitude
        var sumV: Float = 0
        for row in out {
            for v in row {
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                sumV += v
            }
        }
        print("[MLHelpers] multiArrayTo2D -> shape=(\(h),\(w)) min=\(minV) max=\(maxV) sum=\(sumV)")

        return out
    }

    static func detectPeaks(density: [[Float]], minDistance: Int = 10, relThresh: Float = 0.12) -> [(x:Int,y:Int,conf:Float)] {
        let h = density.count
        let w = density.first?.count ?? 0
        var peaks: [(Int,Int,Float)] = []
        var maxVal: Float = 0
        for row in density { for v in row { if v > maxVal { maxVal = v } } }
        let thresh = max(relThresh * maxVal, 0.01)

        print("[MLHelpers] detectPeaks -> density shape=\(h)x\(w), maxVal=\(maxVal), thresh=\(thresh)")

        func isLocalMax(x:Int, y:Int) -> Bool {
            let v = density[y][x]
            if v < thresh { return false }
            let y0 = max(0, y - 1)
            let y1 = min(h - 1, y + 1)
            let x0 = max(0, x - 1)
            let x1 = min(w - 1, x + 1)
            for yy in y0...y1 {
                for xx in x0...x1 {
                    if density[yy][xx] > v { return false }
                }
            }
            return true
        }

        for y in 0..<h {
            for x in 0..<w {
                if isLocalMax(x: x, y: y) {
                    peaks.append((x,y,density[y][x]))
                    print("[MLHelpers] peak at (x=\(x),y=\(y)) val=\(density[y][x])")
                }
            }
        }

        // non-maximum suppression by minDistance: greedy
        peaks.sort { $0.2 > $1.2 }
        var kept: [(Int,Int,Float)] = []
        for p in peaks {
            var ok = true
            for k in kept {
                let dx = k.0 - p.0
                let dy = k.1 - p.1
                if dx*dx + dy*dy <= minDistance * minDistance { ok = false; break }
            }
            if ok { kept.append(p) }
        }
        return kept.map { (x:$0.0, y:$0.1, conf:$0.2) }
    }

    static func peaksToFixedBoxes(peaks: [(x:Int,y:Int,conf:Float)], boxSize: Int = 40, imgShape: (h:Int,w:Int)? = nil) -> [(CGRect, Float)] {
        var boxes: [(CGRect, Float)] = []
        let half = boxSize/2
        let h = imgShape?.h ?? (peaks.first.map { _ in 384 } ?? 384)
        let w = imgShape?.w ?? (peaks.first.map { _ in 384 } ?? 384)
        for p in peaks {
            var x1 = p.x - half
            var y1 = p.y - half
            var x2 = x1 + boxSize
            var y2 = y1 + boxSize
            if w > 0 {
                if x2 > w { x2 = w; x1 = max(0, x2 - boxSize) }
                if x1 < 0 { x1 = 0; x2 = min(w, boxSize) }
            }
            if h > 0 {
                if y2 > h { y2 = h; y1 = max(0, y2 - boxSize) }
                if y1 < 0 { y1 = 0; y2 = min(h, boxSize) }
            }
            boxes.append((CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1), p.conf))
        }
        return boxes
    }

    static func drawBoxesOnImage(_ image: UIImage, boxes: [(CGRect, Float)], color: UIColor = .green, thickness: CGFloat = 2.0, showConfidence: Bool = true, fontSize: CGFloat = 12.0) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, true, image.scale)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        let ctx = UIGraphicsGetCurrentContext()!
        ctx.setStrokeColor(color.cgColor)
        ctx.setLineWidth(thickness)

        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.alignment = .left
        let attrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: fontSize),
            .paragraphStyle: paragraphStyle,
            .foregroundColor: color
        ]

        // Assuming boxes are in image pixel coordinates (same size as image)
        for (box, conf) in boxes {
            ctx.stroke(box)
            if showConfidence {
                let label = String(format: "%.2f", conf)
                // draw label at top-left of box
                let labelRect = CGRect(x: box.origin.x + 2, y: max(0, box.origin.y - fontSize - 4), width: 80, height: fontSize + 4)
                label.draw(in: labelRect, withAttributes: attrs)
            }
        }
        let out = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return out
    }
}

// UIImage convenience
extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        self.draw(in: CGRect(origin: .zero, size: size))
        let img = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return img
    }

}
