//
//  MNISTDataSet.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//
import MetalPerformanceShaders
import zlib
import Foundation

fileprivate let imagePrefixLength = 16

fileprivate let labelPrefixLength = 8

protocol Label {
   // associatedtype Item
    static func label(with value: Int, length: Int, device: MTLDevice) -> Self
}

extension Int: Label {
    static func label(with value: Int, length: Int, device: MTLDevice) -> Int {
        return value
    }
}

extension MPSCNNLossLabels: Label {
    static func label(with value: Int, length: Int, device: MTLDevice) -> Self {
        var values = [Float](repeating: 0, count: length)
        values[value] = 1
        let data = NSData(bytes: &values, length: length * MemoryLayout<Float>.size)
        let labelsDescriptor = MPSCNNLossDataDescriptor(data: data as Data, layout: .HeightxWidthxFeatureChannels, size: MTLSize(width: 1, height: 1, depth: length))!
        return MPSCNNLossLabels(device: device, labelsDescriptor: labelsDescriptor) as! Self
    }

}
protocol DataSet {
    
    var count: Int {get}
    
    var outputCount: Int {get}
    
    var imageWidth: Int {get}
    
    var imageHeight: Int {get}
    
    var numberOfChannels: Int {get}
    
    func enumerateBatches<T: Label>(device: MTLDevice, batchSize: Int, isRandom: Bool, block: @escaping (Int, [MPSImage], [T]) -> Void)
    
}

extension DataSet {
    func batchCount(with batchSize: Int) -> Int {
        return count / batchSize + (count % batchSize == 0 ? 0 : 1)
    }
    
    var imageSize: Int {
        return imageWidth * imageHeight * numberOfChannels
    }
}

class MNISTDataSet: DataSet {
    
    let imageWidth = 28
    
    let imageHeight = 28
    
    let numberOfChannels = 1
    
    let outputCount = 10
    
    private static let baseURL = "http://yann.lecun.com/exdb/mnist/"
    private static let names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte" , "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    
    var imageData: Data!
    var labelData: Data!
    var isTrain = false
    init(isTrain: Bool = true) {
        self.isTrain = isTrain
    }
    
    
    func dataPath(with name: String) -> String {
        return Bundle.main.bundlePath + "/" + name
    }
    
    func load() throws {
        let imageIndex = isTrain ? 0 : 2
        let labelIndex = isTrain ? 1 : 3
        let imageFileName = Self.names[imageIndex]
        let labelFileName = Self.names[labelIndex]
        let group = DispatchGroup()
        group.enter()
        try downloadFile(name: imageFileName) { [weak self] (data) in
            self?.imageData = data
            group.leave()
        }
        group.wait()
        group.enter()
        try downloadFile(name: labelFileName, completion: { [weak self] data in
            self?.labelData = data
            group.leave()
        })
        group.wait()
    }
    
    private func downloadFile(name: String, completion: @escaping (Data)->Void) throws {
        let path = dataPath(with: name)
        if FileManager.default.fileExists(atPath: path) {
            let data =  try Data(contentsOf: URL(fileURLWithPath: path))
            completion(data)
        } else {
            let url = URL(string: "\(Self.baseURL)\(name).gz")!
            print("Downloading \(url.absoluteString)")
            let request = URLRequest(url: url, timeoutInterval: 3600)
            URLSession.shared.dataTask(with: request) { (data, response, error) in
                if error == nil, let data = data {
                    FileManager.default.createFile(atPath: path + ".gz", contents: data, attributes: nil)
                    _ = execute(launchPath: gzipPath(), currentDirectory: Bundle.main.bundlePath, arguments: ["-d", path + ".gz"])
                    print("Save to \(path)")
                    let data =  try! Data(contentsOf: URL(fileURLWithPath: path))
                    completion(data)
                } else {
                    print("Error:\(error)")
                }
            }.resume()
            
        }
    }
    
    
    var count: Int {
        return (imageData.count - imagePrefixLength) / imageSize
    }
    
    func enumerateBatches<T: Label>(device: MTLDevice, batchSize: Int, isRandom: Bool, block: @escaping (Int, [MPSImage], [T]) -> Void) {
        let descriptor = MPSImageDescriptor(channelFormat: .unorm8, width: imageWidth, height: imageHeight, featureChannels: numberOfChannels, numberOfImages: 1, usage: isTrain ? [.shaderWrite, .shaderRead] : [.shaderRead])
        var indices = [Int](0..<count)
        if isRandom {
            indices.shuffle()
        }
        let labelLength = 12
        var index = 0
        var start = 0
        var end = 0
        var currentIndex = 0
        var images = [MPSImage]()
        var labels = [T]()
        for i in 0..<batchCount(with: batchSize) {
            currentIndex = batchSize * i
            images.removeAll()
            labels.removeAll()
            for j in currentIndex..<min(currentIndex + batchSize, count) {
                index = indices[j]
                let image = MPSImage(device: device, imageDescriptor: descriptor)
                start = imagePrefixLength + index * imageSize
                end = start + imageSize
                var items = [UInt8](imageData[start..<end])
                image.writeBytes(&items, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
                images.append(image)
                let label = Int(labelData[labelPrefixLength + index])
                labels.append(T.label(with: label, length: labelLength, device: device))
            }
            block(i, images, labels)
        }
    }
    
}
