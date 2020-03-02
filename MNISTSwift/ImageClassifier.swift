//
//  ImageClassifier.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//
import Accelerate
import Foundation
import MetalPerformanceShaders

fileprivate let signBit = UInt16(0b1000000000000000)
fileprivate let exponentBit = UInt16(0b0111110000000000)
fileprivate let fractionBit = UInt16(0b0000001111111111)
fileprivate let powf2_minus10 = powf(2, -10)
fileprivate let float_minius_1 = Float(-1)
fileprivate let float_1 = Float(1)

@available(OSX 10.15, *)
class ImageClassifier: NSObject {
    var batchSize: Int
    private var trainSet: DataSet
    private var testSet: DataSet
    var net: Net
    init(net: Net, batchSize: Int, trainSet: DataSet, testSet: DataSet) {
        self.net = net
        self.batchSize = batchSize
        self.trainSet = trainSet
        self.testSet = testSet
        super.init()
    }
    
    fileprivate let semaphore = DispatchSemaphore(value: 2)
    
    func train(epochCount: Int) {
        for epoch in 0..<epochCount {
            let date = Date()
            var latestCommandBuffer: MPSCommandBuffer?
            trainSet.enumerateBatches(device: net.device, batchSize: batchSize, isRandom: true) {[unowned self] (i, images, labels: [MPSCNNLossLabels]) in
                autoreleasepool {
                    self.semaphore.wait()
                    if i > 0 && i % 100 == 0 {
                        latestCommandBuffer?.waitUntilCompleted()
                        let correctRate = self.evaluateAccuracy()
                        print("iteration \(i): correct rate: \(correctRate)")
                    }
                    let commandBuffer = MPSCommandBuffer(from: self.net.queue)
                    let returnBatch = self.net.encodeTrainingBatchToCommandBuffer(commandBuffer, sourceImages: images, labels: labels)
                    let outputBatch = labels.map({$0.lossImage()})
                    commandBuffer.addCompletedHandler {[unowned self] (buffer) in
                        self.semaphore.signal()
                        let loss = self.getLoss(batch: outputBatch)
                        if i % 50 == 0 || i == self.trainSet.batchCount(with: self.batchSize) - 1 {
                            let elapsedTime = Date().timeIntervalSince(date)
                            print("iteration: \(i) Training loss = \(loss) elapsed time: \(elapsedTime)s Estimated Time:\(elapsedTime / (Double(i + 1) / Double(self.trainSet.batchCount(with: self.batchSize))))s")
                        }
                        
                        if let error = buffer.error {
                            print("Error:\(error)")
                        }
                    }
                    MPSImageBatchSynchronize(returnBatch, commandBuffer)
                    MPSImageBatchSynchronize(outputBatch, commandBuffer)
                    commandBuffer.commit()
                    latestCommandBuffer = commandBuffer
                }
            }
            let correctRate = self.evaluateAccuracy()
            print("Epoch:\(epoch) elapsed time: \(Date().timeIntervalSince(date)) s test accuracy: \(correctRate)")
        }
    }
    
    func getLoss(batch: [MPSImage]) -> Float {
        var result: Float = 0
        batch.forEach { (image) in
            var val: [Float] = [0]
            image.readBytes(&val, dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)
            result += (val[0] / Float(batch.count))
        }
        return result
    }
    
    func evaluateAccuracy() -> Double {
        net.inferenceGraph.reloadFromDataSources()
        var correctCount = 0
        var latestComandBuffer: MPSCommandBuffer?
        var index: Int = 0
        var max: Float = 0
        testSet.enumerateBatches(device: net.device, batchSize: batchSize, isRandom: false) { [unowned self] (i, images, labels: [Int]) in
             autoreleasepool {
                 self.semaphore.wait()
                let commandBuffer = MPSCommandBuffer(from: self.net.queue)
                let output = self.net.encodeInferenceBatchToCommandBuffer(commandBuffer, sourceImages: images)
                 MPSImageBatchSynchronize(output, commandBuffer)
                 commandBuffer.addCompletedHandler { (commandBuffer) in
                     self.semaphore.signal()
                     output.enumerated().forEach { (item) in
                         var data = [UInt16](repeating: 0, count: 10)
                         item.element.readBytes(&data, dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)
                         max = 0
                         index = 0
                         var values = [Float]()
                         for i in 0..<data.count {
                             let s = (data[i] & signBit) >> 15
                             let e = (data[i] & exponentBit) >> 10
                             let m = (data[i] & fractionBit)
                             let value = (s == 1 ? float_minius_1 : float_1) * powf(2, Float(e) - 15) * Float(1 + Float(m) * powf2_minus10)
                             if value > max {
                                 max = value
                                 index = i
                             }
                             values.append(value)
                        }
                        if index == labels[item.offset] {
                            correctCount += 1
                        }
                     }
                 }
                 commandBuffer.commit()
                 latestComandBuffer = commandBuffer
             }
        }
        if let buffer = latestComandBuffer {
             buffer.waitUntilCompleted()
        }
        return Double(correctCount) / Double(self.testSet.count)
    }
}
