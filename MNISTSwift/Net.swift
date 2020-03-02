//
//  Net1.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//
import MetalPerformanceShaders
fileprivate let sameConvolutionPadding = MPSNNDefaultPadding(method: [.topLeft, .sizeSame, .centered])
fileprivate let validConvPadding = MPSNNDefaultPadding(method: [.centered, .topLeft, .validOnly])
fileprivate let samePoolingPadding = MPSNNDefaultPadding.forTensorflowAveragePooling()
fileprivate let validPoolingPadding = MPSNNDefaultPadding.forTensorflowAveragePoolingValidOnly()

class Net {
    
    var trainingGraph: MPSNNGraph!
    
    var inferenceGraph: MPSNNGraph!
    
    var device: MTLDevice
    
    var queue: MTLCommandQueue
    
    var outputCount: Int
    
    var batchSize: Int
    
    private var conv1Weights: ConvolutionWeights!
    
    private var conv2Weights: ConvolutionWeights!
    
    private var fc1Weights: ConvolutionWeights!
    
    private var fc2Weights: ConvolutionWeights!

    init(device: MTLDevice, queue: MTLCommandQueue, outputCount: Int, batchSize: Int, optimizerParams params: [String: Any]) {
        self.device = device
        self.queue = queue
        self.outputCount = outputCount
        self.batchSize = batchSize
        conv1Weights = ConvolutionWeights(parameter: .init(kernelWidth: 5, kernelHeight: 5, inputFeatureChannels: 1, outputFeatureChannels: 32, stride: 1, label: "conv1", optimizerParams: params), device: device, commandQueue: queue)
        conv2Weights = ConvolutionWeights(parameter: .init(kernelWidth: 5, kernelHeight: 5, inputFeatureChannels: 32, outputFeatureChannels: 64, stride: 1, label: "conv2", optimizerParams: params), device: device, commandQueue: queue)
        fc1Weights = ConvolutionWeights(parameter: .init(kernelWidth: 7, kernelHeight: 7, inputFeatureChannels: 64, outputFeatureChannels: 1024, stride: 1, label: "fc1", optimizerParams: params), device: device, commandQueue: queue)
        fc2Weights = ConvolutionWeights(parameter: .init(kernelWidth: 1, kernelHeight: 1, inputFeatureChannels: 1024, outputFeatureChannels: 10, stride: 1, label: "fc2", optimizerParams: params), device: device, commandQueue: queue)
        if let finalNode = filterNode(training: true) {
            let lossExitPoints = finalNode.trainingGraph(withSourceGradient: nil) { (gradientNode, inferenceNode, inferenceSource, gradientSource) in
                gradientNode.resultImage.format = .float32
            }!
            assert(lossExitPoints.count == 1)
            trainingGraph = MPSNNGraph(device: device, resultImage: lossExitPoints[0].resultImage, resultImageIsNeeded: true)!
            trainingGraph.format = .float16
        }
        
        if let finalNode = filterNode(training: false) {
            inferenceGraph = MPSNNGraph(device: device, resultImage: finalNode.resultImage, resultImageIsNeeded: true)!
            inferenceGraph.format = .float16
        }
    }
    func filterNode(training: Bool) -> MPSNNFilterNode? {
        let conv1Node =  MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil), weights: conv1Weights)
        conv1Node.paddingPolicy = sameConvolutionPadding
        let relu1 = MPSCNNNeuronReLUNode(source: conv1Node.resultImage, a: 0)
        let pool1 = MPSCNNPoolingMaxNode(source: relu1.resultImage, filterSize: 2, stride: 2)
        pool1.paddingPolicy = samePoolingPadding
        let conv2Node = MPSCNNConvolutionNode(source: pool1.resultImage, weights: conv2Weights)
        conv2Node.paddingPolicy = sameConvolutionPadding
        let relu2 = MPSCNNNeuronReLUNode(source: conv2Node.resultImage, a: 0)
        let pool2 = MPSCNNPoolingMaxNode(source: relu2.resultImage, filterSize: 2, stride: 2)
        pool2.paddingPolicy = samePoolingPadding
        let fc1Node = MPSCNNFullyConnectedNode(source: pool2.resultImage, weights: fc1Weights)
        let relu3 = MPSCNNNeuronReLUNode(source: fc1Node.resultImage, a: 0)
        var fcInputNode: MPSNNFilterNode = relu3
        if training {
            let node = MPSCNNDropoutNode(source: relu3.resultImage, keepProbability: 0.5, seed: 1,  maskStrideInPixels: .init(width: 1, height: 1, depth: 1))
            fcInputNode = node
        }
        let fc2Node = MPSCNNFullyConnectedNode(source: fcInputNode.resultImage, weights: fc2Weights)
        if training {
            let lossDescriptor = MPSCNNLossDescriptor(type: .softMaxCrossEntropy, reductionType: .sum)
            lossDescriptor.weight = 1 / Float(batchSize)
            return MPSCNNLossNode(source: fc2Node.resultImage, lossDescriptor: lossDescriptor)
        } else {
            return MPSCNNSoftMaxNode(source: fc2Node.resultImage)
        }
    }

}


extension Net {
    func encodeTrainingBatchToCommandBuffer(_ commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage], labels: [MPSCNNLossLabels]) -> [MPSImage] {
        let returnImage = trainingGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: [labels])!
        MPSImageBatchSynchronize(returnImage, commandBuffer)
        return returnImage
    }
    
    func encodeInferenceBatchToCommandBuffer(_ commandBuffer: MTLCommandBuffer, sourceImages: [MPSImage]) -> [MPSImage] {
        return inferenceGraph.encodeBatch(to: commandBuffer, sourceImages: [sourceImages], sourceStates: nil)!
    }
}
