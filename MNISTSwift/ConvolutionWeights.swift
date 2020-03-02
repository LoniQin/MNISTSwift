//
//  ConvolutionWeights.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//

import MetalPerformanceShaders
@available(OSX 10.15, *)
class ConvolutionWeights: NSObject, MPSCNNConvolutionDataSource {
    struct Parameter {
        var kernelWidth: Int
        var kernelHeight: Int
        var inputFeatureChannels: Int
        var outputFeatureChannels: Int
        var stride: Int
        var label: String
        var optimizerParams: [String: Any]
    }
    typealias Vectors = (MPSVector, MPSVector, MPSVector)
    
    typealias Pointers = (pointer: UnsafeMutableRawPointer, momentumPointer: UnsafeMutableRawPointer, velocityPointer: UnsafeMutableRawPointer)
    
    var parameter: Parameter!
    
    private var device: MTLDevice!
    
    private var optimizer: MPSNNOptimizer!
    
    private var convelutionDescriptor: MPSCNNConvolutionDescriptor!
    
    private var weightVectorDescriptor: MPSVectorDescriptor!
    
    private var weightMomentumVector: MPSVector!
    
    private var weightVelocityVector: MPSVector!
    
    private var weightVector: MPSVector!
    
    private var biasVectorDescriptor: MPSVectorDescriptor!
    
    private var biasMomentumVector: MPSVector!
    
    private var biasVelocityVector: MPSVector!
    
    private var biasVector: MPSVector!
    
    private var convolutionWeightAndBias: MPSCNNConvolutionWeightsAndBiasesState!
    
    private var weightPointers: Pointers!
    
    private var biasPointers: Pointers!
    
    private var randomDescriptor: MPSMatrixRandomDistributionDescriptor!
    
    private var randomKernel: MPSMatrixRandomMTGP32!
    
    private var commandBuffer: MPSCommandBuffer!
    
    private var commandQueue: MTLCommandQueue!
    
    init(parameter: Parameter, device: MTLDevice, commandQueue: MTLCommandQueue) {
        super.init()
        self.parameter = parameter
        self.device = device
        self.commandQueue = commandQueue
        convelutionDescriptor = .init(kernelWidth: parameter.kernelWidth, kernelHeight: parameter.kernelHeight, inputFeatureChannels: parameter.inputFeatureChannels, outputFeatureChannels: parameter.outputFeatureChannels)
        convelutionDescriptor.strideInPixelsX = parameter.stride
        convelutionDescriptor.strideInPixelsY = parameter.stride
        convelutionDescriptor.fusedNeuronDescriptor = .cnnNeuronDescriptor(with: .none)
        optimizer = optimizerWithParams(parameter.optimizerParams)!
        weightVectorDescriptor = MPSVectorDescriptor(length: lenthOfWeights(), dataType: dataType())
        weightMomentumVector = MPSVector(device: device, descriptor: weightVectorDescriptor)
        weightVelocityVector = MPSVector(device: device, descriptor: weightVectorDescriptor)
        weightVector = MPSVector(device: device, descriptor: weightVectorDescriptor)
        biasVectorDescriptor = MPSVectorDescriptor(length: parameter.outputFeatureChannels, dataType: dataType())
        biasMomentumVector = MPSVector(device: device, descriptor: biasVectorDescriptor)
        biasVelocityVector = MPSVector(device: device, descriptor: biasVectorDescriptor)
        biasVector = MPSVector(device: device, descriptor: biasVectorDescriptor)
        convolutionWeightAndBias = MPSCNNConvolutionWeightsAndBiasesState(weights: weightVector.data, biases: biasVector.data)
        weightPointers = (weightVector.data.contents(), weightMomentumVector.data.contents(), weightVelocityVector.data.contents())
        var zero: Float = 0
        memset_pattern4(weightPointers.momentumPointer, &zero, sizeOfWeights())
        memset_pattern4(weightPointers.velocityPointer, &zero, sizeOfWeights())
        biasPointers = (biasVector.data.contents(), biasMomentumVector.data.contents(), biasVelocityVector.data.contents())
        memset_pattern4(biasPointers.momentumPointer, &zero, sizeOfBias())
        memset_pattern4(biasPointers.velocityPointer, &zero, sizeOfBias())
        var bias: Float = 0.1
        memset_pattern4(biasPointers.pointer, &bias, sizeOfBias())
        randomDescriptor = MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptor(withMinimum: -0.2, maximum: 0.2)
        randomKernel = MPSMatrixRandomMTGP32(device: device, destinationDataType: dataType(), seed: 0, distributionDescriptor: randomDescriptor)
        commandBuffer = MPSCommandBuffer(from: commandQueue)
        randomKernel.encode(commandBuffer: commandBuffer, destinationVector: weightVector)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        weightMomentumVector.data.didModifyRange(0..<sizeOfWeights())
        weightVelocityVector.data.didModifyRange(0..<sizeOfWeights())
        biasVector.data.didModifyRange(0..<sizeOfBias())
        biasMomentumVector.data.didModifyRange(0..<sizeOfBias())
        biasVelocityVector.data.didModifyRange(0..<sizeOfBias())
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return convelutionDescriptor
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return weightPointers.pointer
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return biasPointers.pointer.assumingMemoryBound(to: Float.self)
    }
    
    func load() -> Bool {
        autoreleasepool {
            let commandBuffer = MPSCommandBuffer(from: commandQueue)
            convolutionWeightAndBias.synchronize(on: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        return true
    }
    
    func update(with commandBuffer: MTLCommandBuffer, gradientState: MPSCNNConvolutionGradientState, sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
        if optimizer is MPSNNOptimizerAdam {
            (optimizer as! MPSNNOptimizerAdam).encode(commandBuffer: commandBuffer,
                                                      convolutionGradientState: gradientState,
                                                      convolutionSourceState: sourceState,
                                                      inputMomentumVectors: [weightMomentumVector, biasMomentumVector],
                                                      inputVelocityVectors: [weightVelocityVector, biasVelocityVector],
                                                      resultState: convolutionWeightAndBias)
        }
        if optimizer is MPSNNOptimizerStochasticGradientDescent {
            (optimizer as! MPSNNOptimizerStochasticGradientDescent).encode(commandBuffer: commandBuffer, convolutionGradientState: gradientState, convolutionSourceState: sourceState, inputMomentumVectors: [weightMomentumVector, biasMomentumVector], resultState: convolutionWeightAndBias)
        }
        return convolutionWeightAndBias
    }
    
    
    func purge() {
        
    }
    
    func label() -> String? {
        return parameter.label
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        return self
    }
    
    func optimizerWithParams(_ params: [String: Any]) -> MPSNNOptimizer? {
        guard
            let type = params["type"] as? String,
            let learningRate = params["learningRate"] as? Double else  {
            return nil
        }
        if
            type == "adam" {
            if
                let beta1 = params["beta1"] as? Double,
                let beta2 = params["beta2"] as? Double,
                let epsilon = params["epsilon"] as? Double,
                let timeStep = params["timeStep"] as? Int {
                let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: Float(learningRate), gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
                return MPSNNOptimizerAdam(device: device,
                                          beta1: beta1,
                                          beta2: beta2,
                                          epsilon: Float(epsilon),
                                          timeStep: timeStep,
                                          optimizerDescriptor: optimizerDescriptor)
                }
        }
        if type == "sgd" {
            let optimizerDescriptor = MPSNNOptimizerDescriptor(learningRate: Float(learningRate), gradientRescale: 1.0, regularizationType: .None, regularizationScale: 1.0)
            return MPSNNOptimizerStochasticGradientDescent(device: device, momentumScale: 0.1, useNestrovMomentum: true, optimizerDescriptor: optimizerDescriptor)
        }
        return nil
    }
    
    func lenthOfWeights() -> Int {
        return self.parameter.inputFeatureChannels * self.parameter.kernelWidth * self.parameter.kernelHeight * self.parameter.outputFeatureChannels
    }
    
    func sizeOfWeights() -> Int {
        return lenthOfWeights() * MemoryLayout<Float>.size
    }
    
    func sizeOfBias() -> Int {
        return parameter.outputFeatureChannels * MemoryLayout<Float>.size
    }
    
}
