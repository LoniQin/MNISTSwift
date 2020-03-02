//
//  main.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//

import MetalPerformanceShaders
guard
    let device = MTLCreateSystemDefaultDevice(),
    let queue = device.makeCommandQueue() else {
        exit(0)
}
do {
    let params: [String: Any] = [
        "type": "adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 10e-8,
        "learningRate": 1e-3,
        "timeStep": 0
    ]
    /*
    let params: [String: Any] = [
        "type": "sgd",
        "learningRate": 0.1
    ]
    */
    let batchSize = 40
    let trainSet = MNISTDataSet(isTrain: true)
    let testSet = MNISTDataSet(isTrain: false)
    try trainSet.load()
    try testSet.load()
    let net = Net(device: device, queue: queue, outputCount: trainSet.outputCount, batchSize: batchSize, optimizerParams: params)
    let classfier = ImageClassifier(net: net,
                                    batchSize: batchSize,
                                    trainSet: trainSet,
                                    testSet: testSet)
    classfier.train(epochCount: 1)
} catch let error {
    print(error)
}

