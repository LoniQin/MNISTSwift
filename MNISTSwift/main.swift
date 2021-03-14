//
//  main.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/2.
//  Copyright © 2020 lonnie. All rights reserved.
//
import MLCompute
import MetalPerformanceShaders
guard
    let device = MTLCreateSystemDefaultDevice(),
    let queue = device.makeCommandQueue() else {
        exit(0)
}
do {
    var paths = #file.components(separatedBy: "/")
    paths.removeLast()
    let basePath = paths.joined(separator: "/")
    let params = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: basePath + "/" + "configuration_sgd.json")), options: .allowFragments) as! [String: Any]
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
    classfier.train(epochCount: 3)
} catch let error {
    print(error)
}

