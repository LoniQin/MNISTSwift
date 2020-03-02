//
//  functions.swift
//  MNISTSwift
//
//  Created by lonnie on 2020/3/3.
//  Copyright © 2020 lonnie. All rights reserved.
//

import Foundation

func execute(launchPath: String, currentDirectory: String? = nil, arguments: [String] = []) -> String {
    let pipe = Pipe()
    let file = pipe.fileHandleForReading
    let task = Process()
    task.launchPath = launchPath
    task.arguments = arguments
    task.standardOutput = pipe
    if let currentDirectory = currentDirectory  {
        task.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
    }
    task.launch()
    let data = file.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)!
}

func gzipPath() -> String {
    return execute(launchPath: "/usr/bin/which", arguments: ["gzip"]).components(separatedBy: .newlines).first!
}
