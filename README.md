# MNISTSwift
This sample shows how to train MNIST dataset in Swift. It's based on **MetalPerformanceShaders** framework.
## Requirement
You should have **Xcode 11** installed. And the OS version should not be lower than **OSX 10.15**. And terminal should have **gzip** command installed to decompress gz file.
## Result
This sample can reach 98% test accuracy in 1 epoch.
```
iteration: 0 Training loss = 0.65278214 elapsed time: 0.6494189500808716s Estimated Time:974.1284251213074s
iteration: 50 Training loss = 0.0136705 elapsed time: 15.995146989822388s Estimated Time:470.44549970065844s
iteration 100: correct rate: 0.9419
iteration: 100 Training loss = 0.0051596183 elapsed time: 49.99788498878479s Estimated Time:742.542846368091s
iteration: 150 Training loss = 0.0057025035 elapsed time: 67.73050093650818s Estimated Time:672.8195457269024s
iteration 200: correct rate: 0.9545
iteration: 200 Training loss = 0.003107817 elapsed time: 103.74873399734497s Estimated Time:774.2442835622759s
iteration: 250 Training loss = 0.008943876 elapsed time: 119.94079601764679s Estimated Time:716.7776654441043s
iteration 300: correct rate: 0.9671
iteration: 300 Training loss = 0.007212986 elapsed time: 153.6869260072708s Estimated Time:765.8816910661336s
iteration: 350 Training loss = 0.0015372412 elapsed time: 169.74360394477844s Estimated Time:725.4000168580275s
iteration 400: correct rate: 0.9716
iteration: 400 Training loss = 0.007528485 elapsed time: 203.50220799446106s Estimated Time:761.2302044680589s
iteration: 450 Training loss = 0.001789755 elapsed time: 219.41161501407623s Estimated Time:729.7503825301869s
iteration 500: correct rate: 0.9681
iteration: 500 Training loss = 0.0070172334 elapsed time: 252.75309693813324s Estimated Time:756.7457992159677s
iteration: 550 Training loss = 0.0011382578 elapsed time: 268.74554097652435s Estimated Time:731.612180516854s
iteration 600: correct rate: 0.9736
iteration: 600 Training loss = 0.0011637404 elapsed time: 302.2500560283661s Estimated Time:754.3678603037423s
iteration: 650 Training loss = 0.00023202729 elapsed time: 318.53383898735046s Estimated Time:733.948937758872s
iteration 700: correct rate: 0.9751
iteration: 700 Training loss = 0.004278217 elapsed time: 352.7821159362793s Estimated Time:754.8832723315535s
iteration: 750 Training loss = 0.0026151112 elapsed time: 369.0782699584961s Estimated Time:737.1736417280214s
iteration 800: correct rate: 0.9743
iteration: 800 Training loss = 0.0052886275 elapsed time: 402.90297198295593s Estimated Time:754.4999475336253s
iteration: 850 Training loss = 0.003424872 elapsed time: 419.05810499191284s Estimated Time:738.6453084463799s
iteration 900: correct rate: 0.9783
iteration: 900 Training loss = 0.0038121094 elapsed time: 452.914381980896s Estimated Time:754.019503852768s
iteration: 950 Training loss = 0.0048718997 elapsed time: 469.18338894844055s Estimated Time:740.0368910858683s
iteration 1000: correct rate: 0.9741
iteration: 1000 Training loss = 0.0015581696 elapsed time: 503.3764899969101s Estimated Time:754.3104245707943s
iteration: 1050 Training loss = 0.009864102 elapsed time: 519.7115260362625s Estimated Time:741.7386194618399s
iteration 1100: correct rate: 0.9822
iteration: 1100 Training loss = 0.00035098594 elapsed time: 553.5216670036316s Estimated Time:754.1167125390076s
iteration: 1150 Training loss = 0.0024814794 elapsed time: 570.2696709632874s Estimated Time:743.1837588574554s
iteration 1200: correct rate: 0.9821
iteration: 1200 Training loss = 0.002272077 elapsed time: 604.570739030838s Estimated Time:755.0841869660758s
iteration: 1250 Training loss = 0.0011157334 elapsed time: 620.8732179403305s Estimated Time:744.4522996886457s
iteration 1300: correct rate: 0.9821
iteration: 1300 Training loss = 0.0002519559 elapsed time: 655.0903239250183s Estimated Time:755.292456485417s
iteration: 1350 Training loss = 0.0025631394 elapsed time: 673.3472139835358s Estimated Time:747.609786066102s
iteration 1400: correct rate: 0.9803
iteration: 1400 Training loss = 0.0028889023 elapsed time: 707.9034960269928s Estimated Time:757.9266552751528s
iteration: 1450 Training loss = 0.0037754804 elapsed time: 724.2509009838104s Estimated Time:748.7087191424642s
iteration: 1499 Training loss = 0.0006261967 elapsed time: 740.2294780015945s Estimated Time:740.2294780015945s
Epoch:0 elapsed time: 755.2867410182953 s test accuracy: 0.9817
```
