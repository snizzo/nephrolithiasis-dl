import torch
import torch.nn as nn

class NNUtils:
    '''
    Some utility functions to generate blocks of arbitrary size for CNNs
    '''

    def getConvBlockRelu(self, in_channels, out_channels, kernel_size, padding, numblocks):
        b = nn.Sequential()

        for i in range(numblocks-1):
            b.append(nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding
            ))
            b.append(nn.ReLU())
        
        b.append(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding
            ))
        b.append( nn.ReLU() )

        b.append( nn.MaxPool2d(kernel_size=2, stride=2) )

        return b
    
    def getTestMultiBlock(self, inc, outc, ks, pad, nb):
        testblock = nn.Sequential ()

        for i in range(nb-1):
            testblock.append(
            nn.Conv2d(
                in_channels=inc,
                out_channels=inc,
                kernel_size=ks,
                padding=pad
            ))
            testblock.append(
                nn.BatchNorm2d(inc)
            )
            testblock.append(
                nn.ReLU())

        testblock.append(
            nn.Conv2d(
                in_channels=inc,
                out_channels=outc,
                kernel_size=ks,
                padding=pad
            ))
        testblock.append(
                nn.BatchNorm2d(outc)
            )
        testblock.append(
            nn.ReLU())
        return testblock