#############################################################################
#                                                                           #
#   Synthetic GPR Image Generation using Generative Adversarial Networks    #
#   Copyright (C) 2020  Jan Rottmayer                                       #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #  
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
#############################################################################
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse

# Preprocessing
import glob
from PIL import Image, ImageOps
import random

# model dependencies
import time
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad, Variable
from utils.parser import *
from utils.utils import *
from utils.dataset import *
from models import Generator, Discriminator



class WGAN_GPR():
    def __init__(self, batchSize=64, learningRate=0.0001, epochs=10,\
        dataDir= './data', hiddenDim = 100, discUpdates = 10, gpWeight = 10, runName = "test", imgSize = 128):
        self.dataDir = dataDir
        self.imgSize = imgSize
        self.useCuda = True if torch.cuda.is_available() else False
        self.hidden = hiddenDim
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.discUpdates = discUpdates
        self.gpWeight = gpWeight
        self.checkpointDir = './checkpoints/' + runName + "/"
        self.sampleDir = './samples/' + runName + "/"
        self.dataLoader = ""
        if not os.path.exists(self.checkpointDir):
            os.makedirs(self.checkpointDir)
        if not os.path.exists(self.sampleDir): 
            os.makedirs(self.sampleDir)
        self.filePathG = self.checkpointDir + "G_state_{}.pth".format(0)
        self.filePathD = self.checkpointDir + "D_state_{}.pth".format(0)
        self.device = getDevice(2)
        self.buildModel()
        self.fixedLatent = torch.rand(self.batchSize,self.hidden).to(self.device)
        self.loadData()
        
    def loadData(self):
        self.dataLoader = makeDataLoader(self.dataDir, self.batchSize, self.imgSize)
        print("[*] Data has been loaded successfully.")
        
    def  buildModel(self, optimizer = "adam"):
        self.G = Generator(self.hidden)
        self.D = Discriminator()
        if self.useCuda:
            self.G.to(self.device)
            self.D.to(self.device)
        if optimizer == "adam":
            self.gOptim = optim.Adam(self.G.parameters(),lr=self.learningRate, betas=(0.0, 0.9))
            self.dOptim = optim.Adam(self.D.parameters(),lr=self.learningRate, betas=(0.0, 0.9))
        elif optimizer == "rms":
            self.gOptim = optim.RMSprop(self.G.parameters(),lr=self.learningRate)
            self.dOptim = optim.RMSprop(self.D.parameters(),lr=self.learningRate)
        else:
            print("Unrecognized Optimizer !")
            return 0
        
        
    def saveModel(self, epoch):
        self.filePathG = self.checkpointDir + "G_state_{}.pth".format(epoch)
        self.filePathD = self.checkpointDir + "D_state_{}.pth".format(epoch)
        torch.save(self.D.state_dict(), self.filePathD)
        torch.save(self.G.state_dict(), self.filePathG)
    
    def loadModel(self, directory = ''):
        if not directory:
            directory = self.checkpointDir
        listG = glob.glob(directory + "G*.pth")
        listD = glob.glob(directory + "D*.pth")
        if  len(listG) == 0 or len(listD) == 0:
            print("[*] No Checkpoint found! Starting from scratch.")
            return 1
        Gfile = max(listG, key=os.path.getctime)
        Dfile = max(listD, key=os.path.getctime)
        epochFound = int( (Gfile.split('_')[-1]).split('.')[0])
  
        print("[*] Checkpoint {} found at {}!".format(epochFound, directory))
        dState = torch.load(Dfile)
        gState = torch.load(Gfile)
        
        self.D.load_state_dict(dState)
        self.G.load_state_dict(gState)
        return epochFound
    
    def logProcess(self, epoch, step, stepPerEpoch, gLog, dLog, wDist, gradPen, lossesF):
        summaryStr = 'Epoch [{}], Step [{}/{}], Losses: G [{:4f}], D [{:4f}], W-dist [{:4f}], gp: [{:4f}]'.format(epoch, step, stepPerEpoch, gLog, dLog, wDist, gradPen)
        print(summaryStr)
        lossesF.write(summaryStr)
    
    def plot_losses(lossesList, legendsList, fileOut):
        assert len(lossesList) == len(legendsList)
        for i, loss in enumerate(lossesList):
            plt.plot(loss, label=legendsList[i])
        plt.legend()
        plt.savefig(fileOut)
        plt.close()
    
    def interpolate(self, realImages, fakeImages):
        n = realImages.shape[0]
        theta = torch.tensor(np.random.uniform(size = n), dtype = torch.float).view(n, 1, 1, 1).to(self.device)
        sample = theta * realImages + (1 - theta) * fakeImages
        return sample
        
    def gradientNorm(self, realImages, fakeImages):
        n = realImages.shape[0]
        _input = self.interpolate(realImages, fakeImages)
        _input = Variable(_input, requires_grad = True)
        score = self.D(_input)
        outputs = torch.ones(score.shape).to(self.device)
        gradient = grad(outputs = score, 
                        inputs = _input, 
                        grad_outputs = outputs,
                        create_graph = True,
                        retain_graph = True)[0]
        gradNorm = torch.sqrt(torch.sum(gradient.view(n, -1) ** 2, dim=1) + 1e-12) # to stop 0 gradients!!!!
        return ((gradNorm - 1) ** 2).mean()
        
    def wassersteinLoss(self, labels, predictions):
        return torch.mean(labels * predictions)
    
    def sample(self, epoch):
        self.G.eval()
        generated = self.G(self.fixedLatent)
        generated = denormalize(generated)
        path = self.sampleDir + "sampled_{}.png".format(epoch)
        torchvision.utils.save_image(generated, path ,nrow = 10)
        self.G.train()
    
    def discriminatorStep(self, realImages):
        self.dOptim.zero_grad()            # Setting Current Gradient to Zero
        # Generate Images from Latent Representation
        latentVectors = torch.randn(self.batchSize, self.hidden).to(self.device)
        fakeImages = self.G(latentVectors)
        
        # Label Smoothing
        labelNoise = torch.randn(self.batchSize) * 0.1 
        
        # Assign Data Labels
        realLabels = (torch.ones(self.batchSize) - labelNoise).to(self.device)
        fakeLabels = -torch.ones(self.batchSize).to(self.device)
        
        # Score Images
        realScore = self.D(realImages)
        fakeScore = self.D(fakeImages)

        # Calculate Gradient Penalty
        gradPenalty = self.gradientNorm(realImages, fakeImages)

        # Discriminator Losses 
        realLoss = self.wassersteinLoss(realLabels, realScore)
        fakeLoss = -self.wassersteinLoss(fakeLabels, fakeScore)
        dLoss = - realLoss + fakeLoss + (self.gradientNorm(realImages, fakeImages)) * self.gpWeight

        # Optimization Step        
        dLoss.backward()              # Calculate Gradients trough Backpropagation
        self.dOptim.step()            # Optimization Step
        return dLoss.item(), realLoss.item(), fakeLoss.item(), gradPenalty.item()
    
    def generatorStep(self):
        # Freeze Discriminator
        for parameter in self.D.parameters():
            parameter.requires_grad_(False)        
        self.gOptim.zero_grad()
        # Generate Images from Latent Representation
        latentVectors = torch.randn(self.batchSize, self.hidden).to(self.device)
        fakeImages = self.G(latentVectors)
        
        # Create Fake Labels
        fakeLabels = -torch.ones(self.batchSize).to(self.device)
        # Score Fake Images
        fakeScore = self.D(fakeImages)
        
        gLoss = self.wassersteinLoss(fakeLabels, fakeScore)
        gLoss.backward()
        self.gOptim.step()
        for parameter in self.D.parameters():
            parameter.requires_grad_(True)
        return gLoss.item()
        
    def trainModel(self, loadDir = ""):
        self.epoch = self.loadModel(loadDir)
        lossesF = open(self.checkpointDir + "losses.txt",'a+')
        dLosses, gLosses, gradPen = [],[],[]
        fakeLosses, realLosses, wDist = [],[],[]
        stepPerEpoch = len(self.dataLoader)
        gLoss = 0
        count = 1
        for epoch in range(self.epoch,self.epochs):
            for batch, realImages in enumerate(self.dataLoader):
                realImages = realImages.to(self.device)

                # Discriminator Training
                dLoss, realLoss, fakeLoss, gradPenalty = self.discriminatorStep(realImages)
                
                dLosses.append(dLoss)
                gradPen.append(gradPenalty)
                wDist.append(realLoss - fakeLoss)
                fakeLosses.append(fakeLoss)
                realLosses.append(realLoss)
                
                if count % self.discUpdates == 0:
                    gLoss = self.generatorStep()

                count += 1
                gLosses.append(gLoss)
                if batch % 8 ==0:
                    self.logProcess(epoch, batch, stepPerEpoch, gLosses[-1], dLosses[-1], wDist[-1], gradPen[-1], lossesF)



            if epoch % 10 == 0:
                # self.logProcess(epoch, batch, stepPerEpoch, gLosses[-1], dLosses[-1], wDist[-1], gradPen[-1], lossesF)
                plotLosses([gLosses, dLosses], ["gen", "disc"], self.sampleDir + "losses.png")
                plotLosses([wDist], ["w_dist"], self.sampleDir + "dist.png")
                plotLosses([gradPen],["grad_penalties"], self.sampleDir + "grad.png")
                plotLosses([fakeLosses, realLosses],["d_fake", "d_real"], self.sampleDir + "d_loss_components.png")
            
            if epoch == 1:
                saveImg = denormalize(realImages)
                torchvision.utils.save_image(saveImg,os.path.join(self.sampleDir,'real.png'),nrow = 10)     

            if epoch % 100 == 0:
                self.saveModel(epoch)
                self.sample(epoch)
            
def main():
    args = parameter_parser()
    runName = args.runName
    with open('config_{}.txt'.format(runName), 'w') as f:
        for item in vars(args):
            f.write("{}:{}\n".format(item,getattr(args,item)))
    model = WGAN_GPR(batchSize=args.batchSize, learningRate=args.learningRate, epochs=args.epochs,\
                    dataDir= args.dataDir, hiddenDim = args.hiddenDim, discUpdates = args.discUpdates,
                    gpWeight = args.gpWeight, runName = args.runName, imgSize = args.imgSize)
    model.trainModel()

if __name__ == '__main__':
    main()
