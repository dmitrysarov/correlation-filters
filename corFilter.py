# -*- coding: utf-8 -*-
"""
@author: Dmitrii Shaulskii
"""

import numpy as np
import os
from scipy import misc
from sklearn.metrics import roc_auc_score
   
class corFilter():
    def __init__(self,corFilterName = 'QCF', 
                         posEigen = 1,
                         negEigen = 1,
                         corFilterSize = 64):
        self.corFilterName = corFilterName
        if corFilterName in ['QCF','RQQCF'] :
            self.posEigen = posEigen
            self.negEigen = negEigen
            self.corFilterSize = corFilterSize
            
    def prepareTrueImages(self, intialTrueImagesFolder, finalTrueImagesFolder):
        '''
        resize images to proper size. Size of final images is set by .corFilterSize parameter. 
        '''
        if not os.path.exists(finalTrueImagesFolder):
            os.mkdir(finalTrueImagesFolder)
        filesName = [name for name in os.listdir(intialTrueImagesFolder) 
                            if os.path.splitext(name)[1] in ['.jpg','png','tif']]
        for i in filesName:
            img = misc.imread(os.path.join(intialTrueImagesFolder,i), flatten = True)            
            new_img = misc.imresize(img, (self.corFilterSize, self.corFilterSize), interp = 'nearest')      
            misc.imsave(os.path.join(finalTrueImagesFolder,i), new_img)
        self.scale = float(self.corFilterSize)/img.shape[0]
        
    def prepareFalseImages(self, intialFalseImagesFolder, finalFalseImagesFolder, falseChipsNum = 2000):
        '''
        truncate images to proper size and scale. Input image can be of arbitrary size
        '''
        if not os.path.exists(finalFalseImagesFolder):
            os.mkdir(finalFalseImagesFolder)
        filesName = [name for name in os.listdir(intialFalseImagesFolder) 
                            if os.path.splitext(name)[1] in ['.jpg','png','tif']]
        for i in xrange(falseChipsNum):
            f = filesName[np.random.randint(len(filesName))]
            img = misc.imread(os.path.join(intialFalseImagesFolder,f), flatten = True)            
            img = misc.imresize(img, self.scale, interp = 'nearest')      
            x = np.random.randint(img.shape[1]-self.corFilterSize + 1)
            y = np.random.randint(img.shape[0]-self.corFilterSize + 1)
            new_img = img[y:y+self.corFilterSize,x:x+self.corFilterSize]
            misc.imsave(os.path.join(finalFalseImagesFolder,os.path.splitext(f)[0] + '_' + str(i) + '.jpg'), new_img)        
     
    def getImageList(self,folder):
        '''
        form the list of images adresess
        '''
        filesPath = [os.path.join(folder,name) for name 
        in os.listdir(folder) if os.path.splitext(name)[1]
        in ['.jpg','png','tif']]
        return np.array(filesPath)
        
    def getTrueClassFilesList(self, trueClass):
        self.trueClassFilesList = self.getImageList(trueClass)
        
    def getFalseClassFilesList(self, trueClass):
        self.falseClassFilesList = self.getImageList(trueClass)        
        
    def getImage(self, imagePath, asRow = True, subMean = True, histEq = False, normVar = True):
        image = misc.imread(imagePath, flatten = True) #read image as grayscale
        self.imageSize = image.shape
        if image.shape[0] != self.corFilterSize:
            image = misc.imresize(image,(self.corFilterSize,self.corFilterSize), interp = 'nearest')
        if asRow == True:
            image = image.flatten()
        if histEq == True:
            image = image # for fueture 
        if subMean == True:
            image = image - np.mean(image)
        if normVar == True:
            image = image / np.var(image)
        return image
        
    def fCor(self,image1,image2, fullField = False):
        '''
        Xcorrelation by mean of FFT
        '''
        self.fullField = fullField
        def myfft(a):
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))
        def myifft(a):
            return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))
        def cutOddMat(a):
            for ax,d in enumerate(a.shape):
                if d%2 != 0:                   
                    a = np.delete(a,0,axis = ax)
            return a
        image1 = cutOddMat(image1)
        image2 = cutOddMat(image2)
        # set images to same size
        difInShape = np.array(image1.shape) - np.array(image2.shape)
        if difInShape[0]>0:
            image2 = np.pad(image2, ((difInShape[0]/2,difInShape[0]/2),(0,0)), 'constant', constant_values = 0)
        elif difInShape[0]<0:
            image1 = np.pad(image1, ((-difInShape[0]/2,-difInShape[0]/2),(0,0)), 'constant', constant_values = 0)
        if difInShape[1]>0:
            image2 = np.pad(image2, ((0,0),(difInShape[1]/2,difInShape[1]/2)), 'constant', constant_values = 0)
        elif difInShape[1]<0:
            image1 = np.pad(image1, ((0,0),(-difInShape[1]/2,-difInShape[1]/2)), 'constant', constant_values = 0)            
        if fullField == True:
            image1 = np.pad(image1, ((image1.shape[0]/2, image1.shape[0]/2),(image1.shape[1]/2, image1.shape[1]/2)), 'constant', constant_values = 0)
            image2 = np.pad(image2, ((image2.shape[0]/2, image2.shape[0]/2),(image2.shape[1]/2, image2.shape[1]/2)), 'constant', constant_values = 0)
        xcorr = myifft(myfft(image1)*np.conj(myfft(image2)))
        return xcorr
        
    def quadraticCorrelation(self,image, crop = True):
        '''
        Quadratic filter output calculation
        '''
        for i in xrange(self.posEigen):
            eigenVector = np.reshape(self.eigenVector[i], (self.corFilterSize,self.corFilterSize))
            if i == 0:
                qxcorr = np.abs(self.fCor(image,eigenVector))**2
                continue
            qxcorr = qxcorr + np.abs(self.fCor(image,eigenVector))**2
        for i in xrange(self.negEigen):
            eigenVector = np.reshape(self.eigenVector[-i-1], (self.corFilterSize,self.corFilterSize))
            qxcorr = qxcorr - np.abs(self.fCor(image,eigenVector))**2
        if self.fullField is True:
            if crop is True:
                qxcorr = qxcorr[qxcorr.shape[0]/4:qxcorr.shape[0]*3/4,qxcorr.shape[1]/4:qxcorr.shape[1]*3/4]
        return qxcorr 
        
    def get_roc_auc(self):
        '''
        Use AUC ROC metric for evaluation of filter performance
        '''
        self.trueCorrelation = np.zeros(len(self.trueClassFilesList))
        self.falseCorrelation = np.zeros(len(self.falseClassFilesList))
        for i,path in enumerate(self.trueClassFilesList):
            img = self.getImage(path, asRow = False)
            if self.corFilterName == 'QCF':
                self.trueCorrelation[i] = np.max(self.quadraticCorrelation(img))
        for i,path in enumerate(self.falseClassFilesList):
            img = self.getImage(path, asRow = False)
            if self.corFilterName == 'QCF':
                self.falseCorrelation[i] = np.max(self.quadraticCorrelation(img))
        score = roc_auc_score(np.hstack((np.ones(len(self.trueClassFilesList)),np.zeros(len(self.falseClassFilesList)))),np.hstack((self.trueCorrelation,self.falseCorrelation)))
        return score
        
    def train(self,trueClass,falseClass = None):
        '''
        Train filter on certain condition
        '''
        self.getTrueClassFilesList(trueClass)
        trueImgeNumbers = np.random.choice(range(len(self.trueClassFilesList)), size = self.posEigen, replace=False)
        if falseClass is not None:
            self.getFalseClassFilesList(falseClass)
#            falseImgeNumbers =  np.random.choice(range(len(self.falseClassFilesList)), size = self.negEigen, replace=False)          
            score = 0.4
            score_new = 0.5
            while(True):
                score = score_new
                self.assembleFilter(self.trueClassFilesList[trueImgeNumbers], self.falseClassFilesList)        
                score_new = self.get_roc_auc()
                trueMin = np.argmin(self.trueCorrelation)
                if any(trueImgeNumbers ==  trueMin):
                    print 'image number %d (%s) repeated twice'%(trueMin,self.trueClassFilesList[trueMin])
                    break
                trueImgeNumbers = np.append(trueImgeNumbers,trueMin)
#                falseMax = np.argmax(self.falseCorrelation)
#                falseImgeNumbers = np.append(falseImgeNumbers, falseMax)
                self.trueImgeNumbers = trueImgeNumbers
                print score_new
                print trueImgeNumbers
#            self.assembleFilter(self.trueClassFilesList[trueImgeNumbers[:-1]], self.falseClassFilesList) 
            
    def assembleFilter(self, trueClassFilesList, falseClassFilesList = None, fastQC = False):
        '''
        Assemble the filter from availble images in trueClassFilesList falseClassFilesList
        '''
        if isinstance(trueClassFilesList, (str, unicode)):
             trueClassFilesList = [trueClassFilesList] 
        if falseClassFilesList is not None:
             if isinstance(falseClassFilesList, (str, unicode)):
                  falseClassFilesList = [falseClassFilesList] 
        if self.corFilterName == 'QCF':
            if fastQC == False:
                 for i,imagePath in enumerate(trueClassFilesList):
                     if i == 0:
                         tempImg = self.getImage(imagePath)
                         trueImagesMat = np.zeros((len(tempImg),len(trueClassFilesList)))
                         trueImagesMat[:,i] = tempImg
                         continue
                     trueImagesMat[:,i] = self.getImage(imagePath)
                 trueImagesMat = trueImagesMat/np.sqrt(i+1)
                 trueCorMat = np.dot(trueImagesMat,np.transpose(trueImagesMat)) #true correlation matrix
                 del trueImagesMat, tempImg
                 difCorMat = trueCorMat #if there is no false class then correlation matrix difference is a simple true cor matrix
                 del trueCorMat
                 if falseClassFilesList is not None: 
                     for i,imagePath in enumerate(falseClassFilesList):
                         if i == 0:
                             tempImg = self.getImage(imagePath)
                             falseImagesMat = np.zeros((len(tempImg),len(falseClassFilesList)))
                             falseImagesMat[:,i] = tempImg
                             continue
                         falseImagesMat[:,i] = self.getImage(imagePath)
                     falseImagesMat = falseImagesMat/np.sqrt(i+1)
                     falseCorMat = np.dot(falseImagesMat,np.transpose(falseImagesMat)) #false correlation matrix
                     del falseImagesMat, tempImg
                     difCorMat = difCorMat - falseCorMat #difference of correlation matrices
                     del falseCorMat
                 eigenValue, eigenVector = np.linalg.eig(difCorMat)
                 eigenVector = np.real(eigenVector)
                 del difCorMat
                 self.eigenValues, self.eigenVector = zip(*sorted(zip(eigenValue, np.transpose(eigenVector)),reverse=True))
                 del eigenValue, eigenVector
            '''
            FAST EIGEN VALUE CALCULATION. Uses the correspondence of outer and inner matrices eigen vectors
            Use it when training set size less then number of pixels
            '''     
            if fastQC == True:                 
                 for trueClassNum,imagePath in enumerate(trueClassFilesList):
                     if trueClassNum == 0:
                         tempImg = self.getImage(imagePath)
                         trueImagesMat = np.zeros((len(tempImg),len(trueClassFilesList)))
                         trueImagesMat[:,trueClassNum] = tempImg
                         continue
                     trueImagesMat[:,trueClassNum] = self.getImage(imagePath)
                 A = trueImagesMat/np.sqrt(trueClassNum+1)
                 del trueImagesMat
                 if falseClassFilesList is not None: 
                     for falseClassNum,imagePath in enumerate(falseClassFilesList):
                         if falseClassNum == 0:
                             tempImg = self.getImage(imagePath)
                             falseImagesMat = np.zeros((len(tempImg),len(falseClassFilesList)))
                             falseImagesMat[:,falseClassNum] = tempImg
                             continue
                         falseImagesMat[:,falseClassNum] = self.getImage(imagePath)
                     falseImagesMat = 1j*falseImagesMat/np.sqrt(falseClassNum+1)
                     A = np.hstack((A,falseImagesMat)) 
                     del falseImagesMat
                 eigenValue, eigenVector = np.linalg.eig(np.dot(np.transpose(A),A))
                 eigenVector = np.dot(A,eigenVector)
                 eigenVector = map(lambda x,y: x/y, eigenVector.T, np.sqrt(eigenValue))
                 del A
                 eigenValues, eigenVector = zip(*sorted(zip(eigenValue, eigenVector),reverse=True))
                 eigenVector = np.real(eigenVector) + np.imag(eigenVector)
                 self.eigenValues = eigenValues
                 self.eigenVector = eigenVector
                 del eigenValue, eigenVector
        if self.corFilterName == 'RQQCF':
            if fastQC == False:
                 for i,imagePath in enumerate(trueClassFilesList):
                     if i == 0:
                         tempImg = self.getImage(imagePath)
                         trueImagesMat = np.zeros((len(tempImg),len(trueClassFilesList)))
                         trueImagesMat[:,i] = tempImg
                         continue
                     trueImagesMat[:,i] = self.getImage(imagePath)
                 trueImagesMat = trueImagesMat/np.sqrt(i+1)
                 trueCorMat = np.dot(trueImagesMat,np.transpose(trueImagesMat)) #true correlation matrix
                 del trueImagesMat, tempImg
                 difCorMat = trueCorMat #if there is no false class then correlation matrix difference is a simple true cor matrix
                 del trueCorMat
                 if falseClassFilesList is not None: 
                     for i,imagePath in enumerate(falseClassFilesList):
                         if i == 0:
                             tempImg = self.getImage(imagePath)
                             falseImagesMat = np.zeros((len(tempImg),len(falseClassFilesList)))
                             falseImagesMat[:,i] = tempImg
                             continue
                         falseImagesMat[:,i] = self.getImage(imagePath)
                     falseImagesMat = falseImagesMat/np.sqrt(i+1)
                     falseCorMat = np.dot(falseImagesMat,np.transpose(falseImagesMat)) #false correlation matrix
                     del falseImagesMat, tempImg
                     difCorMat = difCorMat - falseCorMat #difference of correlation matrices
                     del falseCorMat
                 eigenValue, eigenVector = np.linalg.eig(difCorMat)
                 eigenVector = np.real(eigenVector)
                 del difCorMat
                 self.eigenValues, self.eigenVector = zip(*sorted(zip(eigenValue, np.transpose(eigenVector)),reverse=True))
                 del eigenValue, eigenVector
    
    def detect(self,image,thresh=1.3, rawImage = False, scale = True):
        '''
        apply filter
        '''
        if scale == True:
            image = misc.imresize(image, self.scale, interp = 'nearest')
        if rawImage == False:
            image = image - np.mean(image)
            image = image/np.var(image)
        if self.corFilterName == 'QCF':
            xcor = self.quadraticCorrelation(image)
            xcor = xcor - np.min(xcor)
        xcor = xcor/np.mean(xcor)
        thresh_xcor = np.zeros(xcor.shape)
#        thresh = np.max(xcor)*thresh
        thresh_xcor[xcor>thresh] = xcor[xcor>thresh]
        thresh_xcor_crop = thresh_xcor[10:-10, 10:-10] # crop due to window of maximum value elimination
        first = True
        self.correlationField = xcor.copy()
        points = None
        x_list = None
        y_list = None
        while(np.sum(thresh_xcor_crop) != 0):
            thresh_xcor_crop = thresh_xcor[10:-10, 10:-10]
            if first == True:
                first = False
                y_croped,x_croped = zip(*np.where(thresh_xcor_crop == np.max(thresh_xcor_crop)))[0]
                y = y_croped + 10
                x = x_croped + 10
                thresh_xcor[y-9:y+9,x-9:x+9] = 0 # maximum value window elimination
                y_list, x_list = np.array([int(y/self.scale)]),np.array([int(x/self.scale)])
            else:
                y_croped,x_croped = zip(*np.where(thresh_xcor_crop == np.max(thresh_xcor_crop)))[0]
                y = y_croped + 10
                x = x_croped + 10
                y_list = np.append(y_list,int(y/self.scale))
                x_list = np.append(x_list,int(x/self.scale))
                thresh_xcor[y-9:y+9,x-9:x+9] = 0 # maximum value window elimination
#                print x,y
        if x_list is not None and  y_list is not None:      
            points = zip(x_list, y_list) 
        return points