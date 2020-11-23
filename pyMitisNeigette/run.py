import numpy as np
import os

import pyRiverPTV.PreProcess as ptv
import pyMitisNeigette.SpaceTimeMatrices as stm
import pyMitisNeigette.SpaceTimeMatricesDrone as stmd
import pyMitisNeigette.ScalarFrequencyCompare as sf
import pyMitisNeigette.FreeSurfaceVectorPlot as vec
import pyMitisNeigette.PIVsurfaceVectorPlot as p


class Run():
    
    def __init__(self,path,vtp,dd,dCorrs):
        
        self.path=path
        self.vtp=vtp
        self.dd=dd
        self.dCorrs=dCorrs
        #only to access functions from other packages
        self.ptv=ptv.PreProcess(self.path)
        
        self.setupFolders()
#        self.makeCFDSpaceTimeFig()
#        self.makeDroneSpaceTimeFig()
#        self.makeScalarFreqCompareFig()
#        self.makeFreeSurfaceVectorPlots()
#        self.makePIVcomparePlot()


    def setupFolders(self):

        """
        Creates folders for I/O
        """

        self.imageFolderPath = self.path+"\\"+"Images"        
        try:
            os.stat(self.imageFolderPath)
            print('"Images" directory already exists.')
        except:
            print('Made the "Images" directory')
            os.system("mkdir %s " % self.imageFolderPath)        

        
        self.DataFolderPath = self.path+"\\"+"Data"      
        try:
            os.stat(self.DataFolderPath)
            print('"Data" directory already exists.')
        except:
            print('Made the "Data" directory')
            os.system("mkdir %s " % self.DataFolderPath)      


    def makeCFDSpaceTimeFig(self):
        
        """
        Extract and plot data for SpaceTimeCorrelation figures
        """
        self.stm=stm.SpaceTimeMatrices(self)
        self.stm.getExtractRectLimits()
        
        #extract data from vtp files of surface
        
        if self.vtp == True:
            
            self.stm.extractDataFromVTPfiles()
            linePoints=np.arange(0,10.5,0.025) #10.5 m long line divided by 0.025 m sections
            self.stm.treatData(self.stm.profiles,self.stm.planeOrigins,self.stm.ends,linePoints)
            self.stm.saveTreatedData2H5py('extractRectData') 
            print('made it here')
        elif self.vtp == 'h5':
            print('Reading CFD rectangle data from h5 file')
            self.stm.rereadTreatedDataFromH5py('extractRectData')


        if self.dCorrs == True:
            print('Doing CFD correlations')
            self.stm.createPlottingData()
            self.stm.setupCrossCorrelationsData()
            self.stm.extractSelection()
            self.stm.doCrossCorrelations()
            
            
        elif self.dCorrs == 'h5':
            self.stm.createPlottingData()
            self.stm.setupCrossCorrelationsData()
            self.stm.rereadCorrsDataFromH5py()
            

        
        #self.stm.plotCFDCrossCorrelations()
        
        

    def makeDroneSpaceTimeFig(self):
        
        self.stmd=stmd.SpaceTimeMatricesDrone(self)
        
        #use this only rotate images for PIV work
        self.stmd.rotateStabilizedImages()
        
#        if self.dd == True:
#            
#            self.stmd.extractDataFromRectangles()
#            self.stmd.processMIdata()
#            self.stmd.takeMeans()
#            self.stmd.doCorrelations() 
#            self.stmd.plotDroneCorrelations()
#
#            
#        elif self.dd == 'h5':
#            
#            self.stmd.rereadHstackDataFromH5py()        
#            self.stmd.rereadCorrsDataFromH5py() 
#            self.stmd.plotDroneCorrelations()
#            
#        else:
#            print('Missing argument for "droneData"')
#            
#            return
        
        
    def makeScalarFreqCompareFig(self):
        
        self.sf=sf.ScalarFrequencyCompare(self)
        self.sf.processData()
        self.sf.plotFrequencyComapre()

   
    def makeFreeSurfaceVelocityPlots(self):
        
        self.vec=vec.FreeSurfaceVelocityPlots(self)
        self.vec.treatTracTracData()
        self.vec.treatCFDvectorData()
        self.vec.makePTVvectorPlot()
        self.vec.uMagProfilePlots()
        

    def makePIVcomparePlot(self):
        
        self.p=p.PIVsurfaceVectorPlot(self)
        
        
        