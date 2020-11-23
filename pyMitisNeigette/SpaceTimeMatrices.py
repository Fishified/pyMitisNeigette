import numpy as np
import vtk
import pandas as pd
from vtk.util import numpy_support as VN
from matplotlib import pyplot as plt
from scipy import signal
from matplotlib import rc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

rc('font',**{'family':'serif','serif':['Times New Roman'],'size':'9'})
rc('text', usetex=True)


class SpaceTimeMatrices():
    
    def __init__(self,run):
        
        #instance of run
        self.run=run
        
        print('SpaceTimeMatrices module loaded')
        print(self.run.path+"\\"+"postProcessingOrdered\\surfaces\\top\\")
        self.vtpFileNames=self.run.ptv.getImagesInDirectory(self.run.path+"//"+"postProcessingOrdered//surfaces//top//","vtp",sort=True)
        

    def getExtractRectLimits(self):
        
        """
        Returns coordinates related to extraction rectangles
        
        Rectangles are 8 m long by 0.5 m wide
        Rectangles are spaced by 6 m center to center
        """
        
        rwCoords=[[56.59,99.07],[94.13,115.36]] #real-world coords
        imgCoords=[(686,131),(1001,1358)]       #corresponding img coords
        sf=0.032285                             #scaling factor
        
        basePoint=[180+83,880]                  #top-left corner of 1st rect on image
        longSpacing=186                         #0.032285 m/px = 6.002 m
        nbrSpacings=5
        
        usLeft=[]
        usMid=[]
        usRight=[]
        
        for count, spacing in enumerate(range(nbrSpacings)):
            usLeftPoints=(basePoint[1]+count*longSpacing,basePoint[0])
            usLeft.append(usLeftPoints)
            
        for count, spacing in enumerate(range(nbrSpacings)):
            usMidPoints=(basePoint[1]+count*longSpacing,(180+207))
            usMid.append(usMidPoints)

        for count, spacing in enumerate(range(nbrSpacings)):
            usRightPoints=(basePoint[1]+count*longSpacing,180+331)
            usRight.append(usRightPoints)
        
        #get upstream left corners of rectangles on drone images
        self.usLeftx,self.usLefty,angle=self.run.ptv.getRealWorldCoords(imgCoords,rwCoords,usLeft,sf)
        print(angle)
        #get upstream midpoints of rectanges on drone images
        self.usMidx,self.usMidy,angle=self.run.ptv.getRealWorldCoords(imgCoords,rwCoords,usMid,sf)        
        
        #get upstream right corners of rectangles on drone images 
        self.usRightx,self.usRighty,angle=self.run.ptv.getRealWorldCoords(imgCoords,rwCoords,usRight,sf)
        

    def extractDataFromVTPfiles(self):
        
        self.planeOrigins=[]
        self.planeNormals=[]
        self.ends=[]
        self.planes=[]
        
        
        for count, i in enumerate(self.usLeftx):
            self.planeOrigins.append([i,self.usLefty[count],2.2])
            self.planeNormals.append([-0.15746,0.9875,0])
        
        for count, i in enumerate(self.usRightx):
            self.ends.append([i,self.usRighty[count],2.2])
        
        for count,origin in enumerate(self.planeOrigins):
            self.planes.append(self.createPlane(origin,self.planeNormals[count])) # Create the line        


        self.profiles={'plane1':[],'plane2':[],'plane3':[],'plane4':[],'plane5':[]}
        self.extractData()


    def extractData(self):
        
        names=[]
        for key in self.profiles.keys():
            names.append(key)
            
        for file in self.vtpFileNames:
            print('Extracting profiles from: %s' %file)
            reader = self.readVTK(file) # read the VTKfile
            data = reader.GetOutput()
            
            #slice data on plane
            for count, plane in enumerate(self.planes):
                
                cutEdges = vtk.vtkCutter()
                cutEdges.SetInputConnection(reader.GetOutputPort())
                cutEdges.SetCutFunction(plane)
                cutEdges.Update()
            
                probe = vtk.vtkProbeFilter()
                probe.SetInputConnection(cutEdges.GetOutputPort())
                probe.SetSourceData(data)
                probe.Update()
                
                scalar=VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray('s'))
                points=self.returnPoints(probe)
    
                self.profiles[names[count]].append(np.stack((points[:,0],points[:,1],points[:,2],scalar),axis=1))
                

    def treatData(self, profiles,startPoints,endPoints,intrplPos):
        
        #clip, label and sort data
        self.treatedProfiles={}
        
        for count, key in enumerate(self.profiles.keys()):
            
            self.treatedProfiles[key]=[]
            
            for profile in self.profiles[key]:
                
                profile=profile[(profile[:,0] > startPoints[count][0]) & (profile[:,0] < endPoints[count][0])]
                distances=((profile[:,0]-startPoints[count][0])**2+(profile[:,1]-startPoints[count][1])**2)**0.5
                
                df=pd.DataFrame((np.stack((distances,profile[:,3]),axis=1)))
                df.columns=['l','s']
                df=df.sort_values(by=['l'])
                interpolatedPoints=np.interp(intrplPos,df['l'],df['s'])
    
                self.treatedProfiles[key].append(interpolatedPoints)

    def saveTreatedData2H5py(self,fileName):

        df=pd.DataFrame(self.treatedProfiles)
        df.to_hdf('%s\\%s.h5' % (self.run.path,fileName),'data')
        

    def rereadTreatedDataFromH5py(self,fileName):
        
        reread = pd.read_hdf('%s\\%s.h5' % (self.run.path,fileName))
        self.treatedProfiles=reread.to_dict(orient='list')

    def rereadCorrsDataFromH5py(self):
        
        self.corrs=[]
        
        for count in range(5):    
            reread = pd.read_hdf('%s\\%s.h5' % (self.run.path+'\\Data','cfdCorrs'+"_%d" %count))
            self.corrs.append(reread.to_numpy())
        

    def readVTK(self,filename):
        
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        
        return reader
    
    def createPlane(self,origin,normal):
    
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        
        return plane
    
    def returnPoints(self,cutData):
    
        numPoints = cutData.GetOutput().GetNumberOfPoints() 
    
        x = np.zeros(numPoints)
        y = np.zeros(numPoints)
        z = np.zeros(numPoints)
        points = np.zeros((numPoints , 3))
    
        for i in range(numPoints):
            x[i],y[i],z[i] = cutData.GetOutput().GetPoint(i)
            points[i,0]=x[i]
            points[i,1]=y[i]
            points[i,2]=z[i]
        return points
    
    def setZeroToNaN(self,array):
        # In case zero-values in the data, these are set to NaN.
        array[array==0]=np.nan
        return array
    
    
    def createPlottingData(self):
        
        self.dataForPlotting={}
    
        for count, key in enumerate(self.treatedProfiles):
            
            array=[]
            
            for profile in self.treatedProfiles[key]:
                array.append(profile)
                
            stacked=np.stack(array, axis=1)
        
            self.dataForPlotting[key]=stacked


    def setupCrossCorrelationsData(self):
        """
        Cross-correlation analysis
        """
        self.dataForPlotting1mean = self.dataForPlotting['plane1'] - self.dataForPlotting['plane1'].mean()
        self.dataForPlotting2mean = self.dataForPlotting['plane2'] - self.dataForPlotting['plane2'].mean()
        self.dataForPlotting3mean = self.dataForPlotting['plane3'] - self.dataForPlotting['plane3'].mean()
        self.dataForPlotting4mean = self.dataForPlotting['plane4'] - self.dataForPlotting['plane4'].mean()
        self.dataForPlotting5mean = self.dataForPlotting['plane5'] - self.dataForPlotting['plane5'].mean()
        
        #makes all pixels less than the mean equal to zero
        self.dataForPlotting1mean[self.dataForPlotting1mean < 0] = 0
        self.dataForPlotting2mean[self.dataForPlotting2mean < 0] = 0
        self.dataForPlotting3mean[self.dataForPlotting3mean < 0] = 0
        self.dataForPlotting4mean[self.dataForPlotting4mean < 0] = 0
        self.dataForPlotting5mean[self.dataForPlotting5mean < 0] = 0


    def extractSelection(self):
        
        #KH1=np.copy(dataForPlotting1mean[0:500, 13:71])
        self.template=np.copy(self.dataForPlotting3mean[0:500, 225:283])


    def doCrossCorrelations(self):
        
        """
        Provide 
        """
        
        self.corrs=[]
        
        print("Start performing cross-correlations")

        self.corrs.append(signal.correlate2d(self.dataForPlotting1mean, self.template, boundary='symm', mode='same'))
#        self.yKH1_1, self.xKH1_1 = np.unravel_index(np.argmax(self.corrKH1_1), self.corrKH1_1.shape)
        
        print("Done xcorrelations in rect 1.")
        
        self.corrs.append(signal.correlate2d(self.dataForPlotting2mean, self.template, boundary='symm', mode='same'))
#        self.yKH1_2, self.xKH1_2 = np.unravel_index(np.argmax(self.corrKH1_2), self.corrKH1_2.shape)
        
        print("Done xcorrelations in rect 2.")
        
        self.corrs.append(signal.correlate2d(self.dataForPlotting3mean, self.template, boundary='symm', mode='same'))
#        self.yKH1_3, self.xKH1_3 = np.unravel_index(np.argmax(self.corrKH1_3), self.corrKH1_3.shape)
        
        print("Done xcorrelations in rect 3.")
        
        self.corrs.append(signal.correlate2d(self.dataForPlotting4mean, self.template, boundary='symm', mode='same'))
#        self.yKH1_4, self.xKH1_4 = np.unravel_index(np.argmax(self.corrKH1_4), self.corrKH1_4.shape)
        
        print("Done xcorrelations in rect 4.")
        
        self.corrs.append(signal.correlate2d(self.dataForPlotting5mean, self.template, boundary='symm', mode='same'))
#        self.yKH1_5, self.xKH1_5 = np.unravel_index(np.argmax(self.corrKH1_5), self.corrKH1_5.shape)        

        print("Done xcorrelations in rect 5.")


        for count, corr in enumerate(self.corrs):
            df=pd.DataFrame(corr)
            df.to_hdf('%s\\%s.h5' % (self.run.path+'\\Data','cfdCorrsNew'+"_%d" %count),'data')  


    def plotCFDCrossCorrelations(self):

        """
        Plot data
        """

        x, y = np.meshgrid(np.linspace(0,self.dataForPlotting1mean.shape[1],self.dataForPlotting1mean.shape[1]), 
                           np.linspace(0,self.dataForPlotting1mean.shape[0],self.dataForPlotting1mean.shape[0]))
        
        fig, ax = plt.subplots(10,1,sharex=True,sharey=True)
        fig.set_figwidth(7,forward=True)
        fig.set_figheight(7,forward=True)
        ax[0].pcolormesh(x, y, self.dataForPlotting1mean, cmap='gray')
        ax[2].pcolormesh(x, y, self.dataForPlotting2mean, cmap='gray')
        ax[4].pcolormesh(x, y, self.dataForPlotting3mean, cmap='gray')
        ax[6].pcolormesh(x, y, self.dataForPlotting4mean, cmap='gray')
        ax[8].pcolormesh(x, y, self.dataForPlotting5mean, cmap='gray')
        
        
        colorMap='Blues'
        ax[1].pcolormesh(x, y, self.corrs[0], cmap=colorMap)
        ax[3].pcolormesh(x, y, self.corrs[1], cmap=colorMap)
        ax[5].pcolormesh(x, y, self.corrs[2], cmap=colorMap)
        ax[7].pcolormesh(x, y, self.corrs[3], cmap=colorMap)
        ax[9].pcolormesh(x, y, self.corrs[4], cmap=colorMap)
        
        peaks=[(234,140),(255,152),(273,184),(287,168)]
        xLength=58
        
        faceColors=['red','yellow','yellow','yellow','yellow']
        axes=[2,4,6,8]
        
        for count, peak in enumerate(peaks):
            
            rect=[]
            rect.append(Rectangle((peak[0]-xLength/2, 0), xLength, 500))
            ax[axes[count]].add_collection(PatchCollection(rect, facecolor=faceColors[count], alpha=0.3,edgecolor=''))
        
        ax[0].set_xlim([0,600])
        #ax[0].set_ylim([40,374])
        fig.subplots_adjust(wspace=0, hspace=0)
        
        ax[0].yaxis.set_ticks([28*2,68*2,108*2,148*2,188*2])
        
        ax[0].xaxis.set_ticks([0,100,200,300,400,500,600])
        ax[0].set_xticklabels(['0','50','100','150','200','250','300'])
        
        ax[0].set_yticklabels(['4','2','0','-2','-4'])
        ax[0].invert_yaxis()
        
        fig.text(0.51, 0.05, 't (s)', ha='center')
        fig.text(0.06, 0.5, r'$y_{\xi}$ (m)', va='center', rotation='vertical')
        
        fig.text(0.13, 0.815, 'a)', va='center', color='white', fontsize=11)
        fig.text(0.16, 0.815, r'$x_{\xi}$ = 0 m', va='center', color='white', fontsize=11)
        
        fig.text(0.13, 0.66, 'b)', va='center', color='white',fontsize=11)
        fig.text(0.16, 0.66, r'$x_{\xi}$ = 6 m', va='center', color='white',fontsize=11)
        
        fig.text(0.13, 0.505, 'c)', va='center', color='white',fontsize=11)
        fig.text(0.16, 0.505, r'$x_{\xi}$ = 12 m', va='center', color='white',fontsize=11)
        
        fig.text(0.13, 0.35, 'd)', va='center', color='white',fontsize=11)
        fig.text(0.16, 0.35, r'$x_{\xi}$ = 18 m', va='center', color='white',fontsize=11)
        
        fig.text(0.13, 0.20, 'e)', va='center', color='white',fontsize=11)
        fig.text(0.16, 0.20, r'$x_{\xi}$ = 24 m', va='center', color='white',fontsize=11)
        
        ax[2].annotate('KH1 pattern',
                    xy=(234, 220), xycoords='data',
                    xytext=(280, 220), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'),color='white')
        ax[3].annotate('KH1 peak ($t$ = 117, $\delta t = 0 s$)',
                    xy=(234, 140), xycoords='data',
                    xytext=(280, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'),color='black',horizontalalignment='left', verticalalignment='top')
        ax[5].annotate('KH1 peak ($t$ = 127.5, $\delta t = 10.5 s$)',
                    xy=(255, 152), xycoords='data',
                    xytext=(280, 250), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'),horizontalalignment='left', verticalalignment='top')
        ax[7].annotate('KH1 peak ($t$ = 136.5, $\delta t = 9 s$)',
                    xy=(273,184), xycoords='data',
                    xytext=(300, 250), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'),horizontalalignment='left', verticalalignment='top')
        ax[9].annotate('KH1 peak ($t$ = 143.5, $\delta t = 7 s$)',
                    xy=(287,168), xycoords='data',
                    xytext=(287, 250), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'),horizontalalignment='left', verticalalignment='top')

        #output file
        plt.savefig(self.run.path+"\\"+"\\Images\\fig_cfdXcorrelations.png", format='png', dpi=600)

