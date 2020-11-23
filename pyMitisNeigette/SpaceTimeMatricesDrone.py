import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib import rc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

rc('font',**{'family':'serif','serif':['Times New Roman'],'size':'9'})
rc('text', usetex=True)


class SpaceTimeMatricesDrone():
    
    def __init__(self, run):
        
        self.run=run
        
        

    def extractDataFromRectangles(self):
        
        print("Extracting data from drone images in rectangles ...")

        rotatedPaths=self.run.ptv.getImagesInDirectory(r"Z:\Jason\Projects\2020\MitisNeigette\Analysis\FieldDataAnalysis\MixingAnalysis\video_DJI_0378\rotated","png",sort=True)
        
        self.seriesMeans=[]

        self.basePoint=[180,880]
        self.width=20
        self.height=500
        self.longSpacing=93     #0.032285 m/px = 3.002 m
        nbrSamples=10
            
        
        for path in rotatedPaths:
        
            self.samples=[]
            extracts=[]
            means=[]
            
            #extract information over n rectangles along mixing-layer
            image = cv2.imread(path, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            
            for i in range(nbrSamples):
                self.samples.append([self.basePoint[0],self.basePoint[1]+i*self.longSpacing,self.height,self.width])
            
            for count, sample in enumerate(self.samples):
                
                extracts.append(gray[sample[0]:sample[0]+sample[2], sample[1]:sample[1]+sample[3]])
                means.append(extracts[count].mean(axis=1))
                
            self.seriesMeans.append(means)

    def rotateStabilizedImages(self):
        
        coloredPaths=self.run.ptv.getImagesInDirectory(r"C:/Users/Jason/Desktop/HigherFrequencyDJI378/stabilized_8fps","png",sort=True)
        
        for path in coloredPaths:
            print(path)
            image=self.run.ptv.rotateImage(path,4,'./',False)
            rotatedImagePath="%s/%s_rot.png" %(r"C:/Users/Jason/Desktop/HigherFrequencyDJI378/colorRotated", os.path.splitext(os.path.basename(path))[0])
            print('Rotated and moved to: %s' % rotatedImagePath)
            cv2.imwrite(rotatedImagePath, image)
#            print('Writing rectangle image')


    def drawRectanglesOnImages(self):
        
        coloredPaths=self.run.ptv.getImagesInDirectory(r"Z:\Jason\Projects\2020\MitisNeigette\Analysis\FieldDataAnalysis\MixingAnalysis\video_DJI_0378\stabilized_2fps","png",sort=True)
        
        for path in coloredPaths:
        
            image=self.run.ptv.rotateImage(path,4,'./',False)
        
            for count, sample in enumerate(self.samples):
                if count in [0,2,4,6,8]:
                    image=cv2.rectangle(image, (self.basePoint[1]+count*self.longSpacing, self.basePoint[0]+83), 
                                        (self.basePoint[1]+count*self.longSpacing+self.width, self.basePoint[0]+331), 255, 2)
                else:
                    pass
            
            rectangleImagePath='%s\%s_rect.png' %('./rectangles', os.path.splitext(os.path.basename(path))[0])
            cv2.imwrite(rectangleImagePath, image) 
            print('Writing rectangle image')

    
    def processMIdata(self):
        
        print("Processing mixing-interface data for correlations ...")
        
        arrays=[]
        self.hstacks=[]
        
        for i in range(len(self.seriesMeans[0])):
            
            array=[]
            for x in range(len(self.seriesMeans)):
                array.append(self.seriesMeans[x][i])
            arrays.append(array)
            
        for array in arrays:  
            self.hstacks.append(np.stack(array, axis=1))

        for count,i in enumerate(self.hstacks):
    
            df=pd.DataFrame(i)
            df.to_hdf('%s\\%s.h5' % (self.run.path+'\\Data','hstacks'+"_%d" %count),'data')
        
        #measured from averaged lateral plots
#        self.MIpositions=[205.234,183.101,191.94,223,232] #207.05 average


    def rereadHstackDataFromH5py(self):
        
        self.hstacks=[]
        for count in range(10):    
            reread = pd.read_hdf('%s\\%s.h5' % (self.run.path+'\\Data','hstacks'+"_%d" %count))
            self.hstacks.append(reread.to_numpy())
 
    
    def rereadCorrsDataFromH5py(self):
        
        self.corrs=[]
        
        for count in range(5):    
            reread = pd.read_hdf('%s\\%s.h5' % (self.run.path+'\\Data','corrs'+"_%d" %count))
            self.corrs.append(reread.to_numpy())


    def takeMeans(self):

        #templates for cross-correlations
        self.KH1D = np.copy(self.hstacks[2][0:500, 125:194])
        self.KH1D -= self.KH1D.mean()
        
        self.hstacks0mean = self.hstacks[0] - self.hstacks[0].mean()
        self.hstacks2mean = self.hstacks[2] - self.hstacks[2].mean()
        self.hstacks4mean = self.hstacks[4] - self.hstacks[4].mean()
        self.hstacks6mean = self.hstacks[6] - self.hstacks[6].mean()
        self.hstacks8mean = self.hstacks[8] - self.hstacks[8].mean()
            
        
    def doCorrelations(self): 
        

        print("Starting cross-correlations ...")

        self.corrs=[]
        
        self.corrs.append(signal.correlate2d(self.hstacks0mean, self.KH1D, boundary='symm', mode='same'))
        yKH1D_0, xKH1D_0 = np.unravel_index(np.argmax(self.corrs[0]), self.corrs[0].shape)
        
        print("Done xcorrelations for rect 1.")
        
        self.corrs.append(signal.correlate2d(self.hstacks2mean, self.KH1D, boundary='symm', mode='same'))
        yKH1D_1, xKH1D_1 = np.unravel_index(np.argmax(self.corrs[1]), self.corrs[1].shape)
        
        print("Done xcorrelations for rect 2.")
        
        self.corrs.append(signal.correlate2d(self.hstacks4mean, self.KH1D, boundary='symm', mode='same'))
        yKH1D_2, xKH1D_2 = np.unravel_index(np.argmax(self.corrs[2]), self.corrs[2].shape)
        
        print("Done xcorrelations for rect 3.")
        
        self.corrs.append(signal.correlate2d(self.hstacks6mean, self.KH1D, boundary='symm', mode='same'))
        yKH1D_3, xKH1D_3 = np.unravel_index(np.argmax(self.corrs[3]), self.corrs[3].shape)
        
        print("Done xcorrelations for rect 4.")
        
        self.corrs.append(signal.correlate2d(self.hstacks8mean, self.KH1D, boundary='symm', mode='same'))
        yKH1D_4, xKH1D_4 = np.unravel_index(np.argmax(self.corrs[4]), self.corrs[4].shape)  
                
        print("Done xcorrelations for rect 5.")
        
        
        for count, corr in enumerate(self.corrs):
            df=pd.DataFrame(corr)
            df.to_hdf('%s\\%s.h5' % (self.run.path+'\\Data','corrs'+"_%d" %count),'data')        
        

    def plotDroneCorrelations(self):
        
        """
        Plots of 2D cross-correlations
        """
        x, y = np.meshgrid(np.linspace(0,self.hstacks[0].shape[1],self.hstacks[0].shape[1]), np.linspace(0,self.hstacks[0].shape[0],self.hstacks[0].shape[0]))
        
        fig, ax = plt.subplots(10,1,sharex=True,sharey=True)
        fig.set_figwidth(7,forward=True)
        fig.set_figheight(7,forward=True)
        ax[0].pcolormesh(x, y, self.hstacks[0], cmap='gray', vmin=0, vmax=255)
        ax[2].pcolormesh(x, y, self.hstacks[2], cmap='gray', vmin=0, vmax=255)
        ax[4].pcolormesh(x, y, self.hstacks[4], cmap='gray', vmin=0, vmax=255)
        ax[6].pcolormesh(x, y, self.hstacks[6], cmap='gray', vmin=0, vmax=255)
        ax[8].pcolormesh(x, y, self.hstacks[8], cmap='gray', vmin=0, vmax=255)
        
        colorMap='Blues'
        ax[1].pcolormesh(x, y, self.corrs[0], cmap=colorMap)
        ax[3].pcolormesh(x, y, self.corrs[1], cmap=colorMap)
        ax[5].pcolormesh(x, y, self.corrs[2], cmap=colorMap)
        ax[7].pcolormesh(x, y, self.corrs[3], cmap=colorMap)
        ax[9].pcolormesh(x, y, self.corrs[4], cmap=colorMap)
        
        peaks=[(139,251),(159,249),(179,250),(195,285),(211,301)]
        
        xLength=69
        
        faceColors=['red','yellow','yellow','yellow','yellow']
        for count, peak in enumerate(peaks):
            
            rect=[]
            rect.append(Rectangle((peak[0]-xLength/2, 0), xLength, 500))
            ax[count*2].add_collection(PatchCollection(rect, facecolor=faceColors[count], alpha=0.3,edgecolor=''))
        
        ax[0].set_xlim([0,600])
        ax[0].set_ylim([40,374])
        fig.subplots_adjust(wspace=0, hspace=0)
        
        ax[0].yaxis.set_ticks([83,145,207,269,331])
        
        ax[0].xaxis.set_ticks([0,100,200,300,400,500,600])
        ax[0].set_xticklabels(['0','50','100','150','200','250','300'])
        ax[0].invert_yaxis()
        
        ax[0].set_yticklabels(['4','2','0','-2','-4'])
        
        fig.text(0.51, 0.05, 't (s)', ha='center')
        fig.text(0.05, 0.5, r'$y_{\xi}$ (m)', va='center', rotation='vertical')
        
        fig.text(0.13, 0.86, 'a)', va='center', color='black', fontsize=11)
        fig.text(0.16, 0.86, r'$x_{\xi}$ = 0 m', va='center', color='black', fontsize=11)
            
        fig.text(0.13, 0.705, 'b)', va='center', color='black',fontsize=11)
        fig.text(0.16, 0.705, r'$x_{\xi}$ = 6 m', va='center', color='black',fontsize=11)
        
        fig.text(0.13, 0.553, 'c)', va='center', color='black',fontsize=11)
        fig.text(0.16, 0.55, r'$x_{\xi}$ = 12 m', va='center', color='black',fontsize=11)
        
        fig.text(0.13, 0.40, 'd)', va='center', color='black',fontsize=11)
        fig.text(0.16, 0.40, r'$x_{\xi}$ = 18 m', va='center', color='black',fontsize=11)
        
        fig.text(0.13, 0.245, 'e)', va='center', color='black',fontsize=11)
        fig.text(0.16, 0.245, r'$x_{\xi}$ = 24 m', va='center', color='black',fontsize=11)
        
        ax[0].annotate('KH1 pattern',
                    xy=(140, 140), xycoords='data',
                    xytext=(200, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        ax[1].annotate('KH1 peak ($t$ = 69.5 s, $\delta t = 0$)',
                    xy=(140, 245), xycoords='data',
                    xytext=(200, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        ax[3].annotate('KH1 peak ($t$ = 79, $\delta t = 9.5 s$)',
                    xy=(159, 260), xycoords='data',
                    xytext=(200, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        ax[5].annotate('KH1 peak ($t$ = 89, $\delta t = 10s$)',
                    xy=(179, 250), xycoords='data',
                    xytext=(220, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        ax[7].annotate('KH1 peak ($t$ = 97,  $\delta t = 8s$)',
                    xy=(195, 275), xycoords='data',
                    xytext=(236, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        ax[9].annotate('KH1 peak ($t$ = 105, $\delta t = 8s$)',
                    xy=(211, 275), xycoords='data',
                    xytext=(252, 140), textcoords='data',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='red'))
        
        plt.savefig(self.run.path+"\\"+"\\Images\\fig_droneXcorrelations.png", format='png', dpi=600)
        
        
        
        
        

