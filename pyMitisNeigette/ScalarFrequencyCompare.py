
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times New Roman'],'size':'9'})
rc('text', usetex=True)


class ScalarFrequencyCompare():
    
    def __init__(self,run):
        
        self.run=run

    def smooth(self,y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def processData(self):

        self.drone=self.run.stmd.hstacks[4].mean(axis=0)
        self.drone=self.drone/self.drone.mean()-1
        self.cfd=self.run.stm.dataForPlotting['plane3'].mean(axis=0) #data from cfdMixingAnalysis.py run before!!
        self.cfd=self.cfd/self.cfd.mean()-1
        self.drone=self.drone[0:600]
        
        self.droneSmooth=self.smooth(self.drone,6)
        self.cfdSmooth=self.smooth(self.cfd,6)

    def plotFrequencyComapre(self):

        fig, ax = plt.subplots(nrows=2,sharex=False,sharey=False)
        fig.set_figwidth(3,forward=True)
        fig.set_figheight(4.5,forward=True)
        
        #axis 0
        ax[0].plot(self.droneSmooth,c='black',lw=0.5,label='drone')
        ax[0].plot(self.cfdSmooth,c='blue',lw=0.5,label='WMLES')
        
        ax[0].set_xlim([0,600])
        ax[0].set_ylim([-1,1])
        ax[0].xaxis.set_ticks([0,100,200,300,400,500,600])
        ax[0].set_xticklabels(['0','50','100','150','200','250','300'])
        ax[0].yaxis.set_ticks([-1,-0.5,0,0.5,1])
        ax[0].set_yticklabels(['-1','-0.5','0','0.5','1'])
        ax[0].set_xlabel('$t$ (s)')
        
        ax[0].set_ylabel(r'$S_{o}^{+} [-]$' )
        #= \frac{\langle S_{o}^{\prime} \rangle}{\langle \overline{S_{o}} \rangle - 1}
        #axis 1
        fDrone, pxx_denDrone = signal.welch(x=self.droneSmooth, fs=2, detrend='constant',nfft=512)
        fCFD, pxx_denCFD = signal.welch(x=self.cfdSmooth, fs=2, detrend='constant',nfft=512)
        
        ax[1].plot(fDrone,pxx_denDrone,c='black',lw=0.5, label='drone')
        ax[1].plot(fCFD,pxx_denCFD,c='blue',lw=0.5,label='WMLES')
        
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        
        ax[1].set_xlim([0.01,1])
        
        ax[1].set_xlabel('Hz')
        ax[1].set_ylabel(r'PSD [$S_{o}^{+}Hz^{-1}]$') #$[\frac{\langle S_{o}^{\prime} \rangle_{y^{\prime}}}{\langle \overline{S_{o}} \rangle_{y^{\prime}} -1 }]^{2} Hz^{-1}$')
        ax[0].text(0.02, 0.06, 'a)', va='center', color='black', fontsize=9,transform=ax[0].transAxes)
        ax[1].text(0.02, 0.06, 'b)', va='center', color='black', fontsize=9,transform=ax[1].transAxes)
        ax[0].text(0.05, 0.92, r'$x_{\xi} = 12 m$', va='center', color='black', fontsize=7,transform=ax[0].transAxes)
        
            
        ax[0].text(0.5, 0.1, r'$S_{o}^{+} = \frac{\langle S_{o}^{\prime} \rangle_{y_{\xi}}}{\langle \overline{S_{o}} \rangle_{y_{\xi}} -1 }$', va='center', color='black', fontsize=9,transform=ax[0].transAxes)
        
        ax[0].legend(loc='upper right')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.2)
        
    
        ax[1].annotate('0.055 Hz',
                    xy=(0.40, 0.95), xycoords='axes fraction',
                    xytext=(0.55, 0.90), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='blue'))
        
        ax[1].annotate('0.043 Hz',
                    xy=(0.34, 0.9), xycoords='axes fraction',
                    xytext=(0.17, 0.7), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->",fc='red',ec='black'))



        plt.savefig(self.run.path+"\\"+"\\Images\\fig_freqCompare.pdf", format='pdf', dpi=600)


