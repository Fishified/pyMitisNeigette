import h5py
import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # added 
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':'9'})
rc('text', usetex=True)



class FreeSurfaceVelocityPlots():
    
    def __init__(self,run):
        
        self.run=run
        self.f = h5py.File(self.run.path+"\\Data\\pngseq_track.hdf5", 'r')


    def treatTracTracData(self):
        
        data=pd.DataFrame()
        data['Frame'] = np.array(self.f['Frame'])
        data['Id']=np.array(self.f['Id']).astype(str)
        data['x']=np.array(self.f['x'][:,0])
        data['y']=np.array(self.f['x'][:,1])
        data['u']=np.array(self.f['u'][:,0])
        data['v']=np.array(self.f['u'][:,1])
        
        #rearrange dataframe to sort by Id and then frame number
        particle=data.groupby(['Id','Frame']).mean().reset_index()     
        
        self.xbins=80
        self.ybins=60
        self.binx = np.arange(0,2760,self.xbins)
        self.biny = np.arange(0,1560,self.ybins)
        
        #compute 2D binned statistics, the 0.008 is the m/px scaling parameter, 20 is for 20 fps
        #U is in the 'up' direction on the raw drone images and therefore corresponds to the local streamwise velocity
        retMeanU = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['u']*0.008*20, statistic='mean', bins=[self.binx, self.biny])
        retStdU = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['u']*0.008*20, statistic='std', bins=[self.binx, self.biny])
        retCountU = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['u']*0.008*20, statistic='count', bins=[self.binx, self.biny])
        
        retMeanV = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['v']*0.008*20, statistic='mean', bins=[self.binx, self.biny])
        retStdV = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['v']*0.008*20, statistic='std', bins=[self.binx, self.biny])
        retCountV = stats.binned_statistic_2d(particle['x'], particle['y'], values=particle['v']*0.008*20, statistic='count', bins=[self.binx, self.biny])
        
        #means
        self.u=retMeanU.statistic
        self.v=retMeanV.statistic
        self.M = np.sqrt(self.u*self.u+self.v*self.v) 
        
        #std
        self.uStd=retStdU.statistic
        self.vStd=retStdV.statistic
        
        #count
        self.uCount=retCountU.statistic
        self.vCount=retCountV.statistic

    def treatCFDvectorData(self):
        
#        self.cfd=pd.read_csv(r"C:\Users\Jason\Google Drive (mitis.neigette@gmail.com)\Data\CFDdata\ptvZone_600s_main_split2.csv")
        self.cfd=pd.read_csv(self.run.path+"\\Data\\CFDvectors.csv")
        self.cfd=self.cfd.rename(columns={"Points:0":"x","Points:1":"y","Points:2":"z","UMean:0":"u","UMean:1":"v","UMean:2":"w"})

#        self.cfd=self.cfd.rename(columns={"Points:0":"x","Points:1":"y","UMean:0":"u","UMean:1":"v"})
        
        #rotate the extracted surface to align with the orthorectified photos
        def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
            """Rotate a point around a given point.
            
            I call this the "high performance" version since we're caching some
            values that are needed >1 time. It's less readable than the previous
            function but it's faster.
            """
            x, y = xy
            offset_x, offset_y = origin
            adjusted_x = (x - offset_x)
            adjusted_y = (y - offset_y)
            cos_rad = math.cos(radians)
            sin_rad = math.sin(radians)
            qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        
            return qx, qy
        
        xy=self.cfd[['x','y']].to_numpy().transpose()
        uv=self.cfd[['u','v']].to_numpy().transpose()
        
        qx, qy=rotate_around_point_highperf(xy,0.2108,origin=(76.88,116.92))
        qu, qv=rotate_around_point_highperf(uv,0.2108,origin=(0,0))
        
        self.cfd['qx']=qx
        self.cfd['qy']=qy
        
        self.cfd['qu']=qu
        self.cfd['qv']=qv
        
        #rotate GRPs to align with image coordinates
        #positions of GRPs in real-world units (m) (input data)
        #this data was obtained from dGPS data at the site (minus 5376400 and 258000)
        xPos=[76.883,74.928,62.176,59.743]
        yPos=[116.919,126.034,122.626,113.033]
        
        xyGRP=np.asarray([xPos,yPos])
        self.qxPos, self.qyPos = rotate_around_point_highperf(xyGRP,0.2108,origin=(76.88,116.92))
        
        #find pixel coords of GRPs
        scalingFactor=0.008
        self.qxPosPx=(self.qxPos-56.4)/scalingFactor
        self.qyPosPy=(128.28-self.qyPos)/scalingFactor
        
        #place cfd data on same coordinate as images and PTV vectors (local pixel coordinates 2720 x 1530, origin top-left)
        scalingFactor=0.008
        self.cfd['qx']=(self.cfd['qx']-56.4)/scalingFactor
        self.cfd['qy']=(128.28-self.cfd['qy'])/scalingFactor
        
        
        self.cfdU = stats.binned_statistic_2d(self.cfd['qx'], self.cfd['qy'], values=self.cfd['qu'], statistic='mean', bins=[self.binx, self.biny]).statistic
        self.cfdV = stats.binned_statistic_2d(self.cfd['qx'], self.cfd['qy'], values=self.cfd['qv'], statistic='mean', bins=[self.binx, self.biny]).statistic
        
        binx2 = np.arange(self.xbins/2,2720,self.xbins)
        biny2 = np.arange(self.ybins/2,1530,self.ybins)
        self.X, self.Y = np.meshgrid(binx2,biny2)
        self.cfdM = np.sqrt(self.cfdU*self.cfdU+self.cfdV*self.cfdV)         
        

    def makePTVvectorPlot(self):

        fig, ax = plt.subplots(2,1,sharex=False,sharey=False,frameon=False)
        fig.set_figwidth(4,forward=True)
        fig.set_figheight(5.09,forward=True)
        
        colors=['Spectral_r','black']
        ptv=ax[0].contourf(self.X.transpose(), self.Y.transpose(),self.M,cmap=colors[0],clim=[0,1.5],width=0.003,scale=50,levels=14)
        ax[1].contourf(self.X.transpose(), self.Y.transpose(),self.cfdM,cmap=colors[0],clim=[0,1.5],width=0.003,scale=50,levels=14)
        
        self.ptvVec=ax[0].quiver(self.X.transpose(), self.Y.transpose(),self.u,-self.v,color=colors[1],clim=[0,1.5],width=0.003,scale=50)
        self.OFVec=ax[1].quiver(self.X.transpose(), self.Y.transpose(),self.cfdU,self.cfdV,color=colors[1],clim=[0,1.5],width=0.003,scale=50)
        
        
        xticks = [40,540,1040,1540,2040,2540]
        xlabels = ['0','4','8','12','16','20']
        
        yticks = [1470,1220,970,720,470,220]
        ylabels = ['0','2','4','6','8','10']
        
        ax[0].set_xlim([40,2675])
        ax[1].set_xlim([40,2675])
        
        ax[0].set_ylim([30,1470])
        ax[1].set_ylim([30,1470])
        
        ax[0].invert_yaxis()
        ax[1].invert_yaxis()
        
        
        ax[0].set_xticks([])
        ax[0].set_xticklabels(xlabels, minor=False)
        
        
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(ylabels, minor=False)
        
        ax[1].set_xlim([40,2675])
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels(xlabels, minor=False)
        
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(ylabels, minor=False)
        
        #ax.set_title("Mean particle tracking velocimetry vectors")
        ax[0].set_ylabel("$x_{\zeta}$ (m)", labelpad=5)
        ax[1].set_ylabel("$x_{\zeta}$ (m)", labelpad=5)
        ax[1].set_xlabel("$y_{\zeta}$ (m)", labelpad=5)
        
        for i in [0,1]:
            ax[i].add_artist(plt.Circle((2560, 1420), 40, color='black'))
            ax[i].add_artist(plt.Circle((2560, 255), 40, color='black'))
            ax[i].add_artist(plt.Circle((910, 337), 40, color='black'))
            ax[i].add_artist(plt.Circle((361, 1441), 40, color='black'))
            ax[i].add_artist(plt.Circle((2560, 1420), 25, color='w'))
            ax[i].add_artist(plt.Circle((2560, 255), 25, color='w'))
            ax[i].add_artist(plt.Circle((910, 337), 25, color='w'))
            ax[i].add_artist(plt.Circle((361, 1441), 25, color='w'))
        
        ax[0].add_patch(patches.Rectangle((0,200 ), 200, 200,facecolor='white'))
        ax[0].text(0.01, 0.90, 'a)', fontsize=10, color='black',transform=ax[0].transAxes)
        ax[1].add_patch(patches.Rectangle((0,200 ), 200, 200,facecolor='white'))
        ax[1].text(0.01, 0.90, 'b)', fontsize=10, color='black',transform=ax[1].transAxes)
        fig.text(0.15, 0.765, 'PTV', fontsize=10, color='black')
        fig.text(0.15, 0.41, 'WMLES', fontsize=10, color='black')
        
        
        #plt.tight_layout()
        fig.colorbar(ptv,ax=ax.ravel().tolist(),location='top',shrink=0.5,pad=0.01,label=r'$|\langle \overline{U_{\zeta}} \rangle|$ (m/s)')
        
        
        plt.savefig(self.run.path+"\\"+"\\Images\\fig_vectors2.pdf", format='pdf', dpi=600)


    def uMagProfilePlots(self):
        
        
        """
        extract data on lines
        """
        
        #axis data
        x=self.X[0,:]
        
        #u_4=u[:,16]
        #u_8=u[:,7]
        #
        #v_4=v[:,16]
        #v_8=v[:,7]
        
        #cfdU_4=self.cfdU[:,16]
        #cfdU_8=self.cfdU[:,7]
        
        #cfdV_4=self.cfdV[:,16]
        #cfdV_8=self.cfdV[:,7]
        
        Ur_4=self.M[:,16]
        UrCFD_4=self.cfdM[:,16]
        
        Ur_8=self.M[:,7]
        UrCFD_8=self.cfdM[:,7]
        
        #std_4=self.uStd[:,16]
        #std_8=self.uStd[:,7]        

        """
        Plot figures
        """

        fig, ax = plt.subplots(1,2,sharex=True,sharey=False,frameon=False)
        fig.set_figwidth(5,forward=True)
        fig.set_figheight(2.5,forward=True)
        
        ax[0].plot(x,Ur_4,'k',label="PTV",lw=0.5) #CONTAINS YOUR 3RD ROW
        ax[0].plot(x,UrCFD_4,'blue', marker='s',mfc="None",ms=3,ls='--',label="WMLES",lw=0.5) #CONTAINS YOUR 4TH ROW
        
        ax[1].plot(x,Ur_8,'k',label="PTV",lw=0.5) #CONTAINS YOUR 3RD ROW
        ax[1].plot(x,UrCFD_8,'blue',marker='s',mfc="None",ms=3,ls='--',label="WMLES",lw=0.5) #CONTAINS YOUR 4TH ROW
        
        ax[0].set_xlim(0, 2500)
        ax[0].set_ylim(0, 1.5)
        
        ax[1].set_xlim(0, 2500)
        ax[1].set_ylim(0, 1.5)
        
        xticks = [0,500,1000,1500,2000,2500]
        xlabels = ['0','4','8','12','16','20']
        
        yticks = [0,0.25,0.5,0.75,1,1.25,1.5]
        ylabels = ['0','0.25','0.50','0.75','1.00','1.25','1.50']
        
        ax[0].set_xticks(xticks)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(ylabels, minor=False)
        ax[0].set_xticklabels(xlabels, minor=False)
        
        ax[0].set_ylabel(r'$|\langle \overline{U_{\zeta}} \rangle|$ (m/s)')
        ax[1].set_ylabel(r'$|\langle \overline{U_{\zeta}} \rangle|$ (m/s)')
        
        ax[0].set_xlabel("$y_{\zeta}$ (m)")
        ax[1].set_xlabel("$y_{\zeta}$ (m)")
        
        
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels(ylabels, minor=False)
        
        nbins = len(ax[0].get_xticklabels())
        #ax[0].xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        
        ax[1].legend(loc='lower right',edgecolor='black', frameon=False)
        fig.text(0.15, 0.87, 'a)', va='center', ha='center', fontsize=9)
        fig.text(0.64, 0.87, 'b)', va='center', ha='center', fontsize=9)
        
        fig.text(0.25, 0.87, "$x_{\zeta}$ = 4 m", va='center', ha='center', fontsize=10)
        fig.text(0.74, 0.87, "$x_{\zeta}$ = 8 m", va='center', ha='center', fontsize=10)
        
        #fig.text(0.5, 0.02, 'longitudinal distance (m)', va='center', ha='center', fontsize=9)
        #fig.text(0.02, 0.5, '$U_{r}$ (m/s)', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.tight_layout()
        
#        plt.savefig(r'C:\Users\Jason\Dropbox\Apps\ShareLaTeX\MitisNeigette_Article\Imgs\fig_Mcompare.pdf', format='pdf', dpi=300)
        plt.savefig(self.run.path+"\\"+"\\Images\\fig_vectors2.pdf", format='pdf', dpi=600)
                
        
        
        
        
        
