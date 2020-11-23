import pyMitisNeigette.run as run

proc=run.Run(r'Z:\Jason\Projects\2020\MitisNeigette\CFD\MainSims\main_split2Smag',vtp=True,dCorrs=True,dd='h5')

#proc.makeFreeSurfaceVelocityPlots()
proc.makeCFDSpaceTimeFig()

print(proc.stm.vtpFileNames)
