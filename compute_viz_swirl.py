""" Generate visualizations using BSL tools.
"""

import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree as KDTree 
import imageio
import pyvista as pv
from make_swirl_figs import Dataset
import numpy as np 
import vtk
import math
import re
import gc

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def images_to_movie(imgs, outpath, fps=30):
    """ Write images to movie.
    Format inferred from outpath extension, 
    see imageio docs.
    """
    writer = imageio.get_writer(outpath, format='FFMPEG',fps=fps)

    for im in imgs:
        writer.append_data(imageio.imread(im))
    writer.close()

def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())

def viz_(dd, output_folder, clim=None, cpos=None, window_size=[768, 768], indices=None):
    """ Visualize a contour of a field variable.
    """
    output_folder = Path(output_folder)
    output_folder = output_folder / dd.folder.stem

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = range(len(dd.swirl_files))
    '''
    pl=pv.Plotter(off_screen=True, window_size=window_size)
    pl.camera_position = cpos
    pl.add_mesh(dd.mesh)
    pl.show(screenshot=output_folder / (dd.folder.stem + 'mesh'))
    '''
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.image_scale = 2
    if dd.HIT==False:
        p.camera_position = cpos

    silhouette = dict(color='black', line_width=1.0, decimate=None)
        
    ct = 0
    mesh=dd.mesh.copy()
    surf=dd.surf.copy()
    #surf.save(output_folder / (dd.folder.stem + 'surf.vtp'))
    grid = pv.ImageData()
    if 'PerturbNewt' in dd.folder.stem:
        bounds=np.array(surf.bounds)#np.array([0.5,0.5,0.5,0.5, 0.25,0.25])*np.array(surf.bounds)
        bounds_diff=np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        grid.origin = (-19.05,surf.center[1], surf.center[2])#(surf.center[0],surf.center[1], surf.center[2]-0.5*bounds_diff[2])
        grid.dimensions=(int(600*bounds_diff[0] / bounds_diff[0]), int(600*bounds_diff[1] / bounds_diff[0]), int(600*bounds_diff[2]/bounds_diff[0]))#(int(600*bounds_diff[0] / bounds_diff[1]), int(600*bounds_diff[1] / bounds_diff[1]), int(600*bounds_diff[2]/bounds_diff[1]))
        grid.spacing=(bounds_diff[0] / grid.dimensions[0], bounds_diff[1] / grid.dimensions[1], bounds_diff[2] / grid.dimensions[2])
    elif dd.folder.stem == 'PTSeg043' or dd.folder.stem == 'PTSeg028' or dd.folder.stem == 'PTSeg106':
        bounds=np.array(surf.bounds)
        bounds_diff=np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        grid.origin = (surf.center[0]-0.3*bounds_diff[0],surf.center[1]-0.3*bounds_diff[1], surf.center[2]-0.3*bounds_diff[2])
        grid.dimensions=(int(600*bounds_diff[0] / bounds_diff[1]), int(600*bounds_diff[1] / bounds_diff[1]), int(600*bounds_diff[2]/bounds_diff[1]))
        grid.spacing=(bounds_diff[0] / grid.dimensions[0], bounds_diff[1] / grid.dimensions[1], bounds_diff[2] / grid.dimensions[2])
    elif dd.folder.stem == 'Groccia':
        bounds=np.array(surf.bounds)
        bounds_diff=np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        grid.origin = (surf.center[0]-0.2*bounds_diff[0],surf.center[1]-0.3*bounds_diff[1], surf.center[2]-0.3*bounds_diff[2])
        grid.dimensions=(int(600*bounds_diff[0] / bounds_diff[1]), int(600*bounds_diff[1] / bounds_diff[1]), int(600*bounds_diff[2]/bounds_diff[1]))
        grid.spacing=(bounds_diff[0] / grid.dimensions[0], bounds_diff[1] / grid.dimensions[1], bounds_diff[2] / grid.dimensions[2])
    elif dd.folder.stem == 'Slow' or dd.folder.stem == 'JHData' :
        bounds=np.array(surf.bounds)
        bounds_diff=np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        grid.origin = (surf.center[0]-0.5*bounds_diff[0],surf.center[1]-0.5*bounds_diff[1], surf.center[2]-0.5*bounds_diff[2])
        grid.dimensions=(int(600*bounds_diff[0] / bounds_diff[1]), int(600*bounds_diff[1] / bounds_diff[1]), int(600*bounds_diff[2]/bounds_diff[1]))
        grid.spacing=(bounds_diff[0] / grid.dimensions[0], bounds_diff[1] / grid.dimensions[1], bounds_diff[2] / grid.dimensions[2])
    newmesh=grid.copy()
    #newmesh.save(output_folder / (dd.folder.stem + 'newmesh.vtk'))
    #pl=pv.Plotter(off_screen=True, window_size=window_size)

    #if dd.HIT==False:
    #    pl.camera_position = cpos

    #pl.add_mesh(newmesh)
    #pl.show(screenshot=output_folder / (dd.folder.stem + 'newmesh'))

    for idx in indices:
        gc.collect()
        if not Path(output_folder / (dd.folder.stem + '_volume_{:04d}'.format(idx))).exists():
            mesh.point_data['S'] = dd(idx)
            newmesh = grid.sample(mesh)
            if dd.HIT==False:
                cull='back'
            else:
                cull='front'

            if ct == 0:
                p.add_mesh(surf,
                color='w',
                opacity=1,
                show_scalar_bar=False,
                lighting=True,
                smooth_shading=True,
                specular=0.00,
                diffuse=0.9,
                ambient=0.5,
                culling=cull,
                silhouette=silhouette,
                name='surf',
                )

                p.add_volume(newmesh,
                    cmap='cool',#'jet_r',
                    opacity='linear',
                    scalars='S',
                    clim=clim,
                    name = 'prime',
                    show_scalar_bar=True,
                    #shade=True
                    )
                p.show(auto_close=False)
                p.screenshot(filename=output_folder / (dd.folder.stem + '_volume_{:04d}.png'.format(idx)),transparent_background=True, return_img=False)
                actors = [x for x in p.renderer.actors.keys()]
                for a in actors:
                    p.remove_actor(a)
            else:
                p.update()

    p.close()
    comm.Barrier()
    if rank == 0:
        keyl = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]
        _imgs = sorted(output_folder.glob('*volume*.png'), key=keyl)
        images_to_movie(_imgs, output_folder / (dd.folder.stem + '.mp4'), fps=30)


if __name__ == "__main__":
    folder = sys.argv[1] #results folder eg. results/art_
    case_name = sys.argv[2] #eg. case_028_low
    S_folder = sys.argv[3] #'/scratch/s/steinman/ahaleyyy/triple_decomp/swirl_files/case_A/case_028_low'
    S_out_folder = sys.argv[4] #eg spec_imgs
    cps=sys.argv[5]
    cpos_str = cps.split(',')[:]
    cpos_flt = [float(x) for x in cpos_str]
    it = iter(cpos_flt)
    cpos=list(zip(it,it, it))
    #print(cpos)
    if len(sys.argv)>6:
        if sys.argv[6]=='mono':
            mono=True
            HIT=False
        elif sys.argv[6]=='HIT':
            HIT=True
            mono=False
    else:
        mono=False
        HIT=False
    dd = Dataset(S_folder, folder, mono=mono, HIT=HIT)#, case_name = case_name)
    dd = dd.assemble_mesh() #gets mesh info from results/art_/PT_Seg028_low.h5
    #arr = dd(math.floor(len(S_files)/3), file=S_files[math.floor(len(S_files)/3)])
    if case_name == 'Groccia_refined_0p64':
        clim = (0, 100)
        #dd.mesh.points *= 10**-3
        #dd.surf.points *= 10**-3
    elif case_name =='PTSeg043_base_0p64':
        clim = (0, 500)
        #dd.mesh.points *= 10**3
        #dd.surf.points *= 10**3
    elif case_name=='PerturbNewt400':
        clim = (0,50)
    elif case_name=='PerturbNewt500':
        clim = (0,200)
    elif case_name =='Slow_case16_m06_ramp':
        clim = (0,60)
    elif case_name=='JHData':
        clim = (0,40)
    elif case_name == 'Case_028_low' or case_name == 'Case_028_low_orig' or case_name == 'PTSeg028_base_0p64':
        clim = (0,300)    
    elif case_name == 'PTSeg106_base_0p64':
        clim = (0,500) 
        #dd.mesh.points *= 10**3
        #dd.surf.points *= 10**3
        #pl=pv.Plotter(off_screen=True)
        #pl.camera_position = cpos
        #pl.add_mesh(dd.mesh)
        #pl.show(screenshot= S_out_folder +'/'+dd.folder.stem + 'mesh')
    #cpos = [(-29.740970708775073, -56.96481868777567, 13.630348460231204),(15.210040301620888, -21.221124634187493, 4.619872560702723),(0.20284467265235034, -0.00825965400652369, 0.9791760908499827)]
    if size != 1:
        if rank==size-1:
            n = math.floor(len(dd.swirl_files)/(size-1))
            n_end = len(dd.swirl_files)-n*(size-1)
            ind = range(rank*n, rank*n+n_end)
        else:
            n = math.floor(len(dd.swirl_files)/(size-1))
            ind = range(rank*n,rank*n+n)
    else:
        ind=None
    viz_(dd, output_folder=S_out_folder, clim=clim, cpos=cpos, indices=ind) 
   
