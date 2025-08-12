""" Generate visualizations using BSL tools.
"""

import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.spatial import cKDTree as KDTree 
import imageio
import pyvista as pv
from make_swirl_figs import Dataset
import numpy as np 
import vtk
import itertools

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

def viz_contour(dd, array_name, contour_val, output_folder, indices=None,
                    parallel_scale=0.2, color='r', cpos=None, scalars=None, clip_surf=None,
                    window_size=[768, 768], show_scalar_bar=False,scalar_bar_args=None, case_name=None,pos=None):
    """ Visualize a contour of a field variable.
    """
    output_folder = Path(output_folder)
    output_folder = output_folder / dd.folder.stem

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = range(len(dd.up_files))

    p = pv.Plotter(off_screen=True, window_size=window_size, image_scale=4)

    silhouette = dict(color='black', line_width=1.0, decimate=None)
    silhouette_c = dict(color='black', line_width=0.1, decimate=None)

    array_dict = {}
    array_dict[array_name] = array_name
        
    if clip_surf is not None:
        surf = dd.surf.clip_surface(clip_surf, invert=True).extract_largest()
    else:
        surf = dd.surf

    p.add_mesh(surf,
                color='w',
                opacity=1,
                show_scalar_bar=False,
                lighting=True,
                smooth_shading=True,
                specular=0.00,
                diffuse=0.9,
                ambient=0.5,
                culling='back',
                silhouette=silhouette,
                name='surf',
                )
    indices=indices[::15]
    cmap = mpl.colormaps['spring']
    colors = cmap(np.linspace(0, 1, len(indices)))
    #print(colors)
    color=itertools.cycle(colors)
    #opacities = np.linspace(1, 0.2, len(indices))

    for i, idx in enumerate(indices):
        if not Path(output_folder / (case_name + '_contours.png')).exists():
            dd.mesh.point_data[array_name]=dd(idx, array=array_name)
            contour = dd.mesh.contour([contour_val], scalars=array_name)

            print('n_points in contour:', contour.n_points)
            
            if contour.n_points > 0:
                contour = vtk_taubin_smooth(contour)

                clim = None
                c=next(color)
                c=c[:-1]
                #print(c)
                p.add_mesh(contour,
                    color=c,
                    #opacity=opacities[i],
                    smooth_shading=True,
                    show_scalar_bar=show_scalar_bar,
                    scalar_bar_args=scalar_bar_args,
                    silhouette=silhouette_c ,
                    specular=0.0,
                    diffuse=1.0,
                    ambient=1.0,
                    )
    for i,cpos in enumerate(cposi):
        p.camera_position = cpos
        p.show(auto_close=False)
        p.screenshot(filename=output_folder / (case_name + '_contours_{}.png'.format(pos[i])))#,transparent_background=True, return_img=False)
    p.close()

if __name__ == "__main__":
    folder = Path(sys.argv[1]) #swirl files folder
    meshfolder = sys.argv[2]
    case_name = sys.argv[3] #eg. PTSeg028
    out_folder = Path(sys.argv[4])
    
    dd = Dataset(folder,meshfolder)#, case_name = case_name)
    dd = dd.assemble_mesh() #gets mesh info from results/art_/PT_Seg028_low.h5
    #cps = sys.argv[5]
    #cpos_str = cps.split(',')[:]
    #cpos_flt = [float(x) for x in cpos_str]
    #it = iter(cpos_flt)
    #cpos=list(zip(it,it, it))
    ind=range(int(sys.argv[6]),int(sys.argv[7]))
    if 'PTSeg028' in case_name:
        #front
        cpos0 = [(-29.740970708775073,-56.96481868777567,13.630348460231204),
                (15.210040301620888,-21.221124634187493,4.619872560702723),
                (0.20284467265235034,-0.00825965400652369,0.9791760908499827)]
        #back
        cpos1=[(109.90422854692072, 15.97590922505639, 3.0723431319907357),
            (14.793379089877371, -23.499034576897408, 3.492523241021754),
            (-0.1937458008426502, 0.4759421035012653, 0.8578704323909472)]
        #top
        cpos2=[(20.68000207941282, 17.5017969541395, 44.96099912780623),
            (41.79731155816988, -20.62737943029021, -10.241470531439822),
            (-0.6665228620392106, -0.7078289928809927, 0.233934587472949)]
        pos=['front','back','top']
    elif 'PTSeg043' in case_name:
        #front
        cpos0=[(-29.81932643611117,-42.47205087731038,50.331720431206506),
            (44.71233178141023,-17.22396712584402,40.18388401050693),
            (-0.0425387466145877,0.47815894682806126,0.8772425414927958)]
        #back/top
        cpos1=[(72.19571358481821, 1.3149928467741194, 56.67106308781252),
            (46.1343248324221, -12.28444364911261, 22.882809121854592),
            (-0.5599949678557962, -0.5230985761980056, 0.6424745252193313)]
        #under    
        cpos2=[(69.84268111114989, -6.565946935406767, -0.013742196961246123),
            (56.70333398876734, -9.696924055383855, 15.925899556674484),
            (-0.4593457551548993, -0.7200751790507487, -0.520089620869526)]
        pos=['front','back','under']
    elif 'PTSeg106' in case_name:
        #side
        cpos0 = [(31.88855204906848,78.92607275293116,-585.0249383973434),
                (30.604051091312275,-31.804910265644445,-603.1068411131863),
                (-0.20271561231268886,-0.15552420547197116,0.9668084619183904)]
        #back/under
        cpos1=[(82.65603614980529, -9.865184419475417, -606.8994834930071),
                (23.153458509129976, -16.49175771330144, -586.2335773751289),
                (0.290426265173412, 0.2621753616716892, 0.9202807529388748)]
        #front
        cpos2=[(-3.6765628439737945, -19.5640847466169, -589.2626468907002),
            (20.616896108605463, -18.870755759919852, -591.6371321332596),
            (0.09241482845044043, 0.13873207585231767, 0.986008575323857)]
        pos=['side','back','front']
    cposi=[cpos0,cpos1,cpos2]
    viz_contour(dd,'S', contour_val=float(sys.argv[8]), output_folder=out_folder, cpos=cposi, indices=ind, scalars=None, clip_surf=None, show_scalar_bar=False,scalar_bar_args=None, case_name=case_name,pos=pos)
    
   
