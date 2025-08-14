""" Generate visualizations using BSL tools.
"""

import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree as KDTree 
import pyvista as pv
sys.path.insert(1, '/project/s/steinman/ahaleyyy/Swirl/scripts/')
from make_swirl_figs import Dataset
import numpy as np 

def viz_(dd, output_folder, clim=None, cpos=None, window_size=[768, 768], ind=None,pos=None):
    """ Visualize a contour of a field variable.
    """
    output_folder = Path(output_folder)
    output_folder = output_folder / dd.folder.stem

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    idx=ind
    silhouette = dict(color='black', line_width=1.0, decimate=None)
        
    ct = 0
    mesh=dd.mesh.copy()
    surf=dd.surf.copy()
    #surf.save(output_folder / (dd.folder.stem + 'surf.vtp'))
    grid = pv.ImageData()
    if dd.folder.stem == 'PTSeg043' or dd.folder.stem == 'PTSeg028' or dd.folder.stem == 'PTSeg106':
        bounds=np.array(surf.bounds)
        bounds_diff=np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        grid.origin = (surf.center[0]-0.5*bounds_diff[0],surf.center[1]-0.5*bounds_diff[1], surf.center[2]-0.5*bounds_diff[2])
        grid.dimensions=(int(600*bounds_diff[0] / bounds_diff[1]), int(600*bounds_diff[1] / bounds_diff[1]), int(600*bounds_diff[2]/bounds_diff[1]))
        grid.spacing=(bounds_diff[0] / grid.dimensions[0], bounds_diff[1] / grid.dimensions[1], bounds_diff[2] / grid.dimensions[2])
    newmesh=grid.copy()

    mesh.point_data['S'] = dd(idx)
    newmesh = grid.sample(mesh)

    for i,cpos in enumerate(cposi):
        print(pos[i])
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.image_scale = 2

        if pos[i]=='front' and 'PTSeg106' not in case_name:
            show_scalar_bar=True
            scalar_bar_args=None
        elif pos[i]=='side' and 'PTSeg106' in case_name:
            show_scalar_bar=True
            scalar_bar_args=dict(position_x=0.05, position_y=0.05)
        else:
            show_scalar_bar=False
            scalar_bar_args=None

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

        p.add_volume(newmesh,
            cmap='cool',#'jet_r',
            opacity='linear',
            scalars='S',
            clim=clim,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args=scalar_bar_args
            )
        p.camera_position=cpos
        p.show(auto_close=False)
        p.screenshot(filename=output_folder / (dd.folder.stem + '_volume_{}_{}.png'.format(idx,pos[i])),transparent_background=True, return_img=False)
        p.close()

if __name__ == "__main__":
    folder = sys.argv[1] #results folder eg. results/art_
    case_name = sys.argv[2] #eg. case_028_low
    S_folder = sys.argv[3] #'/scratch/s/steinman/ahaleyyy/triple_decomp/swirl_files/case_A/case_028_low'
    S_out_folder = sys.argv[4] #eg spec_imgs
    
    dd = Dataset(S_folder, folder)#, case_name = case_name)
    dd = dd.assemble_mesh() #gets mesh info from results/art_/PT_Seg028_low.h5
    ind = int(sys.argv[5])

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
        clim = (0, 300)
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
        clim = (0, 300)
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
        clim = (0, 500)
    cposi=[cpos0,cpos1,cpos2]
    print(case_name)
    viz_(dd, output_folder=S_out_folder, clim=clim, cpos=cposi, ind=ind,pos=pos) 
   
