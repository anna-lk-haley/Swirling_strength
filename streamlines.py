import sys
sys.path.insert(1, '/project/s/steinman/ahaleyyy/analysis/')
import os
from pathlib import Path
import numpy as np
import DMD
import pyvista as pv

output_folder='swirl_files/streamlines'
if not Path(output_folder).exists():    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
cases=['106']#['028','043',
for case in cases:
    if case=='028':
        results='../DPQ/case_A/PTSeg028_base_0p64/results/'
        dd = DMD.Dataset(Path((results + os.listdir(results)[0])))
        seg_name = 'PTSeg028'
        main_folder = Path(results).parents[0]
        dd = dd.assemble_mesh()
        dd.surf.points *=10**3
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
        rad = 0.009
        center = np.array([40.4234,-45.9637,14.2515])*10**-3
        idx=1262
        pos=['front','back','top']
        n_lines=1500
    elif case=='043':
        if not Path(output_folder).exists():    
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        results='../DPQ/case_B/PTSeg043_base_0p64/results/'
        dd = DMD.Dataset(Path((results + os.listdir(results)[0])))
        seg_name = 'PTSeg043'
        main_folder = Path(results).parents[0]
        dd = dd.assemble_mesh()
        dd.surf.points *=10**3
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
        rad = 0.009
        center = np.array([66.3222,-14.2323,36.5867])*10**-3
        idx=1333
        pos=['front','back','under']
        n_lines=1250
    elif case=='106':
        results='../DPQ/case_C/PTSeg106_base_0p64/results/'
        dd = DMD.Dataset(Path((results + os.listdir(results)[0])))
        seg_name = 'PTSeg106'
        main_folder = Path(results).parents[0]
        dd = dd.assemble_mesh()
        #dd.mesh.points *=10**3 #get back into large size
        dd.surf.points *=10**3
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
        rad = 0.009
        center = np.array([40.529,-35.4031,-586.856])*10**-3
        idx=1444
        pos=['side','back','front']
        n_lines=1250
    cposes=[cpos0,cpos1,cpos2]
    silhouette = dict(color='black', line_width=1.0, decimate=None)

    dd.mesh.point_data['v']=dd(idx) #40% through the cycle
    dd.mesh.set_active_scalars("v")
    streamlines = dd.mesh.streamlines(
        integrator_type=45,
        terminal_speed=0.001,
        n_points=n_lines,
        source_radius=rad,
        source_center=center,
    )
    streamlines.points *=10**3
    
    for i, cposi in enumerate(cposes):
        p=pv.Plotter(off_screen=True, window_size=[768, 768], image_scale=4)
        if pos[i]=='front' and 'PTSeg106' not in case_name:
            show_scalar_bar=True
            scalar_bar_args=None
        elif pos[i]=='side' and 'PTSeg106' in case_name:
            show_scalar_bar=True
            scalar_bar_args=dict(position_x=0.05, position_y=0.05)
        else:
            show_scalar_bar=False
            scalar_bar_args=None
        p.add_mesh(dd.surf,
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

        p.add_mesh(streamlines.tube(radius=0.008),cmap='plasma', show_scalar_bar=show_scalar_bar,scalar_bar_args=scalar_bar_args)
        p.camera_position = cposi
        p.show(screenshot=output_folder + '/{}_streamlines_{}_{}.png'.format(seg_name,idx,pos[i]), auto_close=False)
        p.close()