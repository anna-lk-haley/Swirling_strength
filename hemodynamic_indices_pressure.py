import sys, os
import h5py, glob
import numpy as np
import gc
import vtk
import pyvista as pv
from numpy.fft import fftfreq, fft
from pathlib import Path
import multiprocessing
from multiprocessing import sharedctypes
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def assemble_mesh(mesh_file):
    """ Create UnstructuredGrid from h5 mesh file. """
    with h5py.File(mesh_file, 'r') as hf:
        coord = np.array(hf['Mesh/Wall/coordinates'])
        elems = np.array(hf['Mesh/Wall/topology'])
        pts = np.array(hf['Mesh/Wall/pointIds'])
        elem_type = np.ones((elems.shape[0], 1), dtype=int) * 3
        elems = np.concatenate([elem_type, elems], axis = 1)
        surf = pv.PolyData(coord,elems.ravel())
        surf.point_data['vtkOriginalPtIds']=pts

    return surf

def create_shared_var(size, dtype=np.float64):
    a = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(a._type_, a,  lock=False)

def get_shared_var(a):
    return sharedctypes.Array(a._type_, a,  lock=False)

def get_array(shared_var):
    return np.ctypeslib.as_array(shared_var)

# read all h5 files
def read_h5_files(ids, wallids, h5_files, _press_):
    press_ = get_array(_press_)
    for i in ids:
        with h5py.File(h5_files[i], 'r') as hw:
            pn = np.array(hw['Solution']['p'])[wallids]
            pn=pn.flatten()
            for j in range(pn.shape[0]):
                press_[j][i] = pn[j]

################################################################
def filter_SPI(U, W_low_cut, tag):
    #for HI
    if tag=="withmean":
        U_fft = fft(U)
    else:
        U_fft = fft(U-np.mean(U))
    # filter any amplitude corresponding frequency equal to 0Hz
    U_fft[W_low_cut[0]] = 0
    # filter any amplitude corresponding frequency lower to 25Hz
    U_fft_25Hz = U_fft.copy()
    U_fft_25Hz[W_low_cut[1]] = 0
    #Compute the absolute value
    Power_25Hz = np.sum ( np.power( np.absolute(U_fft_25Hz),2))
    Power_0Hz  = np.sum ( np.power( np.absolute(U_fft     ),2))
    if Power_0Hz < 1e-5:
        return 0
    return Power_25Hz/Power_0Hz
################################################################
def compute_hemo(ids,_press,_SPI,W_low_cut):
    press_ = get_array(_press)

    SPI = get_array(_SPI)

    print ('    working on', len(ids), 'points:', ids)

    for j in ids:
        SPI[j] = filter_SPI(press_[j],W_low_cut,"withoutmean")

def get_ts(h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])
################################################################
def hemodynamics(surf, input_folder, outfolder, case_name, nproc, period):
    h5_files = sorted(Path(input_folder).glob('*_curcyc_*up.h5'),key=get_ts)
    file_count = len(h5_files)
    if file_count==0:
        print('No files found in {}!'.format(input_folder))
        sys.exit()
    
    wall_ids=surf.point_data['vtkOriginalPtIds']
    number_of_points=len(surf.points)

    print ('   found', file_count, 'pressure files for', number_of_points, 'points.')
    print ('Allocating %5.2f'%(number_of_points*file_count*8/(1024*1024))+'MB of memory ...', end='')

    press_ = create_shared_var([number_of_points, file_count])

    print ('done.', flush=True)

    print ('Reading', len(h5_files), 'files into 1 array of shape', [number_of_points, file_count],' ...')#, end='')

    # make group and divide the procedure
    step = max(int(file_count / nproc), 1)
    rng = list(range(0,file_count))
    groups = [rng[i:i+step] for i  in range(rng[0], rng[-1]+1, step)]
    
    p_list=[]
    for i,g in enumerate(groups):
        p = multiprocessing.Process(target=read_h5_files, name='Process'+str(i), args=(g,wall_ids,h5_files,press_))
        p_list.append(p)
    for p in p_list: p.start()
    # Wait for all the processes to finish
    for p in p_list: p.join()
    gc.collect()

    print (' done.', flush=True)
    dt = period/float(file_count-1)
    W = fftfreq(file_count, d=dt)
    
    W_low_cut = np.where( np.abs(W) == 0 ) + np.where( np.abs(W) < 25.0 )

    print ('period=',period, '  file_count=',file_count, '  dt=', dt, '  frequencies=',W)
    print ('max frequency=',np.max(W))
    print ('min frequency=',np.min(W))

    np.set_printoptions(threshold=sys.maxsize)
    print ( 'w_low_cut:', W_low_cut )

    # now compute desired variables
    print ('Now, computing  SPI')

    SPI = create_shared_var([number_of_points])

    # make group and divide the procedure
    step = max(int(number_of_points / nproc), 1)
    rng = list(range(0,number_of_points))
    groups = [rng[i:i+step] for i  in range(rng[0], rng[-1]+1, step)]

    for i,g in enumerate(groups):
        compute_hemo(g,press_,SPI, W_low_cut)
    gc.collect()

    SPI = get_array(SPI)
    print ('done.', flush=True)
    #print to file
    surf.point_data['SPI_p']=SPI
    surf.save(outfolder+'/{}_SPI_p_25.vtp'.format(case_name))
    print (' done.')


if __name__ == '__main__':
    nargs = len(sys.argv)
    folder = sys.argv[1] #results/art*
    meshfolder = sys.argv[2]
    case_name =  sys.argv[3]
    outfolder = sys.argv[4]
    if not Path(outfolder).exists():
        Path(outfolder).mkdir(parents=True, exist_ok=True)

    mesh_file=list(Path(meshfolder).glob('*.h5'))[0]
    surf = assemble_mesh(mesh_file)

    ncore = 40
    period=0.915
    print ('Performing hemodynamics computation on %d core%s.'%(ncore,'s' if ncore>1 else '') )
    hemodynamics(surf, folder, outfolder, case_name, ncore, period)

