#Script to compute local swirling strength written by ALKH

import numpy as np
import math
import sys
sys.path.insert(1, '/project/s/steinman/ahaleyyy/analysis/')
import DMD
import os
import psutil
from pathlib import Path
import h5py
import pyvista as pv
import gc
import resource
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank_ = comm.Get_rank()
size_tot = comm.Get_size()

def swirling_strength(dV):
    """
    input: velocity gradient tensor at a point
    """
    dV = dV.reshape((3,3))
    vals, _ = np.linalg.eig(dV)
    lam_ci = 0
    if np.sum(np.iscomplex(vals))==2: #exactly two complex values
        for _, val in enumerate(vals):
            lam_ci = abs(val.imag)
            if lam_ci > 0:  
                lam_cr = val.real
                return lam_ci, lam_cr
            else:
                return 0, 0
    else:
        return 0, 0

def derivs(dd, idx):
    dd.mesh.point_data['u'] = dd(idx)
    deriv = dd.mesh.compute_derivative(scalars = 'u', gradient=True) #use this for now since it is faster, but l8r use ZZ
    return deriv['gradient']

def Compute_Swirl(dd, rank, size, idx):
    if rank_ == rank*4:
        dv = derivs(dd,idx)
        #divide domain of dvs into four pieces per super rank and send to sub ranks (4)
        N = len(dv)
        n = math.floor(N/3)
        last_n = N-n*3
        list_n = [n,n,last_n]
        ranges = [range(n),range(n,2*n), range(2*n, 3*n), range(3*n, 3*n+last_n)]
        for i, r in enumerate(range(rank*4+1,rank*4+4)):
            comm.send(list_n[i], r, tag=0) #send the number of dv entries to expect
            comm.Send(dv[ranges[i+1],:], r, tag=1) #send the range of dvs to look at
        dV=dv[ranges[0],:] #for the first subrank of the super rank
    elif rank_ in range(rank*4+1,(rank+1)*4):
        n = comm.recv(source=rank*4, tag=0)
        dV = np.empty((n,9), dtype=np.float64)
        comm.Recv(dV, source=rank*4, tag=1) #for all other sub ranks

    S=np.zeros((n,1))
    R=np.zeros((n,1))
    for ndx in range(n):
        S[ndx], R[ndx] = swirling_strength(dV[ndx]) #get the swirling strength and real part for this index
    if rank_== rank*4:
        #make an empty array
        S_tot = np.zeros((N,1))
        R_tot = np.zeros((N,1))
        #fill up the S_tot array
        for i, r in enumerate(range(rank*4+1,rank*4+4)):
            S_recv= np.empty((list_n[i],1), dtype = np.float64)
            comm.Recv(S_recv, source=r, tag=2)
            R_recv= np.empty((list_n[i],1), dtype = np.float64)
            comm.Recv(R_recv, source=r, tag=3)
            S_tot[ranges[i+1]] = S_recv
            R_tot[ranges[i+1]] = R_recv
        S_tot[ranges[0]]=S
        R_tot[ranges[0]]=R
        return S_tot, R_tot
    elif rank_ in range(rank*4+1,(rank+1)*4):
        comm.Send(S, dest=rank*4, tag=2)
        comm.Send(R, dest=rank*4, tag=3)
        return 0, 0

if __name__=="__main__":
    folder = sys.argv[1]
    if len(sys.argv)>2:
        if sys.argv[2]=='mono':
            mono=True
            random=False
        else:
            mono=False
            random=True
    else:
        mono=False
        random=False

    if mono:
        case_names=['PerturbNewt400','PerturbNewt500']
    elif random:
        case_names=['JHData']
    else:
        case_names = ['PTSeg106_base_0p64','PTSeg028_base_0p64']#[ name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) and '106' in name]#['Slow_case16_m06_ramp']#
    for case_name in case_names: #[s for s in case_names if "high" not in s]:#
        print(case_name)
        gc.collect()
        results = folder+'/'+ case_name + '/results/'
        if mono:
            results_folder=Path(results)
            fs = 5
        elif random:
            results_folder=Path(results)
            fs = 1
        else:
            results_folder = Path((results + os.listdir(results)[0])) #results folder eg. results/art_
            if rank_ == 0:
                print(results + os.listdir(results)[0])
            fs = round(len(list(results_folder.glob('*_curcyc_*')))/3000)
        #print(fs)
        dd = DMD.Dataset(results_folder, file_stride=fs, mono=mono,random=random)
        #if rank_==0:
        #    print('{} GiB of Memory used.'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3), flush=True)
        outfolder='swirl_files/{}'.format(case_name.split('_')[0])
        print(outfolder)
        if not Path(outfolder).exists():
            Path(outfolder).mkdir(parents=True, exist_ok=True)
        if rank_ == 0:
            print('Beginning computations!', flush=True)
        total_tsteps=dd.tsteps
        size = size_tot/4 #bundles of 4 procs per tstep bunch (40/4=10)
        pieces = math.floor(total_tsteps/size) #dividing by 10
        if pieces*size != total_tsteps:
            rem = total_tsteps-pieces*size
            last_piece = int(pieces + rem)
        else:
            last_piece = pieces
        #1. Divide up timesteps between super ranks (10)
        rank = math.floor(rank_/4) # 0 1 2 3 4 5 6 7 8 9(last one is 39/4-->9)

        if rank < size-1:
            tsteps = range(rank*pieces,(rank+1)*pieces)
        else:
            tsteps = range(rank*pieces,rank*pieces+last_piece)

        if rank_ == rank*4:
            dd = dd.assemble_mesh()
            #print('{} GiB of Memory used.'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3), flush=True)

        for idx in tsteps:
            if mono or random:
                file = None
            else:
                file=dd.up_files[idx]
            if not Path(outfolder +'/Swirl_{}.h5'.format(dd._get_ts(file, idx))).exists():
                S, R = Compute_Swirl(dd, rank, size, idx)
                #print(S)
                if rank_ == rank*4:
                    with h5py.File(outfolder +'/Swirl_{}.h5'.format(dd._get_ts(file,idx)), 'w') as f:
                        f.create_dataset(name='S', data=S)
                        f.create_dataset(name='R', data=R)
                if (idx%100==0) and (rank_ == 0):
                    print('{}% complete'.format(round(100*idx/len(tsteps))), flush=True)
        print('Super-rank {} finished case!'.format(rank))
        comm.Barrier()