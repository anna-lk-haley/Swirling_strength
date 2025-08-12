""" Generate spectrograms using BSL tools.
"""

import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree as KDTree 
from scipy import signal

import pyvista as pv
from bsl.dataset import Dataset
from bsl import spectral 
import numpy as np 
import h5py

size = 12
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

def assemble_matrix(dd, quantity='p', pt_id=None, mask=None):
    """ Create a N * ts matrix of scalar u_mag or p data.

    Args:
        array (array or None): Supplying array will overwrite array.
        ind (array or None): Indicies for a subset.

    Used for spectrograms.
    """
    array = np.zeros((dd.mesh.n_points, len(dd.up_files)), dtype=np.float64)
    # Get indices of mask
    if mask is None:
        ind=dd.surf.point_arrays['vtkOriginalPtIds'][pt_id]
    else:
        ind = dd.surf.point_arrays['vtkOriginalPtIds'][dd.mesh.point_arrays[mask]==1]
    array = array[ind]
    key = quantity        

    for idx in range(len(dd.up_files)):
        if idx % 100 == 0:
            print(idx, '/', len(dd.up_files))

        arr = dd(idx, array=key)[ind]
        if mask is None:
            array[idx] = arr
        else:
            array[:,idx] = arr.reshape((-1,))
    return array

def spectrogram(dd, array_key, n_fft=None):
    """ Compute spectrogram from an array, usually u_mag.

    Args:
        array (array or None): Array containing u magnitude or QoI.
        indices (list or None): Location indices of, for example, sac.
        spec_file (path or None): Where to save spectrogram data.
        spec_img_file (path or None): Path to save spectrogram img.
    """
    array = dd.arrays[array_key]

    period = dd.get_period()

    n_samples = array.shape[0]
    sr = array.shape[0] / period 

    if n_fft is None:
        n_fft = spectral.shift_bit_length(int(n_samples / 10))
    
    spec_args = {}
    spec_args['sr'] = sr
    spec_args['n_fft'] = n_fft
    spec_args['hop_length'] = int(0.25*n_fft)
    spec_args['win_length'] = n_fft
    spec_args['detrend'] = 'linear'
    spec_args['pad_mode'] = 'cycle'

    S, bins, freqs = average_spectrogram(
        data=array, 
        **spec_args
        )
        
    # Remove last frame
    S = S[:,:-1]
    bins = bins[:-1]

    spec_data = {}
    spec_data['S'] = S
    spec_data['bins'] = bins
    spec_data['freqs'] = freqs
    spec_data['sr'] = sr 
    spec_data['n_fft'] = n_fft

    dd.spectrogram_data[array_key] = spec_data

def average_spectrogram(data, sr, n_fft=None, hop_length=None, win_length=None, 
                        window='hann', pad_mode='cycle', detrend='linear', print_progress=False):
    """ Compute the average spectrogram of a dataset.
    
    Args: 
        data (array): N * ts array.
        For other args, see librosa.stft docs.
    
    Returns:
        array: Average spectrogram of data.

    """
    pad_size = win_length // 2
    front_pad = data[-pad_size:]
    back_pad = data[:pad_size]
    data = np.concatenate([front_pad, data, back_pad])
    boundary = None 
    
    n_samples = data.shape[0]
    
    if n_fft is None:
        n_fft = spectral.shift_bit_length(int(n_samples / 10))
    
    if hop_length is None:
        hop_length = int(n_fft / 4)
    
    if win_length is None:
        win_length = n_fft
      
    stft_params = {
        'fs' : sr,
        'window' : window,
        'nperseg' : win_length,
        'noverlap' : win_length - hop_length,
        'nfft' : n_fft,
        'detrend' : detrend,
        'return_onesided' : True,
        'boundary' : boundary,
        'padded' : True,
        'axis' : -1,
        }

    freqs, bins, S_ = signal.stft(x=data, **stft_params)
    S = np.log(np.abs(S_)**2)

    if pad_mode in ['cycle', 'even', 'odd']:
        bins = bins - bins[0]

    return S, bins, freqs

if __name__ == "__main__":
    folder = Path(sys.argv[1]) #results folder eg. results/art_
    case_name = sys.argv[2] #eg. PTSeg028_low
    spec_data_out_folder = Path(sys.argv[3]) #eg. spec_data
    spec_img_out_folder = Path(sys.argv[4]) #eg spec_imgs
    stride = int(sys.argv[5])
    dd = Dataset(folder, file_stride=stride, mesh_glob_key='*.h5')#, case_name = case_name)
    print(dd.mesh_file)
    with h5py.File(dd.mesh_file, 'r') as hf:
        coord = np.array(hf['Mesh/Wall/coordinates'])
        elems = np.array(hf['Mesh/Wall/topology'])
        pts = np.array(hf['Mesh/Wall/pointIds'])
        elem_type = np.ones((elems.shape[0], 1), dtype=int) * 3
        elems = np.concatenate([elem_type, elems], axis = 1)
        dd.surf = pv.PolyData(coord,elems.ravel())
        dd.surf.point_arrays['vtkOriginalPtIds']=pts

    if not Path(spec_data_out_folder / (case_name)).exists():
        Path(spec_data_out_folder / (case_name)).mkdir(parents=True, exist_ok=True)
    #single point:
    pt_id=11762
    radius = 0.15
    sphere = pv.Sphere(radius = radius, center = dd.surf.points[pt_id])
    spec_out_file = spec_data_out_folder / (case_name+'/' + case_name + 'pt_id_11762.npz') #eg. spec_data/PTSeg028/PTSeg028_spectrosphere0.npz
    if spec_out_file.exists():
        spec_data = np.load(spec_out_file)
        bins = spec_data['bins']
        freqs = spec_data['freqs']
        S = spec_data['S']
        S[S < -20] = -20 
    else:
        #just for the point
        dd.mesh=dd.surf
        dd.arrays['p']=assemble_matrix(dd, quantity='p', pt_id=pt_id)
        # Spectrograms
        spectrogram(dd, array_key='p')
        #for a sphere around the point
        #mesh_sel=dd.surf.select_enclosed_points(sphere, tolerance=0.01)
        #dd.mesh=mesh_sel
        #dd.arrays['p']=assemble_matrix(dd, quantity='p', mask='SelectedPoints')
        # Spectrograms
        #dd.spectrogram(array_key='p')
        spec_data = dd.spectrogram_data['p']
        np.savez(spec_out_file, **dd.spectrogram_data['p'])
        spec_data = dd.spectrogram_data['p']
        np.savez(spec_out_file, **dd.spectrogram_data['p'])

        bins = spec_data['bins']
        freqs = spec_data['freqs']
        S = spec_data['S']

        S[S < -20] = -20 
    '''

    thedir=folder/('../../..')
    surf_folder = os.path.join(thedir,[name for name in os.listdir(thedir) if 'surf_spectrospheres' in name and os.path.isdir(os.path.join(thedir, name))][0]) #eg. Refinement/PTSeg028_cl_mapped_spectrospheres
    print(surf_folder)
    for sphere_file in os.listdir(surf_folder):
        spec_out_file = spec_data_out_folder / (case_name+'/' + case_name + sphere_file.split('.')[0].split('_')[-1]+ '.npz') #eg. spec_data/PTSeg028/PTSeg028_spectrosphere0.npz
        if spec_out_file.exists():
            spec_data = np.load(spec_out_file)
            bins = spec_data['bins']
            freqs = spec_data['freqs']
            S = spec_data['S']
            S[S < -20] = -20 
        else:
            surf_file = surf_folder + '/' + sphere_file
            surf = pv.read(surf_file)
            mesh_sel=dd.surf.select_enclosed_points(surf, tolerance=0.01)
            dd.mesh=mesh_sel
            dd.arrays['p']=assemble_matrix(dd, quantity='p', mask='SelectedPoints')
            # Spectrograms
            dd.spectrogram(array_key='p')
            spec_data = dd.spectrogram_data['p']
            np.savez(spec_out_file, **dd.spectrogram_data['p'])

            bins = spec_data['bins']
            freqs = spec_data['freqs']
            S = spec_data['S']

            S[S < -20] = -20 
        
                # Plotting
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        ax.pcolormesh(bins, freqs, S, shading='gouraud')
        ax.set_xlabel('Time (s)', labelpad=-5)
        ax.set_ylabel('Freq (Hz)', labelpad=-10)
        ax.set_xticks([0, 0.9])
        ax.set_xticklabels(['0.0', '0.9'])
        ax.set_yticks([0, 600,800])
        ax.set_yticklabels(['0', '600', '800'])
        ax.set_ylim([0, 800])

        title = case_name + sphere_file.split('.')[0].split('_')[-1]

        ax.set_title(title)
        plt.tight_layout
        plt.savefig(spec_img_out_folder / (title + '.png'))#, transparent=True)
        '''
    # Plotting
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    ax.pcolormesh(bins, freqs, S, shading='gouraud')
    ax.set_xlabel('Time (s)', labelpad=0)
    ax.set_ylabel('Freq (Hz)', labelpad=-10)
    period=dd.get_period()
    ax.set_xticks([])
    #ax.set_xticklabels(['0.0', '{}'.format(period)])
    ax.set_yticks([0, 600,800])
    ax.set_yticklabels(['0', '600', '800'])
    ax.set_ylim([0, 800])

    title = case_name + 'pt_id_11762'

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(spec_img_out_folder / (title + '.png'))#, transparent=True)

