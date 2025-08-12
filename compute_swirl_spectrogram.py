""" Generate spectrograms using BSL tools.
"""

import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
import h5py 
import numpy as np
import pyvista as pv 
import vtk 
import sys 
import re
import gc
from bsl import spectral 
from scipy.spatial import cKDTree as KDTree 

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, meshfolder=None, file_stride=1, mesh_glob_key=None):

        self.folder = Path(folder)
        
        if mesh_glob_key is None:
            mesh_glob_key = '*h5'

        keyl = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]
        self.swirl_files = sorted(Path(folder).glob('Swirl_*.h5'), key=keyl)
        #print(self.swirl_files)  

        self.mesh_file=list(Path(meshfolder).glob('*.h5'))[0]

        self.arrays = {}
        self.spectrogram_data = {}

    def __call__(self, idx, array='u_pAvg'):
        h5_file = self.swirl_files[idx]    
        with h5py.File(h5_file, 'r') as hf:
            val = np.array(hf[array])
        return val
        
    def assemble_mesh(self):
        """ Create UnstructuredGrid from h5 mesh file. """
        
        with h5py.File(self.mesh_file, 'r') as hf:
            points = np.array(hf['Mesh']['coordinates'])
            cells = np.array(hf['Mesh']['topology'])

            celltypes = np.empty(cells.shape[0], dtype=np.uint8)
            celltypes[:] = vtk.VTK_TETRA

            cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
            cells = np.concatenate([cell_type, cells], axis = 1)
            self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
            self.surf = self.mesh.extract_surface()
    
        # self.assemble_surface()
        return self

    def assemble_matrix(self, array_key='u_p', quantity='u_p', array=None, mask=None):
        """ Create a N * ts matrix of scalar u_mag or p data.

        Args:
            array (array or None): Supplying array will overwrite array.
            ind (array or None): Indicies for a subset.

        Used for spectrograms.
        """
        self.arrays[array_key] = np.zeros((self.mesh.n_points, len(self.swirl_files)), dtype=np.float64)
        # Get indices of mask
        ind = np.where(self.mesh.point_arrays[mask])
        self.arrays[array_key] = self.arrays[array_key][ind]
        key = quantity        

        for idx in range(len(self.swirl_files)):
            if idx % 100 == 0:
                print(idx, '/', len(self.swirl_files))

            arr = self(idx, array=key)[ind]
            self.arrays[array_key][:,idx] = arr.reshape((-1,))
        
        return self
    
    def spectrogram(self, array_key, n_fft=None, period=0.915):
        """ Compute spectrogram from an array, usually u_mag.

        Args:
            array (array or None): Array containing u magnitude or QoI.
            indices (list or None): Location indices of, for example, sac.
            spec_file (path or None): Where to save spectrogram data.
            spec_img_file (path or None): Path to save spectrogram img.
        """
        array = self.arrays[array_key]
        n_samples = array.shape[1]
        sr = array.shape[1] / period 

        if n_fft is None:
            n_fft = spectral.shift_bit_length(int(n_samples / 10))
        
        spec_args = {}
        spec_args['sr'] = sr
        spec_args['n_fft'] = n_fft
        spec_args['hop_length'] = int(0.25*n_fft)
        spec_args['win_length'] = n_fft
        spec_args['detrend'] = 'linear'
        spec_args['pad_mode'] = 'cycle'

        S, bins, freqs = spectral.average_spectrogram(
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

        self.spectrogram_data[array_key] = spec_data

        return self

size = 12
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

if __name__ == "__main__":
    figs_data_out_folder = Path(sys.argv[1]) #eg. spec_data
    figs_img_out_folder = Path(sys.argv[2]) #eg spec_imgs

    figs_data_out_folder.mkdir(parents=True, exist_ok=True)
    figs_img_out_folder.mkdir(parents=True, exist_ok=True)

    case_names=['PerturbNewt400','PerturbNewt500', 'Groccia_refined_0p64', 'PTSeg043_base_0p43']
    #folder = sys.argv[1] #stats
    #case_name = sys.argv[2] #eg. c10
    #spec_data_out_folder = Path(sys.argv[3]) #eg. spec_data
    #spec_img_out_folder = Path(sys.argv[4]) #eg spec_imgs
    #stride = int(sys.argv[5])
    #meshfolder = '../../../../mesh_rez/cases/case_A/case_028_low/results/art_PTSeg028_low_I1_FC_VENOUS_Q557_Per915_Newt370_ts15660_cy2_uO1/'
    for case_name in case_names: #[s for s in case_names if "high" not in s]:#
        gc.collect()
        folder=case_name
        project=os.environ["PROJECT"]
        meshfolder=project+'/Swirl/swirl_cases/'+case_name +'/data'
        dd = Dataset(folder, meshfolder, mesh_glob_key='*.h5')#, case_name = case_name)
        dd = dd.assemble_mesh() #gets mesh info from results/art_/PT_Seg028_low.h5

        spec_out_file = figs_img_out_folder / ('{}.npz'.format(case_name))
        if spec_out_file.exists():
            spec_data = np.load(spec_out_file)
            bins = spec_data['bins']
            freqs = spec_data['freqs']
            S = spec_data['S']
            S[S < -20] = -20   
        else:
            surf_file=meshfolder+'/../spectro_sigmoid.vtp'
            surf = pv.read(surf_file)
            bounds=surf.bounds

            def generate_points(bounds, subdivisions=50):
                x_points=np.linspace(bounds[0], bounds[1],num=subdivisions)
                y_points=np.linspace(bounds[2], bounds[3],num=subdivisions)
                z_points=np.linspace(bounds[4], bounds[5],num=subdivisions)
                points = np.array([[x_points[0], y_points[0], z_points[0]],[x_points[0], y_points[0], z_points[1]]])
                for i in range(subdivisions):
                    for j in range(subdivisions):
                        for k in range(subdivisions):
                            points=np.append(points,[[x_points[i], y_points[j], z_points[k]]], axis=0)
                return points[2:,:]
                
            points = generate_points(bounds,subdivisions=50)
            point_cloud=pv.PolyData(points)
            surf_sel = point_cloud.select_enclosed_points(surf, tolerance=0.01)
            tree = KDTree(dd.mesh.points)
            _, idx = tree.query(surf_sel.points[surf_sel.point_arrays['SelectedPoints']==1]) #find closest node to the points in the equispaced points        

            dd.mesh.point_arrays['EquispacedPoints']=np.zeros(dd.mesh.n_points)
            dd.mesh.point_arrays['EquispacedPoints'][idx]=1
            dd.assemble_matrix(array_key='swirl', quantity='S', mask='EquispacedPoints')
            # Spectrograms
            dd.spectrogram(array_key='swirl')
            spec_data = dd.spectrogram_data['swirl']
            np.savez(spec_out_file, **dd.spectrogram_data['swirl'])

            bins = spec_data['bins']
            freqs = spec_data['freqs']
            S = spec_data['S']
            S[S < -20] = -20

            fig, ax = plt.subplots(1,1, figsize=(4,4))
            ax.pcolormesh(bins, freqs, S, shading='gouraud')
            ax.set_xlabel('Time (s)')#, labelpad=-5)
            ax.set_ylabel('Freq (Hz)', labelpad=-10)
            ax.set_xticks([])#([0, 0.9])
            ax.set_xticklabels([])#(['0.0', '0.9'])
            ax.set_yticks([0, 600,800])
            ax.set_yticklabels(['0', '600', '800'])
            ax.set_xlim([0.04, 0.88])
            ax.set_ylim([0, 800])

            title = case_name
            #ax.set_title(title)
            plt.tight_layout
            plt.savefig(figs_data_out_folder / (title + '.png'))#, transparent=True)
