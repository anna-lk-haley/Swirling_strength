import sys
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = '/scratch/s/steinman/ahaleyyy/.config/mpl'
import matplotlib.pyplot as plt 
from scipy.spatial import cKDTree as KDTree 

import pyvista as pv
from bsl.dataset import Dataset
import numpy as np 

size = 12
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend

if __name__ == "__main__":
    case_name = sys.argv[1] #eg. PTSeg028_low
    PSD_data_file = sys.argv[2]
    PSD_img_out_folder = sys.argv[3] #eg spec_imgs

    #do entire one first
    entire_out_file = Path(PSD_data_file)
    if entire_out_file.exists():
        fig_data = np.load(PSD_data_file)
        SS_bins = fig_data['SS_bins']
        time_bins = fig_data['t_bins']
        n_points = fig_data['n_points'] 

    # Plotting
    fig, ax = plt.subplots(1,1, figsize=(6,4.5))
    ax.loglog(SS_bins[:-1],n_points[int(time_bins.size*0.35),:]) #1/3 of the way through the plot
    ax.set_xlabel('Swirling Strength (Hz)')
    ax.set_ylabel('# points')
    #ax.set_xticks([0,100,200,300])

    title = case_name + '_PSD'

    #ax.set_title(title)
    #plt.tight_layout
    plt.savefig(PSD_img_out_folder +'/'+ title + '.png')#, transparent=True)
