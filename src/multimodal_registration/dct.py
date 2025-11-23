import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import shift

class DCT:
    
    def __init__(self, dct_ds_path:str, ref_pct_path:str):
        with h5py.File(dct_ds_path, 'r') as hin:
            keys = list(hin['DS'].keys())
#             print(self.keys)
            for key in keys:
                entry = f'DS/{key}'
                setattr(self, key, hin[entry][:])
        self.shape = self.GIDvol.shape
        self.voxel_size = self.VoxSize[0, 0]
        
    def plot_ipf(self, plane:str = 'xz', slice_n = None):
        
        plane = plane.lower()
        plt.figure()
        if slice_n == None:
            if plane == 'xy' or plane == 'yx':
                slice_n = self.shape[2]//2
                plt.imshow(self.IPF001[:, :, slice_n])
            elif plane == 'yz' or plane=='zy':
                slice_n = self.shape[1]//2
                plt.imshow(self.IPF001[:, slice_n, :])
            elif plane == 'xz' or plane == 'zx':
                slice_n = self.shape[0]//2
                plt.imshow(self.IPF001[slice_n, :, :])
            else:
                print('I do not know how to proceed')
        else:
            if plane == 'xy' or plane == 'yx':
                plt.imshow(self.IPF001[:, :, slice_n])
            elif plane == 'yz' or plane == 'zy':
                plt.imshow(self.IPF001[:, slice_n, :])
            elif plane == 'xz' or plane == 'zx':
                plt.imshow(self.IPF001[slice_n, :, :])
            else:
                print('I do not know how to proceed')
        plt.show()
        
    def upscale(self, factor:float):

        pass

    def shift(self, shifts:tuple[float]):

        pass

    def pad(self, pad_width:tuple[tuple[float]]):
        
        pass

    def _calculate_pad_width(self, ):
        pass
