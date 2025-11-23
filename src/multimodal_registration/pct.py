import os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import skimage as sk

from .utils import get_extension
class Reference_PCT:

    def __init__(self, pct_ref_path:str, h5_key:str|None = None):
        
        self.path = pct_ref_path
        self.im_format = get_extension(self.path)
        self.h5_key = h5_key

        if self.im_format in ['h5', 'hdf5'] and self.h5_key==None:
            print('Please provide an h5 key to load your volume from h5/hdf5!')
            sys.exit(1)
        self.vol = self._load_volume()

        self.shape = self.vol.shape

    def _load_volume(self):
        if self.im_format == 'tiff' or self.im_format == 'tif':
            try:
                vol = sk.io.imread(self.path)
                return vol
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
            
        elif self.im_format =='h5' or self.im_format == 'hdf5':
            try:
                with h5py.File(self.path, 'r') as hin:
                    vol = hin[self.key][:]
                return vol
            except FileNotFoundError as e:
                print(e)
                sys.exit(1)
        else:
            print('Something went wrong with the format of your reference volume. Please check it.')
            sys.exit(1)
        
            