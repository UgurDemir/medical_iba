import os
from os.path import join, isfile, basename, exists

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

class FolderIterator():
    def __init__(self, input_dir_or_file, output_dir, xform=None):
        self.fileiter_list = []

        if isfile(input_dir_or_file):
            scan_list = [input_dir_or_file]
        else:
            scan_list = [join(input_dir_or_file, f) for f in os.listdir(input_dir_or_file)]
        
        for scan_path in scan_list:
            if scan_path.endswith('.nii.gz') or scan_path.endswith('.nii'):
                self.fileiter_list.append(NiftiIterator(scan_path, output_dir, xform))
            elif scan_path.endswith('.img'):
                self.fileiter_list.append(MRIIterator(scan_path, output_dir, xform))
            elif scan_path.endswith('.npy'):
                raise NotImplementedError('.npy file support is not implemented yet for the file {}'.format(scan_path))

    def __len__(self):
        return len(self.fileiter_list)

    def __iter__(self):
        for fiter in self.fileiter_list:
            for slce in fiter:   
                yield slce, fiter.output_handler
            fiter.save_all()

class SingleSliceIterator():
    def __init__(self, input_file, output_dir, xform=None):
        self.input_path = input_file
        self.patient_name = basename(input_file).split('.')[0]
        self.file_extension = '.'.join(basename(input_file).split('.')[1:])
        self.output_dir = join(output_dir, self.patient_name)
        self.xform = xform
        
        self.heatmap_dict = {}

    def __iter__(self):
        input_3d = self.get_data_as_np() # [slice, H,W]

        nslices = input_3d.shape[0]
        for i in range(nslices):
            slice_raw = input_3d[i:i+1, ...]
            if self.xform is not None:
                slice_preproc = self.xform(slice_raw.transpose((1,2,0)))
            else:
                slice_preproc = torch.from_numpy(slice_raw)

            # Add dummy batch dimension
            slice_preproc = slice_preproc[np.newaxis,...]

            yield slice_preproc

    def output_handler(self, target, heatmap):
        if target not in self.heatmap_dict:
            self.heatmap_dict[target] = []    
        self.heatmap_dict[target].append(heatmap)

    def save_all(self):
        for target, hlist in self.heatmap_dict.items():
            if not exists(self.output_dir):
                os.makedirs(self.output_dir)
            target_filename = join(self.output_dir, '{}.{}'.format(target, self.file_extension))
            np_data = np.array(hlist)
            self.save(np_data, target_filename)

    def save(self, np_data, filename):
        # It should save the raw numpy array in the form of original input format with meta data
        pass

    def get_data_as_np(self):
        # It should return the image as numpy array with shape [slice, height, width]
        pass


class NiftiIterator(SingleSliceIterator):
    def __init__(self, input_file, output_dir, xform=None):
        super(NiftiIterator, self).__init__(input_file, output_dir, xform)
        self.nifti_img = nib.load(input_file)

    def get_data_as_np(self):
        return self.nifti_img.get_fdata().astype(np.float32).transpose((2,0,1))

    def save(self, np_data, filename):
        nifti_out = nib.Nifti1Image(np_data.transpose((1,2,0)), self.nifti_img.affine)
        nib.save(nifti_out, filename)

class MRIIterator(SingleSliceIterator):
    def __init__(self, input_file, output_dir, xform=None):
        super(MRIIterator, self).__init__(input_file, output_dir, xform)
        self.mri_img = sitk.ReadImage(input_file, imageIO="NiftiImageIO")

    def get_data_as_np(self):
        return sitk.GetArrayFromImage(self.mri_img).astype(np.float32)

    def save(self, np_data, filename):
        #mri_orig = sitk.ReadImage(self.input_path, imageIO="NiftiImageIO")
        #mri_img.set
        #sitk.WriteImage(image, filename)
        pass

