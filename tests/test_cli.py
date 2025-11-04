import subprocess
import sys
import os
import numpy as np
import nibabel as nib
import tempfile
import shutil
import pytest


def create_dummy_nifti(path, shape=(2, 2, 2, 4), value=1.0):
    data = np.full(shape, value, dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)
    return path


def create_text_file(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')
    return path


def test_tpil_calculate_tstat_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, 'input.nii.gz')
        subj_path = os.path.join(tmpdir, 'subjects.txt')
        group_path = os.path.join(tmpdir, 'groups.txt')
        tstat_out = os.path.join(tmpdir, 'tstat.nii.gz')
        pval_out = os.path.join(tmpdir, 'pval.nii.gz')
        create_dummy_nifti(nifti_path, (2, 2, 2, 4), value=1.0)
        create_text_file(subj_path, ['s1', 's1', 's2', 's2'])
        create_text_file(group_path, ['A', 'A', 'B', 'B'])
        cmd = [sys.executable, 'tpil_calculate_tstat.py',
               '--nifti_4d', nifti_path,
               '--subject_file', subj_path,
               '--group_file', group_path,
               '--output_tstat', tstat_out,
               '--output_pval', pval_out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert os.path.isfile(tstat_out)
        assert os.path.isfile(pval_out)
        tstat_img = nib.load(tstat_out)
        assert tstat_img.shape == (2, 2, 2)


def test_tpil_calculate_i2c2_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, 'input.nii.gz')
        mask_path = os.path.join(tmpdir, 'roi_mask.nii.gz')
        subj_path = os.path.join(tmpdir, 'subjects.txt')
        visit_path = os.path.join(tmpdir, 'visits.txt')
        # Create 2x2x2x4 nifti, mask, and text files
        # 2 subjects, 2 visits, 2 scans per subject
        create_dummy_nifti(nifti_path, (2, 2, 2, 4), value=1.0)
        create_dummy_nifti(mask_path, (2, 2, 2), value=1.0)
        create_text_file(subj_path, ['s1', 's1', 's2', 's2'])
        create_text_file(visit_path, ['v1', 'v2', 'v1', 'v2'])
        cmd = [sys.executable, 'tpil_calculate_i2c2.py',
               '--nifti_file', nifti_path,
               '--roi_mask', mask_path,
               '--subject_file', subj_path,
               '--visit_file', visit_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert 'I2C2 within ROI' in result.stdout


def test_tpil_i2c2_filter_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, 'input.nii.gz')
        subj_path = os.path.join(tmpdir, 'subjects.txt')
        group_path = os.path.join(tmpdir, 'groups.txt')
        visit_path = os.path.join(tmpdir, 'visits.txt')
        mask_path = os.path.join(tmpdir, 'mask.nii.gz')
        out_path = os.path.join(tmpdir, 'output.nii.gz')
        create_dummy_nifti(nifti_path, (2, 2, 2, 4), value=1.0)
        create_dummy_nifti(mask_path, (2, 2, 2), value=1.0)
        create_text_file(subj_path, ['s1', 's1', 's2', 's2'])
        create_text_file(group_path, ['A', 'A', 'B', 'B'])
        create_text_file(visit_path, ['v1', 'v2', 'v1', 'v2'])
        cmd = [sys.executable, 'tpil_i2c2_filter.py',
               '--nifti_4d', nifti_path,
               '--subject_file', subj_path,
               '--group_file', group_path,
               '--visit_file', visit_path,
               '--output_file', out_path,
               '--stat_threshold', '0.0',
               '--size_threshold', '1',
               '--i2c2_threshold', '0.0',
               '--mask', mask_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert os.path.isfile(out_path)
        img = nib.load(out_path)
        assert img.shape == (2, 2, 2)
