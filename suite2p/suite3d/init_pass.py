import os
import numpy as n
import copy

from ..io import lbm as lbmio 
from ..registration import register
from . import utils

def default_log(string, val): 
    print(string)

def choose_init_tifs(tifs, n_init_files, init_file_pool_lims=None, method='even', seed=2358):

    init_file_pool = []
    if init_file_pool_lims is not None:
        for limits in init_file_pool_lims:
            init_file_pool += tifs[limits[0]:limits[1]]
    else:
        init_file_pool = tifs
    init_file_pool = n.array(init_file_pool)
    
    if method == 'even':
        sample_file_ids = n.linspace(0, len(init_file_pool), n_init_files + 2, dtype=int)[1:-1]
        sample_tifs = n.array(init_file_pool)[sample_file_ids]
    elif method == 'random':
        n.random.seed(seed)
        sample_tifs = n.random.choice(init_file_pool, n_init_files, replace=False)

    return sample_tifs

def load_init_tifs(init_tifs, planes, filter_params):
    full_mov = lbmio.load_and_stitch_tifs(init_tifs, planes = planes, 
                                        filt=filter_params, concat=False)

    mov_lens = [mov.shape[1] for mov in full_mov]
    full_mov = n.concatenate(full_mov, axis=1)

    return full_mov

def prep_registration(full_mov, reg_ops, log_callback=default_log, filter_pcorr=0):
    nz, nt, ny, nx = full_mov.shape
    ref_img_3d = []
    log_callback("Computing reference images")
    for i in range(nz):
        ref_img = register.compute_reference(reg_ops, full_mov[i])
        ref_img_3d.append(ref_img)
        log_callback("  Computed reference for plane %d" % i,2)
    ref_img_3d = n.array(ref_img_3d)

    tvecs = n.concatenate([[[0,0]], utils.get_shifts_3d(ref_img_3d, filter_pcorr=filter_pcorr)])
    log_callback("Tvecs: %s" % str(tvecs), 2)

    ref_img_3d_aligned = utils.register_movie(ref_img_3d[:,n.newaxis], tvecs=tvecs)[:,0]

    all_ref_and_masks = []
    all_ops = []
    for plane_idx in range(nz):
        ref_img = ref_img_3d_aligned[plane_idx].copy()
        plane_ops = copy.deepcopy(reg_ops)
        if plane_ops.get('norm_frames',False):
            plane_ops['rmin'], plane_ops['rmax'] = n.int16(n.percentile(ref_img,1)), n.int16(n.percentile(ref_img,99))
            ref_img = n.clip(ref_img, plane_ops['rmin'], plane_ops['rmax'])    
        ref_and_masks = register.compute_reference_masks(ref_img, plane_ops)
        all_ref_and_masks.append(ref_and_masks)
        all_ops.append(plane_ops)

    return tvecs, ref_img_3d_aligned, all_ops, all_ref_and_masks


def register_sample_movie(full_mov, all_ops, all_refs, in_place=True, log_callback=default_log):
    nz = full_mov.shape[0]
    if not in_place:
        full_mov = full_mov.copy()
    log_callback("Registering sample movie")
    all_offsets = []
    for plane_idx in range(nz):
        ref_and_masks = all_refs[plane_idx]
        plane_ops = all_ops[plane_idx]
        log_callback("  Registering plane %d" % plane_idx)
        full_mov[plane_idx], ym, xm, cm, ym1, xm1, cm1 = \
            register.register_frames(ref_and_masks, full_mov[plane_idx], ops=plane_ops)
        all_offsets.append((ym, xm, cm, ym1, xm1, cm1))

    return full_mov, all_offsets


def run_init_pass(job):
    tifs = job.tifs
    params = job.params

    summary_path = os.path.join(job.dirs['summary'], 'summary.npy')
    job.log("Saving summary to %s" % summary_path,0)
    if not os.path.isdir(job.dirs['summary']):
        job.log("Summary dir does not exist!!")
        assert False

    init_tifs = choose_init_tifs(tifs, params['n_init_files'], params['init_file_pool'], 
                                       params['init_file_sample_method'])
    init_mov = load_init_tifs(init_tifs, params['planes'], params['notch_filt'])
    nz, nt, ny, nx = init_mov.shape
    im3d = init_mov.mean(axis=1)

    if params['subtract_crosstalk']:
        if params['override_crosstalk'] is not None:
            cross_coeff = params['override_crosstalk']
            job.log("Subtracting crosstalk with forced coefficient %0.3f" % cross_coeff)
        else:
            __, __, cross_coeff = utils.calculate_crosstalk_coeff(im3d,
                                                    estimate_from_last_n_planes=params['n_planes_for_crosstalk_est'],
                                                    show_plots=False, save_plots=params['job_summary_dir'],
                                                    verbose=(job.verbosity == 2))
            job.log("Subtracting with estimated coefficient %0.3f" % cross_coeff)
        for plane in params['planes']:
            if plane > 14:
                plane_idx = n.where(n.array(params['planes']) == plane)[0][0]
                sub_plane_idx = n.where(n.array(params['planes']) == plane - 15)[0][0]
                
                job.log("    Subtracting plane %d from %d" % (plane-15, plane), 2)
                job.log("        Corresponds to index %d from %d" % (sub_plane_idx, plane_idx))

                init_mov[plane_idx] = init_mov[plane_idx] - init_mov[sub_plane_idx] * cross_coeff
    else:
        job.log("No crosstalk estimation or subtraction")
        cross_coeff = None

    job.log("Building ops file")
    reg_ops = utils.build_ops('', {}, {'smooth_sigma' : job.params['smooth_sigma'],
                                        'maxregshift' : job.params['maxregshift'],
                                        'Ly' : ny, 'Lx' : nx,
                                        'nonrigid' : job.params['nonrigid']})

    job.log("Aligning planes and calculating reference images")
    tvecs, ref_img_3d, all_ops, all_refs_masks = prep_registration(init_mov, reg_ops, job.log, filter_pcorr=params['reg_filter_pcorr'])

    summary = {
        'ref_img_3d' : ref_img_3d,
        'crosstalk_coeff' : cross_coeff,
        'plane_shifts' : tvecs,
        'refs_and_masks' : all_refs_masks,
        'all_ops' : all_ops,
        'plane_mean' : init_mov.mean(axis=(1,2,3), dtype=job.params['dtype']),
        'plane_std' : init_mov.std(axis=(1,2,3), dtype=job.params['dtype']),
    }
    summary_path = os.path.join(job.dirs['summary'], 'summary.npy')
    job.log("Saving summary to %s" % summary_path)
    n.save(summary_path, summary)

    if job.params['generate_sample_registered_bins']:
        sample_bin_path = os.path.join(job.dirs['summary'], 'sample_reg_movie.npy')
        sample_off_path = os.path.join(job.dirs['summary'], 'sample_offsets.npy')
        job.log("Registering sample files and saving them to %s for verification" % sample_bin_path)
        job.log("Offsets will be saved in summary.npy")

        init_mov, all_offsets = register_sample_movie(init_mov, all_ops, all_refs_masks, log_callback=job.log)

        n.save(sample_bin_path, init_mov)
        summary['all_offsets'] : all_offsets
        n.save(summary_path, summary)


    job.log("Initial pass complete. See %s for details" % job.dirs['summary'])