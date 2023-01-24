from dask_image.ndfilters import uniform_filter as dask_uniform_filter
import cProfile
from ..suite3d import utils as utils3d
from ctypes import util
from http.client import MOVED_PERMANENTLY
import multiprocessing
import numpy as n
np = n
from dask import array as darr
from scipy.ndimage import maximum_filter, gaussian_filter, uniform_filter
from .utils import threshold_reduce

def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Lz, Ly, Lx = mov.shape
    mov = mov[:(n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Lz, Ly, Lx).astype(np.float32).mean(axis=1)

def standard_deviation_over_time(mov: np.ndarray, batch_size: int,sqrt=True, dask=False) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Lz x Ly x Lx
        The statistics for each pixel
    """
    nbins = mov.shape[0]
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros(mov.shape[1:], 'float32')
    if dask: 
        sdmov = darr.zeros(*mov.shape[1:], dtype=n.float32)
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix+batch_size], axis=0) ** 2).sum(axis=0))
    if sqrt: 
        sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov

# def npsub_worker(filt_size, mode, c1):
#     frame = n.zeros((8,1202,1015))
#     return frame - uniform_filter(frame, size=filt_size, mode=mode) / c1
    # return frame - frame / c1
def npsub_worker(mov_params, idxs, filt_size, mode, c1):
    print("RUNNING idxs: %s" % str(idxs))
    cp = cProfile.Profile()
    cp.enable()
    shmem, mov = utils3d.load_shmem(mov_params)
    for idx in idxs:
        # mov[idx] = mov[idx] - (uniform_filter(mov[idx], size=filt_size, mode = mode) / c1)
        mov[idx] = mov[idx] - mov[idx] / c1
    cp.disable()
    cp.print_stats(sort='cumtime')

def neuropil_subtraction(mov: np.ndarray, filter_size: int, filter_size_z: int, mode='constant') -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filt_size, mode=mode) / c1)
    return movt


def neuropil_subtraction_dask(mov: np.ndarray, filter_size: int, filter_size_z: int, mode='constant') -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    
    for i in range(nt):
        mov[i] = mov[i] - dask_uniform_filter(mov[i], size=filt_size, mode=mode) / c1
    return mov

def neuropil_subtraction_debug(mov: np.ndarray, filter_size: int, filter_size_z: int, mode='constant') -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""

    nt, Lz, Ly, Lx = mov.shape
    filt_size = (filter_size_z, filter_size, filter_size)
    # print('Neuropil filter size:', filt_size)
    c1 = uniform_filter(np.ones((Lz, Ly, Lx)), size=filt_size, mode=mode)
    movt = np.zeros_like(mov)
    lpfs = []
    for frame, framet in zip(mov, movt):
        lpf = (uniform_filter(frame, size=filt_size, mode=mode) / c1)
        framet[:] = frame - lpf
        lpfs.append(lpf)
    return movt, n.array(lpfs)


def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """
    Returns a pixel-downsampled movie from 'mov', tapering the edges of 'taper_edge' is True.

    Parameters
    ----------
    mov: nImg x Lz x Ly x Lx
        The frames to downsample
    taper_edge: bool
        Whether to taper the edges

    Returns
    -------
    filtered_mov:
        The downsampled frames
    """
    n_frames, Lz, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, Lz, int(np.ceil(Ly / 2)), Lx), 'float32')
    movd[:,:, :Ly//2, :] = np.mean([mov[:,:, 0:-1:2, :], mov[:,:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:,:, -1, :] = mov[:,:, -1, :] / 2 if taper_edge else mov[:,:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, Lz,  int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), 'float32')
    mov2[:,:, :, :Lx//2] = np.mean([movd[:,:, :, 0:-1:2], movd[:,:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:,:, :, -1] = movd[:,:, :, -1] / 2 if taper_edge else movd[:,:, :, -1]

    return mov2

def square_convolution_2d(mov: np.ndarray, filter_size: int, filter_size_z: int) -> np.ndarray:
    """Returns movie convolved by uniform kernel with width 'filter_size'."""
    movt = np.zeros_like(mov, dtype=np.float32)
    filt_size = (filter_size_z, filter_size, filter_size)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filt_size, mode='constant')
    return movt

def get_vmap3d(movu0,intensity_threshold=None, fix_edges=True, sqrt=True,mean_subtract=True):
    nt, nz, ny, nx = movu0.shape
    vmap = n.zeros((nz,ny,nx))
    for i in range(nz):
        vmap[i] = threshold_reduce(movu0[:, i], intensity_threshold,
                                   mean_subtract=mean_subtract, fix_edges=fix_edges, sqrt=sqrt)
    return vmap

def get_vmap3d_shmem_w(shmem_in, shmem_vmap, z_idx, intensity_threshold, fix_edges, sqrt):
    shin, mov_in = utils3d.load_shmem(shmem_in)
    shvmap, vmap_z = utils3d.load_shmem(shmem_vmap)

    vmap_z[z_idx] = threshold_reduce(
        mov_in[:, z_idx], intensity_threshold, fix_edges, sqrt)

def get_vmap3d_shmem(shmem_in, shmem_vmap, intensity_threshold=None, fix_edges=True, sqrt=True, n_proc=15, pool=None):
    nt, nz, ny, nx = shmem_in['shape']
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
    pool.starmap(get_vmap3d_shmem_w, [(shmem_in, shmem_vmap, z_idx, intensity_threshold, fix_edges, sqrt) for z_idx in range(nz)])

def np_sub_and_conv3d_shmem_w(in_par, idxs, np_filt_size,conv_filt_size, c1):
    shin, mov_in = utils3d.load_shmem(in_par)
    for idx in idxs:
        mov_in[idx] = mov_in[idx] - \
            (uniform_filter(mov_in[idx], size=np_filt_size, mode='constant') / c1)
        mov_in[idx] = uniform_filter(mov_in[idx], size=conv_filt_size, mode='constant')

def np_sub_and_conv3d_shmem(shmem_in, np_filt_size, conv_filt_size, n_proc=8, batch_size=50, pool=None):
    """
    WARNING: this is not optimal because the process startup time is quite long for each process
    seems like this is because each subprocess imports suite2p, which takes about 1-2 seconds
    this should all be in parallel, but in reality it causes memory bottlenecks so lasts about ~10 seconsds
    so if you have a mp.pool().starmap() that does nothing, it takes 10 seconds to run
    this is because windows does not have forking and python uses the "spawn" start method
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    in the future: either maintain the same processes for the whole thing, or figure out something else
    """
    nt, Lz, Ly, Lx = shmem_in['shape']
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), np_filt_size, mode='constant')

    batches = [n.arange(idx, min(nt, idx+batch_size))
               for idx in n.arange(0, nt, batch_size)]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close=False
    pool.starmap(np_sub_and_conv3d_shmem_w, [
                 (shmem_in, b, np_filt_size, conv_filt_size, c1) for b in batches])
    if close:
        pool.close()
        pool.terminate()



def np_sub_and_conv3d_split_shmem_w(sub_par, filt_par, idxs, np_filt_size,conv_filt_size, c1, c2):
    sub_sh, mov_sub = utils3d.load_shmem(sub_par)
    filt_sh,   mov_filt = utils3d.load_shmem(filt_par)
    for idx in idxs:
        mov_sub[idx] = mov_sub[idx] - \
            (uniform_filter(mov_sub[idx], size=np_filt_size, mode='constant') / c1)
        mov_filt[idx] = uniform_filter(mov_sub[idx], size=conv_filt_size, mode='constant') / c2
    sub_sh.close(); filt_sh.close()

def np_sub_and_conv3d_split_shmem(shmem_sub, shmem_filt, np_filt_size, conv_filt_size, n_proc=8, batch_size=50, pool=None):
    nt, Lz, Ly, Lx = shmem_sub['shape']
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), np_filt_size, mode='constant')
    c2 = uniform_filter(n.ones((Lz, Ly, Lx)), conv_filt_size, mode='constant')

    batches = [n.arange(idx, min(nt, idx+batch_size))
               for idx in n.arange(0, nt, batch_size)]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close=False
    pool.starmap(np_sub_and_conv3d_split_shmem_w, [
                 (shmem_sub, shmem_filt, b, np_filt_size, conv_filt_size, c1, c2) for b in batches])
    if close:
        pool.close()
        pool.terminate()



def np_sub_shmem_w(in_par, idxs, np_filt_size, c1):
    shin, mov_in = utils3d.load_shmem(in_par)
    for idx in idxs:
        mov_in[idx] = mov_in[idx] - \
            (uniform_filter(mov_in[idx],
             size=np_filt_size, mode='constant') / c1)



def np_sub_shmem(shmem_in, np_filt_size, n_proc=8, batch_size=50, pool=None):
    """
    WARNING: this is not optimal because the process startup time is quite long for each process
    seems like this is because each subprocess imports suite2p, which takes about 1-2 seconds
    this should all be in parallel, but in reality it causes memory bottlenecks so lasts about ~10 seconsds
    so if you have a mp.pool().starmap() that does nothing, it takes 10 seconds to run
    this is because windows does not have forking and python uses the "spawn" start method
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    in the future: either maintain the same processes for the whole thing, or figure out something else
    """
    nt, Lz, Ly, Lx = shmem_in['shape']
    c1 = uniform_filter(n.ones((Lz, Ly, Lx)), np_filt_size, mode='constant')

    batches = [n.arange(idx, min(nt, idx+batch_size))
               for idx in n.arange(0, nt, batch_size)]
    close = True
    if pool is None:
        pool = multiprocessing.Pool(n_proc)
        close = False
    pool.starmap(np_sub_shmem_w, [
                 (shmem_in, b, np_filt_size, c1) for b in batches])
    if close:
        pool.close()
        pool.terminate()


def square_convolution_2d(mov: np.ndarray, filter_size: int, filter_size_z: int) -> np.ndarray:
    """Returns movie convolved by uniform kernel with width 'filter_size'."""
    movt = np.zeros_like(mov, dtype=np.float32)
    filt_size = (filter_size_z, filter_size, filter_size)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * \
            uniform_filter(frame, size=filt_size, mode='constant')
    return movt




# def np_sub_shmem_w(in_par, out_par, idxs, size, c1):
#     shin, mov_in = utils3d.load_shmem(in_par)
#     shout, mov_out = utils3d.load_shmem(out_par)
#     for idx in idxs:
#         mov_out[idx] = mov_in[idx] - \
#             (uniform_filter(mov_in[idx], size=size, mode='constant') / c1)


# def np_sub_shmem(shmem_in, shmem_out, size, n_proc, batch_size=50):
#     nt, Lz, Ly, Lx = shmem_in['shape']
#     c1 = uniform_filter(n.ones((Lz, Ly, Lx)), size, mode='constant')

#     batches = [n.arange(idx, min(nt, idx+batch_size))
#                for idx in n.arange(0, nt, batch_size)]
#     pool = multiprocessing.Pool(n_proc)
#     pool.starmap(np_sub_shmem_w, [
#                  (shmem_in, shmem_out, b, size, c1) for b in batches])
