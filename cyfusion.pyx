cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()



cdef extern from "fusion.h":
  cdef cppclass Views:
    Views()
    int n_views_;
    float* depthmaps_;
    int* rgb_images_;
    int rows_;
    int cols_;
    float* Ks_;
    float* Rs_;
    float* Ts_;

    float* Ks_inv_;
    float* cam2world_;

  cdef cppclass Volume:
    Volume()
    int depth_;
    int height_;
    int width_;
    float* data_;

    int* rgb_data_;
    float* color_kernel_;
    int kernel_size_;

  void fusion_tsdf_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume& vol);

cdef class PyViews:
  cdef Views views
  # need to keep reference, otherwise it could get garbage collected
  cdef float[:,:,::1] depthmaps_
  cdef int[:,:,:,::1] rgb_images_
  cdef float[:,:,::1] Ks_
  cdef float[:,:,::1] Rs_
  cdef float[:,::1] Ts_

  def __init__(self, float[:,:,::1] depthmaps, int[:,:,:,::1] rgb_images, float[:,:,::1] Ks, float[:,:,::1] Rs, float[:,::1] Ts, float[:,:,::1] Ks_inv, float[:,:,::1] cam2world):
    cdef int n = depthmaps.shape[0]
    if n != Ks.shape[0]:
      raise Exception('number of depthmaps and Ks differ')
    if n != Rs.shape[0]:
      raise Exception('number of depthmaps and Rs differ')
    if n != Ts.shape[0]:
      raise Exception('number of depthmaps and Ts differ')

    if Ks.shape[1] != 3 or Ks.shape[2] != 3:
      raise Exception('Ks have to be nx3x3')
    if Rs.shape[1] != 3 or Rs.shape[2] != 3:
      raise Exception('Rs have to be nx3x3')
    if Ts.shape[1] != 3:
      raise Exception('Ts have to be nx3')
    if Ks_inv.shape[1] != 3 or Ks_inv.shape[2] != 3:
      raise Exception('Ks_inv have to be nx3x3')
    if cam2world.shape[1] != 3 or cam2world.shape[2] != 4:
      raise Exception('cam2world have to be nx3x4')


    self.depthmaps_ = depthmaps
    self.rgb_images_ = rgb_images
    self.Ks_ = Ks
    self.Rs_ = Rs
    self.Ts_ = Ts

    self.views.depthmaps_ = &(depthmaps[0,0,0])
    self.views.rgb_images_ = &(rgb_images[0,0,0,0])
    self.views.n_views_ = depthmaps.shape[0]
    self.views.rows_ = depthmaps.shape[1]
    self.views.cols_ = depthmaps.shape[2]
    self.views.Ks_ = &(Ks[0,0,0])
    self.views.Rs_ = &(Rs[0,0,0])
    self.views.Ts_ = &(Ts[0,0])

    self.views.Ks_inv_ = &(Ks_inv[0,0,0])
    self.views.cam2world_ = &(cam2world[0,0,0])


cdef class PyVolume:
  cdef Volume vol

  def __init__(self, float[:,:,::1] data, int[:,:,:,::1] rgb_data, float[:,:,::1] color_kernel, int kernel_size):
    self.vol = Volume()
    self.vol.rgb_data_ = &(rgb_data[0,0,0,0])
    self.vol.data_ = &(data[0,0,0])
    self.vol.color_kernel_ = &(color_kernel[0,0,0])
    self.vol.kernel_size_ = kernel_size;
    self.vol.depth_ = data.shape[0]
    self.vol.height_ = data.shape[1]
    self.vol.width_ = data.shape[2]

def tsdf_cpu(PyViews views, np.ndarray np_color_kernel, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, int n_threads=8, rgb=True):
  vol = np.empty((depth, height, width), dtype=np.float32)
  rgb_vol = np.zeros((depth, height, width, 3), dtype=np.int32)
  cdef float[:,:,::1] vol_view = vol
  cdef float[:,:,::1] vol_color_kernel = np_color_kernel
  cdef int[:,:,:,::1] rgb_vol_view = rgb_vol
  cdef int kernel_size = np_color_kernel.shape[0]
  print(vol.shape)
  cdef PyVolume py_vol = PyVolume(vol_view, rgb_vol, vol_color_kernel, kernel_size)
  fusion_tsdf_cpu(views.views, vx_size, truncation, unknown_is_free, n_threads, py_vol.vol)
  return vol, rgb_vol
