# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np

def im2col_optimized_cython(np.ndarray[np.float32_t, ndim=4] batch,
                             np.ndarray[np.float32_t, ndim=4] kernels,
                             np.ndarray[np.float32_t, ndim=1] biases = None,
                             int padding = 0,
                             int stride = 1,
                             bint applyReLU = True):

    cdef int out_channels = kernels.shape[0]
    cdef int in_channels = kernels.shape[1]
    cdef int kH = kernels.shape[2]
    cdef int kW = kernels.shape[3]
    cdef int N = batch.shape[0]
    cdef int C = batch.shape[1]
    cdef int H = batch.shape[2]
    cdef int W = batch.shape[3]

    if in_channels != C:
        raise ValueError("Mismatch nei canali input")

    cdef int out_H = (H + 2 * padding - kH) // stride + 1
    cdef int out_W = (W + 2 * padding - kW) // stride + 1

    if padding > 0:
        batch = np.pad(batch, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    cdef tuple shape = (N, C, out_H, out_W, kH, kW)
    cdef tuple strides = (
        batch.strides[0],
        batch.strides[1],
        stride * batch.strides[2],
        stride * batch.strides[3],
        batch.strides[2],
        batch.strides[3],
    )

    patches = np.lib.stride_tricks.as_strided(batch, shape=shape, strides=strides)
    patches = patches.reshape(N * out_H * out_W, C * kH * kW)

    kernels_flat = kernels.reshape(out_channels, -1).T  # shape: (C*kH*kW, OC)
    output_col = patches @ kernels_flat
    output = output_col.reshape(N, out_H, out_W, out_channels).transpose(0, 3, 1, 2)

    if biases is not None:
        output += biases.reshape(1, -1, 1, 1)

    mask = output > 0

    if applyReLU:
        output *= mask
    else:
        mask[:] = True

    return output.astype(np.float32), mask.astype(np.float32)


def im2col_gradient_optimized_cython(np.ndarray[np.float32_t, ndim=4] X,
                                      np.ndarray[np.float32_t, ndim=4] d_out,
                                      np.ndarray[np.float32_t, ndim=4] kernels,
                                      np.ndarray[np.float32_t, ndim=4] mask,
                                      int padding,
                                      int stride):

    cdef int OC = kernels.shape[0]
    cdef int IC = kernels.shape[1]
    cdef int kH = kernels.shape[2]
    cdef int kW = kernels.shape[3]
    cdef int N = X.shape[0]
    cdef int H = X.shape[2]
    cdef int W = X.shape[3]
    cdef int OH = d_out.shape[2]
    cdef int OW = d_out.shape[3]

    cdef np.ndarray[np.float32_t, ndim=4] dZ = d_out * mask
    cdef np.ndarray[np.float32_t, ndim=1] db = dZ.sum(axis=(0, 2, 3))

    if padding > 0:
        X_padded = np.pad(X, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    else:
        X_padded = X

    shape = (N, IC, OH, OW, kH, kW)
    strides = (
        X_padded.strides[0],
        X_padded.strides[1],
        stride * X_padded.strides[2],
        stride * X_padded.strides[3],
        X_padded.strides[2],
        X_padded.strides[3],
    )
    X_cols = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)
    X_cols = X_cols.reshape(N * OH * OW, IC * kH * kW)

    dZ_col = dZ.transpose(0, 2, 3, 1).reshape(N * OH * OW, OC)
    dW_flat = X_cols.T @ dZ_col
    dW = dW_flat.reshape(IC, kH, kW, OC).transpose(3, 0, 1, 2)

    from cython.parallel import prange
    dZ_dilated = np.zeros((N, OC, (OH - 1) * stride + 1, (OW - 1) * stride + 1), dtype=np.float32)
    for n in range(N):
        for c in range(OC):
            for i in range(OH):
                for j in range(OW):
                    dZ_dilated[n, c, i * stride, j * stride] = dZ[n, c, i, j]

    flipped = np.rot90(kernels, 2, axes=(2, 3)).transpose(1, 0, 2, 3)
    pad_dx = kH - 1 - padding
    dX_raw, _ = im2col_optimized_cython(dZ_dilated, flipped, padding=pad_dx, stride=1, applyReLU=False)

    dX = dX_raw[:, :, :H, :W]
    return dX, dW, db
