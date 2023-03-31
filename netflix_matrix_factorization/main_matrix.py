import os
import numpy as np
import math
import pyopencl as cl
from dataset import read_data

os.environ["PYOPENCL_CTX"] = "0"

train_m, train_u, train_r, train_J, test_m, test_u, test_r, test_J, u2uid, m2mid = read_data()

dimuser = 50
dimmovie = 10

round_number = 0

if os.path.isfile(f"model_uvec_{round_number}.npy"):
    uvec = np.load(f"model_uvec_{round_number}.npy")
else:
    uvec = np.random.rand(len(u2uid) * dimuser).astype(np.float32)
    uvec *= np.float32(0.1)

if os.path.isfile(f"model_mvec_{round_number}.npy"):
    mvec = np.load(f"model_mvec_{round_number}.npy")
else:
    mvec = np.random.rand(len(m2mid) * dimmovie).astype(np.float32)
    mvec *= np.float32(0.1)

uvecd = np.zeros((len(u2uid) * dimuser)).astype(np.float32)
mvecd = np.zeros((len(m2mid) * dimmovie)).astype(np.float32)

if os.path.isfile(f"model_matrix_{round_number}.npy"):
    matrix = np.load(f"model_matrix_{round_number}.npy")
else:
    matrix = np.random.rand(dimuser * dimmovie).astype(np.float32)
    matrix *= np.float32(0.1)

matrixd = np.zeros((dimuser * dimmovie)).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

d_uvec = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=uvec)
d_uvecd = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=uvecd)
d_mvec = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mvec)
d_mvecd = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mvecd)
d_matrix = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrix)
d_matrixd = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrixd)

d_train_m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_m)
d_train_u = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_u)
d_train_r = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_r)
d_train_J = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=train_J)

d_test_m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_m)
d_test_u = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_u)
d_test_r = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_r)
d_test_J = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=test_J)

prg = cl.Program(ctx, open('kernels.cl', mode='rt').read()).build()

ss = 0.0000001
laste = 100.0
while True:
    knl = prg.train_matrix
    knl.set_scalar_arg_dtypes( [None, None, None, None, None, None, None, None, None, None, np.int32, np.int32] )
    knl(queue, train_m.shape, None, d_train_m, d_train_u, d_train_r, d_uvec, d_uvecd, d_mvec, d_mvecd, d_matrix, d_matrixd, d_train_J, np.int32(dimuser), np.int32(dimmovie))

    res_train_J = np.empty_like(train_J)
    cl.enqueue_copy(queue, res_train_J, d_train_J)
    etrain = np.sqrt(np.sum(res_train_J)/train_m.shape[0])

    if etrain <= laste:
        ss *= 1.3
    else:
        ss /= 30.0

    knl = prg.dostep
    knl.set_scalar_arg_dtypes( [None, None, np.float32] )
    knl(queue, uvec.shape, None, d_uvec, d_uvecd, np.float32(ss))

    knl = prg.dostep
    knl.set_scalar_arg_dtypes( [None, None, np.float32] )
    knl(queue, mvec.shape, None, d_mvec, d_mvecd, np.float32(ss))

    knl = prg.dostep
    knl.set_scalar_arg_dtypes( [None, None, np.float32] )
    knl(queue, matrix.shape, None, d_matrix, d_matrixd, np.float32(ss))
    
    knl = prg.test_matrix
    knl.set_scalar_arg_dtypes( [None, None, None, None, None, None, None, np.int32, np.int32] )
    knl(queue, test_m.shape, None, d_test_m, d_test_u, d_test_r, d_uvec, d_mvec, d_matrix, d_test_J, np.int32(dimuser), np.int32(dimmovie))

    res_test_J = np.empty_like(test_J)
    cl.enqueue_copy(queue, res_test_J, d_test_J)
    etest = np.sqrt(np.sum(res_test_J)/test_m.shape[0])

    print(round_number, etrain, etest, ss)
    file1 = open("debug.txt", "at")
    file1.write(f"{round_number},{etrain},{etest},{ss}\n") 
    file1.close()
    
    if round_number % 10 == 0:
        res_uvec = np.empty_like(uvec)
        cl.enqueue_copy(queue, res_uvec, d_uvec)
        np.save(f"model_uvec_{round_number}.npy", res_uvec)

        res_mvec = np.empty_like(mvec)
        cl.enqueue_copy(queue, res_mvec, d_mvec)
        np.save(f"model_mvec_{round_number}.npy", res_mvec)

        res_matrix = np.empty_like(matrix)
        cl.enqueue_copy(queue, res_matrix, d_matrix)
        np.save(f"model_matrix_{round_number}.npy", res_matrix)
    
    laste = etrain
    round_number += 1
