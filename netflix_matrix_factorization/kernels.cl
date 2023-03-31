#define dimM 10
#define dimU 50

inline float atomicAdd(volatile __global float* address, const float value){
    float old = value;
    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
    return old;
}

inline float atomicAdd2(volatile __local float* address, const float value){
    float old = value;
    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
    return old;
}

__kernel void train_quad(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec, __global float *uvecd,
    __global float *mvec, __global float *mvecd,
    __global float *train_J, 
    int dimension)
{
  int gid = get_global_id(0);
  
  float res = 0.0f;
  for(int i=0;i<dimension;i++)
  {
    float tmp = sqrt(3.0f) + uvec[dimension * train_u[gid] + i] + mvec[dimension * train_m[gid] + i];
    res += tmp*tmp/((float)dimension);
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
  for(int i=0;i<dimension;i++)
  {
    float tmp = 4.0f * res * (sqrt(3.0f) + uvec[dimension * train_u[gid] + i] + mvec[dimension * train_m[gid] + i]) / ((float)dimension);
    atomicAdd(&(uvecd[dimension * train_u[gid] + i]), tmp);
    atomicAdd(&(mvecd[dimension * train_m[gid] + i]), tmp);
  }
}

__kernel void test_quad(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec,
    __global float *mvec,
    __global float *train_J, 
    int dimension)
{
  int gid = get_global_id(0);
  
  float res = 0.0f;
  for(int i=0;i<dimension;i++)
  {
    float tmp = sqrt(3.0f) + uvec[dimension * train_u[gid] + i] + mvec[dimension * train_m[gid] + i];
    res += tmp*tmp/((float)dimension);
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
}

__kernel void train_matrix_factorization(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec, __global float *uvecd,
    __global float *mvec, __global float *mvecd,
    __global float *train_J, 
    int dimension)
{
  int gid = get_global_id(0);
  
  float res = 3.0f;
  for(int i=0;i<dimension;i++)
  {
    res += uvec[dimension * train_u[gid] + i] * mvec[dimension * train_m[gid] + i];
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
  for(int i=0;i<dimension;i++)
  {
    atomicAdd(&(uvecd[dimension * train_u[gid] + i]), 2.0 * res * mvec[dimension * train_m[gid] + i]);
    atomicAdd(&(mvecd[dimension * train_m[gid] + i]), 2.0 * res * uvec[dimension * train_u[gid] + i]);
  }
}

__kernel void test_matrix_factorization(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec,
    __global float *mvec,
    __global float *train_J, 
    int dimension)
{
  int gid = get_global_id(0);
  
  float res = 3.0f;
  for(int i=0;i<dimension;i++)
  {
    res += uvec[dimension * train_u[gid] + i] * mvec[dimension * train_m[gid] + i];
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
}

__kernel void train_matrix(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec, __global float *uvecd,
    __global float *mvec, __global float *mvecd,
    __global float *matrix, __global float *matrixd,
    __global float *train_J, 
    int dimuser, int dimmovie)
{
  int gid = get_global_id(0);
  
  float tempmovie[dimM];
  float tempmovied[dimM];

  float tempuser[dimU];
  float tempuserd[dimU];
  
  __local float tempmatrix[dimU * dimM];
  __local float tempmatrixd[dimU * dimM];
  int ltid = get_local_id(0);
  int ltidi = get_local_size(0);
  for(int i=ltid;i<dimU*dimM;i+=ltidi)
  {
	tempmatrix[i] = matrix[i];
  	tempmatrixd[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for(int i=0;i<dimuser;i++)
  {
    tempuser[i] = uvec[dimuser * train_u[gid] + i];
    tempuserd[i] = 0.0f;
  }
  
  for(int j=0;j<dimmovie;j++)
  {
    tempmovie[j] = mvec[dimmovie * train_m[gid] + j];
    tempmovied[j] = 0.0f;
  }
  
  float res = 3.0f;
  for(int i=0;i<dimuser;i++)
  {
    for(int j=0;j<dimmovie;j++)
    {
        res += tempuser[i] * tempmovie[j] * tempmatrix[i * dimmovie + j];
    }
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
  for(int i=0;i<dimuser;i++)
  {
    for(int j=0;j<dimmovie;j++)
    {
        tempuserd[i] += 2.0f * res * tempmovie[j]* tempmatrix[i * dimmovie + j];
        tempmovied[j] += 2.0f * res * tempuser[i] * tempmatrix[i * dimmovie + j];
        atomicAdd2(&(tempmatrixd[i * dimmovie + j]), 2.0f * res * tempuser[i] * tempmovie[j]);
    }
  }
  
  for(int i=0;i<dimuser;i++)
  {
    atomicAdd(&(uvecd[dimuser * train_u[gid] + i]), tempuserd[i]);
  }
  
  for(int j=0;j<dimmovie;j++)
  {
    atomicAdd(&(mvecd[dimmovie * train_m[gid] + j]), tempmovied[j]);
  }  
  
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i=ltid;i<dimU*dimM;i+=ltidi)
  {
  	atomicAdd(&(matrixd[i]), tempmatrixd[i]);
  }  
}

__kernel void test_matrix(
    __global const int *train_m, __global const int *train_u, __global const int *train_r, 
    __global float *uvec,
    __global float *mvec,
    __global float *matrix,
    __global float *train_J, 
    int dimuser, int dimmovie)
{
  int gid = get_global_id(0);
  
  float tempmovie[dimM];

  float tempuser[dimU];

  __local float tempmatrix[dimU * dimM];
  int ltid = get_local_id(0);
  int ltidi = get_local_size(0);
  for(int i=ltid;i<dimU*dimM;i+=ltidi)
  {
	tempmatrix[i] = matrix[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
    
  for(int i=0;i<dimuser;i++)
  {
    tempuser[i] = uvec[dimuser * train_u[gid] + i];
  }
  
  for(int j=0;j<dimmovie;j++)
  {
    tempmovie[j] = mvec[dimmovie * train_m[gid] + j];
  }
  
  float res = 3.0f;
  for(int i=0;i<dimuser;i++)
  {
    for(int j=0;j<dimmovie;j++)
    {
        res += tempuser[i] * tempmovie[j] * tempmatrix[i * dimmovie + j];
    }
  }
  res -= (float)train_r[gid];

  train_J[gid] = res * res;
}

__kernel void dostep(
    __global float *vec, __global float *vecd, float ss)
{
  int gid = get_global_id(0);
  vec[gid] -= vecd[gid] * ss;
  vecd[gid] = 0;
}

__kernel void dostepregu(
    __global float *vec, __global float *vecd, float ss, float regu)
{
  int gid = get_global_id(0);
  vec[gid] -= (vecd[gid] + regu * 2.0f * vec[gid]) * ss;
  vecd[gid] = 0;
}

