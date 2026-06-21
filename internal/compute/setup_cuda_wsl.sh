#!/usr/bin/env bash
# setup_cuda_wsl.sh — stand up a NO-SUDO CUDA 12.6 toolchain in WSL for the -tags cuda
# build of internal/compute. This box has working WSL2 GPU passthrough (nvidia-smi sees
# the RTX 4070, /usr/lib/wsl/lib/libcuda.so present) and a C compiler, but no system CUDA
# toolkit and no passwordless sudo — so we install a complete user-space toolkit (real
# nvcc front-end, cuBLAS-dev, headers, cmake) via micromamba. This is the build/CI
# orchestration seam DIRECTION.md sanctions: it produces a toolchain, it is off the
# request path, and the default `go build` needs none of it.
#
# Idempotent. After it runs, `source ~/cudaenv.env` then `bash build_cuda.sh test`.
set -euo pipefail
MAMBA=$HOME/bin/micromamba
ENVDIR=$HOME/cudaenv
mkdir -p "$HOME/bin"

if [ ! -x "$MAMBA" ]; then
  echo "[setup] downloading micromamba ..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -C "$HOME" -xj bin/micromamba
fi

if [ ! -x "$ENVDIR/bin/nvcc" ]; then
  echo "[setup] creating CUDA 12.6 env (cuda-nvcc + cudart-dev + cublas-dev + nvrtc-dev + cmake) ..."
  "$MAMBA" create -y -p "$ENVDIR" -c nvidia -c conda-forge \
    cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-nvrtc-dev=12.6 libcublas-dev=12.6 cuda-cccl=12.6 \
    cmake ninja make
fi

cat > "$HOME/cudaenv.env" <<EOF
export CUDA_HOME=$ENVDIR
export CUDAToolkit_ROOT=$ENVDIR
export PATH=$ENVDIR/bin:\$PATH
export LD_LIBRARY_PATH=$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}
export CPATH=$ENVDIR/include:$ENVDIR/targets/x86_64-linux/include:\${CPATH:-}
EOF
. "$HOME/cudaenv.env"

echo "[setup] nvcc: $(command -v nvcc)"
nvcc --version | tail -2

# on-box witness: compile + run a kernel on the GPU
cat > /tmp/fak_probe.cu <<'CU'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void axpy(float a,float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a*x[i]+y[i];}
int main(){cudaDeviceProp p;if(cudaGetDeviceProperties(&p,0)!=cudaSuccess){printf("NO_DEVICE\n");return 1;}
printf("DEVICE %s sm_%d%d %.1fGB\n",p.name,p.major,p.minor,p.totalGlobalMem/1e9);
int n=1<<20;size_t s=n*4;float*x=(float*)malloc(s),*y=(float*)malloc(s),*dx,*dy;
for(int i=0;i<n;i++){x[i]=1;y[i]=2;}cudaMalloc(&dx,s);cudaMalloc(&dy,s);
cudaMemcpy(dx,x,s,cudaMemcpyHostToDevice);cudaMemcpy(dy,y,s,cudaMemcpyHostToDevice);
axpy<<<(n+255)/256,256>>>(3,dx,dy,n);cudaDeviceSynchronize();cudaMemcpy(y,dy,s,cudaMemcpyDeviceToHost);
printf("RESULT y[0]=%.1f (want 5.0)\n",y[0]);return y[0]==5.0f?0:2;}
CU
if nvcc -arch=sm_89 -o /tmp/fak_probe /tmp/fak_probe.cu -lcudart && /tmp/fak_probe; then
  echo "SETUP_OK"
else
  echo "SETUP_FAIL"
  exit 1
fi
