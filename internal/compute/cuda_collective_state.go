//go:build cuda

package compute

// cudaNCCLWorld is the rank count the NCCL communicator is up over (0 = not initialized).
// Plain `-tags cuda` builds keep it at zero; `-tags cuda,nccl` sets it after fcuda_nccl_init.
var cudaNCCLWorld int32
