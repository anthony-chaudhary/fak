// pdl_chain.cuh - structure-only declaration of a PDL-chained CUDA graph.
//
// ADAPT of tie-pilot-qxw/pdl-megakernel-reconstruction (MIT), demos/low-latency-llama/llama.cu and
// docs/reconstructing-a-megakernel-with-pdl.md @8664720072fefb62cc3a097c9c5bba8c936294c5: a decode
// step expressed as a chain of kernels (81 for Llama-3.2-1B: 5 per layer x 16 layers + 1) where
// each edge uses programmatic dependent launch, so a successor's prologue may overlap its
// predecessor but its body waits on griddepcontrol.wait. No upstream code is copied.
//
// SKELETON ONLY: constants. No hardware run, no tuning, no speed claim; included only under
// FAK_CUDA_PDL_CHAIN. The Go oracle in pdl_chain_test.go reads these #defines.
#ifndef FAK_PDL_CHAIN_CUH
#define FAK_PDL_CHAIN_CUH

#define FAK_PDL_KERNELS_PER_LAYER 5
#define FAK_PDL_TAIL_KERNELS 1
#define FAK_PDL_EDGE_FULL 0  // successor launches after predecessor completes
#define FAK_PDL_EDGE_PDL 1   // successor prologue overlaps; body waits on dependency sync

#endif  // FAK_PDL_CHAIN_CUH
