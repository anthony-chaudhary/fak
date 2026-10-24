package compute

// nvfp4CUBLASLtPolicy reports whether an NVIDIA architecture and cuBLAS
// runtime are eligible for the NVFP4 cuBLASLt path, and whether heuristic
// candidate algorithm 66 must be skipped.
//
// cublasGetVersion uses CUBLAS_VERSION's major*10000+minor*100+patch encoding;
// CUBLAS_VER_BUILD is deliberately absent from that macro:
// https://docs.nvidia.com/cuda/cublas/index.html#cublasgetversion
//
// Primary release evidence for the architecture floors:
//   - CUDA 12.8 GA / cuBLAS 12.8.3.14 introduced microscaled FP4 for SM100+:
//     https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
//   - CUDA 12.8 Update 1 / cuBLAS 12.8.4.1 added Blackwell GeForce FP4:
//     https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html
//   - CUDA 12.9 GA / cuBLAS 12.9.0.13 added the SM103 compiler target:
//     https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
//
// The SM103 exclusion windows conservatively skip all stages of algorithm 66
// across NVIDIA's documented affected branches. cuBLAS 13.6.2 fixes the first
// branch, cuBLAS 13.7.0 regresses affected B300 NVFP4 cases, and CUDA Toolkit
// 13.4 Update 1 bundles fixed cuBLAS 13.8.0.4:
// https://docs.nvidia.com/cuda/cublas-patch-release-notes/#cublas-patch-release-13-6-2
// https://docs.nvidia.com/cuda/archive/13.4.2/cuda-toolkit-release-notes/index.html
//
// Official package versions include a fourth build component, but the runtime
// API omits it. This policy therefore assumes official NVIDIA release builds.
func nvfp4CUBLASLtPolicy(sm, libraryVersion int) (eligible, skipAlgo66 bool) {
	var minimum int
	switch sm {
	case 100:
		minimum = 120803
	case 103:
		minimum = 120900
	case 120:
		minimum = 120804
	default:
		return false, false
	}
	if libraryVersion < minimum {
		return false, false
	}
	if sm != 103 {
		return true, false
	}
	skipAlgo66 = libraryVersion >= 130501 &&
		(libraryVersion < 130602 || libraryVersion >= 130700 && libraryVersion < 130800)
	return true, skipAlgo66
}
