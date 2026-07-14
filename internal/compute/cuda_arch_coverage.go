package compute

import "strconv"

// cuda_arch_coverage.go — the always-compiled, hardware-independent half of the CUDA (NVIDIA)
// backend's arch-admission logic (issue #4184). Before any kernel runs, a CUDA build must answer
// one honest question about the GPU in front of it: does THIS binary run NATIVELY here (an
// embedded SASS cubin matches), can the embedded PTX floor JIT-compile FORWARD onto the device,
// or is the card UNCOVERED (neither, so it will not run)? cuda.go answers this blindly today — it
// sets tier = "sm_" + deviceSM (cuda.go:95), so a single-arch sm_89 binary dropped on a Blackwell
// sm_100 card reports a confident bare "sm_100" and then silently JITs-or-dies. That decision needs
// no GPU to make: it follows from the cubin set this build embedded (cuda_arch.txt) plus the two
// fixed CUDA compatibility rules, so it is pure Go with no build tag, unit-witnessed on any host —
// the same split rocm_arch.go uses (ship the exact, host-tractable part; the on-device probe that
// feeds Classify a real (major,minor) is the //go:build cuda twin's job, deferred to a GPU node).
//
// The two rules this encodes, both forward-only:
//   - SASS (cubin) binary compatibility: a cubin built for sm_XY runs on a device of compute
//     capability X.Z iff the MAJOR matches and Z >= Y (minor forward-compat within a major). So an
//     sm_80 cubin covers 8.0–8.9 but never 9.x; there is no backward or cross-major SASS compat.
//   - PTX (JIT) compatibility: the embedded PTX floor compute_XY can be JIT-compiled by the driver
//     onto any device of compute capability >= X.Y, ACROSS majors (forward only). So a compute_120
//     floor JITs onto 12.0 and every later part, but NOT down onto an 11.x device.
// A single-arch build (FAK_CUDA_ARCH=<one target>) emits that one cubin and NO PTX floor, so its
// only reach is that cubin's native range — the case that makes the sm_89-on-Blackwell gap visible
// instead of silent.

// CUDACoverage is how a CUDA build reaches (or fails to reach) a given GPU.
type CUDACoverage uint8

const (
	// CUDAUncovered: no embedded cubin matches and the PTX floor cannot JIT onto the device — the
	// binary will not run here. The honest, fail-closed answer #4184 wants instead of a bare tier.
	CUDAUncovered CUDACoverage = iota
	// CUDANative: an embedded SASS cubin runs directly on the device (same major, device minor >=
	// cubin minor). The fast path — no driver JIT, no first-launch stall.
	CUDANative
	// CUDAJITFromPTX: no native cubin, but the embedded PTX floor JIT-compiles forward onto the
	// device (device cc >= PTX floor cc). Runs after a one-time driver JIT, not natively.
	CUDAJITFromPTX
)

// String returns the short coverage label.
func (c CUDACoverage) String() string {
	switch c {
	case CUDANative:
		return "native"
	case CUDAJITFromPTX:
		return "jit-from-ptx"
	default:
		return "uncovered"
	}
}

// cudaCC is a CUDA compute capability (major.minor): e.g. 8.9 = Ada, 9.0 = Hopper, 10.0 = Blackwell
// datacenter, 12.0 = Blackwell consumer. It is both a device's reported capability and a
// cubin/PTX compile target.
type cudaCC struct {
	Major int
	Minor int
}

// atLeast reports whether this capability is >= o, comparing major then minor. It is the PTX
// forward-JIT test (cross-major >=); the SASS test uses nativeCovers, which checks major equality.
func (c cudaCC) atLeast(o cudaCC) bool {
	if c.Major != o.Major {
		return c.Major > o.Major
	}
	return c.Minor >= o.Minor
}

// cudaEmbeddedArches is the SASS cubin set this build embeds, ascending, mirroring
// internal/compute/cuda_arch.txt (sm_80, sm_89, sm_90, sm_100, sm_120). TestCUDAArchCoverageMatchesFile
// asserts this Go table stays byte-equal to cuda_arch.txt so the two can never drift.
var cudaEmbeddedArches = []cudaCC{
	{8, 0},  // sm_80  — A100 (Ampere datacenter)
	{8, 9},  // sm_89  — Ada (L4/L40, RTX 40xx)
	{9, 0},  // sm_90  — H100/H200 (Hopper)
	{10, 0}, // sm_100 — Blackwell datacenter (B100/B200)
	{12, 0}, // sm_120 — Blackwell consumer (RTX 50xx)
}

// cudaPTXFloor is the embedded PTX virtual-arch floor (compute_120): the highest arch in
// cuda_arch.txt, matching build_cuda.sh's `-gencode arch=compute_${PTX_CC},code=compute_${PTX_CC}`
// where PTX_CC is the last entry of the file. The driver can JIT this PTX onto any device of
// compute capability >= this, across majors.
var cudaPTXFloor = cudaCC{12, 0}

// nativeCovers reports whether embedded cubin `cubin` runs natively on device `dev`: same major and
// device minor >= cubin minor (SASS minor-forward-compat within a major).
func nativeCovers(cubin, dev cudaCC) bool {
	return cubin.Major == dev.Major && dev.Minor >= cubin.Minor
}

// ClassifyCUDAArch reports how the full (all-cubins + PTX-floor) build reaches a device of the given
// compute capability: CUDANative if any embedded cubin runs directly, else CUDAJITFromPTX if the
// compute_120 PTX floor JITs forward onto it, else CUDAUncovered. This is the honest answer
// cuda.go:95's blind `tier = "sm_" + deviceSM` cannot give — e.g. an 11.x device is UNCOVERED (no
// cubin, and the 12.0 PTX floor is too high to JIT down to it), not a confident bare "sm_110".
func ClassifyCUDAArch(devMajor, devMinor int) CUDACoverage {
	return classifyCUDA(cudaCC{devMajor, devMinor}, cudaEmbeddedArches, true)
}

// ClassifyCUDAArchSingle reports coverage for a SINGLE-ARCH build (FAK_CUDA_ARCH=<arch>) that embeds
// only the one cubin sm_(archMajor.archMinor) and NO PTX floor: CUDANative iff that one cubin runs
// on the device, else CUDAUncovered — there is no PTX to JIT. This is the #4184 headline case: a
// single-arch sm_89 build on a Blackwell sm_100 (10.0) card is UNCOVERED, the fail-closed truth the
// blind tier string hides.
func ClassifyCUDAArchSingle(archMajor, archMinor, devMajor, devMinor int) CUDACoverage {
	return classifyCUDA(cudaCC{devMajor, devMinor}, []cudaCC{{archMajor, archMinor}}, false)
}

// classifyCUDA is the shared rule: native if any cubin in `cubins` covers `dev`, else — only when a
// PTX floor is present — JIT-from-PTX if the device is at or above the floor, else uncovered.
func classifyCUDA(dev cudaCC, cubins []cudaCC, ptxFloor bool) CUDACoverage {
	for _, c := range cubins {
		if nativeCovers(c, dev) {
			return CUDANative
		}
	}
	if ptxFloor && dev.atLeast(cudaPTXFloor) {
		return CUDAJITFromPTX
	}
	return CUDAUncovered
}

// KnownCUDAArches returns the embedded SASS cubin set as "sm_XY" strings in ascending order — the
// enumerator a `fak` diagnostic prints to say exactly which NVIDIA parts this build has native
// cubins for (the honest "does fak have a cubin for my card?", the CUDA twin of KnownROCmArches).
// The "sm_XY" spelling is the nvcc convention: the last digit is the minor, the rest the major, so
// {10,0} renders "sm_100" and {8,9} renders "sm_89".
func KnownCUDAArches() []string {
	out := make([]string, len(cudaEmbeddedArches))
	for i, c := range cudaEmbeddedArches {
		out[i] = "sm_" + strconv.Itoa(c.Major*10+c.Minor)
	}
	return out
}
