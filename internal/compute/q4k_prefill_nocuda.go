//go:build !cuda || !cgo

package compute

// SetCUDAQ4KPrefillArm has no device GEMM to switch in the non-cuda build. It still refuses an
// unknown or unimplemented arm, so a host sees the same typed errors either way.
func SetCUDAQ4KPrefillArm(arm CUDAQ4KPrefillArm) error { return CheckCUDAQ4KPrefillArm(arm, 0) }

// CUDAQ4KPrefillArmInUse reports the default arm in the non-cuda build.
func CUDAQ4KPrefillArmInUse() (CUDAQ4KPrefillArm, error) { return CUDAQ4KPrefillCompensated, nil }
