//go:build !cuda || !cgo

package compute

// SetCUDAQ4KPrefillArm has no device GEMM to switch in the non-cuda build. It still refuses an
// unknown or unimplemented arm, so a host sees the same typed errors either way.
func SetCUDAQ4KPrefillArm(arm CUDAQ4KPrefillArm) error { return CheckCUDAQ4KPrefillArm(arm, 0) }

// CUDAQ4KPrefillArmInUse reports the default arm in the non-cuda build.
func CUDAQ4KPrefillArmInUse() (CUDAQ4KPrefillArm, error) { return CUDAQ4KPrefillCompensated, nil }

// SetCUDAQ4KDecodeArm has no device GEMV to switch in the non-cuda build; it still refuses an
// unknown arm.
func SetCUDAQ4KDecodeArm(arm CUDAQ4KDecodeArm) error { return CheckCUDAQ4KDecodeArm(arm) }

// CUDAQ4KDecodeArmInUse reports the default arm in the non-cuda build.
func CUDAQ4KDecodeArmInUse() CUDAQ4KDecodeArm { return CUDAQ4KDecodeDefault }
