package fp4meta

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CUDACapabilities returns DefaultCapabilities extended with "nvidia/sm<sm>"
// only when compute's cuBLASLt NVFP4 policy admits that architecture at the
// given runtime cuBLAS version (major*10000+minor*100+patch). An sm_120 device
// on cuBLAS >= 12.8.4 is therefore recognized; below the floor, or on an
// unqualified architecture, the defaults are returned unchanged so a measured
// envelope still abstains with ReasonHardwareUnverified.
func CUDACapabilities(sm, cublasVersion int) Capabilities {
	caps := DefaultCapabilities()
	if !compute.NVFP4CUBLASLtEligible(sm, cublasVersion) {
		return caps
	}
	key := fmt.Sprintf("nvidia/sm%d", sm)
	if !contains(caps.Hardware, key) {
		caps.Hardware = append(caps.Hardware, key)
	}
	return caps
}
