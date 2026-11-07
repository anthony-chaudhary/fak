// The driver-package coverage types moved to pkg/gpubinding; these aliases keep
// the historical compute.* names for existing callers.
package compute

import (
	"github.com/anthony-chaudhary/fak/pkg/gpubinding"
)

// GPUDriverPackage answers, for an extracted NVIDIA driver package, which
// package INF lists each present GPU's hardware ID and which one Windows would
// rank first, plus the ordered pnputil plan that would bind every covered GPU.
type GPUDriverPackage = gpubinding.DriverPackage

// GPUPackageDevice is one present GPU's coverage by the package.
type GPUPackageDevice = gpubinding.PackageDevice
