// The binding-doctor topology types moved to pkg/gpubinding; these aliases keep
// the historical compute.* names for existing callers.
package compute

import (
	"github.com/anthony-chaudhary/fak/pkg/gpubinding"
)

// GPUPnPHop is one ancestor of a GPU in the device tree, nearest first.
type GPUPnPHop = gpubinding.PnPHop

// GPUUSB4HostRouter is a USB4 host router (Windows PCI\USB4_MS_CM device).
type GPUUSB4HostRouter = gpubinding.USB4HostRouter
