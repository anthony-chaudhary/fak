//go:build vulkan && (windows || linux) && cgo

package compute

func (v *vulkanBackend) v41HostInputBorrowSafe(original Backend) bool {
	canonical, ok := original.(*vulkanBackend)
	// uploadClass's F32 activation arm allocates separate device storage and
	// completes fvk_h2d before returning. Neither it nor canonical Read retains
	// an alias to the caller's Go slice. This grants no device capability.
	return ok && v != nil && canonical == v
}
