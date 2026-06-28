//go:build !linux

package compute

// loadPathFSMagic fails open on every non-Linux platform: macOS reports the filesystem by
// name (statfs Fstypename) rather than the numeric magic this probe classifies, and Windows
// has no statfs at all. Returning known=false makes ProbeLoadPath yield LoadPathUnknown, so
// the load proceeds without a (possibly wrong) warning. The load-path tax this guards is a
// Linux lab-host concern (da33 NFS vs local NVMe, #1062); the dev/verify nodes never hit it.
func loadPathFSMagic(path string) (magic int64, known bool) {
	return 0, false
}
