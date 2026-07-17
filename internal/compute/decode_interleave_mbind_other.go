//go:build !(linux && amd64)

package compute

import "errors"

// errInterleaveUnsupported is returned when the mbind apply is compiled for a platform
// without the raw SYS_MBIND shim (anything but linux/amd64). The planner refuses with
// DecodeInterleaveUnsupported before this is reached on the auto/force paths, so it is a
// belt-and-suspenders guard rather than a live error path.
var errInterleaveUnsupported = errors.New("compute: NUMA interleave placement unsupported on this platform (linux/amd64 only)")

func mbindInterleave(region []byte, nodes []int) error { return errInterleaveUnsupported }
