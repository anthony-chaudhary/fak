package compute

import (
	"io/fs"
	"path"
	"strconv"
	"strings"
)

// cgroupV1UnlimitedFloor: cgroup v1 reports "no limit" as a page-rounded max int64
// (9223372036854771712), not a sentinel string; anything at or above 2^62 is treated as unlimited.
const cgroupV1UnlimitedFloor = int64(1) << 62

// clampFreeToCgroup lowers a host-wide free figure (MemAvailable) to the calling process's
// cgroup headroom when a memory limit applies. A container or systemd slice with memory.max set
// sees the whole box in /proc/meminfo, so without this clamp the CPU-offload fit admits
// allocations the cgroup OOM-killer will refuse. The clamp only ever lowers free.
func clampFreeToCgroup(free int64, fsys fs.FS) int64 {
	if avail, limited := cgroupMemoryAvailable(fsys); limited && avail < free {
		return avail
	}
	return free
}

// cgroupMemoryAvailable returns the tightest (limit - usage) across the process's memory cgroup
// and its ancestors, reading from fsys rooted at "/" (os.DirFS("/") in production; a MapFS in
// tests). limited is false when no level carries a limit (v2 "max", v1 page-rounded max int64)
// or no cgroup information is readable. Handles cgroup v2 (memory.max / memory.current) and v1
// (memory.limit_in_bytes / memory.usage_in_bytes under the memory controller mount).
func cgroupMemoryAvailable(fsys fs.FS) (avail int64, limited bool) {
	raw, err := fs.ReadFile(fsys, "proc/self/cgroup")
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(raw), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) != 3 {
			continue
		}
		if parts[0] == "0" && parts[1] == "" {
			if a, ok := cgroupWalk(fsys, "sys/fs/cgroup", parts[2], "memory.max", "memory.current"); ok {
				return a, true
			}
			continue
		}
		for _, ctrl := range strings.Split(parts[1], ",") {
			if ctrl == "memory" {
				if a, ok := cgroupWalk(fsys, "sys/fs/cgroup/memory", parts[2], "memory.limit_in_bytes", "memory.usage_in_bytes"); ok {
					return a, true
				}
			}
		}
	}
	return 0, false
}

// cgroupWalk walks from the cgroup leaf up to the mount root, returning the minimum headroom of
// every level that carries a limit. When the leaf directory is not visible (a cgroup namespace
// shows the host path while mounting only the namespace root), only the mount root is consulted.
func cgroupWalk(fsys fs.FS, mount, cgPath, limitFile, usageFile string) (int64, bool) {
	rel := strings.Trim(path.Clean("/"+cgPath), "/")
	if rel != "" {
		if _, err := fs.Stat(fsys, path.Join(mount, rel)); err != nil {
			rel = ""
		}
	}
	var best int64
	found := false
	for {
		dir := mount
		if rel != "" {
			dir = path.Join(mount, rel)
		}
		if limit, ok := readCgroupInt(fsys, path.Join(dir, limitFile)); ok && limit < cgroupV1UnlimitedFloor {
			usage, _ := readCgroupInt(fsys, path.Join(dir, usageFile))
			head := limit - usage
			if head < 0 {
				head = 0
			}
			if !found || head < best {
				best, found = head, true
			}
		}
		if rel == "" {
			return best, found
		}
		if i := strings.LastIndex(rel, "/"); i >= 0 {
			rel = rel[:i]
		} else {
			rel = ""
		}
	}
}

// readCgroupInt parses a single-integer cgroup file; "max" (and anything unparseable) is not ok.
func readCgroupInt(fsys fs.FS, name string) (int64, bool) {
	raw, err := fs.ReadFile(fsys, name)
	if err != nil {
		return 0, false
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil || v < 0 {
		return 0, false
	}
	return v, true
}
