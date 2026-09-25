package procguard

import "github.com/anthony-chaudhary/fak/internal/harnessres"

// ResourceSnapshot is one point-in-time resource observation for an owned
// process tree. I/O is cumulative; callers derive rates from successive
// snapshots. Presence bits are conservative: an axis is true only when every
// process in the tree produced that counter, so a partial read cannot look like
// a safe low value.
type ResourceSnapshot struct {
	RootPID          int
	RSSBytes         uint64
	CPUSeconds       float64
	ReadBytes        uint64
	WriteBytes       uint64
	ProcessCount     int
	HaveRSS          bool
	HaveCPU          bool
	HaveIO           bool
	HaveProcessCount bool
}

// CollectResourceSnapshot samples CPU, resident memory, and cumulative file
// I/O for the same descendant tree as CollectMemorySnapshot. It intentionally
// returns supported=true when the process census succeeds even if a platform
// lacks an individual counter; the corresponding Have bit remains false.
func CollectResourceSnapshot(rootPID int) (ResourceSnapshot, bool, string) {
	memory, supported, detail := CollectMemorySnapshot(rootPID)
	if !supported {
		return ResourceSnapshot{RootPID: rootPID}, false, detail
	}
	out := ResourceSnapshot{
		RootPID:          rootPID,
		ProcessCount:     len(memory.Processes),
		HaveProcessCount: true,
	}
	if len(memory.Processes) == 0 {
		return out, true, detail
	}
	allRSS, allCPU, allIO := true, true, true
	for _, process := range memory.Processes {
		usage, ok := harnessres.ReadProcessResource(process.PID)
		if !ok {
			allRSS, allCPU, allIO = false, false, false
			continue
		}
		if usage.HaveRSS {
			out.RSSBytes += usage.RSSBytes
		} else {
			allRSS = false
		}
		if usage.HaveCPU {
			out.CPUSeconds += usage.CPUSeconds
		} else {
			allCPU = false
		}
		if usage.HaveIO {
			out.ReadBytes += usage.IOReadBytes
			out.WriteBytes += usage.IOWriteBytes
		} else {
			allIO = false
		}
	}
	out.HaveRSS = allRSS
	out.HaveCPU = allCPU
	out.HaveIO = allIO
	return out, true, detail
}
