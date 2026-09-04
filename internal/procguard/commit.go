package procguard

import (
	"strconv"
	"strings"
)

// MemoryMetric names the operating-system quantity represented by a
// MemorySnapshot. Commit charge and resident set size are different metrics and
// must never share a metric-specific field or receipt label.
type MemoryMetric string

const (
	MemoryMetricCommit MemoryMetric = "commit"
	MemoryMetricRSS    MemoryMetric = "rss"

	// DefaultSystemCommitHeadroomBytes is the reserve managed worker launches
	// keep below the operating-system commit limit. Guard monitoring and launch
	// preflight must derive policy from this one value.
	DefaultSystemCommitHeadroomBytes = uint64(16) << 30
)

const SystemCommitHeadroomReason = "SYSTEM_COMMIT_HEADROOM"

// SystemCommitHeadroom is the side-effect-free admission result shared by the
// launch-time check and the running child guard.
type SystemCommitHeadroom struct {
	Supported              bool
	Refuse                 bool
	Reason                 string
	ObservedBytes          uint64
	RequiredBytes          uint64
	SystemBytes            uint64
	SystemLimit            uint64
	PhysicalAvailableBytes uint64
	PhysicalTotalBytes     uint64
}

// RequiredSystemCommitHeadroom reads the guard's positive-megabyte override.
// Empty, malformed, zero, signed, and overflowing values preserve the default;
// an invalid setting can never silently disable containment.
func RequiredSystemCommitHeadroom(getenv func(string) string) uint64 {
	raw := strings.TrimSpace(getenv("FAK_SYSTEM_COMMIT_HEADROOM_MB"))
	mb, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || mb == 0 || mb > ^uint64(0)>>20 {
		return DefaultSystemCommitHeadroomBytes
	}
	return mb << 20
}

// EvaluateSystemCommitHeadroom applies the exact guard boundary to one
// metric-typed snapshot. Unsupported metrics/zero limits abstain, matching the
// guard's platform contract. At the boundary (observed == required) admission
// refuses so a child cannot consume the reserve itself.
func EvaluateSystemCommitHeadroom(snapshot MemorySnapshot, required uint64) SystemCommitHeadroom {
	result := SystemCommitHeadroom{
		Supported:              snapshot.Metric == MemoryMetricCommit && snapshot.SystemLimit > 0,
		RequiredBytes:          required,
		SystemBytes:            snapshot.SystemBytes,
		SystemLimit:            snapshot.SystemLimit,
		PhysicalAvailableBytes: snapshot.HostPhysicalAvailableBytes,
		PhysicalTotalBytes:     snapshot.HostPhysicalBytes,
	}
	if snapshot.SystemLimit >= snapshot.SystemBytes {
		result.ObservedBytes = snapshot.SystemLimit - snapshot.SystemBytes
	}
	if result.Supported && required > 0 && result.ObservedBytes <= required {
		result.Refuse = true
		result.Reason = SystemCommitHeadroomReason
	}
	return result
}

// MemorySnapshot is a point-in-time accounting of one owned process tree.
// TreeBytes and each process Bytes use Metric's semantics. SystemBytes and
// SystemLimit are populated only when the platform exposes the same metric at
// system scope (Windows commit charge); HostPhysicalBytes is informational and
// is not treated as current RSS usage.
type MemorySnapshot struct {
	Metric                     MemoryMetric
	RootPID                    int
	TreeBytes                  uint64
	SystemBytes                uint64
	SystemLimit                uint64
	HostPhysicalBytes          uint64
	HostPhysicalAvailableBytes uint64
	Processes                  []MemoryProcess
}

type MemoryProcess struct {
	PID         int
	PPID        int
	Name        string
	CommandLine string
	Bytes       uint64
}

// CollectMemorySnapshot returns native, metric-typed memory accounting for
// rootPID's descendant tree. supported=false means the platform has no native
// implementation and managed launchers must decide whether that is acceptable.
func CollectMemorySnapshot(rootPID int) (snapshot MemorySnapshot, supported bool, detail string) {
	return collectMemorySnapshot(rootPID)
}

// HostPhysicalMemoryBytes returns the host's installed physical memory when the
// platform exposes it. Callers use this only to size a ceiling, never as a claim
// about current system memory pressure.
func HostPhysicalMemoryBytes() (uint64, string) {
	return hostPhysicalMemoryBytes()
}

// CommitSnapshot is a point-in-time accounting of one owned process tree and
// system commit pressure. It is retained as the Windows-compatible API used by
// existing callers; Darwin RSS is available only through CollectMemorySnapshot
// so resident bytes cannot be mislabeled as commit charge.
type CommitSnapshot struct {
	RootPID           int
	TreeCommitBytes   uint64
	SystemCommitBytes uint64
	SystemCommitLimit uint64
	Processes         []CommitProcess
}

type CommitProcess struct {
	PID         int
	PPID        int
	Name        string
	CommandLine string
	CommitBytes uint64
}

// CollectCommitSnapshot returns native process commit accounting for rootPID's
// descendant tree plus system commit charge/limit. supported=false means the
// platform has no native implementation and managed launchers must decide
// whether that is acceptable.
func CollectCommitSnapshot(rootPID int) (snapshot CommitSnapshot, supported bool, detail string) {
	s, supported, detail := collectMemorySnapshot(rootPID)
	if !supported || s.Metric != MemoryMetricCommit {
		if detail == "" {
			detail = "commit accounting unsupported on this platform"
		}
		return CommitSnapshot{RootPID: rootPID}, false, detail
	}
	out := CommitSnapshot{
		RootPID:           s.RootPID,
		TreeCommitBytes:   s.TreeBytes,
		SystemCommitBytes: s.SystemBytes,
		SystemCommitLimit: s.SystemLimit,
		Processes:         make([]CommitProcess, 0, len(s.Processes)),
	}
	for _, p := range s.Processes {
		out.Processes = append(out.Processes, CommitProcess{
			PID: p.PID, PPID: p.PPID, Name: p.Name, CommandLine: p.CommandLine, CommitBytes: p.Bytes,
		})
	}
	return out, true, detail
}
