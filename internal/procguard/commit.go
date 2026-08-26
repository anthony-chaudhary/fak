package procguard

// MemoryMetric names the operating-system quantity represented by a
// MemorySnapshot. Commit charge and resident set size are different metrics and
// must never share a metric-specific field or receipt label.
type MemoryMetric string

const (
	MemoryMetricCommit MemoryMetric = "commit"
	MemoryMetricRSS    MemoryMetric = "rss"
)

// MemorySnapshot is a point-in-time accounting of one owned process tree.
// TreeBytes and each process Bytes use Metric's semantics. SystemBytes and
// SystemLimit are populated only when the platform exposes the same metric at
// system scope (Windows commit charge); HostPhysicalBytes is informational and
// is not treated as current RSS usage.
type MemorySnapshot struct {
	Metric            MemoryMetric
	RootPID           int
	TreeBytes         uint64
	SystemBytes       uint64
	SystemLimit       uint64
	HostPhysicalBytes uint64
	Processes         []MemoryProcess
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
