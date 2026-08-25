package procguard

// CommitSnapshot is a point-in-time accounting of one owned process tree and
// system commit pressure. Values are bytes. RootPID is included in Processes.
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
	return collectCommitSnapshot(rootPID)
}
