//go:build !windows

package procguard

func collectCommitSnapshot(rootPID int) (CommitSnapshot, bool, string) {
	return CommitSnapshot{RootPID: rootPID}, false, "commit accounting unsupported on this platform"
}
