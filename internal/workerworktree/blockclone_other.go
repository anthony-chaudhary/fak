//go:build !windows && !darwin && !linux

package workerworktree

import "errors"

func nativeIsolationBackend() IsolationBackend { return gitWorktree{} }
func probeBlockClone(string) error {
	return errors.New("block cloning is unavailable on this platform")
}
func cloneFileBlocks(string, string) error {
	return errors.New("block cloning is unavailable on this platform")
}
