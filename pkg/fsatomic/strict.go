package fsatomic

import (
	"os"
	"time"
)

// WriteFileAtomicStrict writes through a synced sibling temporary file and
// replaces path without ever deleting the existing destination first. A failed
// rename leaves the previous file intact, including on Windows.
func WriteFileAtomicStrict(path string, data []byte, perm os.FileMode) error {
	return WriteFileAtomicStrictWithRename(path, data, perm, nil)
}

// WriteFileAtomicStrictWithRename injects only the rename syscall. The caller's
// rename must preserve its destination on error, like os.Rename. A nil rename
// uses os.Rename; transient Windows refusals receive bounded retries.
func WriteFileAtomicStrictWithRename(path string, data []byte, perm os.FileMode, rename func(string, string) error) error {
	if rename == nil {
		rename = os.Rename
	}
	return WriteFileAtomicWithRename(path, data, perm, func(from, to string) error {
		return strictRenameWith(from, to, rename, isTransientRenameError, time.Sleep)
	})
}

func strictRenameWith(from, to string, rename func(string, string) error, transient func(error) bool, sleep func(time.Duration)) error {
	// No remove-and-rename fallback: losing the previous lease state is worse
	// than refusing a write after transient replacement contention.
	return renameRetrying(from, to, rename, transient, sleep)
}
