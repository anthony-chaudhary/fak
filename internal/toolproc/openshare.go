//go:build !windows

package toolproc

import "os"

// OpenShareDelete opens the journal at path for reading. On Windows the open
// must add FILE_SHARE_DELETE so a concurrent session's compaction rename
// (replaceFileAtomic) can supersede the file while this handle is open; the
// default os.Open share mode (READ|WRITE only) makes that rename fail with
// ERROR_ACCESS_DENIED (#3555). On POSIX a rename over an open file always
// succeeds, so this is plain os.Open.
func OpenShareDelete(path string) (*os.File, error) {
	return os.Open(path)
}

// OpenAppendShareDelete opens (creating if absent) the journal at path for
// append-only writes, with the same FILE_SHARE_DELETE requirement on Windows
// as OpenShareDelete: an appender's open window must not block a concurrent
// session's compaction rename (#3555).
func OpenAppendShareDelete(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
}

// renameOverOpenHandles renames oldpath over newpath even while another
// handle holds newpath open. On POSIX that is what rename(2) already does;
// the Windows implementation needs an explicit POSIX-semantics rename (#3555).
func renameOverOpenHandles(oldpath, newpath string) error {
	return os.Rename(oldpath, newpath)
}
