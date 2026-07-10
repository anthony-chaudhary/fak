//go:build windows

package toolproc

import (
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

// OpenShareDelete opens the journal at path for reading with share mode
// FILE_SHARE_READ|WRITE|DELETE. Go's os.Open shares only READ|WRITE, which
// makes a concurrent session's compaction rename (replaceFileAtomic's
// MoveFileEx(MOVEFILE_REPLACE_EXISTING)) fail with ERROR_ACCESS_DENIED for as
// long as the handle is open (#3555). Adding FILE_SHARE_DELETE lets the
// rename supersede the file out from under this handle; the handle keeps
// reading the superseded bytes, which for the journal is exactly the
// "observes the pre-compaction file, complete and fold-clean" case the swap
// already promises.
func OpenShareDelete(path string) (*os.File, error) {
	return openShareDelete(path, syscall.GENERIC_READ, syscall.OPEN_EXISTING)
}

// OpenAppendShareDelete opens (creating if absent) the journal at path for
// append-only writes with share mode FILE_SHARE_READ|WRITE|DELETE, for the
// same reason as OpenShareDelete (#3555). Access is FILE_APPEND_DATA without
// FILE_WRITE_DATA — the same access right Go's own syscall.Open derives from
// O_APPEND|O_WRONLY — so every write lands atomically at end-of-file.
func OpenAppendShareDelete(path string) (*os.File, error) {
	return openShareDelete(path, syscall.FILE_APPEND_DATA, syscall.OPEN_ALWAYS)
}

func openShareDelete(path string, access, createmode uint32) (*os.File, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	h, err := syscall.CreateFile(p, access,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, createmode, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(h), path), nil
}

var procSetFileInformationByHandle = syscall.NewLazyDLL("kernel32.dll").NewProc("SetFileInformationByHandle")

const (
	_DELETE                             = 0x00010000
	_FileRenameInfoEx                   = 22 // FILE_INFO_BY_HANDLE_CLASS
	_FILE_RENAME_FLAG_REPLACE_IF_EXISTS = 0x00000001
	_FILE_RENAME_FLAG_POSIX_SEMANTICS   = 0x00000002
)

// fileRenameInfoEx mirrors FILE_RENAME_INFO with the Flags arm of its leading
// union, as consumed by the FileRenameInfoEx information class. FileName is
// variable-length; the struct is a header over a larger buffer.
type fileRenameInfoEx struct {
	Flags          uint32
	RootDirectory  syscall.Handle
	FileNameLength uint32
	FileName       [1]uint16
}

// renameOverOpenHandles renames oldpath over newpath even while another
// handle holds newpath open. MoveFileEx (os.Rename) performs a legacy
// superseding rename, which Windows refuses with ERROR_ACCESS_DENIED whenever
// the destination has ANY open handle — even one sharing FILE_SHARE_DELETE
// (verified empirically on this host; #3555). Only a POSIX-semantics rename
// (FILE_RENAME_FLAG_POSIX_SEMANTICS, NTFS on Windows 10 1607+) supersedes an
// open destination, and then only if the destination's handles were opened
// with FILE_SHARE_DELETE — which is exactly what OpenShareDelete /
// OpenAppendShareDelete grant. A filesystem without POSIX rename semantics
// (e.g. FAT32) falls back to plain os.Rename, restoring the pre-#3555
// behavior there: fine uncontended, ERROR_ACCESS_DENIED under contention.
func renameOverOpenHandles(oldpath, newpath string) error {
	perr := renamePosixSemantics(oldpath, newpath)
	if perr == nil {
		return nil
	}
	if rerr := os.Rename(oldpath, newpath); rerr == nil {
		return nil
	}
	return perr
}

func renamePosixSemantics(oldpath, newpath string) error {
	linkErr := func(err error) error {
		return &os.LinkError{Op: "rename", Old: oldpath, New: newpath, Err: err}
	}
	// RootDirectory stays NULL, so FileName must be a full path.
	abs, err := filepath.Abs(newpath)
	if err != nil {
		return linkErr(err)
	}
	name, err := syscall.UTF16FromString(abs)
	if err != nil {
		return linkErr(err)
	}
	src, err := syscall.UTF16PtrFromString(oldpath)
	if err != nil {
		return linkErr(err)
	}
	h, err := syscall.CreateFile(src, _DELETE|syscall.SYNCHRONIZE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return linkErr(err)
	}
	defer syscall.CloseHandle(h)

	// Backing storage as []uint64 so the header cast is aligned.
	nameOff := unsafe.Offsetof(fileRenameInfoEx{}.FileName)
	size := nameOff + uintptr(len(name)*2)
	buf := make([]uint64, (size+7)/8)
	info := (*fileRenameInfoEx)(unsafe.Pointer(&buf[0]))
	info.Flags = _FILE_RENAME_FLAG_REPLACE_IF_EXISTS | _FILE_RENAME_FLAG_POSIX_SEMANTICS
	info.FileNameLength = uint32((len(name) - 1) * 2) // bytes, excluding the NUL
	copy(unsafe.Slice((*uint16)(unsafe.Pointer(uintptr(unsafe.Pointer(&buf[0]))+nameOff)), len(name)), name)

	r1, _, e1 := procSetFileInformationByHandle.Call(
		uintptr(h), _FileRenameInfoEx,
		uintptr(unsafe.Pointer(&buf[0])), uintptr(size))
	if r1 == 0 {
		return linkErr(e1)
	}
	return nil
}
