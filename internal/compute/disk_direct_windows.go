//go:build windows

package compute

// The unbuffered open, its cached random-access counterpart and the refusal
// fallback are adapted from magnitudedev/magnitude,
// inference/seismic/io/src/disk/platform.rs at commit
// shared by buffer, offset and length follows the upstream direct-read layout at
// re-expressed in Go against this package's direct-read seam.

import (
	"errors"
	"io"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

// windowsDirectIOAlignment is the one alignment an unbuffered read gives its
// buffer address, file offset and length. An unbuffered handle refuses any of
// the three that is not a multiple of the volume's logical sector size. The
// rule is the larger of that sector size and the VM page size; 16 KiB is the
// value upstream settled on across platforms and over-aligns the 512-byte and
// 4 KiB sectors Windows volumes present, trading a little extra read at each
// unaligned edge for one constant with no per-volume query.
const windowsDirectIOAlignment = 16 << 10

// windowsDirectIOMaxRead keeps one ReadFile call inside its 32-bit length while
// staying a multiple of the alignment.
const windowsDirectIOMaxRead = 1 << 30

// dropFilePagesOS is a no-op on Windows: there is no per-range cache advice, and
// an unbuffered handle never populated the cache in the first place.
func dropFilePagesOS(fd uintptr, offset, length int64) error {
	return nil
}

// windowsFileOpener opens name read-only with extra CreateFile flags. It is a
// parameter so the fallback decision can be exercised without a volume that
// refuses unbuffered opens.
type windowsFileOpener func(name string, flags uint32) (*os.File, error)

func openWindowsFile(name string, flags uint32) (*os.File, error) {
	path, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: name, Err: err}
	}
	// Shares delete as well, which os.Open does not, so a weight file stays
	// replaceable while it is read.
	const shareAll = windows.FILE_SHARE_READ | windows.FILE_SHARE_WRITE | windows.FILE_SHARE_DELETE
	h, err := windows.CreateFile(path, windows.GENERIC_READ, shareAll, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL|flags, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: name, Err: err}
	}
	return os.NewFile(uintptr(h), name), nil
}

// refusesDirectIO reports whether err is a filesystem declining unbuffered I/O
// rather than a storage failure.
func refusesDirectIO(err error) bool {
	return errors.Is(err, windows.ERROR_INVALID_PARAMETER)
}

// openWindowsWeightFile opens name past the cache manager when direct, and
// through it with read-ahead disabled otherwise: read-ahead past a requested
// weight would fill the cache with its neighbours.
func openWindowsWeightFile(open windowsFileOpener, name string, direct bool) (*os.File, error) {
	if direct {
		return open(name, windows.FILE_FLAG_NO_BUFFERING)
	}
	return open(name, windows.FILE_FLAG_RANDOM_ACCESS)
}

// openUnbufferedOrCached opens name unbuffered and reports direct=true, or, when
// the filesystem refuses unbuffered I/O, opens it through the cache and reports
// direct=false. Any other error is returned as is.
func openUnbufferedOrCached(open windowsFileOpener, name string) (f *os.File, direct bool, err error) {
	f, err = openWindowsWeightFile(open, name, true)
	if err == nil {
		return f, true, nil
	}
	if !refusesDirectIO(err) {
		return nil, false, err
	}
	f, err = openWindowsWeightFile(open, name, false)
	return f, false, err
}

// openDirectIOFile returns a handle any reader can use: through the cache with
// read-ahead off. A bare unbuffered handle refuses unaligned reads, so it is
// only handed out wrapped, by openDirectIOReaderAt.
func openDirectIOFile(name string) (*os.File, error) {
	return openWindowsWeightFile(openWindowsFile, name, false)
}

// windowsWeightReader is what a weight read needs of either kind of handle.
type windowsWeightReader interface {
	io.ReaderAt
	io.Closer
	Fd() uintptr
}

// openDirectIOReaderAt opens name for arbitrary-range reads past the cache
// manager, and reports direct=false when the filesystem refused and the reader
// goes through the cache instead.
func openDirectIOReaderAt(open windowsFileOpener, name string) (r windowsWeightReader, direct bool, err error) {
	f, direct, err := openUnbufferedOrCached(open, name)
	if err != nil {
		return nil, false, err
	}
	if direct {
		return &unbufferedReaderAt{f: f}, true, nil
	}
	return f, false, nil
}

// unbufferedReaderAt adapts an unbuffered handle to io.ReaderAt for arbitrary
// ranges by widening each read to windowsDirectIOAlignment.
type unbufferedReaderAt struct {
	f *os.File
}

func (u *unbufferedReaderAt) Fd() uintptr  { return u.f.Fd() }
func (u *unbufferedReaderAt) Close() error { return u.f.Close() }

// ReadAt reads len(p) bytes at off. A range that is already aligned in address,
// offset and length is read in place; any other is read through an aligned
// scratch buffer covering the enclosing aligned span.
func (u *unbufferedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("compute: negative unbuffered read offset")
	}
	if len(p) == 0 {
		return 0, nil
	}
	const align = windowsDirectIOAlignment
	if off%align == 0 && len(p)%align == 0 && uintptr(unsafe.Pointer(&p[0]))%align == 0 {
		n, err := u.readAligned(p, off)
		if err == nil && n < len(p) {
			err = io.EOF
		}
		return n, err
	}
	start := off - off%align
	span := int((off+int64(len(p))+align-1)/align*align - start)
	buf := alignedWindowsBuffer(span)
	filled, err := u.readAligned(buf, start)
	skip := int(off - start)
	n := 0
	if filled > skip {
		n = copy(p, buf[skip:filled])
	}
	if err == nil && n < len(p) {
		err = io.EOF
	}
	return n, err
}

// readAligned fills buf from the aligned offset start and returns how many
// bytes the file held. It issues each ReadFile itself: an unbuffered read that
// ends at an unaligned end of file must not be retried from that offset, which
// the handle would refuse instead of reporting end of file.
func (u *unbufferedReaderAt) readAligned(buf []byte, start int64) (int, error) {
	h := windows.Handle(u.f.Fd())
	filled := 0
	for filled < len(buf) {
		chunk := buf[filled:]
		if len(chunk) > windowsDirectIOMaxRead {
			chunk = chunk[:windowsDirectIOMaxRead]
		}
		at := start + int64(filled)
		ov := windows.Overlapped{Offset: uint32(at), OffsetHigh: uint32(at >> 32)}
		var done uint32
		err := windows.ReadFile(h, chunk, &done, &ov)
		filled += int(done)
		if err != nil {
			if errors.Is(err, windows.ERROR_HANDLE_EOF) {
				break
			}
			return filled, &os.PathError{Op: "read", Path: u.f.Name(), Err: err}
		}
		if int(done) < len(chunk) {
			break
		}
	}
	return filled, nil
}

// alignedWindowsBuffer returns size bytes whose address is a multiple of
// windowsDirectIOAlignment.
func alignedWindowsBuffer(size int) []byte {
	raw := make([]byte, size+windowsDirectIOAlignment)
	skip := 0
	if rem := uintptr(unsafe.Pointer(&raw[0])) % windowsDirectIOAlignment; rem != 0 {
		skip = windowsDirectIOAlignment - int(rem)
	}
	return raw[skip : skip+size : skip+size]
}
