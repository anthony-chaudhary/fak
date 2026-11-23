//go:build windows

package compute

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func writeWindowsDirectFixture(t *testing.T, size int) (string, []byte) {
	t.Helper()
	data := make([]byte, size)
	for i := range data {
		data[i] = byte((i * 131) ^ (i >> 7))
	}
	name := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(name, data, 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return name, data
}

func TestDirectIOReaderWindowsUnbuffered(t *testing.T) {
	t.Run("direct opens unbuffered and cached opens random access", func(t *testing.T) {
		var flags []uint32
		open := func(_ string, f uint32) (*os.File, error) {
			flags = append(flags, f)
			return nil, nil
		}
		if _, err := openWindowsWeightFile(open, "w", true); err != nil {
			t.Fatal(err)
		}
		if _, err := openWindowsWeightFile(open, "w", false); err != nil {
			t.Fatal(err)
		}
		if len(flags) != 2 || flags[0] != windows.FILE_FLAG_NO_BUFFERING || flags[1] != windows.FILE_FLAG_RANDOM_ACCESS {
			t.Fatalf("flags = %#x, want [NO_BUFFERING RANDOM_ACCESS]", flags)
		}
	})

	t.Run("a refused unbuffered open falls back to the cached open", func(t *testing.T) {
		name, _ := writeWindowsDirectFixture(t, 4096)
		var flags []uint32
		refusing := func(n string, f uint32) (*os.File, error) {
			flags = append(flags, f)
			if f&windows.FILE_FLAG_NO_BUFFERING != 0 {
				return nil, &os.PathError{Op: "open", Path: n, Err: windows.ERROR_INVALID_PARAMETER}
			}
			return openWindowsFile(n, f)
		}
		f, direct, err := openUnbufferedOrCached(refusing, name)
		if err != nil {
			t.Fatalf("fallback open: %v", err)
		}
		defer f.Close()
		if direct {
			t.Fatal("direct reported after the unbuffered open was refused")
		}
		if len(flags) != 2 || flags[1] != windows.FILE_FLAG_RANDOM_ACCESS {
			t.Fatalf("flags = %#x, want the refusal followed by a RANDOM_ACCESS open", flags)
		}
	})

	t.Run("any other open error is not swallowed by the fallback", func(t *testing.T) {
		calls := 0
		failing := func(n string, _ uint32) (*os.File, error) {
			calls++
			return nil, &os.PathError{Op: "open", Path: n, Err: windows.ERROR_ACCESS_DENIED}
		}
		if _, _, err := openUnbufferedOrCached(failing, "w"); !errors.Is(err, windows.ERROR_ACCESS_DENIED) || calls != 1 {
			t.Fatalf("err = %v after %d opens, want ACCESS_DENIED from the single direct attempt", err, calls)
		}
		missing := filepath.Join(t.TempDir(), "absent.bin")
		if _, err := openDirectIOFile(missing); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("open of a missing file: err = %v, want not-exist", err)
		}
	})

	t.Run("a refused direct reader open yields the cached handle", func(t *testing.T) {
		name, data := writeWindowsDirectFixture(t, 5000)
		refusing := func(n string, f uint32) (*os.File, error) {
			if f&windows.FILE_FLAG_NO_BUFFERING != 0 {
				return nil, &os.PathError{Op: "open", Path: n, Err: windows.ERROR_INVALID_PARAMETER}
			}
			return openWindowsFile(n, f)
		}
		r, direct, err := openDirectIOReaderAt(refusing, name)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer r.Close()
		if _, wrapped := r.(*unbufferedReaderAt); direct || wrapped {
			t.Fatalf("refused open returned direct=%v %T, want the cached file", direct, r)
		}
		got := make([]byte, 1000)
		if n, err := r.ReadAt(got, 123); err != nil || n != len(got) || !bytes.Equal(got, data[123:1123]) {
			t.Fatalf("cached read: n=%d err=%v", n, err)
		}
	})

	t.Run("the plain file seam stays readable at any range", func(t *testing.T) {
		name, data := writeWindowsDirectFixture(t, 3*windowsDirectIOAlignment+77)
		f, err := openDirectIOFile(name)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer f.Close()
		r, err := NewDirectIOReaderFromFile(f, true)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]byte, 1000)
		if n, err := r.ReadAt(got, 100); err != nil || n != len(got) || !bytes.Equal(got, data[100:1100]) {
			t.Fatalf("unaligned read through the file seam: n=%d err=%v", n, err)
		}
		all := make([]byte, len(data))
		if n, err := io.ReadFull(r, all); err != nil || n != len(data) || !bytes.Equal(all, data) {
			t.Fatalf("sequential read through the file seam: n=%d err=%v", n, err)
		}
	})

	t.Run("an unbuffered handle serves arbitrary ranges through one alignment", func(t *testing.T) {
		const align = windowsDirectIOAlignment
		// An unaligned size puts the end of file inside an aligned span.
		name, data := writeWindowsDirectFixture(t, 5*align+1234)
		opened, direct, err := openDirectIOReaderAt(openWindowsFile, name)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer opened.Close()
		if !direct {
			t.Skip("this volume refuses unbuffered opens; the fallback is pinned above")
		}
		u, ok := opened.(*unbufferedReaderAt)
		if !ok {
			t.Fatalf("a direct open returned %T, want the aligning reader", opened)
		}
		f := u.f

		// The one constant must cover what this volume and this machine demand.
		root, err := windows.UTF16PtrFromString(filepath.VolumeName(name) + `\`)
		if err != nil {
			t.Fatal(err)
		}
		var sectorsPerCluster, bytesPerSector, freeClusters, totalClusters uint32
		getDiskFreeSpace := windows.NewLazySystemDLL("kernel32.dll").NewProc("GetDiskFreeSpaceW")
		if ok, _, err := getDiskFreeSpace.Call(uintptr(unsafe.Pointer(root)),
			uintptr(unsafe.Pointer(&sectorsPerCluster)), uintptr(unsafe.Pointer(&bytesPerSector)),
			uintptr(unsafe.Pointer(&freeClusters)), uintptr(unsafe.Pointer(&totalClusters))); ok == 0 {
			t.Fatalf("GetDiskFreeSpace: %v", err)
		}
		if align != 16<<10 || bytesPerSector == 0 || align%int(bytesPerSector) != 0 || align%os.Getpagesize() != 0 {
			t.Fatalf("alignment %d is not 16 KiB or not a multiple of sector %d and page %d", align, bytesPerSector, os.Getpagesize())
		}

		// The handle really is unbuffered: it refuses an unaligned read that a
		// cached handle would serve.
		if _, err := f.ReadAt(make([]byte, 1000), 1); !refusesDirectIO(err) {
			t.Fatalf("raw unaligned read: err = %v, want the unbuffered refusal", err)
		}

		r := NewDirectIOReader(u, int64(len(data)), align, true)
		if r.Fd() != f.Fd() {
			t.Fatalf("reader fd %v, want the unbuffered handle %v", r.Fd(), f.Fd())
		}
		for _, tc := range []struct {
			name      string
			off, size int
		}{
			{"inside one span", 100, 1000},
			{"across a span boundary", align - 7, 4096},
			{"aligned offset unaligned length", 2 * align, 333},
			{"the unaligned tail", 5*align + 1000, 234},
		} {
			got := make([]byte, tc.size)
			if n, err := r.ReadAt(got, int64(tc.off)); err != nil || n != tc.size {
				t.Fatalf("%s: n=%d err=%v", tc.name, n, err)
			}
			if !bytes.Equal(got, data[tc.off:tc.off+tc.size]) {
				t.Fatalf("%s: bytes differ", tc.name)
			}
		}

		// Past the end: a short read reports io.EOF, never the refusal a retry
		// from the unaligned end of file would raise.
		tail := make([]byte, 4096)
		n, err := r.ReadAt(tail, int64(len(data)-100))
		if n != 100 || err != io.EOF || !bytes.Equal(tail[:n], data[len(data)-100:]) {
			t.Fatalf("read across end of file: n=%d err=%v", n, err)
		}
		if n, err := r.ReadAt(tail, int64(len(data)+align)); n != 0 || err != io.EOF {
			t.Fatalf("read past end of file: n=%d err=%v", n, err)
		}

		// A fully aligned request is read in place, with no scratch copy.
		block := alignedWindowsBuffer(2 * align)
		if uintptr(unsafe.Pointer(&block[0]))%align != 0 {
			t.Fatal("aligned buffer is not aligned")
		}
		if n, err := u.ReadAt(block, align); err != nil || n != len(block) || !bytes.Equal(block, data[align:3*align]) {
			t.Fatalf("aligned in-place read: n=%d err=%v", n, err)
		}
		last := alignedWindowsBuffer(align)
		if n, err := u.ReadAt(last, 5*align); n != 1234 || err != io.EOF || !bytes.Equal(last[:n], data[5*align:]) {
			t.Fatalf("aligned read of the short last span: n=%d err=%v", n, err)
		}

		all := make([]byte, len(data))
		if _, err := r.Seek(0, io.SeekStart); err != nil {
			t.Fatal(err)
		}
		if n, err := io.ReadFull(r, all); err != nil || n != len(data) || !bytes.Equal(all, data) {
			t.Fatalf("sequential read: n=%d err=%v", n, err)
		}
	})
}
