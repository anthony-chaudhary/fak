package model

import (
	"errors"
	"io"
)

// mmap.go — the platform-neutral seam for memory-mapping a single-file safetensors
// checkpoint. The per-platform mmapOpen (mmap_unix.go for darwin/linux/BSD via stdlib
// syscall.Mmap; mmap_other.go elsewhere, incl. Windows) maps the file read-only and hands
// back the mapped bytes; safetensors.go slices each tensor's [start,end) directly out of that
// map (zero-copy) so a single-file checkpoint is never fully resident in the process heap —
// the single-file analogue of the per-shard free invariant in LoadSafetensorsQuantDir. Any
// platform without an mmap impl, or any map that fails, reports errMmapUnsupported and the
// loader falls back to os.Open + per-tensor ReadAt (already streaming, just with a transient
// per-tensor heap copy).

// errMmapUnsupported signals that this platform cannot memory-map the file (or the map
// failed), so the caller should use the portable os.Open + ReadAt fallback. It is returned
// by the mmap_other.go stub and by the per-platform impls on any pre-map error.
var errMmapUnsupported = errors.New("model: mmap unsupported on this platform")

// closerFunc adapts a teardown closure to io.Closer so the per-platform mmapOpen can return
// its munmap/unmap-and-close logic without each platform declaring a bespoke struct.
type closerFunc func() error

// Close runs the wrapped teardown closure (the per-platform munmap/unmap-and-close)
// and returns its error.
func (f closerFunc) Close() error { return f() }

// MmapOpen maps path read-only (PROT_READ) and returns the mapped bytes plus a Closer
// that munmaps the region and closes the file. It is the exported seam over the
// per-platform mmapOpen so sibling weight loaders (internal/ggufload's SSD-offload
// shard readers, #2726/#2722) can share the one mmap implementation instead of growing
// their own per-platform build-tag pair. ok=false with a nil err means this
// platform/file simply cannot be mapped (mmap_other.go stub, zero-size file) — the
// caller should fall back to os.Open + ReadAt, exactly as openSafetensorsFile does. A
// non-nil err is a real mapping failure (open/stat/mmap syscall error) worth surfacing.
// The returned slice aliases OS-managed read-only memory: callers must only READ it and
// must not retain it past the Closer.
func MmapOpen(path string) (data []byte, closer io.Closer, ok bool, err error) {
	data, closer, err = mmapOpen(path)
	if err != nil {
		if errors.Is(err, errMmapUnsupported) {
			return nil, nil, false, nil
		}
		return nil, nil, false, err
	}
	return data, closer, true, nil
}
