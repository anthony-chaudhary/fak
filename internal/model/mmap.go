package model

import "errors"

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
