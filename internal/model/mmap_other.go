//go:build !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package model

import "io"

// mmapOpen is the fallback for every platform without a stdlib-only memory-map implementation
// here — Windows, js/wasm, plan9. It always reports errMmapUnsupported, so openSafetensorsFile
// uses the portable os.Open + per-tensor ReadAt streaming path. That path is already lean (one
// transient tensor buffer at a time, never the whole file resident) — it just keeps the
// per-tensor heap copy the unix mmap path elides. A Windows file-mapping impl is intentionally
// omitted: the only stdlib idiom (MapViewOfFile -> uintptr -> unsafe.Pointer) trips go vet's
// unsafeptr check, which `make ci` gates on; the zero-copy slicing logic it would feed is
// instead witnessed on every platform by the data-backed leg of TestLoadSafetensorsMmapMatchesReadFile.
func mmapOpen(path string) ([]byte, io.Closer, error) {
	return nil, nil, errMmapUnsupported
}
