//go:build cuda && windows

// cuda_windows.go — the native-Windows half of the `-tags cuda` cgo wiring (issue #481).
// It compiles ONLY under `-tags cuda` on GOOS=windows; the default `go build ./cmd/fak`
// excludes it (no `cuda` tag), so the shipped artifact stays one pure-Go binary
// (cuda.go's "DIRECTION.md rule 1 + reviewer check 3"). It carries NO Go logic of its own
// — the backend, the C ABI, and the registration all live in cuda.go (build tag `cuda`,
// every OS). This file's whole job is to layer the Windows-specific link configuration on
// top of that shared wiring so the SAME cuda_kernels.cu / cuda_backend.h seam links against
// a native Windows CUDA Toolkit instead of the WSL user-space toolchain (setup_cuda_wsl.sh).
//
// Why a separate file (and not edits to cuda.go): cuda.go is `//go:build cuda` for ALL
// operating systems, and its `#cgo LDFLAGS` already names the libraries (-lfakcuda -lcudart
// -lcublas -lstdc++ -lm). cgo accumulates the `#cgo` directives of every file in the package,
// so on a `cuda && windows` build cuda.go supplies the library NAMES and this file supplies
// WHERE Windows finds them — additive, with no change to the Linux/WSL path (a `#cgo windows`
// directive is inert on every non-Windows target, and this whole file is absent there).
//
// Toolkit paths come from the environment, not from a hardcoded `#cgo` literal. The default
// Windows CUDA install path —
//
//	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.Y
//
// contains spaces, which a `#cgo` directive cannot carry reliably; so the include/lib
// directories are injected at build time via CGO_CFLAGS / CGO_LDFLAGS by
// tools/build_cuda_windows.ps1. That is the SAME env-injection seam build_cuda.sh uses on
// Linux (it exports CGO_CFLAGS/CGO_LDFLAGS rather than baking paths into cuda.go). What this
// file CAN pin portably — no spaces — is the in-tree search dir for the nvcc-built kernel
// archive (libfakcuda.a, produced beside this file by the build script) and the package
// include dir for cuda_backend.h.
//
// Build + sign (a Windows host with the CUDA Toolkit + a code-signing cert):
//
//	pwsh tools/build_cuda_windows.ps1
//
// Signing is not cosmetic here: the WDAC / Smart-App-Control policy on the reference host
// refuses to fork/exec freshly compiled UNSIGNED binaries (the reason the suite runs in WSL
// today, see issue #481). The produced fak.exe must be Authenticode-signed before WDAC will
// let it — and any test binary it spawns — run. This file does not and cannot perform that
// step; it only makes the native link possible. The signing cert is parameterized via env in
// the build script and is never embedded here.
//
// Toolchain note: the cgo C compiler must understand the GNU-style `-l`/`-L` flags inherited
// from cuda.go (the mingw-w64 gcc Go documents for cgo on Windows, or clang in GNU-driver
// mode). build_cuda_windows.ps1 archives the nvcc object into libfakcuda.a to match cuda.go's
// `-lfakcuda`, so the one static-library name is consistent across both files.
package compute

/*
// Windows-only CUDA link configuration (layered onto cuda.go's shared `#cgo` directives).
// The CUDA Toolkit include/lib dirs are injected by tools/build_cuda_windows.ps1 through
// CGO_CFLAGS / CGO_LDFLAGS (the default install path has spaces a #cgo literal can't carry).
// Pinned here, portably and space-free: the package dir (cuda_backend.h) and the in-tree
// nvcc archive dir (libfakcuda.a). These duplicate cuda.go's ${SRCDIR} entries harmlessly —
// the compiler dedupes include dirs and the linker dedupes library search dirs.
#cgo windows CFLAGS: -I${SRCDIR}
#cgo windows LDFLAGS: -L${SRCDIR}
*/
import "C"

// cudaWindowsBuildTag is a compile-time provenance marker: its presence in a binary's symbol
// table witnesses that the native-Windows CUDA cgo wiring (this file, `-tags cuda` on
// GOOS=windows) was linked in — distinct from both the pure-Go default build (which omits this
// file) and the Linux/WSL `-tags cuda` build (which omits this file's `#cgo windows` paths).
//slop:keep symbol-table provenance marker, intentionally unreferenced by design
const cudaWindowsBuildTag = "cuda+windows-native"
