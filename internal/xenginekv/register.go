package xenginekv

import (
	"fmt"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"

	// Import blob so its init() (the default RegionBackend) runs FIRST: this package's
	// opt-in registration is a last-wins override, so the ordering must be deterministic
	// — exactly the dependency storedrv declares for the same reason.
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// defaultArenaBytes is the co-resident region size used when FAK_XENGINE_KV is a bare
// truthy flag rather than an explicit byte count (256 MiB — a representative single-host
// KV/result working set; a real deployment sizes it to the imported engine region).
const defaultArenaBytes = 256 << 20

// Default is the live Arena once the package opts in (nil when inert). Exposed so a
// caller that already holds the backend (e.g. an engine adapter mapping a real KV
// region) can drive Evict/Clone directly.
var Default *Arena

// init performs the DELIBERATE, reviewed RegionBackend swap (architest
// regionBackendRole "xenginekv"): when FAK_XENGINE_KV opts in, an Arena becomes the
// single live Ref backend and a keyed page-out codec. OPT-IN — unset leaves the
// package inert and the content-addressed blob store stays the singleton RegionBackend,
// so every default build is unchanged. blob's init ran first (imported above), so this
// last-wins override is order-deterministic.
func init() {
	spec := os.Getenv("FAK_XENGINE_KV")
	if spec == "" {
		return // OPT-IN: unset => inert, blob stays the live RegionBackend
	}
	size := defaultArenaBytes
	if n, err := strconv.Atoi(spec); err == nil && n > 0 {
		size = n // an explicit byte count overrides the default region size
	}
	Default = NewArena(size)
	abi.RegisterRegionBackend(backend{Default})
	abi.RegisterPageOutBackend("xenginekv", backend{Default})
	fmt.Fprintf(os.Stderr, "fak: cross-engine KV co-residence arena -> %d bytes (id=xenginekv, cap=%s)\n", size, CapZeroCopy)
}
