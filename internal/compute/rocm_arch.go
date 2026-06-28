package compute

// rocm_arch.go — the always-compiled, hardware-independent half of the ROCm (AMD Linux)
// backend (issue #266 / C-002). It is the device-arch taxonomy a HIP build needs BEFORE
// any kernel runs: which AMD GPU generations fak targets, whether each is a CDNA datacenter
// part or an RDNA consumer part, and the canonical LLVM AMDGPU target string
// (`gfxNNNN`) that `hipcc --offload-arch=<gfx>` compiles for. None of this needs an AMD GPU
// to be correct, so it is pure Go with no build tag and is unit-witnessed on any host — the
// same split PREFILL-B001-NOTES.md uses (ship the exact, host-tractable part; defer the
// device run). The cgo HIP backend itself (the `//go:build rocm` twin that registers an
// Approx backend named "rocm", mirroring cuda.go) lands on an AMD-on-Linux node where it can
// actually compile and be witnessed; see ROCM-C002-NOTES.md for that hand-off.
//
// Why a taxonomy and not a free-form string: hipcc must be told an EXACT offload target, and
// the CDNA/RDNA split is load-bearing for kernel tuning — CDNA/GCN parts execute a 64-lane
// wavefront and carry matrix cores (the MI Instinct datacenter line), RDNA parts execute a
// native 32-lane wavefront (the Radeon consumer line). Picking the wrong family silently
// mistunes occupancy and LDS. This table is the single place that mapping lives.

// ROCmFamily is an AMD GPU architecture generation, in ROCm/HIP terms.
type ROCmFamily uint8

const (
	// ROCmUnknown is the zero value: an arch string fak does not have a target for.
	ROCmUnknown ROCmFamily = iota
	// ROCmGCN5 is Vega 20 (gfx906, Radeon Instinct MI50/MI60) — a ROCm-supported
	// datacenter part that predates CDNA but still runs a 64-lane wavefront.
	ROCmGCN5
	// ROCmCDNA1 is gfx908 (Instinct MI100), the first CDNA datacenter generation.
	ROCmCDNA1
	// ROCmCDNA2 is gfx90a (Instinct MI200 — MI210/MI250/MI250X).
	ROCmCDNA2
	// ROCmCDNA3 is gfx942 (Instinct MI300A/MI300X).
	ROCmCDNA3
	// ROCmRDNA1 is gfx101x (Radeon RX 5000), the first RDNA consumer generation.
	ROCmRDNA1
	// ROCmRDNA2 is gfx103x (Radeon RX 6000).
	ROCmRDNA2
	// ROCmRDNA3 is gfx110x (Radeon RX 7000) — includes gfx1102, the RX 7600 the Vulkan
	// backend already runs on with numerical parity (docs/benchmarks/VULKAN-AMD-RESULTS.md).
	ROCmRDNA3
)

// String returns the short generation label.
func (f ROCmFamily) String() string {
	switch f {
	case ROCmGCN5:
		return "GCN5"
	case ROCmCDNA1:
		return "CDNA1"
	case ROCmCDNA2:
		return "CDNA2"
	case ROCmCDNA3:
		return "CDNA3"
	case ROCmRDNA1:
		return "RDNA1"
	case ROCmRDNA2:
		return "RDNA2"
	case ROCmRDNA3:
		return "RDNA3"
	default:
		return "unknown"
	}
}

// IsCDNA reports whether the family is one of the CDNA datacenter generations (the
// matrix-core Instinct line). GCN5 (gfx906) is datacenter but NOT CDNA — use Datacenter
// for the line distinction and IsCDNA for the matrix-core-architecture distinction.
func (f ROCmFamily) IsCDNA() bool { return f == ROCmCDNA1 || f == ROCmCDNA2 || f == ROCmCDNA3 }

// IsRDNA reports whether the family is one of the RDNA consumer generations.
func (f ROCmFamily) IsRDNA() bool { return f == ROCmRDNA1 || f == ROCmRDNA2 || f == ROCmRDNA3 }

// Datacenter reports whether the part is a server/Instinct GPU (GCN5 + every CDNA), as
// opposed to an RDNA consumer Radeon. The multi-GPU acceptance bullet (#266) is a
// datacenter concern; this is the predicate a fleet planner keys on.
func (f ROCmFamily) Datacenter() bool { return f == ROCmGCN5 || f.IsCDNA() }

// ROCmArch is one supported AMD compile target: its canonical LLVM AMDGPU id, its family,
// the native wavefront width hipcc tunes for, and a representative product.
type ROCmArch struct {
	GFX       string     // canonical `--offload-arch` token, e.g. "gfx90a"
	Family    ROCmFamily // generation
	Wavefront int        // native wavefront lanes: 64 on GCN/CDNA, 32 on RDNA
	Examples  string     // representative product(s)
}

// rocmArches is the supported-target table, declared once in generation order. Datacenter
// (Instinct, 64-lane) first, then consumer Radeon (RDNA, 32-lane).
var rocmArches = []ROCmArch{
	{GFX: "gfx906", Family: ROCmGCN5, Wavefront: 64, Examples: "Instinct MI50/MI60 (Vega 20)"},
	{GFX: "gfx908", Family: ROCmCDNA1, Wavefront: 64, Examples: "Instinct MI100"},
	{GFX: "gfx90a", Family: ROCmCDNA2, Wavefront: 64, Examples: "Instinct MI210/MI250/MI250X"},
	{GFX: "gfx942", Family: ROCmCDNA3, Wavefront: 64, Examples: "Instinct MI300A/MI300X"},
	{GFX: "gfx1010", Family: ROCmRDNA1, Wavefront: 32, Examples: "Radeon RX 5700 (XT)"},
	{GFX: "gfx1030", Family: ROCmRDNA2, Wavefront: 32, Examples: "Radeon RX 6800/6900"},
	{GFX: "gfx1032", Family: ROCmRDNA2, Wavefront: 32, Examples: "Radeon RX 6600 (XT)"},
	{GFX: "gfx1100", Family: ROCmRDNA3, Wavefront: 32, Examples: "Radeon RX 7900 XTX/XT"},
	{GFX: "gfx1102", Family: ROCmRDNA3, Wavefront: 32, Examples: "Radeon RX 7600 (Vulkan-witnessed)"},
}

// rocmByGFX indexes the table by canonical gfx id for O(1) lookup.
var rocmByGFX = func() map[string]ROCmArch {
	m := make(map[string]ROCmArch, len(rocmArches))
	for _, a := range rocmArches {
		m[a.GFX] = a
	}
	return m
}()

// normalizeGFX canonicalizes a device-reported arch string to a bare gfx id. ROCm reports
// targets with case noise and optional feature suffixes — e.g. "GFX90A", "gfx90a:sramecc+:xnack-",
// "gfx1100  " — that all denote the same compile target. It lowercases, trims, and drops the
// feature suffix at the first ':'. It does not invent a target: an unrecognized base id is
// returned as-is for Lookup to reject.
func normalizeGFX(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ':' { // strip target-feature suffix (":sramecc+:xnack-")
			break
		}
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' { // trim whitespace anywhere
			continue
		}
		if c >= 'A' && c <= 'Z' { // lowercase
			c += 'a' - 'A'
		}
		out = append(out, c)
	}
	return string(out)
}

// LookupROCmArch resolves a device-reported arch string (any case, optional ":feature"
// suffix) to its supported target, or (zero, false) if fak has no target for it. This is
// the fail-closed admission a build/runtime path uses so an unknown AMD part is never
// silently compiled for the wrong wavefront.
func LookupROCmArch(gfx string) (ROCmArch, bool) {
	a, ok := rocmByGFX[normalizeGFX(gfx)]
	return a, ok
}

// ROCmOffloadArch returns the canonical `hipcc --offload-arch=<gfx>` token for a
// device-reported arch string, or (\"\", false) if unsupported. The build script
// (build_rocm.sh, deferred to an AMD node) passes the result straight to hipcc.
func ROCmOffloadArch(gfx string) (string, bool) {
	a, ok := LookupROCmArch(gfx)
	if !ok {
		return "", false
	}
	return a.GFX, true
}

// KnownROCmArches returns the supported-target table in declared (generation) order. A
// `fak` diagnostic or the HIP build script enumerates it to print exactly which AMD parts
// this build of fak targets — the honest answer to "does fak support my card?".
func KnownROCmArches() []ROCmArch {
	out := make([]ROCmArch, len(rocmArches))
	copy(out, rocmArches)
	return out
}
