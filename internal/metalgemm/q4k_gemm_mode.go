//go:build darwin && arm64 && cgo

package metalgemm

import (
	"strings"
	"sync"
	"sync/atomic"
)

// Q4KGEMMExecution identifies the exact Q4_K prefill kernel that reached Metal dispatch.
// NotExecuted means selection declined before a command buffer was created or output was touched.
type Q4KGEMMExecution int

const (
	Q4KGEMMNotExecuted Q4KGEMMExecution = iota
	Q4KGEMMExecutedScalar
	Q4KGEMMExecutedMM32
	Q4KGEMMExecutedM5CooperativeSMEM
	// Q4KGEMMExecutedSmallPGEMV is the small-prompt (2<=P<=20) batched multi-token GEMV route
	// (fak#13694): ceil(P/8) evenly split q4k_gemv_multi dispatches that stream each weight row
	// once per chunk instead of staging a mostly idle 64-token GEMM tile.
	Q4KGEMMExecutedSmallPGEMV
	// Q4KGEMMExecutedMulMM is the fak#13692 llama.cpp mul_mm-shaped q4k_mul_mm.
	Q4KGEMMExecutedMulMM
)

// String names the executed kernel for logs and receipts.
func (e Q4KGEMMExecution) String() string {
	switch e {
	case Q4KGEMMNotExecuted:
		return "none"
	case Q4KGEMMExecutedScalar:
		return "scalar"
	case Q4KGEMMExecutedMM32:
		return "mm32"
	case Q4KGEMMExecutedM5CooperativeSMEM:
		return "m5-cooperative-smem"
	case Q4KGEMMExecutedSmallPGEMV:
		return "smallp-gemv"
	case Q4KGEMMExecutedMulMM:
		return "mulmm"
	}
	return "unknown"
}

// Q4KGEMMIdentity binds the candidate selected for this shape to the kernel that actually
// reached Metal dispatch. Credit MM32 only when both fields are Q4KGEMMExecutedMM32; an
// unavailable optional pipeline is reported as requested MM32 / executed none.
type Q4KGEMMIdentity struct {
	Requested Q4KGEMMExecution
	Executed  Q4KGEMMExecution
}

// Q4KGEMMMode selects a Q4_K prefill kernel candidate. MM32 is shape-bounded: only exact P=32
// dispatches may execute it; every other prompt length executes the scalar kernel. The unavailable
// mode is a deterministic fail-closed witness and is never selected by production.
type Q4KGEMMMode int

const (
	Q4KGEMMModeScalar Q4KGEMMMode = iota
	Q4KGEMMModeMM32
	Q4KGEMMModeM5CooperativeSMEM
	// Q4KGEMMModeSmallPGEMV requests the 2<=P<=20 batched multi-token GEMV route. Outside that
	// band, or when the multi-token pipelines are unavailable, the scalar kernel executes and the
	// identity reports executed scalar (fail-closed to the proven kernel, never NotExecuted).
	Q4KGEMMModeSmallPGEMV
	// Q4KGEMMModeMulMM is the fak#13692 simdgroup-MMA prefill kernel (half-staged weights, 64x32
	// tile, one dispatch). It is the production default above the small-P band.
	Q4KGEMMModeMulMM
	Q4KGEMMModeMM32Unavailable              = -1
	Q4KGEMMModeM5CooperativeSMEMUnavailable = -2
	// Q4KGEMMModeSmallPGEMVUnavailable is the deterministic witness for the small-P fallback: it
	// requests the small-P route but the native side treats its pipelines as missing, so the
	// scalar kernel executes. Production selection never emits it.
	Q4KGEMMModeSmallPGEMVUnavailable = -3
	Q4KGEMMModeMulMMUnavailable      = -4
)

// String names the requested candidate for logs and receipts.
func (m Q4KGEMMMode) String() string {
	switch m {
	case Q4KGEMMModeScalar:
		return "scalar"
	case Q4KGEMMModeMM32:
		return "mm32"
	case Q4KGEMMModeM5CooperativeSMEM:
		return "m5-cooperative-smem"
	case Q4KGEMMModeSmallPGEMV:
		return "smallp-gemv"
	case Q4KGEMMModeMulMM:
		return "mulmm"
	case Q4KGEMMModeMM32Unavailable:
		return "mm32-unavailable"
	case Q4KGEMMModeM5CooperativeSMEMUnavailable:
		return "m5-cooperative-smem-unavailable"
	case Q4KGEMMModeSmallPGEMVUnavailable:
		return "smallp-gemv-unavailable"
	case Q4KGEMMModeMulMMUnavailable:
		return "mulmm-unavailable"
	}
	return "unknown"
}

// Q4KMulMMMinPrompt / Q6KMulMMMinPrompt are the smallest prompt lengths the production selector
// routes to the fak#13692 mul_mm kernels, from the on-GPU receipt
// docs/benchmarks/receipts/metalgemm-kquant-mulmm-m3pro-macos27.json (TestKQuantMulMMReceiptVsScalar):
// at every P >= the floor the Qwen2.5-7B per-layer projection time on mul_mm clears the fak#9937
// 1.10x gate against the best pre-#13692 kernel for that P (small-P GEMV inside its band, else
// scalar/naive). Q4_K mul_mm is 3-4.5x the scalar kernel already at P=8. For Q6_K the small v_proj
// alone stays faster on the naive/small-P kernels until P~20, but down_proj dominates the layer
// (1.63 ms vs 5.2-12.8 ms at P=8..20), so the per-layer floor is also 8. P=1 never reaches the
// GEMM path (the model routes it to the decode GEMV).
const (
	Q4KMulMMMinPrompt = 8
	Q6KMulMMMinPrompt = 8
)

// q4kMulMMDisabled is the process-local kill switch for the mul_mm default (SetGEMMUseMulMM).
var q4kMulMMDisabled atomic.Bool

// kquantMulMMReady caches mg_kquant_mulmm_ready (bit 0 Q4_K, bit 1 Q6_K) once the library is up,
// so the per-projection selector never pays a cgo call after the first successful probe.
var kquantMulMMReady struct {
	once sync.Once
	bits int
}

// q4kMulMMSelected reports whether the production selector routes a P-token Q4_K panel to mul_mm.
func q4kMulMMSelected(P int) bool {
	return P >= Q4KMulMMMinPrompt && !q4kMulMMDisabled.Load() && kquantMulMMBits()&1 != 0
}

var q4kUseMM atomic.Bool

// q4kUseM5 opts the panel regime (P>=64) into the wide-tile cooperative-SMEM candidate. It is
// default OFF and only ever set through SetGEMMUseM5, which requires the pinned device/version
// crossover to be present; until a sanctioned [HW-WITNESSED] ratio clears the >=1.10x gate the
// production selector never requests mode 2 (fak#9937 deliberately shipped it explicit-only).
var q4kUseM5 atomic.Bool

// q4kM5Crossover is the device/version/P-band-pinned routing table for the wide-tile
// cooperative-SMEM candidate (fak#9943/#13133/this leaf). A row admits mode 2 for one Apple GPU
// family + macOS major version + prompt-length band only after a physical on-silicon measurement
// showed the candidate/scalar ratio cleared the fak#9937 >=1.10x margin across that band's
// endpoints; MinRatio records the band's measured floor and Witness the source receipt path.
//
// The P band exists because the on-silicon M3 Pro receipt for fak#13124 measured a
// prompt-length-dependent effect (1.49x @ P=64 and 1.41x @ P=128 in the ticket's capture), so a
// row keyed on device identity alone would discard the one measurement we paid hardware time for
// and could not express a P where the candidate is a regression. A row now also carries
// [MinP, MaxP], and the admission predicate requires MinP <= P <= MaxP.
//
// Exactly one row is pinned: the physical Apple M3 Pro / macOS 26 receipt captured on the
// on-silicon M3 Pro box (date 2026-09-15, commit 97cae3629,
// TestQ4KCrossoverReceiptCandidateVsScalar), where the candidate/scalar on-GPU ratio measured
// 1.31-1.83x at P=64 and 1.44-1.56x at P=128 across repeated runs — every recorded sample
// clearing the fak#9937 >=1.10x margin. The band [64,128] is the interval whose BOTH endpoints
// were physically measured clear; the selector routes only panel GEMMs inside that band to mode 2
// on that device/OS. Every P outside a measured band (including every P above 128 until a receipt
// band covers it) and every other device (including other Apple families and macOS majors) stays
// fail-closed scalar because no row matches it.
//
// MinRatio is the MEASURED candidate/scalar floor over the band (fak#13133), transcribed
// conservatively from that receipt as the lower bound of each measured shape: min(1.31 @ P=64,
// 1.44 @ P=128) = 1.31. It is NOT the 1.10 gate constant. The gate remains the floor the row must
// clear — q4kM5CrossoverAt skips any row whose MinRatio is below 1.10 — so a candidate-kernel
// regression that drops the measured floor below the gate fails the witness (which re-derives the
// floor from the physical measurement) instead of silently keeping the row. The band, not the
// ratio, is what changes routing: MinRatio records the margin the band is admitted with, so the
// table carries the measured magnitude rather than an anonymous constant.
type q4kM5CrossoverRow struct {
	Family    string  // Metal device name prefix the row is pinned to (e.g. "Apple M3")
	OSVersion string  // leading macOS major version the row is pinned to (e.g. "26")
	MinP      int     // inclusive lower bound of the measured prompt-length band
	MaxP      int     // inclusive upper bound of the measured prompt-length band (0 = unbounded above)
	MinRatio  float64 // measured candidate/scalar floor over the band (must be >= 1.10)
	Witness   string  // path/commit of the sanctioned on-silicon receipt that justified the row
}

var q4kM5CrossoverTable = []q4kM5CrossoverRow{
	{Family: "Apple M3 Pro", OSVersion: "26", MinP: 64, MaxP: 128, MinRatio: 1.31,
		Witness: "docs/benchmarks/receipts/q4k-m5-crossover-banded-m3pro.json (TestQ4KCrossoverReceiptCandidateVsScalar @97cae3629)"},
	// fak#13692: the macOS 26 row never matched macOS 27, so every prompt fell through to scalar.
	// Re-measured on macOS 27.0 against the single-dispatch scalar kernel over P in
	// {64..2048}: 1.20-1.22x @64, 1.149-1.182x @128 across six loaded-host runs, then 1.02-1.09x
	// @256..2048, so the admitted band stays [64,128] and MinRatio is the lowest observed floor
	// rounded down. With the fak#13692 mul_mm default on, this row only routes when
	// SetGEMMUseMulMM(false).
	{Family: "Apple M3 Pro", OSVersion: "27", MinP: 64, MaxP: 128, MinRatio: 1.14,
		Witness: "docs/benchmarks/receipts/q4k-m5-crossover-banded-m3pro-macos27.json (TestQ4KCrossoverReceiptCandidateVsScalar, fak#13692)"},
}

// q4kRowCoversPrompt reports whether row's [MinP,MaxP] band includes an inclusive prompt length P.
// A MaxP of 0 means the band is open above MinP. The band is fail-closed: a negative/zero MinP, a
// MaxP below MinP (an inverted or zero-width declaration), or a P outside the band never matches,
// so an unmeasured or malformed band can never admit mode 2.
func q4kRowCoversPrompt(row q4kM5CrossoverRow, P int) bool {
	if row.MinP <= 0 || P < row.MinP {
		return false
	}
	if row.MaxP != 0 && (row.MaxP < row.MinP || P > row.MaxP) {
		return false
	}
	return true
}

// q4kM5CrossoverAt reports whether the device/version/P-band-pinned table admits mode 2 for a
// prompt of P tokens, i.e. at least one row matches this device+OS AND covers P in its measured
// band AND clears the fak#9937 >=1.10x routing margin. It is a pure function of the table +
// device identity + P so it can be asserted without a GPU. A P with no measured row — including a
// P above the largest measured band — fails closed.
func q4kM5CrossoverAt(deviceName, osVersion string, P int) bool {
	for _, row := range q4kM5CrossoverTable {
		if row.MinRatio < q4kM5CrossoverMargin {
			continue
		}
		if !strings.HasPrefix(deviceName, row.Family) {
			continue
		}
		if row.OSVersion != "" && !strings.HasPrefix(osVersion, row.OSVersion) {
			continue
		}
		if !q4kRowCoversPrompt(row, P) {
			continue
		}
		return true
	}
	return false
}

// q4kM5CrossoverMargin is the fak#9937 device-pinned routing gate: the candidate is encoded only
// where a physical measurement shows it is at least 10% faster than the scalar kernel.
const q4kM5CrossoverMargin = 1.10

func q4kGEMMModeForPrompt(P int) Q4KGEMMMode {
	// fak#13692: the mul_mm port is the default from Q4KMulMMMinPrompt up. It outranks the
	// small-P GEMV inside the overlap (on-GPU, Qwen2.5-7B shapes, P=8..20: gate/up 1.02 ms vs
	// 1.65-4.23 ms, down 1.47 vs 1.69-4.27 ms) and the older MM32/M5 candidates, which remain
	// reachable only with SetGEMMUseMulMM(false).
	if q4kMulMMSelected(P) {
		return Q4KGEMMModeMulMM
	}
	// Short prompts below the mul_mm floor (2<=P<=7 by default; 2<=P<=20 with mul_mm off,
	// fak#13694) take the batched multi-token GEMV, which streams each weight row once per
	// <=8-token chunk; a GEMM tile is mostly padding there.
	if q4kSmallPEligible(P) {
		return Q4KGEMMModeSmallPGEMV
	}
	// Exact-P32 keeps its shape-bounded MM32 candidate under FAK_Q4K_MM.
	if P == 32 && q4kUseMM.Load() {
		return Q4KGEMMModeMM32
	}
	// The widened-panel regime (P>=64, fak#13041) is the wide-tile candidate's envelope. It is
	// only requested when the operator opt-in is on AND the P-band-pinned crossover admits this
	// device/version at THIS prompt length; otherwise the scalar kernel remains the executed
	// identity (fail-closed). Threading P into the admit check is the point of fak#13133: a row
	// whose measured band does not cover P cannot promote the candidate, so an unmeasured P runs
	// scalar.
	if P >= 64 && q4kUseM5.Load() && q4kM5CrossoverAdmits(P) {
		return Q4KGEMMModeM5CooperativeSMEM
	}
	return Q4KGEMMModeScalar
}

// q4kM5CrossoverAdmits evaluates the pinned crossover against the live device for a prompt of P
// tokens. It is fail-closed: with no usable device identity, an empty table, or a P outside every
// measured band, it returns false and mode 2 is never encoded.
func q4kM5CrossoverAdmits(P int) bool {
	if !Available() {
		return false
	}
	name := DeviceName()
	if name == "" {
		return false
	}
	return q4kM5CrossoverAt(name, OSVersion(), P)
}

func q4kGEMMRequestedExecution(P int, mode Q4KGEMMMode) Q4KGEMMExecution {
	switch mode {
	case Q4KGEMMModeMulMM, Q4KGEMMModeMulMMUnavailable:
		return Q4KGEMMExecutedMulMM
	case Q4KGEMMModeM5CooperativeSMEM, Q4KGEMMModeM5CooperativeSMEMUnavailable:
		return Q4KGEMMExecutedM5CooperativeSMEM
	case Q4KGEMMModeMM32, Q4KGEMMModeMM32Unavailable:
		if P == 32 {
			return Q4KGEMMExecutedMM32
		}
	case Q4KGEMMModeSmallPGEMV, Q4KGEMMModeSmallPGEMVUnavailable:
		if q4kSmallPPromptInBand(P) {
			return Q4KGEMMExecutedSmallPGEMV
		}
	}
	return Q4KGEMMExecutedScalar
}

func q4kGEMMIdentity(P int, mode Q4KGEMMMode, executed Q4KGEMMExecution) Q4KGEMMIdentity {
	return Q4KGEMMIdentity{
		Requested: q4kGEMMRequestedExecution(P, mode),
		Executed:  executed,
	}
}

// Q4KGEMMIdentityForMode returns the typed requested/executed identity for an explicit candidate.
// It is deterministic and does not inspect pipeline availability or create Metal work.
func Q4KGEMMIdentityForMode(P int, mode Q4KGEMMMode, executed Q4KGEMMExecution) Q4KGEMMIdentity {
	return q4kGEMMIdentity(P, mode, executed)
}

// Q4KGEMMRequestedExecution returns the shape-bounded kernel selected by the current opt-in.
// It is deterministic and does not inspect pipeline availability or create Metal work.
func Q4KGEMMRequestedExecution(P int) Q4KGEMMExecution {
	return q4kGEMMRequestedExecution(P, q4kGEMMModeForPrompt(P))
}

// Q4KGEMMModeForPrompt returns the production candidate for a prompt of P tokens under the current
// process opt-ins AND the live device/version/P-band-pinned crossover. It is the exported selector
// the model-side graph encode uses: the small-P multi-token GEMV for 2<=P<=20 (default on,
// FAK_Q4K_SMALLP=0 opts out; fak#13694), scalar otherwise, exact-P32 MM32 under FAK_Q4K_MM, and the
// wide-tile cooperative-SMEM candidate only for P>=64 when FAK_Q4K_M5 is on AND the pinned
// crossover admits this device/OS at THIS P (a measured band must cover P; an unmeasured prompt
// length stays scalar). It creates no Metal work and mutates no state; callers pass the result to
// ProjectionGraph.SetQ4KGEMMMode, which is itself fail-closed.
func Q4KGEMMModeForPrompt(P int) Q4KGEMMMode { return q4kGEMMModeForPrompt(P) }

// SetGEMMUseMM selects the exact-P32 batched-GEMM candidate: true requests q4k_gemm_mm32 only
// when P==32, while P31/P33 and every other prompt length retain q4k_gemm. False is the default.
// The model layer flips this process-local opt-in from FAK_Q4K_MM.
func SetGEMMUseMM(on bool) {
	q4kUseMM.Store(on)
}

// SetGEMMUseM5 selects the widened-panel regime (P>=64) wide-tile cooperative-SMEM candidate
// (fak#13041 panel shapes). It is the compute-side twin of SetGEMMUseMM, but unlike MM32 it is
// ALSO gated at encode time by the device/version/P-band-pinned crossover table:
// q4kGEMMModeForPrompt requests mode 2 only for a P>=64 shape whose live device/OS has a row whose
// measured band COVERS this P and whose measured floor clears the fak#9937 >=1.10x routing margin.
// With no row covering P for the live device the opt-in stays inert and the scalar kernel is the
// executed identity, so flipping the opt-in on can neither promote an unreceipted device nor
// promote a prompt length no physical measurement covers (fak#13133). The model layer now defaults
// this process-local opt-in ON (FAK_Q4K_M5=0 forces it off) once the sanctioned on-silicon M3 Pro
// receipt pinned a banded row (fak#13124/#13133); the crossover gate remains the real safety.
func SetGEMMUseM5(on bool) {
	q4kUseM5.Store(on)
}

// SetGEMMUseMulMM is the process-local switch for the fak#13692 mul_mm default (Q4_K and Q6_K).
// true (the default) routes P >= Q4KMulMMMinPrompt to mul_mm wherever the pipelines compiled;
// false restores the pre-#13692 selector (scalar / MM32 / M5 crossover, naive Q6_K).
func SetGEMMUseMulMM(on bool) { q4kMulMMDisabled.Store(!on) }

// GEMMUseMulMM reports whether the mul_mm default is enabled (it does not probe the pipelines).
func GEMMUseMulMM() bool { return !q4kMulMMDisabled.Load() }

// GEMMUseM5 reports whether the wide-tile opt-in is on. It does not consult the crossover table.
func GEMMUseM5() bool { return q4kUseM5.Load() }

// Q4KM5CrossoverAdmits reports whether the pinned crossover admits the wide-tile candidate on the
// live device/OS for a prompt of P tokens. It is the exported, side-effect-free view of the
// encode-time gate: false (the default, and the only value until a sanctioned on-silicon receipt
// pins a row covering P) means mode 2 is never requested even under the SetGEMMUseM5 opt-in. A P
// outside every measured band — including an unmeasured long prompt — is fail-closed.
func Q4KM5CrossoverAdmits(P int) bool { return q4kM5CrossoverAdmits(P) }

// Q4KM5CrossoverRowCount returns the number of pinned rows currently in the routing table. It is
// 1 once the sanctioned M3 Pro on-silicon receipt has pinned a row, and 0 before (or if the
// measured margin falls back below the gate).
func Q4KM5CrossoverRowCount() int { return len(q4kM5CrossoverTable) }

// Q4KM5CrossoverPredicate evaluates the device/version/P-band pin against an explicit identity
// without touching the live device, so the gate can be asserted on any host. It returns whether
// the pinned table admits mode 2 for deviceName/osVersion at prompt length P.
func Q4KM5CrossoverPredicate(deviceName, osVersion string, P int) bool {
	return q4kM5CrossoverAt(deviceName, osVersion, P)
}

// Q4KM5CrossoverMinimumRatio is the fak#9937 routing gate (>=1.10x) that every pinned row must
// clear. Exposed so the witness can assert the gate value rather than a magic literal.
const Q4KM5CrossoverMinimumRatio = q4kM5CrossoverMargin
