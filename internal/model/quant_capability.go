package model

import (
	"sort"
	"strconv"
	"strings"
)

// quant_capability.go — the fail-closed LOAD gate for resident k-quant weights (issue #12983).
//
// The bug: a mixed-quant checkpoint (e.g. an unsloth UD mix) can land IQ3_XXS / IQ2_S / Q5_K /
// IQ1_M ... expert tensors in the resident kqw store. Only Q6_K and Q2_K have a resident DEVICE
// kernel on Metal; every other kQuantKind falls through to the host CPU dequant path
// (kQuantMatRowsInto / kQuantMatRowsIntoBatch in quant_kquant.go). On a large MoE that
// CPU-dequant coast is an UNBOUNDED first-turn hang: the model loads fine, then the first token
// never returns.
//
// This file makes that a load-time decision instead of a hang: refuse the load with a typed
// refusal that NAMES every offending kind, its tensor count, and the backend, so the operator
// learns "this checkpoint has no Metal kernel for these bands" before a serve wedges. On the CPU
// backend every declared kind has a model-side dequant function, so nothing is refused (bounded).
//
// SCOPE: only the resident kqw store (kQuantKind). q4kw / q8w / q2w are separate stores with
// their own resident kernels and are always served — they are not kQuantKind and never appear in
// this census. kindQ4K is likewise not a kQuantKind constant (it has its own q4kTensor store and
// Metal kernels), so it is deliberately absent from allKQuantKinds.

// allKQuantKinds enumerates EVERY declared kQuantKind, in declaration order. It is the single
// source of truth for "the kinds this gate must consider". We iterate this slice explicitly and
// never lean on kQuantKind.String(), whose default arm maps any unknown value to "Q5_K" — a
// silent lie that would hide an out-of-range kind from the census.
var allKQuantKinds = []kQuantKind{
	kindQ5K,
	kindQ6K,
	kindIQ3XXS,
	kindIQ4XS,
	kindIQ2XXS,
	kindIQ2XS,
	kindIQ1S,
	kindIQ2S,
	kindIQ1M,
	kindQ8_0,
	kindQ4_0,
	kindQ2K,
	kindQ3K,
	kindIQ3S,
}

// metalResidentKQuantKinds is the single source of truth for which kQuantKind values a Metal
// resident device kernel actually serves. It mirrors the dispatch switches in metal_q4k_on.go:
//
//   - kQuantGemmDispatch  (prefill GEMM)  — handles kindQ6K and kindQ2K
//   - kQuantMatRowsIntoDispatch (decode GEMV) — handles kindQ6K and kindQ2K
//
// Everything else falls through those switches to the CPU dequant (kQuantMatRowsIntoBatch /
// kQuantMatRowsInto), which is the unbounded first-turn hazard. Note this is deliberately NOT the
// same set as the Q8_0 Metal lane: Q8_0 tensors held in m.kqw are NOT handled by either dispatch
// above (the q8Tensor/m.q8w store is a separate Metal lane with its own q8GemmDispatch /
// q8MatRowsDispatch), so a Q8_0 entry in kqw still falls through to CPU and belongs in this
// refusal set. kindQ4K is intentionally not represented here: it is not a kQuantKind member (its
// q4kTensor store has its own kernels and is always served).
var metalResidentKQuantKinds = map[kQuantKind]bool{
	kindQ6K: true,
	kindQ2K: true,
}

// hostBoundedKQuantKinds are the kinds a Metal-arm serve may leave on the host CPU GEMV and still
// finish a first turn, when the resident band of that kind is a MINORITY of the model. The boundary
// is a policy about aggregate stream size, not a claim that these kinds are individually fast — the
// measured 4096x4096 f32 GEMV cost (this host, #12983 probe) is within ~1.1-1.5x across ALL kinds:
//
//	Q4_0 2.67ms  Q8_0 4.14ms  Q6_K 4.84ms  Q5_K 5.06ms  Q2_K 5.10ms  IQ3_XXS 5.09ms ... IQ2_S 16.9ms
//
// No kind is 100x another; the first-turn failure in #12983 is the AGGREGATE per-token stream of
// 372/520 compute-dominant tensors each taking a scalar f32 dequant per block — on kinds that have
// neither a Metal kernel nor a host SIMD reduction. The refused kinds are exactly those that, at
// that aggregate scale, have no accelerated host path at all.
//
// Membership rationale:
//   - kindQ6K / kindQ2K: also have a Metal kernel (metalResidentKQuantKinds), so a Metal-arm hit is
//     served on the device; listed here only so a kqw entry on a non-Metal arm is not refused.
//   - kindQ5K: the Q4_K-mix minority (ffn_down / lm_head) is 1-2 tensors, and Q5_K has an int8 SDOT
//     reduction (kQuantSDOTEnabled, quant_kquant_int8.go) — but note that path is OFF by default
//     (FAK_KQ_INT8 unset), so under the default envelope a Q5_K expert falls to the same scalar f32
//     dequant as the refused kinds. metal_q4k_on.go:448 documents Q5_K/Q6_K are DELIBERATELY left on
//     the CPU path. The membership therefore leans on the minority-band assumption; an artifact
//     whose Q5_K band is compute-dominant is NOT protected by this gate.
//   - kindQ4_0 / kindQ8_0: small fixed-size blocks (18B/32w, 34B/32w) with bounded dequant; Q8_0 has
//     no kqw Metal kernel and is correctly host-served.
var hostBoundedKQuantKinds = map[kQuantKind]bool{
	kindQ5K:  true,
	kindQ6K:  true,
	kindQ2K:  true,
	kindQ8_0: true,
	kindQ4_0: true,
}

// QuantKindStat is one row of a model's per-quant-kind census.
type QuantKindStat struct {
	Kind    kQuantKind `json:"-"`
	Name    string     `json:"name"`
	Tensors int        `json:"tensors"`
	Bytes   int64      `json:"bytes"`
}

// QuantDispatchRow is one resident weight band's dispatch verdict: the bytes/tensors held in that
// quant kind and which engine serves it — "device", "cpu", or "unsupported".
type QuantDispatchRow struct {
	Name     string `json:"name"`
	Tensors  int    `json:"tensors"`
	Bytes    int64  `json:"bytes"`
	Dispatch string `json:"dispatch"` // "device" | "cpu" | "unsupported"
}

// QuantDispatchReceipt records, for each resident quant kind present in the model, which dispatch
// (device kernel vs host CPU dequant) will serve it, so criterion 2 of #12983 is met: a receipt/
// test records which dispatch served each weight band.
type QuantDispatchReceipt struct {
	Backend     string             `json:"backend"` // "metal" or "cpu"
	Rows        []QuantDispatchRow `json:"rows"`
	Unsupported []QuantKindStat    `json:"unsupported"` // every kind named, sorted by Bytes desc then Name
	Bounded     bool               `json:"bounded"`     // true iff Unsupported is empty
}

// QuantKindCensus tallies the model's resident k-quant tensors by kind. Only m.kqw is walked:
// q4kw / q8w / q2w are not kQuantKind stores and are always kernel-served. Rows are sorted by
// Bytes desc, then Name. A nil or empty model returns an empty (non-nil) slice.
func (m *Model) QuantKindCensus() []QuantKindStat {
	stats := make(map[kQuantKind]*QuantKindStat)
	if m != nil {
		for _, qt := range m.kqw {
			if qt == nil {
				continue
			}
			s := stats[qt.kind]
			if s == nil {
				s = &QuantKindStat{Kind: qt.kind, Name: qt.kind.String()}
				stats[qt.kind] = s
			}
			s.Tensors++
			s.Bytes += int64(len(qt.raw))
		}
	}
	rows := make([]QuantKindStat, 0, len(stats))
	for _, s := range stats {
		rows = append(rows, *s)
	}
	sortQuantKindStats(rows)
	return rows
}

// sortQuantKindStats orders rows by Bytes desc, then Name asc — the deterministic order every
// receipt and refusal in this file uses.
func sortQuantKindStats(rows []QuantKindStat) {
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Bytes != rows[j].Bytes {
			return rows[i].Bytes > rows[j].Bytes
		}
		return rows[i].Name < rows[j].Name
	})
}

// QuantKindHasResidentKernel reports whether kind is SAFELY SERVABLE — it has either a resident
// device kernel or a bounded host path — on the named backend. backend is "metal" or "cpu".
//
//   - "cpu": every declared kQuantKind has a model-side dequant function (kQuantDequantSuperBlock
//     dispatches all of them), so every kind is servable. Nothing is refused on a pure-CPU serve.
//   - "metal": a kind is servable iff it has a Metal resident kernel (metalResidentKQuantKinds:
//     Q6_K / Q2_K) OR it is a hostBoundedKQuantKind. A kind with neither — the IQ family and Q3_K —
//     is refused, because that is the aggregate scalar-dequant stream that never completes the
//     first turn (#12983).
//
// An unrecognized backend is treated conservatively as "nothing servable" — the gate must fail
// CLOSED, never assume a kernel that does not exist.
func QuantKindHasResidentKernel(backend string, kind kQuantKind) bool {
	switch backend {
	case "cpu":
		return isDeclaredKQuantKind(kind)
	case "metal":
		return metalResidentKQuantKinds[kind] || hostBoundedKQuantKinds[kind]
	default:
		return false
	}
}

// isDeclaredKQuantKind reports whether kind is one of the explicitly enumerated kQuantKind
// members. This is the guard that keeps String()'s Q5_K default from laundering an unknown value
// into a "served" verdict.
func isDeclaredKQuantKind(kind kQuantKind) bool {
	for _, k := range allKQuantKinds {
		if k == kind {
			return true
		}
	}
	return false
}

// QuantCapabilityReceipt builds the dispatch receipt for the model under the named backend. Every
// resident k-quant kind appears exactly once in Rows, tagged "device" (a resident device kernel),
// "cpu" (a bounded host dequant/reduction) or "unsupported" (no device kernel AND no bounded host
// path — the #12983 first-turn hazard); Unsupported lists exactly the "unsupported" kinds, sorted
// by Bytes desc then Name. Bounded is true iff Unsupported is empty. A nil or empty model yields
// Bounded=true with no rows.
func (m *Model) QuantCapabilityReceipt(backend string) *QuantDispatchReceipt {
	r := &QuantDispatchReceipt{Backend: backend}
	census := m.QuantKindCensus()
	r.Rows = make([]QuantDispatchRow, 0, len(census))
	for _, stat := range census {
		dispatch := quantDispatchClass(backend, stat.Kind)
		if dispatch == quantDispatchUnsupported {
			r.Unsupported = append(r.Unsupported, stat)
		}
		r.Rows = append(r.Rows, QuantDispatchRow{
			Name:     stat.Name,
			Tensors:  stat.Tensors,
			Bytes:    stat.Bytes,
			Dispatch: dispatch,
		})
	}
	// Census is already Bytes-desc/Name-asc; the Unsupported subset preserves that order.
	r.Bounded = len(r.Unsupported) == 0
	return r
}

// Dispatch classes a receipt row can carry.
const (
	quantDispatchDevice      = "device"      // served by a resident device kernel
	quantDispatchCPU         = "cpu"         // served by a bounded host dequant/reduction
	quantDispatchUnsupported = "unsupported" // no kernel and no bounded host path
)

// quantDispatchClass is the per-kind dispatch verdict the receipt records. It is the precise form
// of QuantKindHasResidentKernel: a device kernel wins, then a bounded host path, else unsupported.
func quantDispatchClass(backend string, kind kQuantKind) string {
	if backend == "metal" && metalResidentKQuantKinds[kind] {
		return quantDispatchDevice
	}
	if QuantKindHasResidentKernel(backend, kind) {
		return quantDispatchCPU
	}
	return quantDispatchUnsupported
}

// QuantCapabilityRefusal is the typed, fail-closed load refusal naming EVERY quant kind present
// with no active backend kernel. Unsupported is never empty when this is returned.
type QuantCapabilityRefusal struct {
	Backend     string
	Unsupported []QuantKindStat
	Receipt     *QuantDispatchReceipt
}

// Error renders the refusal. It MUST name every offending type with its tensor count, so an
// operator sees exactly which weight bands have no kernel and how much is at stake, e.g.:
//
//	model: refusing load: 2 quant kind(s) have no metal kernel (IQ3_XXS=3 tensors, IQ2_S=2); first turn would hang in CPU dequant (issue #12983)
func (e *QuantCapabilityRefusal) Error() string {
	var b strings.Builder
	b.WriteString("model: refusing load: ")
	b.WriteString(strconv.Itoa(len(e.Unsupported)))
	b.WriteString(" quant kind(s) have no ")
	b.WriteString(e.Backend)
	b.WriteString(" kernel (")
	for i, u := range e.Unsupported {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(u.Name)
		b.WriteString("=")
		b.WriteString(strconv.Itoa(u.Tensors))
		b.WriteString(" tensors")
	}
	b.WriteString("); first turn would hang in CPU dequant (issue #12983)")
	return b.String()
}

// RefuseUnsupportedQuants returns a non-nil *QuantCapabilityRefusal iff any resident k-quant kind
// lacks a kernel on backend; nil otherwise (bounded, safe to serve). It is the fail-closed load
// gate: a nil return is the only verdict that clears a serve to proceed.
func (m *Model) RefuseUnsupportedQuants(backend string) *QuantCapabilityRefusal {
	receipt := m.QuantCapabilityReceipt(backend)
	if receipt.Bounded {
		return nil
	}
	return &QuantCapabilityRefusal{
		Backend:     backend,
		Unsupported: receipt.Unsupported,
		Receipt:     receipt,
	}
}
