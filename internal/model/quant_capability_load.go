package model

// quant_capability_load.go — post-load quant-capability gate for issue #12983. The pre-load
// census in quant_kquant.go (QuantCapabilityCensus / QuantCapabilityRefusal) inspects the GGUF
// header BEFORE any weight is read and only runs on the Metal admission path. This file adds the
// LOAD-TIME twin: a census over the k-quants the loaded Model actually retains (m.kqw), so every
// serve load arm refuses a mixed-quant checkpoint whose retained bulk has no active backend
// kernel and no bounded host route, instead of reaching READY and hanging in unbounded CPU
// dequant on the first turn.
//
// The refusal type here is LoadQuantCapabilityRefusal: trunk's pre-load QuantCapabilityRefusal
// keeps its band-based shape for the admission path, and the two gates classify different inputs
// (header bands vs retained tensors), so they stay separate types with one shared vocabulary.

import (
	"fmt"
	"sort"
	"strings"
)

// allKQuantKinds enumerates every kQuantKind declared in the const block in quant_kquant.go.
// The census must classify each declared kind explicitly; an undeclared or out-of-range value
// fails closed as unsupported on every backend. Adding a kind to the const block REQUIRES
// adding it here (TestAllKQuantKindsIsExhaustive pins the count).
var allKQuantKinds = []kQuantKind{
	kindQ5K, kindQ6K, kindIQ3XXS, kindIQ4XS, kindIQ2XXS, kindIQ2XS,
	kindIQ1S, kindIQ2S, kindIQ1M, kindQ8_0, kindQ4_0, kindQ2K, kindQ3K, kindIQ3S, kindIQ4NL,
}

// metalResidentKQuantKinds are the k-quant kinds with a resident Metal DEVICE kernel: Q6_K and
// Q2_K run through the native Metal GEMM/GEMV dispatch (kQuantGemmDispatch /
// kQuantMatRowsIntoDispatch in metal_q4k_on.go). Every other kind falls back to the CPU batch
// path on Metal, so the device set is exactly these two.
var metalResidentKQuantKinds = map[kQuantKind]bool{
	kindQ6K: true,
	kindQ2K: true,
}

// hostBoundedKQuantKinds are the k-quant kinds whose host-CPU dequant path stays BOUNDED at
// aggregate scale on the Metal arm: the SDOT-accelerated K-quant families (Q5_K/Q6_K) plus the
// small fixed-size-block formats (Q2_K/Q4_0/Q8_0) whose scalar dequant is a few SIMD lanes per
// 32/256 weights. The IQ super-block family (and Q3_K) is the #12983 hazard: its dequant walks
// grid tables and quarter-scales across the whole expert bulk, which is what wedged the first
// turn of qwen38:27b-q2k. IQ4_NL is deliberately NOT listed: it is a new trunk kind with no
// dispatch evidence in this tree, so the gate fails closed on it until witnessed.
var hostBoundedKQuantKinds = map[kQuantKind]bool{
	kindQ5K:  true,
	kindQ6K:  true,
	kindQ2K:  true,
	kindQ8_0: true,
	kindQ4_0: true,
}

// QuantKindStat is one census row: how many retained tensors of one k-quant kind the loaded
// model holds and how many raw bytes they occupy.
type QuantKindStat struct {
	Name    string
	Kind    kQuantKind
	Tensors int
	Bytes   int64
}

// QuantDispatchRow records where one retained kind dispatches on the named backend.
type QuantDispatchRow struct {
	Name     string
	Dispatch string
	Tensors  int
	Bytes    int64
}

// QuantDispatchReceipt is the full post-load dispatch verdict for one backend: every retained
// kind classified as device-resident, bounded-host, or unsupported, with the bounded flag that
// decides whether the load may proceed.
type QuantDispatchReceipt struct {
	Backend     string
	Bounded     bool
	Rows        []QuantDispatchRow
	Unsupported []QuantKindStat
}

const (
	quantDispatchDevice      = "device"
	quantDispatchCPU         = "cpu"
	quantDispatchUnsupported = "unsupported"
)

func sortQuantKindStats(stats []QuantKindStat) {
	sort.Slice(stats, func(i, j int) bool {
		if stats[i].Bytes != stats[j].Bytes {
			return stats[i].Bytes > stats[j].Bytes
		}
		return stats[i].Name < stats[j].Name
	})
}

// QuantKindCensus counts the retained k-quant tensors by kind, sorted by total bytes descending
// (ties by name) so the largest retained bulk leads the receipt and the refusal text.
func (m *Model) QuantKindCensus() []QuantKindStat {
	if m == nil || len(m.kqw) == 0 {
		return nil
	}
	byKind := map[kQuantKind]*QuantKindStat{}
	for _, qt := range m.kqw {
		if qt == nil {
			continue
		}
		stat := byKind[qt.kind]
		if stat == nil {
			stat = &QuantKindStat{Name: qt.kind.String(), Kind: qt.kind}
			byKind[qt.kind] = stat
		}
		stat.Tensors++
		stat.Bytes += int64(len(qt.raw))
	}
	stats := make([]QuantKindStat, 0, len(byKind))
	for _, stat := range byKind {
		stats = append(stats, *stat)
	}
	sortQuantKindStats(stats)
	return stats
}

// isDeclaredKQuantKind reports whether kind is one of the declared kQuantKind values. An
// out-of-range sentinel must never be laundered into a servable verdict.
func isDeclaredKQuantKind(kind kQuantKind) bool {
	for _, k := range allKQuantKinds {
		if k == kind {
			return true
		}
	}
	return false
}

// QuantKindHasResidentKernel reports whether the named backend can serve the kind without an
// unbounded first-turn CPU dequant: Metal admits the device-kernel kinds plus the bounded host
// kinds; CPU admits every declared kind because the model-side dequant exists for all of them.
// An unknown backend fails closed.
func QuantKindHasResidentKernel(backend string, kind kQuantKind) bool {
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "metal":
		return metalResidentKQuantKinds[kind] || hostBoundedKQuantKinds[kind]
	case "cpu":
		return isDeclaredKQuantKind(kind)
	default:
		return false
	}
}

func quantDispatchClass(backend string, kind kQuantKind) string {
	if !isDeclaredKQuantKind(kind) {
		return quantDispatchUnsupported
	}
	switch strings.ToLower(strings.TrimSpace(backend)) {
	case "metal":
		if metalResidentKQuantKinds[kind] {
			return quantDispatchDevice
		}
		if hostBoundedKQuantKinds[kind] {
			return quantDispatchCPU
		}
		return quantDispatchUnsupported
	case "cpu":
		return quantDispatchCPU
	default:
		return quantDispatchUnsupported
	}
}

// QuantCapabilityReceipt classifies every retained k-quant kind against the named backend and
// reports whether the load is bounded (no unsupported kind retained). A model with no retained
// k-quants is trivially bounded.
func (m *Model) QuantCapabilityReceipt(backend string) *QuantDispatchReceipt {
	stats := m.QuantKindCensus()
	receipt := &QuantDispatchReceipt{
		Backend: backend,
		Bounded: true,
		Rows:    make([]QuantDispatchRow, 0, len(stats)),
	}
	for _, stat := range stats {
		class := quantDispatchClass(backend, stat.Kind)
		receipt.Rows = append(receipt.Rows, QuantDispatchRow{
			Name:     stat.Name,
			Dispatch: class,
			Tensors:  stat.Tensors,
			Bytes:    stat.Bytes,
		})
		if class == quantDispatchUnsupported {
			receipt.Bounded = false
			receipt.Unsupported = append(receipt.Unsupported, stat)
		}
	}
	return receipt
}

// LoadQuantCapabilityRefusal is the typed post-load refusal returned when a loaded model
// retains a quant kind the active backend cannot serve boundedly. It names EVERY offending kind
// with its tensor count so an operator sees the full unsupported inventory, not just the first
// hit. Trunk's pre-load QuantCapabilityRefusal (band-based, admission path) is a distinct type.
type LoadQuantCapabilityRefusal struct {
	Backend     string
	Unsupported []QuantKindStat
	Receipt     *QuantDispatchReceipt
}

func (e *LoadQuantCapabilityRefusal) Error() string {
	parts := make([]string, len(e.Unsupported))
	for i, s := range e.Unsupported {
		parts[i] = fmt.Sprintf("%s=%d tensors", s.Name, s.Tensors)
	}
	return fmt.Sprintf("model: refusing load: %d quant kind(s) have no active backend kernel or bounded host route on %s (%s); first turn would hang in CPU dequant (issue #12983)",
		len(e.Unsupported), e.Backend, strings.Join(parts, ", "))
}

// RefuseUnsupportedQuants returns the typed refusal when the loaded model retains a quant kind
// the named backend cannot serve boundedly, or nil when every retained kind dispatches to a
// device kernel or a bounded host route. Callers must convert the concrete pointer to error
// through a nil check (see cmd/fak serveRefuseUnsupportedQuants) so a bounded load never
// surfaces a non-nil error interface wrapping a nil pointer.
func (m *Model) RefuseUnsupportedQuants(backend string) *LoadQuantCapabilityRefusal {
	receipt := m.QuantCapabilityReceipt(backend)
	if receipt.Bounded {
		return nil
	}
	return &LoadQuantCapabilityRefusal{Backend: backend, Unsupported: receipt.Unsupported, Receipt: receipt}
}
