package vllmcompile

import (
	"errors"
	"fmt"
	"strings"
)

// Block is the `vllm_compile` artifact block recorded at benchmark start for one
// served-engine row. A bench artifact carries one Block per compared row (raw
// vLLM, fak-fronted vLLM, SGLang/llama when comparable) so no row is silently
// cold. Pointer fields distinguish "observed false" from "not observed": a nil
// CompileCacheEnabled means the harness never captured the cache state, which is
// itself gate-relevant — an unobserved cache cannot certify a tuned baseline.
type Block struct {
	// Engine names the served engine this block describes: "vllm",
	// "vllm+fak" (fak-fronted vLLM), "sglang", "llama", ...
	Engine string `json:"engine,omitempty"`

	// EngineCommit is the engine commit/version measured (e.g. a vLLM commit or
	// version string), pinning the baseline to a known build.
	EngineCommit string `json:"engine_commit,omitempty"`

	// CompileCacheEnabled reports whether the torch.compile artifact cache was
	// enabled. nil = not observed; *false = disabled (cold); *true = enabled.
	CompileCacheEnabled *bool `json:"compile_cache_enabled,omitempty"`

	// CompileCacheKey is the compile cache key/hash when the engine exposes it.
	CompileCacheKey string `json:"compile_cache_key,omitempty"`

	// CUDAGraphMode is the runtime CUDA-graph dispatch mode: "full",
	// "piecewise", or "none". Recorded for provenance; not itself a gate axis
	// ("none" is a legitimate tuned configuration).
	CUDAGraphMode string `json:"cuda_graph_mode,omitempty"`

	// CaptureSizes are the CUDA-graph capture batch sizes, when captured.
	CaptureSizes []int `json:"cuda_graph_capture_sizes,omitempty"`

	// WarmupComplete reports whether warmup finished before the measured
	// window. nil = not observed; *false = warmup did not complete (cold).
	WarmupComplete *bool `json:"warmup_complete,omitempty"`

	// RequestTimeCompilation is true when any request triggered compilation
	// during the measured window — the defining signature of a cold reading.
	RequestTimeCompilation bool `json:"request_time_compilation"`
}

// Class is the tuned-baseline classification of a recorded compile block.
type Class string

const (
	// ClassTuned: cache enabled, warmup complete, no request-time compilation —
	// a legitimate tuned baseline the net-true-value standard may quote.
	ClassTuned Class = "tuned"
	// ClassColdStart: the engine paid compile latency inside the measured
	// window — cache disabled, warmup incomplete, or a request-time compilation
	// event. A diagnostic reading, never a tuned baseline.
	ClassColdStart Class = "cold-start"
	// ClassDiagnostic: compile/warmup state was not observed, so the row cannot
	// be certified tuned. Report it; do not quote it as tuned.
	ClassDiagnostic Class = "diagnostic"
)

// ErrNotTuned is the sentinel a gate failure wraps, so callers can errors.Is it
// while still reading the specific reason from the message.
var ErrNotTuned = errors.New("vllm_compile: not a tuned baseline")

// Classify folds the recorded state into a tuned/cold-start/diagnostic class.
// Explicit cold signals (a request-time compilation event, a disabled cache, an
// incomplete warmup) take precedence over unobserved state, so a fixture that
// both compiled at request time and left warmup unknown is reported cold, not
// diagnostic.
func (b Block) Classify() Class {
	switch {
	case b.RequestTimeCompilation:
		return ClassColdStart
	case b.CompileCacheEnabled != nil && !*b.CompileCacheEnabled:
		return ClassColdStart
	case b.WarmupComplete != nil && !*b.WarmupComplete:
		return ClassColdStart
	case b.CompileCacheEnabled == nil || b.WarmupComplete == nil:
		return ClassDiagnostic
	default:
		return ClassTuned
	}
}

// Tuned reports whether the block certifies a tuned baseline.
func (b Block) Tuned() bool { return b.Classify() == ClassTuned }

// Reason states, in one clause, why the block is not tuned — or "tuned" when it
// is. It names the highest-precedence cold/diagnostic signal.
func (b Block) Reason() string {
	switch {
	case b.RequestTimeCompilation:
		return "a request triggered compilation during the measured window"
	case b.CompileCacheEnabled != nil && !*b.CompileCacheEnabled:
		return "torch.compile artifact cache disabled"
	case b.WarmupComplete != nil && !*b.WarmupComplete:
		return "warmup did not complete before the measured window"
	case b.CompileCacheEnabled == nil:
		return "compile cache state not observed"
	case b.WarmupComplete == nil:
		return "warmup completion not observed"
	default:
		return "tuned"
	}
}

// Gate fails closed unless the block certifies a tuned baseline. The returned
// error wraps ErrNotTuned and names the engine, class, and specific reason so a
// bench harness can reject (or relabel) the row instead of quoting a cold engine
// as tuned. It returns nil exactly when Classify() == ClassTuned.
func (b Block) Gate() error {
	if b.Tuned() {
		return nil
	}
	return fmt.Errorf("%w: %s row is %s (%s)", ErrNotTuned, engineLabel(b.Engine), b.Classify(), b.Reason())
}

// RowVerdict is one compared row's classification within a GateReport.
type RowVerdict struct {
	Engine string `json:"engine,omitempty"`
	Class  Class  `json:"class"`
	Reason string `json:"reason,omitempty"` // empty when tuned
}

// GateReport is the verdict for a full set of compared engine rows.
type GateReport struct {
	Rows  []RowVerdict `json:"rows"`
	Tuned bool         `json:"tuned"` // true iff there is >=1 row and every row is tuned
}

// GateRows classifies every compared row and reports whether the whole
// comparison is tuned. A comparison is tuned only when it has at least one row
// and EVERY row is tuned: one cold row poisons the A/B, since a cold raw-vLLM
// baseline makes a fak "win" meaningless even when the fak row is warm.
func GateRows(blocks ...Block) GateReport {
	rep := GateReport{Tuned: len(blocks) > 0, Rows: make([]RowVerdict, 0, len(blocks))}
	for _, b := range blocks {
		v := RowVerdict{Engine: b.Engine, Class: b.Classify()}
		if v.Class != ClassTuned {
			v.Reason = b.Reason()
			rep.Tuned = false
		}
		rep.Rows = append(rep.Rows, v)
	}
	return rep
}

// Bool is a small constructor for the pointer fields, so a caller can write
// vllmcompile.Block{CompileCacheEnabled: vllmcompile.Bool(true)} inline.
func Bool(v bool) *bool { return &v }

func engineLabel(engine string) string {
	if engine = strings.TrimSpace(engine); engine != "" {
		return engine
	}
	return "engine"
}
