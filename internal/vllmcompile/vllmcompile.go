package vllmcompile

import (
	"errors"
	"fmt"
	"strings"
)

// Block records the compile, CUDA-graph, and warmup execution state for a served engine.
type Block struct {
	Engine                 string `json:"engine,omitempty"`
	EngineCommit           string `json:"engine_commit,omitempty"`
	CompileCacheEnabled    *bool  `json:"compile_cache_enabled,omitempty"`
	CompileCacheKey        string `json:"compile_cache_key,omitempty"`
	CUDAGraphMode          string `json:"cuda_graph_mode,omitempty"`
	CaptureSizes           []int  `json:"cuda_graph_capture_sizes,omitempty"`
	WarmupComplete         *bool  `json:"warmup_complete,omitempty"`
	RequestTimeCompilation bool   `json:"request_time_compilation"`
}

// Class classifies whether an engine block meets tuned baseline criteria.
type Class string

const (
	// ClassTuned indicates cache was enabled, warmup completed, and no request compilation occurred.
	ClassTuned Class = "tuned"
	// ClassColdStart indicates compilation latency occurred during the measurement window.
	ClassColdStart Class = "cold-start"
	// ClassDiagnostic indicates compilation or warmup state was unobserved.
	ClassDiagnostic Class = "diagnostic"
)

// ErrNotTuned indicates an engine compile block failed tuned baseline requirements.
var ErrNotTuned = errors.New("vllm_compile: not a tuned baseline")

// Classify evaluates the compile block into a tuned, cold-start, or diagnostic class.
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

// Reason returns the failure clause explaining why a block is not tuned.
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

// Gate returns ErrNotTuned unless the block satisfies tuned baseline criteria.
func (b Block) Gate() error {
	if b.Tuned() {
		return nil
	}
	return fmt.Errorf("%w: %s row is %s (%s)", ErrNotTuned, engineLabel(b.Engine), b.Classify(), b.Reason())
}

// RowVerdict records the baseline classification and failure reason for a single row.
type RowVerdict struct {
	Engine string `json:"engine,omitempty"`
	Class  Class  `json:"class"`
	Reason string `json:"reason,omitempty"`
}

// GateReport aggregates evaluation verdicts across all compared engine rows.
type GateReport struct {
	Rows  []RowVerdict `json:"rows"`
	Tuned bool         `json:"tuned"`
}

// GateRows validates that every compared engine block satisfies tuned criteria.
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

// Bool wraps a primitive boolean as a pointer for optional block fields.
func Bool(v bool) *bool { return &v }

func engineLabel(engine string) string {
	if engine = strings.TrimSpace(engine); engine != "" {
		return engine
	}
	return "engine"
}
