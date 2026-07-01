package vllmcompile

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestClassifyAndGate is the executable form of issue #1731's acceptance
// criteria 2 and 3: a missing OR disabled compile cache is labeled cold-start /
// diagnostic (never tuned), and a request-time compilation event fails the
// tuned-baseline gate.
func TestClassifyAndGate(t *testing.T) {
	cases := []struct {
		name      string
		block     Block
		wantClass Class
		wantGate  bool // true => Gate returns an error (not a tuned baseline)
	}{
		{
			name: "fully warm is tuned",
			block: Block{
				Engine:              "vllm",
				EngineCommit:        "28242824e",
				CompileCacheEnabled: Bool(true),
				CompileCacheKey:     "abc123",
				CUDAGraphMode:       "full",
				CaptureSizes:        []int{1, 2, 4, 8},
				WarmupComplete:      Bool(true),
			},
			wantClass: ClassTuned,
			wantGate:  false,
		},
		{
			name: "request-time compilation fails the gate", // criterion 3
			block: Block{
				Engine:                 "vllm",
				CompileCacheEnabled:    Bool(true),
				WarmupComplete:         Bool(true),
				RequestTimeCompilation: true,
			},
			wantClass: ClassColdStart,
			wantGate:  true,
		},
		{
			name: "disabled compile cache is cold-start", // criterion 2 (disabled)
			block: Block{
				Engine:              "vllm",
				CompileCacheEnabled: Bool(false),
				WarmupComplete:      Bool(true),
			},
			wantClass: ClassColdStart,
			wantGate:  true,
		},
		{
			name: "missing compile cache is diagnostic", // criterion 2 (missing)
			block: Block{
				Engine:         "vllm",
				WarmupComplete: Bool(true),
			},
			wantClass: ClassDiagnostic,
			wantGate:  true,
		},
		{
			name: "incomplete warmup is cold-start",
			block: Block{
				Engine:              "vllm",
				CompileCacheEnabled: Bool(true),
				WarmupComplete:      Bool(false),
			},
			wantClass: ClassColdStart,
			wantGate:  true,
		},
		{
			name: "cold signal beats unknown warmup",
			block: Block{
				Engine:                 "vllm",
				RequestTimeCompilation: true,
				// WarmupComplete nil -> would be diagnostic, but the explicit
				// cold signal wins.
			},
			wantClass: ClassColdStart,
			wantGate:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.block.Classify(); got != tc.wantClass {
				t.Fatalf("Classify() = %q, want %q", got, tc.wantClass)
			}
			err := tc.block.Gate()
			if tc.wantGate && err == nil {
				t.Fatalf("Gate() = nil, want a not-tuned error")
			}
			if !tc.wantGate && err != nil {
				t.Fatalf("Gate() = %v, want nil", err)
			}
			if err != nil && !errors.Is(err, ErrNotTuned) {
				t.Fatalf("Gate() error %v does not wrap ErrNotTuned", err)
			}
			if tc.block.Tuned() != (tc.wantClass == ClassTuned) {
				t.Fatalf("Tuned() = %v, inconsistent with class %q", tc.block.Tuned(), tc.wantClass)
			}
		})
	}
}

// TestGateRowsPoisonedByColdBaseline proves an A/B comparison fails as a set
// when the raw baseline row is cold even though the fak-fronted row is warm —
// the honest-comparison guarantee behind the whole gate.
func TestGateRowsPoisonedByColdBaseline(t *testing.T) {
	rawCold := Block{Engine: "vllm", RequestTimeCompilation: true}
	fakWarm := Block{Engine: "vllm+fak", CompileCacheEnabled: Bool(true), WarmupComplete: Bool(true)}

	rep := GateRows(rawCold, fakWarm)
	if rep.Tuned {
		t.Fatalf("GateRows reported tuned with a cold baseline row: %+v", rep)
	}
	if len(rep.Rows) != 2 {
		t.Fatalf("GateRows returned %d rows, want 2", len(rep.Rows))
	}
	if rep.Rows[0].Class != ClassColdStart || rep.Rows[0].Reason == "" {
		t.Fatalf("raw row verdict = %+v, want cold-start with a reason", rep.Rows[0])
	}
	if rep.Rows[1].Class != ClassTuned || rep.Rows[1].Reason != "" {
		t.Fatalf("fak row verdict = %+v, want tuned with no reason", rep.Rows[1])
	}

	// An all-tuned set passes; an empty set is not vacuously tuned.
	if !GateRows(fakWarm).Tuned {
		t.Fatalf("GateRows(warm) should be tuned")
	}
	if GateRows().Tuned {
		t.Fatalf("GateRows() with no rows should not be tuned")
	}
}

// TestBlockJSONSchema is the executable form of criterion 1: the `vllm_compile`
// block is a real, serializable artifact element embeddable in a bench row, and
// its request-time-compilation field is always present (never omitempty) so a
// cold event can never be silently absent from the artifact.
func TestBlockJSONSchema(t *testing.T) {
	type benchRow struct {
		Engine      string `json:"engine"`
		VLLMCompile Block  `json:"vllm_compile"`
	}
	row := benchRow{
		Engine: "vllm",
		VLLMCompile: Block{
			Engine:                 "vllm",
			EngineCommit:           "28242824e",
			CompileCacheEnabled:    Bool(true),
			CUDAGraphMode:          "piecewise",
			WarmupComplete:         Bool(true),
			RequestTimeCompilation: false,
		},
	}
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal bench row: %v", err)
	}
	got := string(b)
	for _, want := range []string{
		`"vllm_compile":`,
		`"compile_cache_enabled":true`,
		`"cuda_graph_mode":"piecewise"`,
		`"warmup_complete":true`,
		`"request_time_compilation":false`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("marshaled artifact %s missing %q", got, want)
		}
	}

	// Round-trip: the block a harness writes is the block the gate reads.
	var back benchRow
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal bench row: %v", err)
	}
	if !back.VLLMCompile.Tuned() {
		t.Fatalf("round-tripped block should be tuned: %+v", back.VLLMCompile)
	}
}
