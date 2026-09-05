// Model-free witnesses for modelbench's per-cell write-ahead checkpoint + resume (#2382).
//
// The full acceptance witness — start a real grid, kill -9 mid-grid, resume — needs a loaded
// model (a multi-hundred-MB checkpoint) and is host-gated. These tests instead exercise the
// exact seam the grid loops use (checkpointCell + the grid fingerprint) with an injected
// measure function, so the reuse-vs-remeasure and mismatch-refusal logic is proven with no
// model, GPU, or network — the same shape as internal/bench's fanrun_checkpoint_test.go.
package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchckpt"
)

func sp(s string) *string { return &s }
func ip(i int) *int       { return &i }

// gridFP is a representative modelbench grid fingerprint for the ledger-level tests.
func gridFP() benchckpt.Fingerprint {
	return benchckpt.Fingerprint{"schema": "modelbench-grid/1", "source": "smollm2-135m", "quant": false, "prefill_reps": 5}
}

// TestCheckpointCellReuseParity is the #2382 modelbench witness at the model-free cell seam: a
// partial grid records its completed cells write-ahead; a resume over the FULL grid reuses those
// cells verbatim (including their measured *_ms) and measures ONLY the missing coordinate —
// exactly the "crash at cell N keeps 1..N-1, resume runs only the rest" contract.
func TestCheckpointCellReuseParity(t *testing.T) {
	ckpt := filepath.Join(t.TempDir(), "modelbench.jsonl")

	// A synthetic measure: stamps MedianMS = tok*1.5 so a reused cell is distinguishable from a
	// re-measured one, and records which keys it was actually asked to measure.
	var measured []int
	measure := func(tok int) func() prefillResult {
		return func() prefillResult {
			measured = append(measured, tok)
			return prefillResult{Tokens: tok, Reps: 5, MedianMS: float64(tok) * 1.5, TokPerSec: 42}
		}
	}

	// Phase A — measure P=16 and P=64, then "crash" (close the ledger). This is the on-disk state
	// a kill leaves: every completed cell write-ahead appended before the next begins.
	la, err := benchckpt.Open(ckpt, gridFP())
	if err != nil {
		t.Fatalf("phase A open: %v", err)
	}
	a16, r16 := checkpointCell(la, "prefill:P=16", measure(16))
	a64, r64 := checkpointCell(la, "prefill:P=64", measure(64))
	if r16 || r64 {
		t.Fatalf("phase A must MEASURE both cells, got reused=%v,%v", r16, r64)
	}
	if len(measured) != 2 {
		t.Fatalf("phase A measured %d cells, want 2", len(measured))
	}
	la.Close()

	// Write-ahead witness: header + 2 cell lines — strictly fewer than a full [16,64,256] grid,
	// which is the "line count > 0, < total" the acceptance names.
	if n := countJSONLines(t, ckpt); n != 3 {
		t.Fatalf("checkpoint lines = %d, want 3 (header + P=16 + P=64)", n)
	}

	// Phase B — resume over the FULL grid. P=16 and P=64 must be REUSED from the checkpoint; only
	// P=256 freshly measured. The reuse-path measure closures pass tok=999 so a mistaken
	// re-measure would be caught (the recorded cells carry tok=16/64, not 999).
	measured = nil
	lb, err := benchckpt.Open(ckpt, gridFP())
	if err != nil {
		t.Fatalf("phase B (resume) open: %v", err)
	}
	defer lb.Close()
	b16, ru16 := checkpointCell(lb, "prefill:P=16", measure(999))
	b64, ru64 := checkpointCell(lb, "prefill:P=64", measure(999))
	b256, ru256 := checkpointCell(lb, "prefill:P=256", measure(256))

	if !ru16 || !ru64 {
		t.Fatalf("resume must REUSE P=16 and P=64, got reused=%v,%v", ru16, ru64)
	}
	if ru256 {
		t.Fatalf("P=256 was never recorded; it must be freshly measured, not reused")
	}
	if len(measured) != 1 || measured[0] != 256 {
		t.Fatalf("resume measured %v, want exactly [256] (only the missing cell)", measured)
	}
	// Reuse witness: the resumed cells equal phase A's byte-for-byte (identical MedianMS proves
	// they were loaded from the checkpoint, not silently re-measured).
	if b16 != a16 || b64 != a64 {
		t.Fatalf("reused cells not byte-identical:\n A=%+v / %+v\n B=%+v / %+v", a16, a64, b16, b64)
	}
	if b256.Tokens != 256 || b256.MedianMS != 384 { // 256 * 1.5
		t.Fatalf("P=256 measured wrong: %+v", b256)
	}
}

// TestCheckpointResumeFingerprintMismatchRefuses is the #2382 item-5 witness for modelbench:
// resuming a checkpoint built for a different model/precision refuses with the typed error
// instead of blending incompatible cells into one artifact.
func TestCheckpointResumeFingerprintMismatchRefuses(t *testing.T) {
	ckpt := filepath.Join(t.TempDir(), "modelbench.jsonl")

	l, err := benchckpt.Open(ckpt, gridFP())
	if err != nil {
		t.Fatalf("seed open: %v", err)
	}
	checkpointCell(l, "prefill:P=16", func() prefillResult { return prefillResult{Tokens: 16, Reps: 5} })
	l.Close()

	// Same path, different precision (quant flipped) -> typed refusal, not a silent merge.
	changed := gridFP()
	changed["quant"] = true
	if _, err := benchckpt.Open(ckpt, changed); !errors.Is(err, benchckpt.ErrFingerprintMismatch) {
		t.Fatalf("resume with changed fingerprint err = %v, want ErrFingerprintMismatch", err)
	}
}

// TestModelbenchFingerprintIdentity locks the grid-identity contract the resume refusal rests on:
// the fingerprint MUST change when a per-cell timing knob changes (so incompatible regimes can't
// be blended), and MUST NOT depend on the prefill-size list (so a superset grid can resume a
// partial run).
func TestModelbenchFingerprintIdentity(t *testing.T) {
	base := fpFlags()
	want := fpKey(modelbenchFingerprint(base, "m"))

	// The prefill-size LIST is not part of the identity (a P=16 cell is grid-independent) — this
	// is what makes "crash on [16,64], resume on [16,64,256]" reuse the recorded cells.
	diffSizes := fpFlags()
	*diffSizes.prefillSizesCSV = "1,2,3,999"
	if got := fpKey(modelbenchFingerprint(diffSizes, "m")); got != want {
		t.Fatalf("fingerprint must NOT depend on the prefill-size list:\n want %s\n got  %s", want, got)
	}

	// Changing what a cell's median is taken over DOES change the identity, so a resume can't
	// reuse cells measured under a different regime.
	for _, tc := range []struct {
		name  string
		mutot func(*benchFlags)
	}{
		{"prefill_reps", func(f *benchFlags) { *f.prefillReps = 9 }},
		{"decode_reps", func(f *benchFlags) { *f.decodeReps = 9 }},
		{"decode_steps", func(f *benchFlags) { *f.decodeSteps = 99 }},
		{"decode_prompt", func(f *benchFlags) { *f.decodePrompt = 99 }},
		{"quant", func(f *benchFlags) { *f.quant = true }},
		{"metal", func(f *benchFlags) { *f.metal = true }},
		{"q4k", func(f *benchFlags) { *f.q4k = true }},
		{"lean", func(f *benchFlags) { *f.lean = true }},
		{"backend", func(f *benchFlags) { *f.backendName = "cpu-ref" }},
		{"workload", func(f *benchFlags) { *f.workloadPath = "w.json" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := fpFlags()
			tc.mutot(f)
			if got := fpKey(modelbenchFingerprint(f, "m")); got == want {
				t.Fatalf("fingerprint must change when %s changes, but it did not", tc.name)
			}
		})
	}

	// The model name is part of the identity too (resuming a different model refuses).
	if fpKey(modelbenchFingerprint(base, "other")) == want {
		t.Fatalf("fingerprint must change with the model name")
	}
}

// fpFlags builds a benchFlags with every pointer the fingerprint reads populated, so
// modelbenchFingerprint can be called without a real flag parse.
func fpFlags() *benchFlags {
	return &benchFlags{
		hf: sp(""), gguf: sp(""), dir: sp("m"), lean: testBool(false), q4k: testBool(false),
		quant: testBool(false), metal: testBool(false), backendName: sp("legacy"), q4kGateUpSlab: testBool(false), vulkanQ4KProfile: testBool(false), vulkanStageQ4K: testBool(false),
		prefillReps: ip(5), decodeReps: ip(5), decodeSteps: ip(32), decodePrompt: ip(16),
		workloadPath: sp(""), workloadPrefillCap: ip(0), prefillSizesCSV: sp("16,64,256"),
		numaReplicas: sp(""),
	}
}

// fpKey is the canonical JSON of a fingerprint (map keys sort in encoding/json), so two
// fingerprints compare equal iff they are the same grid identity.
func fpKey(fp benchckpt.Fingerprint) string {
	b, _ := json.Marshal(fp)
	return string(b)
}

// countJSONLines counts the non-empty lines of a JSONL checkpoint.
func countJSONLines(t *testing.T, path string) int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	n := 0
	for _, ln := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(ln) != "" {
			n++
		}
	}
	return n
}
