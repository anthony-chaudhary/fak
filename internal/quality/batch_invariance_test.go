package quality

import (
	"strings"
	"testing"
)

// TestBatchInvarianceFaithfulPasses is the happy path: a faithful engine decodes
// the target identically at every batch placement, so the sweep matches the alone
// reference and the case passes with no failure bundle. It also pins the sweep
// itself as non-trivial — alone plus leading/middle/trailing placements in a real
// batch — so a pass can never be the vacuous "we only ever ran the target alone".
func TestBatchInvarianceFaithfulPasses(t *testing.T) {
	c := BatchInvarianceCase()

	plan, err := batchParsePlan(c.Prompt)
	if err != nil {
		t.Fatalf("case prompt must carry a batch plan: %v", err)
	}
	batched, positions := 0, map[int]bool{}
	for _, b := range plan.Batches {
		if len(b) > 1 {
			batched++
		}
		for i, id := range b {
			if id == plan.Target {
				positions[i] = true
			}
		}
	}
	if batched < 2 || !positions[0] || len(positions) < 3 {
		t.Fatalf("sweep must embed the target at multiple positions in real batches; batches %v", plan.Batches)
	}

	res, err := RunCase(c, ReferenceRunner{}, BatchInvarianceEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful batched engine should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean sweep must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestBatchInvariancePositionLeakFails is the position-dependence witness: an
// engine whose target token at step 2 depends on the target's batch position
// fails, the first divergence pins the exact token index, the tokens carried are
// the alone vs batched tokens computed independently here, and Detail NAMES the
// offending batch position — position 1, the first non-leading placement.
func TestBatchInvariancePositionLeakFails(t *testing.T) {
	c := BatchInvarianceCase()
	res, err := RunCase(c, ReferenceRunner{}, BatchInvarianceEngine("position-leak"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("position-leak engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing sweep must carry a failure bundle")
	}
	if fb.FailingOracle != "batch-invariance" {
		t.Errorf("first failing oracle = %q, want batch-invariance", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != batchLeakStep {
		t.Fatalf("expected first divergence at token %d, got %+v", batchLeakStep, d)
	}
	wantRef := c.Reference.Tokens[batchLeakStep]
	if d.Reference != wantRef {
		t.Errorf("divergence reference token = %q, want alone token %q", d.Reference, wantRef)
	}
	// The first offending placement in the sweep is the target at position 1 of a
	// batch of three, so the leaked token is the alone token rotated by 1.
	if wantEng := batchRotate(wantRef, 1); d.Engine != wantEng {
		t.Errorf("divergence engine token = %q, want %q", d.Engine, wantEng)
	}
	if !strings.Contains(fb.Detail, "batch position 1") {
		t.Errorf("detail must name the offending batch position; got %q", fb.Detail)
	}
}

// TestBatchInvarianceNeighborLeakFails is the cross-request state-leak witness: a
// specific neighbor scheduled BEFORE the target contaminates the target's stream.
// The leak fires only where that neighbor precedes the target (position 2 in the
// demo sweep), Detail names THAT position, and the placement where the same
// neighbor sits AFTER the target is proven clean — the failure localizes to the
// leaking composition, not to batching in general.
func TestBatchInvarianceNeighborLeakFails(t *testing.T) {
	c := BatchInvarianceCase()
	res, err := RunCase(c, ReferenceRunner{}, BatchInvarianceEngine("neighbor-leak"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("neighbor-leak engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing sweep must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != batchLeakStep {
		t.Fatalf("expected first divergence at token %d, got %+v", batchLeakStep, d)
	}
	wantRef := c.Reference.Tokens[batchLeakStep]
	if wantEng := batchRotate(wantRef, batchNeighborShift); d.Reference != wantRef || d.Engine != wantEng {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q", d.Reference, d.Engine, wantRef, wantEng)
	}
	if !strings.Contains(fb.Detail, "batch position 2") {
		t.Errorf("detail must name the position where the neighbor preceded the target; got %q", fb.Detail)
	}

	// Localization: in the placement where the leaky neighbor comes AFTER the
	// target (target at position 1), the engine's stream still matches the alone
	// decode exactly — only the neighbor-before composition leaks.
	trace, err := BatchInvarianceEngine("neighbor-leak").Run(c)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	placements, err := batchParsePlacements(trace.Text)
	if err != nil {
		t.Fatalf("parse sweep: %v", err)
	}
	for _, p := range placements {
		if p.Position != 1 {
			continue
		}
		for i, tok := range p.Tokens {
			if tok != c.Reference.Tokens[i] {
				t.Fatalf("placement with neighbor after target must be clean; token %d = %q, want %q", i, tok, c.Reference.Tokens[i])
			}
		}
	}
}

// TestBatchInvarianceFailsClosed pins the oracle's refusal modes: an engine trace
// with no parseable sweep cannot prove invariance and fails, and a sweep that
// never actually batches the target (alone-only) fails as too thin instead of
// vacuously passing.
func TestBatchInvarianceFailsClosed(t *testing.T) {
	c := BatchInvarianceCase()

	v := BatchInvariance{}.Judge(c.Reference, Trace{Tokens: c.Reference.Tokens, Text: "not a sweep"}, c)
	if v.Pass {
		t.Fatal("a trace with no batch sweep must not pass")
	}
	if !strings.Contains(v.Detail, "no batch sweep") {
		t.Errorf("detail should say the sweep is missing; got %q", v.Detail)
	}

	alone := `[{"position":0,"batch":["req-target"],"tokens":` + batchTokensJSON(c.Reference.Tokens) + `}]`
	v = BatchInvariance{}.Judge(c.Reference, Trace{Tokens: c.Reference.Tokens, Text: alone}, c)
	if v.Pass {
		t.Fatal("an alone-only sweep must not pass as invariance evidence")
	}
	if !strings.Contains(v.Detail, "too thin") {
		t.Errorf("detail should call the sweep too thin; got %q", v.Detail)
	}
}

// batchTokensJSON renders tokens as a JSON string array for test fixtures.
func batchTokensJSON(toks []string) string {
	quoted := make([]string, len(toks))
	for i, tok := range toks {
		quoted[i] = `"` + tok + `"`
	}
	return "[" + strings.Join(quoted, ",") + "]"
}
