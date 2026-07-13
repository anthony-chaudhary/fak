package quality

import (
	"math"
	"strings"
	"testing"
)

// TestDevMatrixCleanPasses is the fan-in happy path: all five backends decode
// the case faithfully (tokens exact, logits inside tolerance) and the matrix
// judges as ONE passing verdict with no failure bundle.
func TestDevMatrixCleanPasses(t *testing.T) {
	c := devMatrixCase()
	res, err := RunCase(c, ReferenceRunner{}, devMatrixEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful device matrix should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean matrix must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if len(res.Verdicts) != 1 || !strings.Contains(res.Verdicts[0].Detail, "vulkan") {
		t.Fatalf("pass detail should enumerate the full matrix incl. vulkan; got %+v", res.Verdicts)
	}
}

// TestDevMatrixToleranceIsDoingWork proves the clean pass is not a bitwise
// tautology: a faithful backend's logits DIFFER from the golden reference
// bitwise (real device kernels reassociate floats) yet every delta stays
// inside the tolerance the oracle gates on. Without this, "pass" could mean
// "jitter model is a no-op" rather than "tolerance absorbs device noise".
func TestDevMatrixToleranceIsDoingWork(t *testing.T) {
	c := devMatrixCase()
	_, golden := devMatrixGolden(c.Params.MaxTokens)
	b := devMatrixDecode("cuda", c.Params.MaxTokens, "")
	nonzero := 0
	for i := range golden {
		for j := range golden[i] {
			delta := math.Abs(golden[i][j] - b.Logits[i][j])
			if delta > devMatrixTolerance {
				t.Fatalf("faithful cuda jitter at step %d slot %d is %.3g, beyond tolerance %g", i, j, delta, devMatrixTolerance)
			}
			if delta > 0 {
				nonzero++
			}
		}
	}
	if nonzero == 0 {
		t.Fatal("faithful backend logits are bitwise identical to golden; the tolerance path is untested")
	}
	v := devMatrixParity{}.Judge(c.Reference, mustDevMatrixPack(t, []devBackendTrace{
		devMatrixDecode("cpu", c.Params.MaxTokens, ""),
		devMatrixDecode("cuda", c.Params.MaxTokens, ""),
		devMatrixDecode("rocm", c.Params.MaxTokens, ""),
		devMatrixDecode("metal", c.Params.MaxTokens, ""),
		devMatrixDecode("vulkan", c.Params.MaxTokens, ""),
	}), c)
	if !v.Pass {
		t.Fatalf("jittered-but-in-tolerance matrix must pass; got %+v", v)
	}
}

// TestDevMatrixVulkanTokenFlipFails is the localized-defect witness for the
// argmax class: Vulkan alone emits a wrong token at the defect step. The four
// other backends pass, and the verdict names the backend AND pins the first
// divergence to exactly that step with both tokens reported.
func TestDevMatrixVulkanTokenFlipFails(t *testing.T) {
	c := devMatrixCase()
	res, err := RunCase(c, ReferenceRunner{}, devMatrixEngine(devMatrixDefectTokenFlip), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("vulkan token-flip matrix must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing matrix must carry a failure bundle")
	}
	if fb.FailingOracle != "device-matrix-parity" {
		t.Errorf("first failing oracle = %q, want device-matrix-parity", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, `"vulkan"`) {
		t.Errorf("detail must name the diverging backend vulkan; got %q", fb.Detail)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != devMatrixDefectStep {
		t.Fatalf("expected first divergence at step %d, got %+v", devMatrixDefectStep, d)
	}
	wantRef := c.Reference.Tokens[devMatrixDefectStep]
	if d.Reference != wantRef {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, wantRef)
	}
	if wantEng := devMatrixRotate(wantRef); d.Engine != wantEng {
		t.Errorf("divergence engine token = %q, want %q", d.Engine, wantEng)
	}
}

// TestDevMatrixVulkanLogitDriftFails is the numeric-drift witness: Vulkan's
// tokens still match, but one logit at the defect step drifts three orders of
// magnitude beyond tolerance. The gate trips BEFORE the drift is large enough
// to flip a token, and the detail names the backend, step, and tolerance.
func TestDevMatrixVulkanLogitDriftFails(t *testing.T) {
	c := devMatrixCase()
	res, err := RunCase(c, ReferenceRunner{}, devMatrixEngine(devMatrixDefectLogitDrift), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("vulkan logit-drift matrix must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing matrix must carry a failure bundle")
	}
	if !strings.Contains(fb.Detail, `"vulkan"`) || !strings.Contains(fb.Detail, "logit") {
		t.Errorf("detail must name vulkan and the logit surface; got %q", fb.Detail)
	}
	if d := fb.FirstDivergence; d == nil || d.Index != devMatrixDefectStep {
		t.Fatalf("expected logit divergence localized to step %d, got %+v", devMatrixDefectStep, d)
	}
}

// TestDevMatrixMissingBackendFails closes the opt-out hole: a payload that
// silently drops Vulkan is a FAILURE, not a smaller pass — an unchecked
// backend is not a green backend.
func TestDevMatrixMissingBackendFails(t *testing.T) {
	c := devMatrixCase()
	res, err := RunCase(c, ReferenceRunner{}, devMatrixEngine(devMatrixDefectMissing), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("matrix missing vulkan must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("missing-backend matrix must carry a failure bundle")
	}
	if !strings.Contains(fb.Detail, `"vulkan"`) || !strings.Contains(fb.Detail, "missing") {
		t.Errorf("detail must report vulkan as missing; got %q", fb.Detail)
	}
}

// TestDevMatrixUnparseablePayloadFails: an engine trace whose Text is not a
// device-matrix payload is refused, not passed — the fan-in oracle never
// judges an empty matrix green.
func TestDevMatrixUnparseablePayloadFails(t *testing.T) {
	c := devMatrixCase()
	v := devMatrixParity{}.Judge(c.Reference, Trace{Runner: "engine", Text: "not json"}, c)
	if v.Pass {
		t.Fatalf("unparseable payload must not pass; got %+v", v)
	}
	if !strings.Contains(v.Detail, "payload") {
		t.Errorf("detail should explain the payload was unparseable; got %q", v.Detail)
	}
}

// mustDevMatrixPack packs labeled backend traces for direct Judge calls.
func mustDevMatrixPack(t *testing.T, traces []devBackendTrace) Trace {
	t.Helper()
	tr, err := devMatrixPack("test-matrix", traces)
	if err != nil {
		t.Fatalf("devMatrixPack: %v", err)
	}
	return tr
}
