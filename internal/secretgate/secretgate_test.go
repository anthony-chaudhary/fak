package secretgate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the blob PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/leakcheck"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// A canon.SecretPatterns-matching credential used across the tests.
const secretBody = "token: github_pat_11ABCDEFG0aZbYcXdWeVuTs9R8q7P6o5N4m3L2k1J0"

func result(body string) *abi.Result {
	return &abi.Result{Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body)}}
}
func toolCall(tool string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}, Meta: map[string]string{}}
}
func resolve(t *testing.T, ctx context.Context, r abi.Ref) string {
	t.Helper()
	if r.Kind == abi.RefInline {
		return string(r.Inline)
	}
	b, err := abi.ActiveResolver().Resolve(ctx, r)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	return string(b)
}

// withEnabled flips the opt-in toggle for one test and restores it after.
func withEnabled(t *testing.T, v bool) {
	t.Helper()
	prev := enabled
	enabled = v
	t.Cleanup(func() { enabled = prev })
}

// TestDisabledDefersByDefault: with the rung off (the default), a secret-bearing
// result is a no-op Defer — the payload is untouched and nothing is recorded, so
// today's normgate-only secret path is preserved exactly.
func TestDisabledDefersByDefault(t *testing.T) {
	withEnabled(t, false)
	ctx := context.Background()
	g := New()
	r := result(secretBody)
	v := g.Admit(ctx, toolCall("read_webpage"), r)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("disabled rung must Defer, got %v", v.Kind)
	}
	if got := resolve(t, ctx, r.Payload); got != secretBody {
		t.Errorf("disabled rung must not touch the payload: got %q", got)
	}
	if n := len(g.Findings()); n != 0 {
		t.Errorf("disabled rung must record no findings, got %d", n)
	}
}

// TestQuarantinesDiscoveredSecret: with the rung on, a canon-matched secret is
// quarantined with ReasonSecretDiscovered, the payload is stubbed (the secret bytes
// never enter context), and a structured high-confidence Finding under the secrets
// ref namespace is recorded.
func TestQuarantinesDiscoveredSecret(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	g := New()
	r := result(secretBody)
	v := g.Admit(ctx, toolCall("read_webpage"), r)
	if v.Kind != abi.VerdictQuarantine || v.Reason != abi.ReasonSecretDiscovered {
		t.Fatalf("want Quarantine/RESULT_SECRET_DISCOVERED, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	after := resolve(t, ctx, r.Payload)
	if strings.Contains(after, "github_pat_") {
		t.Errorf("secret bytes leaked into the stubbed payload: %q", after)
	}
	if !strings.Contains(after, "_quarantined") {
		t.Errorf("payload was not replaced with the quarantine stub: %q", after)
	}
	fs := g.Findings()
	if len(fs) == 0 {
		t.Fatal("a discovered secret recorded no Finding")
	}
	f := fs[0]
	if f.Kind != "secret" || f.Confidence != "high" || f.Digest == "" {
		t.Errorf("Finding = %+v, want kind=secret confidence=high non-empty digest", f)
	}
	if !strings.HasPrefix(f.Handle, "refs/fak/secrets/") {
		t.Errorf("Finding handle = %q, want refs/fak/secrets/<nonce>", f.Handle)
	}
	if h := r.Meta["secret_handle"]; h != f.Handle {
		t.Errorf("result Meta handle %q != Finding handle %q", h, f.Handle)
	}
	if total, disc, _ := g.Stats(); total != 1 || disc != 1 {
		t.Errorf("Stats = (total %d, discovered %d), want (1, 1)", total, disc)
	}
}

// TestCleanResultAdmitted: a result with no secret is a no-op Defer even with the
// rung on — only secret-bearing results are touched.
func TestCleanResultAdmitted(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	g := New()
	clean := "the build finished and all 412 tests passed in 9.1s"
	r := result(clean)
	v := g.Admit(ctx, toolCall("read_file"), r)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("clean result must Defer, got %v", v.Kind)
	}
	if got := resolve(t, ctx, r.Payload); got != clean {
		t.Errorf("clean result payload was altered: %q", got)
	}
	if n := len(g.Findings()); n != 0 {
		t.Errorf("clean result recorded a finding: %d", n)
	}
}

// TestWitnessDecisionRecorded: when a recorder is wired, a discovery appends a
// witness.Decision with ReasonClass RESULT_SECRET_DISCOVERED. The fake Runner reads
// the `-F` note payload git would have appended and the test parses the Decision.
func TestWitnessDecisionRecorded(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()

	var captured witness.Decision
	var sawAppend bool
	runner := func(_ context.Context, _ string, args ...string) (string, int, error) {
		// args: notes --ref=... append -F <tmp> <sha>
		for i, a := range args {
			if a == "append" {
				sawAppend = true
			}
			if a == "-F" && i+1 < len(args) {
				body, err := os.ReadFile(args[i+1])
				if err != nil {
					return "", 1, err
				}
				line := strings.TrimSpace(string(body))
				if err := json.Unmarshal([]byte(line), &captured); err != nil {
					return "", 1, err
				}
			}
		}
		return "", 0, nil
	}

	g := New()
	g.SetRecorder(witness.NewRecorderWithRunner(runner, ""))
	r := result(secretBody)
	if v := g.Admit(ctx, toolCall("read_webpage"), r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	if !sawAppend {
		t.Fatal("recorder wired but no `git notes append` was issued")
	}
	if captured.ReasonClass != "RESULT_SECRET_DISCOVERED" {
		t.Errorf("Decision.ReasonClass = %q, want RESULT_SECRET_DISCOVERED", captured.ReasonClass)
	}
	if captured.Verdict != witness.VerdictRefuse || captured.Op != "result-admit" {
		t.Errorf("Decision = {Op:%q Verdict:%q}, want {result-admit, refuse}", captured.Op, captured.Verdict)
	}
}

// TestPageInGatedAndReScreenRefuses: the held secret is returned ONLY through the
// gate, and even a witness Clear does not launder a credential back — the page-in
// re-screen still finds the secret and refuses.
func TestPageInGatedAndReScreenRefuses(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	g := New()
	r := result(secretBody)
	if v := g.Admit(ctx, toolCall("read_webpage"), r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	handle := r.Meta["secret_handle"]
	if handle == "" {
		t.Fatal("no secret_handle recorded on the result")
	}

	if _, err := g.PageIn(ctx, handle); err == nil {
		t.Error("page-in of an uncleared handle must be refused")
	}
	if _, err := g.PageIn(ctx, "refs/fak/secrets/deadbeefdeadbeef"); err == nil {
		t.Error("page-in of an unknown handle must be refused")
	}
	g.Clear(handle)
	if _, err := g.PageIn(ctx, handle); err == nil {
		t.Error("page-in after Clear must STILL be refused — a cleared credential does not launder back into context")
	}
}

// TestReasonRegistered confirms the abi vocabulary wiring the rung depends on.
func TestReasonRegistered(t *testing.T) {
	if got := abi.ReasonName(abi.ReasonSecretDiscovered); got != "RESULT_SECRET_DISCOVERED" {
		t.Fatalf("ReasonSecretDiscovered name = %q, want RESULT_SECRET_DISCOVERED", got)
	}
}

// TestReuseEscalation (#886): the FIRST sighting of a secret digest is recorded at
// base confidence; the SAME digest seen again escalates the Finding (Sightings>=2,
// Escalated, confidence high->critical).
func TestReuseEscalation(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	g := New()

	g.Admit(ctx, toolCall("read_webpage"), result(secretBody))
	first := g.Findings()
	if len(first) == 0 {
		t.Fatal("first admit recorded no findings")
	}
	for _, f := range first {
		if f.Sightings != 1 || f.Escalated {
			t.Fatalf("first sighting must be base (Sightings=1, not escalated): %+v", f)
		}
	}

	g.Admit(ctx, toolCall("read_webpage"), result(secretBody))
	second := g.Findings()[len(first):]
	if len(second) == 0 {
		t.Fatal("second admit recorded no findings")
	}
	for _, f := range second {
		if f.Sightings < 2 || !f.Escalated {
			t.Errorf("repeat sighting must escalate (Sightings>=2, Escalated): %+v", f)
		}
		if f.Confidence != "critical" {
			t.Errorf("repeat of a high-confidence secret must escalate to critical, got %q", f.Confidence)
		}
	}
}

// TestDistinctDigestsDoNotCrossTrigger (#886): a DIFFERENT secret is a fresh first
// sighting — distinct digests never escalate each other.
func TestDistinctDigestsDoNotCrossTrigger(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	g := New()

	g.Admit(ctx, toolCall("a"), result(secretBody))
	before := len(g.Findings())

	// A different credential (a Google API key) — distinct digest, must not escalate.
	g.Admit(ctx, toolCall("b"), result("key=AIzaSyD-9tT8d_xQ2mPaLk7vRz0nW4cYh3bUeKfG"))
	fresh := g.Findings()[before:]
	if len(fresh) == 0 {
		t.Fatal("no findings for the second, distinct secret")
	}
	for _, f := range fresh {
		if f.Sightings != 1 || f.Escalated {
			t.Errorf("a distinct secret must be a first sighting, not escalated: %+v", f)
		}
	}
}

// TestDigestSetBounded (#886): the digest set stays under cap across many distinct
// sightings — a long, secret-heavy session cannot grow it without bound.
func TestDigestSetBounded(t *testing.T) {
	withEnabled(t, true)
	ctx := context.Background()
	const limit = 4
	g := NewWithLimit(limit)
	leakcheck.BoundedSize(t, 60, limit,
		func(i int) {
			// Unique per i, matches github_pat_[0-9A-Za-z_]{20,} (one digest each).
			g.Admit(ctx, toolCall("read"), result(fmt.Sprintf("github_pat_%030d", i)))
		},
		func() int { return g.digests.size() },
	)
}
