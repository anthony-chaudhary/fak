package quality

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSchedulerParityFaithfulPoliciesPass is the parity contract on the happy
// path: a faithful scheduler produces identical per-request outputs under every
// policy, so each policy's trace passes against the fcfs reference. It first
// proves the policies genuinely REORDER execution — a pass over three identical
// orders would be a tautology, not evidence that order doesn't leak into output.
func TestSchedulerParityFaithfulPoliciesPass(t *testing.T) {
	reqs := schedDemoRequests()
	if got := schedOrder("priority", reqs)[0].ID; got != "req-b" {
		t.Fatalf("priority order should execute req-b first, got %q", got)
	}
	if got := schedOrder("longest-first", reqs)[0].ID; got != "req-c" {
		t.Fatalf("longest-first order should execute req-c first, got %q", got)
	}

	c := SchedulerParityCase()
	for _, policy := range SchedPolicies {
		res, err := RunCase(c, ReferenceRunner{}, SchedulerEngine(policy, ""), oraclesFor(t, c))
		if err != nil {
			t.Fatalf("RunCase(%s): %v", policy, err)
		}
		if !res.Pass {
			t.Fatalf("faithful %s scheduler must match the reference; got %s", policy, Explain(res))
		}
		if res.FailureBundle != nil {
			t.Fatalf("clean %s run must not carry a failure bundle: %+v", policy, res.FailureBundle)
		}
	}
}

// TestSchedulerParitySharedBufferDefectFails is the injected-defect witness: a
// scheduler that reuses one un-cleared slab across the batch corrupts the
// shortest request's output whenever the policy runs a longer request first.
// Under both reordering policies the first corruption is req-a's length — its
// reference ends at step 3, so the first divergence pins flat token 3 with
// reference "<end>" and a leaked stale token on the engine side, and the Detail
// names the offending policy, the request, and the index.
func TestSchedulerParitySharedBufferDefectFails(t *testing.T) {
	c := SchedulerParityCase()
	for _, policy := range []string{"priority", "longest-first"} {
		res, err := RunCase(c, ReferenceRunner{}, SchedulerEngine(policy, "shared-buffer"), oraclesFor(t, c))
		if err != nil {
			t.Fatalf("RunCase(%s): %v", policy, err)
		}
		if res.Pass {
			t.Fatalf("shared-buffer defect under %s must not pass; got %s", policy, Explain(res))
		}
		fb := res.FailureBundle
		if fb == nil {
			t.Fatalf("failing %s run must carry a failure bundle", policy)
		}
		if fb.FailingOracle != "scheduler-parity" {
			t.Errorf("%s: first failing oracle = %q, want scheduler-parity", policy, fb.FailingOracle)
		}
		d := fb.FirstDivergence
		if d == nil {
			t.Fatalf("%s: shared-buffer failure must localize a first divergence", policy)
		}
		if d.Index != 3 {
			t.Errorf("%s: divergence index = %d, want 3 (req-a's reference ends at step 3)", policy, d.Index)
		}
		if d.Reference != "<end>" {
			t.Errorf("%s: divergence reference = %q, want %q", policy, d.Reference, "<end>")
		}
		if d.Engine == "" || d.Engine == "<end>" {
			t.Errorf("%s: divergence engine = %q, want a leaked stale token", policy, d.Engine)
		}
		if !strings.Contains(fb.Detail, `"`+policy+`"`) {
			t.Errorf("%s: detail must name the offending policy, got %q", policy, fb.Detail)
		}
		if !strings.Contains(fb.Detail, `"req-a"`) {
			t.Errorf("%s: detail must name the corrupted request, got %q", policy, fb.Detail)
		}
	}
}

// TestSchedulerParityDefectInvisibleUnderFCFS documents WHY cross-policy parity
// is its own gate: the same shared-buffer defect is invisible under fcfs,
// because the demo batch's submission order ascends in length and the slab
// never outgrows the running request. Only a reordering policy exposes the bug
// — a suite that tested one policy alone would stay green over it.
func TestSchedulerParityDefectInvisibleUnderFCFS(t *testing.T) {
	c := SchedulerParityCase()
	res, err := RunCase(c, ReferenceRunner{}, SchedulerEngine("fcfs", "shared-buffer"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("shared-buffer defect should be invisible under fcfs on this batch; got %s", Explain(res))
	}
}

// TestSchedulerParityJudgeLocalizesMidRequest exercises the oracle directly on
// a mid-request corruption (not a length change): flipping req-b's step-1 token
// must pin the first divergence to flat index 4 (req-a's 3 tokens + 1) with the
// exact reference and engine tokens reported.
func TestSchedulerParityJudgeLocalizesMidRequest(t *testing.T) {
	c := SchedulerParityCase()
	refBody, err := schedParseBody(c.Reference.Text)
	if err != nil {
		t.Fatalf("parse reference body: %v", err)
	}
	engBody := schedBody{Policy: "priority", Outputs: make([]schedOutput, len(refBody.Outputs))}
	for i, o := range refBody.Outputs {
		engBody.Outputs[i] = schedOutput{ID: o.ID, Tokens: append([]string(nil), o.Tokens...)}
	}
	want := refBody.Outputs[1].Tokens[1]
	corrupt := schedVocab[0]
	if corrupt == want {
		corrupt = schedVocab[1]
	}
	engBody.Outputs[1].Tokens[1] = corrupt

	b, err := json.Marshal(engBody)
	if err != nil {
		t.Fatalf("marshal engine body: %v", err)
	}
	v := SchedulerParity{}.Judge(c.Reference, Trace{Text: string(b)}, c)
	if v.Pass {
		t.Fatal("mid-request corruption must not pass")
	}
	d := v.FirstDivergence
	if d == nil || d.Index != 4 {
		t.Fatalf("expected first divergence at flat token 4, got %+v", d)
	}
	if d.Reference != want || d.Engine != corrupt {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q", d.Reference, d.Engine, want, corrupt)
	}
	if !strings.Contains(v.Detail, `"req-b"`) || !strings.Contains(v.Detail, `"priority"`) {
		t.Errorf("detail must name the request and policy, got %q", v.Detail)
	}
}

// TestSchedulerParityFailsClosedOnMalformedBody is the fail-closed rung: a
// trace that cannot show its per-request outputs is refused, not passed — a
// scheduler run without evidence is not a green run.
func TestSchedulerParityFailsClosedOnMalformedBody(t *testing.T) {
	c := SchedulerParityCase()
	v := SchedulerParity{}.Judge(c.Reference, Trace{Text: "not a scheduler body"}, c)
	if v.Pass {
		t.Fatal("malformed engine body must not pass")
	}
	if !strings.Contains(v.Detail, "engine trace") {
		t.Errorf("detail should blame the engine trace, got %q", v.Detail)
	}
}
