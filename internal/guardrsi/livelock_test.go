package guardrsi

import "testing"

func TestArgsDigestCanonicalizesJSONObjectWhitespaceAndOrder(t *testing.T) {
	a := ArgsDigest(`{"a":1,"b":[true,"x"]}`)
	b := ArgsDigest("{\n  \"b\": [true, \"x\"],\n  \"a\": 1\n}")
	if a != b {
		t.Fatalf("digest changed across semantic JSON reorder/whitespace:\n  %s\n  %s", a, b)
	}
}

func TestLivelockDetectorFiresOnThirdIdenticalFailure(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{
		TraceID:     "trace-1",
		Tool:        "Bash",
		ArgsDigest:  "sha256:abc",
		Verdict:     "DENY",
		Reason:      "POLICY_BLOCK",
		Disposition: "TERMINAL",
	}
	for i := 1; i <= 2; i++ {
		if env, ok := d.ObserveFailure(obs); ok {
			t.Fatalf("turn %d fired early: %+v", i, env)
		}
	}
	env, ok := d.ObserveFailure(obs)
	if !ok {
		t.Fatal("third identical failure did not fire")
	}
	if env.Event != LivelockEvent || env.RepeatCount != 3 || env.Tool != "Bash" || env.ArgsDigest != "sha256:abc" {
		t.Fatalf("envelope = %+v, want LIVELOCK_DETECTED repeat=3 for Bash@sha256:abc", env)
	}
	if env.SuggestedChange != "change_approach_fetch_merge_escalate_or_not_yet_with_witness" {
		t.Fatalf("suggested change = %q", env.SuggestedChange)
	}
}

func TestLivelockDetectorResetsOnDifferentFailureAndClear(t *testing.T) {
	d := NewLivelockDetector(3)
	obs := LivelockObservation{TraceID: "trace-1", Tool: "Bash", ArgsDigest: "sha256:one", Verdict: "DENY", Reason: "POLICY_BLOCK"}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("first failure fired")
	}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("second failure fired")
	}
	changed := obs
	changed.ArgsDigest = "sha256:two"
	if _, ok := d.ObserveFailure(changed); ok {
		t.Fatal("changed call must reset the consecutive run")
	}
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("returning to the first call starts a fresh run, not repeat=3")
	}
	d.Clear("trace-1")
	if _, ok := d.ObserveFailure(obs); ok {
		t.Fatal("clear must reset the run")
	}
}
