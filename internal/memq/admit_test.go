package memq_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

// #2912 acceptance: a #3006-style 74 KiB verbatim write is refused DETERMINISTICALLY
// by the deny-by-structure rule set, with a reason from the closed vocabulary — never
// by asking a model to honor a "Do NOT capture" list.
func TestAdjudicateMemoryWrite_RefusesOversizeVerbatim(t *testing.T) {
	// A skill body copied verbatim into one memory entry (the #3006 shape): ~74 KiB.
	body := []byte("---\nname: leaked-skill\ndescription: a whole skill body\nmetadata:\n  type: reference\n---\n\n" +
		strings.Repeat("## Section\nThis is a verbatim paragraph copied wholesale from a skill body.\n", 1024))
	if len(body) <= memq.MaxDurableFactBytes {
		t.Fatalf("test fixture must exceed the %d-byte bound, got %d", memq.MaxDurableFactBytes, len(body))
	}

	first := memq.AdjudicateMemoryWrite(body)
	if first.Admit {
		t.Fatalf("a %d-byte verbatim write must be refused, got admit", len(body))
	}
	if first.Reason != memq.RefuseOversizeVerbatim {
		t.Fatalf("reason = %q, want %q", first.Reason, memq.RefuseOversizeVerbatim)
	}
	if !strings.Contains(first.Detail, "exceeds") {
		t.Fatalf("detail must name the size overrun, got %q", first.Detail)
	}

	// Deterministic: same body, same verdict, every time.
	for i := 0; i < 3; i++ {
		if got := memq.AdjudicateMemoryWrite(body); got != first {
			t.Fatalf("adjudication is not deterministic: run %d = %+v, want %+v", i, got, first)
		}
	}
}

// The two other junk classes the Hermes prose list names but cannot enforce are
// refused structurally when the fact IS the event (a thin narrative).
func TestAdjudicateMemoryWrite_RefusesTransientAndNegativeTool(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"connection reset", "connection reset by the gateway while streaming; will retry.", memq.RefuseTransientError},
		{"timed out", "the request timed out after 30s talking to the model host.", memq.RefuseTransientError},
		{"rate limited", "rate limited by the provider on this run (429 wall).", memq.RefuseTransientError},
		{"failed invocation", "failed to run the gateway smoke test this session.", memq.RefuseNegativeToolClaim},
		{"exited nonzero", "the build command exited with status 2 just now.", memq.RefuseNegativeToolClaim},
		{"no output", "no output from the probe call this time.", memq.RefuseNegativeToolClaim},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := memq.AdjudicateMemoryWrite([]byte(tc.body))
			if v.Admit {
				t.Fatalf("%q must be refused, got admit", tc.body)
			}
			if v.Reason != tc.want {
				t.Fatalf("reason = %q, want %q (detail %q)", v.Reason, tc.want, v.Detail)
			}
		})
	}
}

// #2836: the fourth "Do NOT capture" class — a one-off session narrative bound to
// THIS run by a temporal deictic — is refused structurally, with its own closed-
// vocabulary reason, before it can land in durable memory. These are the shapes the
// Hermes prose list names ("one-off narratives") but a review prompt cannot enforce.
func TestAdjudicateMemoryWrite_RefusesOneOffNarrative(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"this session", "Spent this session refactoring the dispatch loop."},
		{"this run", "Fixed three flaky tests this run."},
		{"this turn", "The user asked me to summarize the PR this turn."},
		{"just now", "Just now the operator restarted the fleet."},
		{"a moment ago", "Rebased the branch a moment ago."},
		{"earlier today", "Reviewed the gateway diff earlier today."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := memq.AdjudicateMemoryWrite([]byte(tc.body))
			if v.Admit {
				t.Fatalf("%q must be refused as a one-off narrative, got admit", tc.body)
			}
			if v.Reason != memq.RefuseOneOffNarrative {
				t.Fatalf("reason = %q, want %q (detail %q)", v.Reason, memq.RefuseOneOffNarrative, v.Detail)
			}
		})
	}
}

// The conservative half of the contract: a legitimate distilled fact is admitted
// even when it MENTIONS a retry, a missing binary, or a rate-limit header. Over-
// refusing an honest report is the issue's named risk — these must all pass.
func TestAdjudicateMemoryWrite_AdmitsLegitimateFacts(t *testing.T) {
	ok := []string{
		"The gateway retries on a rate-limit header with exponential backoff.",
		"Set the read timeout to 30s for the model host.",
		"grep is not on PATH on this Windows host; use Select-String instead.",
		"raw grep, wc, cat, and find / are NOT on PATH here — use the PowerShell equivalents.",
		"The user prefers concise answers and no preamble.",
		"Commit by explicit path on main; never git add -A in the shared tree.",
		// #2836 near-misses: a durable fact that names a session/run/turn WITHOUT the
		// deictic binding must NOT trip the one-off arm — over-refusing these is the
		// issue's named risk.
		"The session token must be refreshed every 30 minutes.",
		"A dry run of the migration is safe to repeat.",
		"Each turn should start with a memory recall of the task intent.",
		"The runtime resolves the module from the repo root.",
		// A rich, multi-line distilled fact that happens to discuss a failure mode
		// without being a bare failure report — must not trip the thin-fact rules.
		"When the gateway sees a 429 it treats it as retryable and backs off; this is\n" +
			"the durable policy, not a one-off event. The retry budget is three attempts\n" +
			"before the request is surfaced to the caller as a hard error.",
	}
	for _, body := range ok {
		v := memq.AdjudicateMemoryWrite([]byte(body))
		if !v.Admit {
			t.Fatalf("legitimate fact refused (%s): %q => %q %s", "over-refusal", body, v.Reason, v.Detail)
		}
		if v.Reason != memq.AdmitOK {
			t.Fatalf("admitted fact must carry reason %q, got %q", memq.AdmitOK, v.Reason)
		}
	}
}

// The verdict is closed: Reason is always a known token, and AdmitOK iff Admit.
func TestAdjudicateMemoryWrite_ClosedVocabulary(t *testing.T) {
	known := map[string]bool{
		memq.AdmitOK:                 true,
		memq.RefuseOversizeVerbatim:  true,
		memq.RefuseTransientError:    true,
		memq.RefuseNegativeToolClaim: true,
		memq.RefuseOneOffNarrative:   true,
	}
	bodies := [][]byte{
		[]byte(""),
		[]byte("a normal distilled fact"),
		[]byte("connection refused right now"),
		[]byte("failed to run the probe"),
		[]byte("rebased the branch just now"),
		[]byte(strings.Repeat("x", memq.MaxDurableFactBytes+1)),
	}
	for _, b := range bodies {
		v := memq.AdjudicateMemoryWrite(b)
		if !known[v.Reason] {
			t.Fatalf("verdict carried an out-of-vocabulary reason %q for %q", v.Reason, string(b))
		}
		if v.Admit != (v.Reason == memq.AdmitOK) {
			t.Fatalf("Admit=%v must match Reason==AdmitOK (reason %q)", v.Admit, v.Reason)
		}
	}
}

// The seam is governed: AddPromotedIfDurable refuses a #3006-style verbatim durable
// write with ErrWriteRefused BEFORE it reaches storage, while a small clean fact lands.
func TestAddPromotedIfDurable_RefusesByStructure(t *testing.T) {
	m := memq.NewMemStore()

	huge := []byte(strings.Repeat("verbatim skill paragraph\n", 4096)) // ~100 KiB
	cell, err := m.AddIfDurable("assistant", "note", memq.DurabilityDurable, huge, false, memq.ReclassifyNone)
	if err == nil {
		t.Fatal("a 100 KiB durable write must be refused by structure")
	}
	if !errors.Is(err, memq.ErrWriteRefused) {
		t.Fatalf("error = %v, want wrapping ErrWriteRefused", err)
	}
	if cell.ID != "" {
		t.Fatalf("a refused write must return a zero Cell, got %+v", cell)
	}
	if !strings.Contains(err.Error(), memq.RefuseOversizeVerbatim) {
		t.Fatalf("refusal must name the closed-vocabulary reason, got %v", err)
	}

	// Structure decides, not consent: an explicit reclassification cannot buy the
	// oversize write in.
	if _, err := m.AddIfDurable("assistant", "note", memq.DurabilityDurable, huge, false, memq.ReclassifyExplicitConsent); !errors.Is(err, memq.ErrWriteRefused) {
		t.Fatalf("reclassification must not override a structural refusal, got %v", err)
	}

	// A small, distilled fact lands normally.
	good, err := m.AddIfDurable("user", "user", memq.DurabilityDurable, []byte("The user prefers concise answers."), false, memq.ReclassifyNone)
	if err != nil {
		t.Fatalf("a clean distilled fact must be admitted, got %v", err)
	}
	if good.ID == "" {
		t.Fatal("an admitted write must return a real Cell")
	}
}
