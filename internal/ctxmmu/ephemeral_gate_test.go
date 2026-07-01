package ctxmmu_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestEphemeralFactRefusedForDurableMemory is the done-condition witness for #1592:
// a timestamp observation, a current-step observation, and a mood observation are each
// refused by default (no explicit reclassification) — the write-time gate must not let
// any of them cross into durable memory.
func TestEphemeralFactRefusedForDurableMemory(t *testing.T) {
	cases := []struct {
		name string
		text string
		want ctxmmu.SituationalKind
	}{
		{"timestamp", "it's 3pm", ctxmmu.SituationalTimestamp},
		{"current_step", "I'm on step 4", ctxmmu.SituationalCurrentStep},
		{"mood", "I am tired today", ctxmmu.SituationalMood},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := ctxmmu.GateEphemeral(tc.text, ctxmmu.ReclassifyNone)
			if out.Allowed {
				t.Fatalf("GateEphemeral(%q) allowed a situational observation with no reclassification, want refused", tc.text)
			}
			if out.Situational != tc.want {
				t.Fatalf("GateEphemeral(%q).Situational = %v, want %v", tc.text, out.Situational, tc.want)
			}
			if out.Reason == "" {
				t.Fatalf("refusal must carry a non-empty Reason (typed outcome, not a silent drop)")
			}
		})
	}
}

// TestEphemeralFactAllowedWithExplicitReclassification proves the gate's default is
// overridable "unless explicitly reclassified" (#1592's done condition): the SAME three
// fixtures that were refused above are allowed once a caller supplies an explicit
// Reclassification (explicit consent, user confirmation, or an established pattern).
func TestEphemeralFactAllowedWithExplicitReclassification(t *testing.T) {
	fixtures := []string{"it's 3pm", "I'm on step 4", "I am tired today"}
	overrides := []ctxmmu.Reclassification{
		ctxmmu.ReclassifyExplicitConsent,
		ctxmmu.ReclassifyUserConfirmed,
		ctxmmu.ReclassifyEstablishedPattern,
	}

	for _, text := range fixtures {
		for _, r := range overrides {
			out := ctxmmu.GateEphemeral(text, r)
			if !out.Allowed {
				t.Fatalf("GateEphemeral(%q, %v) refused despite explicit reclassification: %+v", text, r, out)
			}
			if out.Reclassification != r {
				t.Fatalf("GateEphemeral(%q, %v).Reclassification = %v, want %v", text, r, out.Reclassification, r)
			}
		}
	}
}

// TestEphemeralGateAllowsGenuineDurableFacts proves the gate does not over-refuse: an
// observation that already classifies durable on its own tense (a stated preference,
// an identity fact) is allowed with NO reclassification needed at all — the gate only
// bites situational facts, never ordinary durable writes.
func TestEphemeralGateAllowsGenuineDurableFacts(t *testing.T) {
	cases := []string{
		"I prefer concise answers",
		"my name is Ada",
		"I always want a confirmation before deletes",
	}
	for _, text := range cases {
		out := ctxmmu.GateEphemeral(text, ctxmmu.ReclassifyNone)
		if !out.Allowed {
			t.Fatalf("GateEphemeral(%q) refused a genuinely durable-shaped fact with no reclassification: %+v", text, out)
		}
		if out.Situational != ctxmmu.SituationalNone {
			t.Fatalf("GateEphemeral(%q).Situational = %v, want SituationalNone", text, out.Situational)
		}
	}
}

// TestClassifySituationalNamesTheRightShape pins ClassifySituational's own verdict
// (independent of the gate) for the issue's three named fixture categories plus the
// catch-all "other" bucket and the "not situational at all" bucket.
func TestClassifySituationalNamesTheRightShape(t *testing.T) {
	cases := []struct {
		text string
		want ctxmmu.SituationalKind
	}{
		{"it's 3pm", ctxmmu.SituationalTimestamp},
		{"today is a holiday", ctxmmu.SituationalTimestamp},
		{"I'm on step 4", ctxmmu.SituationalCurrentStep},
		{"we're on step 3 of the wizard", ctxmmu.SituationalCurrentStep},
		{"I am tired today", ctxmmu.SituationalMood},
		{"I'm frustrated with this bug", ctxmmu.SituationalMood},
		{"row 42 = 17", ctxmmu.SituationalOther},
		{"I prefer to work in the afternoon", ctxmmu.SituationalNone},
	}
	for _, tc := range cases {
		if got := ctxmmu.ClassifySituational(tc.text); got != tc.want {
			t.Errorf("ClassifySituational(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

// TestReclassificationStringNeverEmpty guards the audit-render contract: every
// Reclassification value (including an out-of-range one) renders a non-empty string,
// so a log line can never carry a blank override field.
func TestReclassificationStringNeverEmpty(t *testing.T) {
	vals := []ctxmmu.Reclassification{
		ctxmmu.ReclassifyNone,
		ctxmmu.ReclassifyExplicitConsent,
		ctxmmu.ReclassifyUserConfirmed,
		ctxmmu.ReclassifyEstablishedPattern,
		ctxmmu.Reclassification(99),
	}
	for _, v := range vals {
		if v.String() == "" {
			t.Errorf("Reclassification(%d).String() is empty", int(v))
		}
	}
}
