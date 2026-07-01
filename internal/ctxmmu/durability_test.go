package ctxmmu_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestClassifyDurabilityViaAdmit pins the rung-1 write-time durability classifier
// (S7, issue #82) through the SHIPPED gate: MMU.Admit stamps the class on the OPEN
// Verdict.Meta map, orthogonal to the trust Kind, with no ABI/golden-freeze cost.
func TestClassifyDurabilityViaAdmit(t *testing.T) {
	ctx := context.Background()

	cases := []struct {
		name string
		tool string
		body string
		want string
	}{
		// punctual/progressive deictics + bare clock times => turn.
		{"bare clock time", "clock", "it's 3pm", ctxmmu.DurabilityTurn},
		{"deictic now", "status", "the build is currently running", ctxmmu.DurabilityTurn},
		{"deictic today", "calendar", "today is a holiday", ctxmmu.DurabilityTurn},
		// habitual/stative frames => durable (the earned exception).
		{"first-person preference", "read_memory", "I prefer to work in the afternoon", ctxmmu.DurabilityDurable},
		{"third-person preference", "read_memory", "the user prefers afternoons", ctxmmu.DurabilityDurable},
		{"identity name frame", "read_profile", "my name is Ada", ctxmmu.DurabilityDurable},
		{"identity role frame", "read_profile", "my role is backend engineer", ctxmmu.DurabilityDurable},
		{"team convention", "read_memory", "we prefer dark mode", ctxmmu.DurabilityDurable},
		// a STRONG durable signal stays durable even when it mentions a time (a genuine
		// standing preference is not demoted just for naming a clock).
		{"strong durable with a clock time", "read_memory", "I prefer meetings at 3pm", ctxmmu.DurabilityDurable},
		// WEAK copular/imperative bodies that report TRANSIENT state must NOT class durable
		// — biasing to the cheap error (a poltergeist promotion is worse than absence).
		{"transient copular my-build", "status", "my build is failing right now", ctxmmu.DurabilityTurn},
		{"transient copular my-meeting", "calendar", "my meeting is at 3pm today", ctxmmu.DurabilityTurn},
		{"transient i-am-a", "status", "I am a bit busy right now", ctxmmu.DurabilityTurn},
		{"transient imperative call-me-back", "message", "call me back later", ctxmmu.DurabilityTurn},
		{"transient we-work-schedule", "status", "we work until 5pm today", ctxmmu.DurabilityTurn},
		// explicit session-scoped frame => session (beats the bare "today" deictic).
		{"session task", "task", "today's task is the durability gate", ctxmmu.DurabilitySession},
		// nothing matched => fail-closed turn (the cheap default).
		{"unmatched body fails closed to turn", "read_file", "row 42 = 17", ctxmmu.DurabilityTurn},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := ctxmmu.New()
			c := &abi.ToolCall{Tool: tc.tool}
			r := &abi.Result{
				Call:    c,
				Status:  abi.StatusOK,
				Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(tc.body)},
			}
			v := m.Admit(ctx, c, r)
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("benign body %q: want VerdictAllow, got %v (reason %s)", tc.body, v.Kind, abi.ReasonName(v.Reason))
			}
			if v.Meta == nil {
				t.Fatalf("benign body %q: verdict carries no Meta map (durability tag missing)", tc.body)
			}
			if got := v.Meta[ctxmmu.DurabilityKey]; got != tc.want {
				t.Fatalf("durability of %q: want %q, got %q", tc.body, tc.want, got)
			}
		})
	}
}

// TestClassifyTextAgreesWithAdmitPath proves the exported chat-shaped wrapper runs the
// SAME rung-1 prior as the admit path: a budget-reset carryover builder calling
// ClassifyText on a transcript line gets the identical verdict MMU.Admit would stamp on
// the same words — "it's 3pm" => turn (dropped on reset), "I prefer afternoons" =>
// durable (kept on reset) — so the reset reuses the shipped classifier, not a fork.
func TestClassifyTextAgreesWithAdmitPath(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		role, content, want string
	}{
		{"user", "it's 3pm", ctxmmu.DurabilityTurn},
		{"user", "the build is currently running", ctxmmu.DurabilityTurn},
		{"user", "I prefer to work in the afternoon", ctxmmu.DurabilityDurable},
		{"assistant", "the user prefers afternoons", ctxmmu.DurabilityDurable},
		{"user", "my name is Ada", ctxmmu.DurabilityDurable},
		{"user", "today's task is the durability gate", ctxmmu.DurabilitySession},
		{"tool", "row 42 = 17", ctxmmu.DurabilityTurn}, // unmatched => fail-closed turn
	}
	for _, tc := range cases {
		// ClassifyText verdict...
		got := ctxmmu.ClassifyText(tc.role, tc.content)
		if got != tc.want {
			t.Fatalf("ClassifyText(%q) = %q, want %q", tc.content, got, tc.want)
		}
		// ...must equal the admit-path verdict on the identical bytes.
		m := ctxmmu.New()
		c := &abi.ToolCall{Tool: "x"}
		r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(tc.content)}}
		v := m.Admit(ctx, c, r)
		if admit := v.Meta[ctxmmu.DurabilityKey]; admit != got {
			t.Fatalf("ClassifyText(%q)=%q disagrees with admit-path %q", tc.content, got, admit)
		}
	}
}

func TestDurabilityPolicyCoversAllContextLedgerClasses(t *testing.T) {
	cases := []struct {
		class          string
		wantExpiry     string
		requiresExpiry bool
	}{
		{ctxmmu.DurabilityTurn, ctxmmu.ExpiryPolicyTurn, false},
		{ctxmmu.DurabilitySession, ctxmmu.ExpiryPolicySession, false},
		{ctxmmu.DurabilityBounded, ctxmmu.ExpiryPolicyRequired, true},
		{ctxmmu.DurabilityDurable, ctxmmu.ExpiryPolicyNone, false},
		{"unknown", ctxmmu.ExpiryPolicyTurn, false},
	}
	for _, tc := range cases {
		t.Run(tc.class, func(t *testing.T) {
			p := ctxmmu.PolicyForDurability(tc.class)
			if p.ExpiryPolicy != tc.wantExpiry || p.RequiresExpiry != tc.requiresExpiry {
				t.Fatalf("PolicyForDurability(%q) = %+v, want expiry=%q requires=%v",
					tc.class, p, tc.wantExpiry, tc.requiresExpiry)
			}
			if tc.class == "unknown" && p.Class != ctxmmu.DurabilityTurn {
				t.Fatalf("unknown class normalized to %q, want %q", p.Class, ctxmmu.DurabilityTurn)
			}
			if label := ctxmmu.DurabilityLabel(tc.class); label == "" {
				t.Fatalf("DurabilityLabel(%q) is empty", tc.class)
			}
		})
	}
}

// TestDurabilityTagIsAdditiveOnTransform confirms an oversize-benign Transform verdict
// also carries the durability class (the paged-out result is still a write-time fact),
// not only the Allow path.
func TestDurabilityTagIsAdditiveOnTransform(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.New()

	// Oversize, distinct (non-repeating), durable-classed body => Transform + durable.
	var body []byte
	body = append(body, []byte("I prefer afternoons. ")...)
	for i := 0; len(body) <= ctxmmu.OversizeBytes; i++ {
		body = append(body, []byte("distinct-filler-token-")...)
		body = append(body, byte('A'+i%26), byte('0'+i%10), ';')
	}
	c := &abi.ToolCall{Tool: "dump_prefs"}
	r := &abi.Result{Call: c, Status: abi.StatusOK, Payload: abi.Ref{Kind: abi.RefInline, Inline: body}}

	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("oversize benign: want VerdictTransform, got %v", v.Kind)
	}
	if v.Meta[ctxmmu.DurabilityKey] != ctxmmu.DurabilityDurable {
		t.Fatalf("transform verdict durability: want %q, got %q", ctxmmu.DurabilityDurable, v.Meta[ctxmmu.DurabilityKey])
	}
}

// TestQuarantineCarriesNoDurability confirms a sealed (Quarantine) verdict gets NO
// durability class — sealed bytes never promote, so the gate needs none.
func TestQuarantineCarriesNoDurability(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.New()

	c := &abi.ToolCall{Tool: "read_file"}
	// Secret-shaped body => Quarantine.
	r := &abi.Result{Call: c, Status: abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("api_key=sk-abcdef0123456789abcdef0123 leaked")}}

	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("secret body: want VerdictQuarantine, got %v", v.Kind)
	}
	if _, ok := v.Meta[ctxmmu.DurabilityKey]; ok {
		t.Fatalf("quarantine verdict must carry no durability class, got %q", v.Meta[ctxmmu.DurabilityKey])
	}
}
