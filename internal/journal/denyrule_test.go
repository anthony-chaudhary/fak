package journal

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// denyEvent builds the shape the kernel emits for a refusal: EvDeny carrying the
// refusing rung's verdict (kernel.Fold returns the winning rung's verdict whole,
// Meta included, and kernel.Decide emits it as &v).
func denyEvent(by string, reason abi.ReasonCode, claim string, meta map[string]string) abi.Event {
	return abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"cd fak && rm -rf build"}`)},
		},
		Verdict: &abi.Verdict{
			Kind:    abi.VerdictDeny,
			Reason:  reason,
			By:      by,
			Payload: abi.WitnessPayload{Claim: claim},
			Meta:    meta,
		},
	}
}

// TestDenyRowCarriesRungRuleID is the win: two refusals that are IDENTICAL on
// (tool, verdict, reason, by) — the only machine-routable keys a refusal row
// carried before #5863 — now separate on deny_rule. Both shapes are taken from
// real rows in .dispatch-runs/guard-audit/*.jsonl, where 160 recursive-delete
// refusals and 4 out-of-tree-write refusals share ("monitor","POLICY_BLOCK") and
// can only be told apart by prose-matching the witness.
func TestDenyRowCarriesRungRuleID(t *testing.T) {
	j := OpenMemory()
	j.Emit(denyEvent("monitor", abi.ReasonPolicyBlock,
		"Bash.command rm_rf recursive/forced delete",
		map[string]string{abi.MetaDenyRule: abi.DenyRuleRmRf}))
	j.Emit(denyEvent("monitor", abi.ReasonPolicyBlock,
		"Bash.command out_of_tree_write",
		map[string]string{abi.MetaDenyRule: abi.DenyRuleOutOfTreeWrite}))
	// The gitgate cluster #5863 measured: five terminal deaths whose args_label
	// read only "command=cd fak", all citing ("gitgate","POLICY_BLOCK").
	j.Emit(denyEvent("gitgate", abi.ReasonPolicyBlock,
		"skip-hooks refused: `git config core.hooksPath ...` persistently redirects the hooks directory",
		map[string]string{"fix": "…", abi.MetaDenyRule: abi.DenyRuleSkipHooks}))
	j.Emit(denyEvent("gitgate", abi.ReasonPolicyBlock,
		"off-trunk refused: `git checkout -b` opens an unmanaged branch",
		map[string]string{"fix": "…", abi.MetaDenyRule: abi.DenyRuleOffTrunk}))

	rows := j.Recent(0)
	if len(rows) != 4 {
		t.Fatalf("wrote %d rows, want 4", len(rows))
	}
	want := []string{
		abi.DenyRuleRmRf, abi.DenyRuleOutOfTreeWrite,
		abi.DenyRuleSkipHooks, abi.DenyRuleOffTrunk,
	}
	seen := map[string]bool{}
	for i, r := range rows {
		if r.DenyRule != want[i] {
			t.Errorf("row %d DenyRule = %q, want %q", i, r.DenyRule, want[i])
		}
		key := r.By + "/" + r.Reason
		if seen[key] && r.DenyRule == "" {
			t.Errorf("row %d collapses onto %q with no rule id — the #5863 gap", i, key)
		}
		seen[key] = true
	}
	if n, err := VerifyRows(rows); err != nil || n != 4 {
		t.Fatalf("DenyRule must not affect hash-chain verification: n=%d err=%v", n, err)
	}
}

// TestDenyRuleDropsUnregisteredValueWhole is the adversarial disclosure test. A
// rung (or anything that can influence Verdict.Meta) stamping a secret-shaped
// value must leave the field EMPTY — not scrubbed, not truncated, not partially
// filtered. A character-filter defense would emit "MYVARhunter2" here, which is
// precisely the live leak #5863's first commit fixed in args_label. The whole row
// is re-marshalled and searched so a leak cannot hide in another field.
func TestDenyRuleDropsUnregisteredValueWhole(t *testing.T) {
	hostile := []struct{ name, val, needle string }{
		{"env assignment", "MYVAR=hunter2", "hunter2"},
		{"api key", "sk-ant-api03-NOTAREALKEYBUTSHAPED", "NOTAREALKEY"},
		{"bearer header", "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9", "eyJhbGciOiJIUzI1NiJ9"},
		{"password kv", "password=correct-horse-battery", "correct-horse-battery"},
		{"absolute path", "/home/agent/.ssh/id_rsa", "id_rsa"},
		{"near miss id", "rm_rfx", "rm_rfx"},
		{"id plus payload", "rm_rf=hunter2", "hunter2"},
	}
	for _, h := range hostile {
		t.Run(h.name, func(t *testing.T) {
			j := OpenMemory()
			j.Emit(denyEvent("monitor", abi.ReasonPolicyBlock, "Bash.command bound",
				map[string]string{abi.MetaDenyRule: h.val}))
			rows := j.Recent(0)
			if len(rows) != 1 {
				t.Fatalf("wrote %d rows, want 1", len(rows))
			}
			if rows[0].DenyRule != "" {
				t.Fatalf("DenyRule = %q for unregistered %q; want \"\" (drop whole, never scrub)",
					rows[0].DenyRule, h.val)
			}
			b, err := json.Marshal(rows[0])
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if strings.Contains(string(b), h.needle) {
				t.Fatalf("row leaked %q: %s", h.needle, b)
			}
			if strings.Contains(string(b), `"deny_rule"`) {
				t.Fatalf("empty DenyRule must be omitted from the row: %s", b)
			}
		})
	}
}

// TestDenyRuleCannotExceedArgsLabelBound holds the new field to the bound the
// existing redacted field already obeys. It cannot be violated by construction —
// every value is a declared literal — so this is a REGRESSION GUARD on the
// vocabulary, not a fix for an observed overflow: it fails the moment someone
// declares a long id or swaps membership for a character filter.
func TestDenyRuleCannotExceedArgsLabelBound(t *testing.T) {
	for _, id := range abi.DenyRuleIDs() {
		if len(id) > maxArgsLabelLen {
			t.Errorf("declared rule id %q is %d bytes, over the %d-byte bounded-disclosure budget",
				id, len(id), maxArgsLabelLen)
		}
	}
	// A pathologically long stamped value never lands long: it either resolves to a
	// declared literal (dropping every trailing byte) or is rejected outright.
	// Truncating a long value INTO range — the shape a scrub-and-bound field uses —
	// is what this forbids.
	declared := map[string]bool{}
	for _, id := range abi.DenyRuleIDs() {
		declared[id] = true
	}
	oversize := []string{
		strings.Repeat("rm_rf ", 4000),         // 24000 bytes whose head is a real id
		strings.Repeat("a", 24000),             // 24000 bytes that are not
		"rm_rf" + strings.Repeat("Z", 24000),   // a real id fused to a payload, no separator
		strings.Repeat("sk-ant-api03-x", 2000), // an oversize secret shape
	}
	for _, val := range oversize {
		j := OpenMemory()
		j.Emit(denyEvent("monitor", abi.ReasonPolicyBlock, "Bash.command bound",
			map[string]string{abi.MetaDenyRule: val}))
		rows := j.Recent(0)
		if len(rows) != 1 {
			t.Fatalf("wrote %d rows, want 1", len(rows))
		}
		got := rows[0].DenyRule
		if len(got) > maxArgsLabelLen {
			t.Fatalf("DenyRule = %d bytes for a %d-byte stamp, over the %d-byte budget",
				len(got), len(val), maxArgsLabelLen)
		}
		if got != "" && !declared[got] {
			t.Fatalf("DenyRule = %q for a %d-byte stamp; not a declared literal — "+
				"the field must never be a truncation of its input", got, len(val))
		}
	}
	// The one shape that must be rejected outright: a real id FUSED to a payload
	// with no separator. Admitting it would mean the function is prefix-matching
	// rather than testing set membership.
	j := OpenMemory()
	j.Emit(denyEvent("monitor", abi.ReasonPolicyBlock, "Bash.command bound",
		map[string]string{abi.MetaDenyRule: "rm_rf" + strings.Repeat("Z", 24000)}))
	if got := j.Recent(0)[0].DenyRule; got != "" {
		t.Fatalf("DenyRule = %q for a fused id+payload stamp; want \"\"", got)
	}
}

// TestDenyRuleAbsentWhenRungStampsNone keeps every pre-#5863 refusal shape valid:
// a rung that declares no rule id (ctxmmu, normgate, ifc-sink today) still writes
// its row, unchanged, with the field omitted.
func TestDenyRuleAbsentWhenRungStampsNone(t *testing.T) {
	j := OpenMemory()
	j.Emit(denyEvent("ctxmmu", abi.ReasonTrustViolation,
		"ctxmmu TRUST_VIOLATION injection_marker quarantine_id=q1", nil))
	j.Emit(denyEvent("monitor", abi.ReasonSelfModify, ".git/",
		map[string]string{"fix": "edit a file outside the guarded tree"}))
	rows := j.Recent(0)
	if len(rows) != 2 {
		t.Fatalf("wrote %d rows, want 2", len(rows))
	}
	for i, r := range rows {
		if r.DenyRule != "" {
			t.Errorf("row %d DenyRule = %q, want \"\" (no rule stamped)", i, r.DenyRule)
		}
		if r.Witness == "" {
			t.Errorf("row %d lost its witness claim", i)
		}
	}
}

// TestDenyRuleLeavesExistingChainVerifiable is the compatibility guard: deny_rule
// is appended AFTER Hash and is not in chainHash's explicit pre-image, so a
// journal written before this field verifies byte-for-byte, and a row that gains
// the field hashes identically to the same row without it.
func TestDenyRuleLeavesExistingChainVerifiable(t *testing.T) {
	base := Row{Seq: 1, TSUnixNano: 7, Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "monitor"}
	withRule := base
	withRule.DenyRule = abi.DenyRuleRmRf
	if a, b := chainHash("", base), chainHash("", withRule); a != b {
		t.Fatalf("deny_rule changed the chain pre-image: %s vs %s", a, b)
	}
}
