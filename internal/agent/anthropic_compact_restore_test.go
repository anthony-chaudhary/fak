package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// anthropic_compact_restore_test.go — witnesses for the restore HANDLE the tombstone now carries.
// The tombstone excerpt is orientation; this feature adds a content-address (id=<sha256hex>) to the
// stub AND surfaces the dropped turn's bytes on the outcome (CompactOutcome.RestoreID /
// RestoreExcerpt / RestoreBytes), so a gateway can stash digest→bytes and a resuming model can page
// the FULL originating task back in via fak_context_restore(id).

// TestCompactEmitsRestoreHandle: a fired tombstone embeds an id=<hex> handle in the stub, and the
// outcome carries the SAME id, the excerpt, and the verbatim dropped-turn bytes — with the id being
// the sha256 content-address OF those bytes (so the handle a gateway stashes and the one the model
// reads in the stub are one address).
func TestCompactEmitsRestoreHandle(t *testing.T) {
	const task = "rotate the auth tokens and update the runbook — the ORIGINATING task"
	raw := tombstoneTaskBody(t, 14, task)

	fired := false
	for _, budget := range []int{120, 200, 350, 550} {
		out, oc := CompactAnthropicHistoryWithOutcome(raw, budget)
		if oc.Reason != CompactReasonNone {
			continue
		}
		fired = true

		if oc.RestoreID == "" {
			t.Fatalf("budget=%d: fired tombstone carried no RestoreID", budget)
		}
		if len(oc.RestoreBytes) == 0 {
			t.Fatalf("budget=%d: fired tombstone carried no RestoreBytes", budget)
		}
		// The id must be the content-address OF the bytes it hands back — a stashing gateway keys on
		// exactly this, and the model reads the same hex in the stub.
		if want := originatingTaskDigestID(oc.RestoreBytes); oc.RestoreID != want {
			t.Fatalf("budget=%d: RestoreID %q is not the digest of RestoreBytes %q", budget, oc.RestoreID, want)
		}
		// The excerpt on the outcome is the same one embedded in the stub.
		if !strings.Contains(oc.RestoreExcerpt, "rotate the auth tokens") {
			t.Fatalf("budget=%d: RestoreExcerpt missing the task text: %q", budget, oc.RestoreExcerpt)
		}
		// The RestoreBytes are the verbatim FIRST user turn — decodable, role user, carrying the task.
		var msg struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(oc.RestoreBytes, &msg); err != nil {
			t.Fatalf("budget=%d: RestoreBytes not valid JSON: %v", budget, err)
		}
		if msg.Role != "user" {
			t.Fatalf("budget=%d: RestoreBytes role = %q, want user", budget, msg.Role)
		}
		if txt, ok := elementTextContent(oc.RestoreBytes); !ok || !strings.Contains(txt, "rotate the auth tokens") {
			t.Fatalf("budget=%d: RestoreBytes do not carry the originating task: ok=%v text=%q", budget, ok, txt)
		}

		// The id=<hex> handle rides IN the stub, on the same tombstone line as the excerpt.
		content := stubContentText(t, out)
		if !strings.Contains(content, compactRestoreIDField+oc.RestoreID) {
			t.Fatalf("budget=%d: stub missing the id=<hex> handle; got %q", budget, content)
		}
	}
	if !fired {
		t.Fatalf("tombstone body never compacted across the swept budgets")
	}
}

// TestCompactNoRestoreHandleWhenTaskProtected: when the originating task survives in the protected
// prefix, there is no tombstone — so no id is embedded and the outcome carries no restore handle.
func TestCompactNoRestoreHandleWhenTaskProtected(t *testing.T) {
	raw := realisticBody(t, 16)
	fired := false
	for _, budget := range []int{150, 250, 500} {
		out, oc := CompactAnthropicHistoryWithOutcome(raw, budget)
		if oc.Reason != CompactReasonNone {
			continue
		}
		fired = true
		if oc.RestoreID != "" || len(oc.RestoreBytes) != 0 {
			t.Fatalf("budget=%d: protected task must mint no restore handle; got id=%q bytes=%d", budget, oc.RestoreID, len(oc.RestoreBytes))
		}
		if strings.Contains(string(out), compactRestoreIDField) {
			t.Fatalf("budget=%d: no tombstone ⇒ the stub must carry no id= handle", budget)
		}
	}
	if !fired {
		t.Fatalf("realisticBody never compacted across the swept budgets")
	}
}

// TestOriginatingTaskDigestIDStable: the content-address is a pure, stable function of the bytes —
// identical input ⇒ identical id, and it is the 64-char sha256 hex the ctxplan.Digest scheme uses.
func TestOriginatingTaskDigestIDStable(t *testing.T) {
	b := []byte(`{"role":"user","content":"the task"}`)
	id1, id2 := originatingTaskDigestID(b), originatingTaskDigestID(b)
	if id1 != id2 {
		t.Fatalf("digest not stable: %q vs %q", id1, id2)
	}
	if len(id1) != 64 {
		t.Fatalf("digest length = %d, want 64 (sha256 hex)", len(id1))
	}
	if originatingTaskDigestID([]byte(`{"role":"user","content":"other"}`)) == id1 {
		t.Fatalf("distinct bytes produced the same digest")
	}
}
