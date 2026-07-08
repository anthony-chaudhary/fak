package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDispatchRolloutStatusDemoJSON drives the demo fixture through the JSON
// surface and asserts the load-bearing SHADOW invariant — a dry-run applies
// NOTHING (any_applied stays false, every row applied=false) — plus the delta
// tally and that exactly the routine cheaper item is a canary candidate.
func TestDispatchRolloutStatusDemoJSON(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runDispatchRolloutStatus(&out, &errBuf, []string{"--demo", "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	var rep struct {
		Schema         string `json:"schema"`
		Mode           string `json:"mode"`
		Items          int    `json:"items"`
		Same           int    `json:"same"`
		Cheaper        int    `json:"cheaper"`
		Refused        int    `json:"refused"`
		CanaryEligible int    `json:"canary_eligible"`
		AnyApplied     bool   `json:"any_applied"`
		Rows           []struct {
			ID            string `json:"id"`
			Delta         string `json:"delta"`
			InCanaryScope bool   `json:"in_canary_scope"`
			Applied       bool   `json:"applied"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, out.String())
	}
	if rep.Schema == "" || rep.Mode != "shadow" {
		t.Fatalf("report header wrong: schema=%q mode=%q", rep.Schema, rep.Mode)
	}
	if rep.Items != 5 {
		t.Fatalf("items = %d, want 5", rep.Items)
	}
	// The whole point of a shadow dry-run: it applies nothing, ever.
	if rep.AnyApplied {
		t.Fatalf("a shadow readout must apply nothing (any_applied=true):\n%s", out.String())
	}
	for _, row := range rep.Rows {
		if row.Applied {
			t.Fatalf("row %q applied in a shadow readout", row.ID)
		}
	}
	// watchdog-1 (T1->T3) and impl-3 (T1->T3) are cheaper; status-2 (T3->T3) and
	// release-4 (T1->T1) are same; starved-5 has no capable seat -> refused.
	if rep.Cheaper != 2 || rep.Same != 2 || rep.Refused != 1 {
		t.Fatalf("delta tally: want cheaper=2 same=2 refused=1, got cheaper=%d same=%d refused=%d",
			rep.Cheaper, rep.Same, rep.Refused)
	}
	// Only the routine cheaper item (watchdog-1) is a canary candidate; the cheaper
	// impl-3 is normal-impl, OUT of canary scope — scope is class-gated, not price.
	if rep.CanaryEligible != 1 {
		t.Fatalf("canary-eligible: want 1 (routine cheaper only), got %d", rep.CanaryEligible)
	}
	for _, row := range rep.Rows {
		if row.ID == "impl-3" && row.InCanaryScope {
			t.Fatalf("impl-3 (normal-impl) must NOT be in canary scope even though it is cheaper")
		}
	}
}

// TestDispatchRolloutStatusDemoRender checks the human readout names the mode, the
// applied-nothing invariant, the columns, and the pending-parity caveat.
func TestDispatchRolloutStatusDemoRender(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runDispatchRolloutStatus(&out, &errBuf, []string{"--demo"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	for _, want := range []string{
		"SHADOW readout", "applied=false", "WOULD-CHOOSE", "CANARY",
		"canary-eligible=", "PENDING PARITY", "cheaper",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, out.String())
		}
	}
}

// TestDispatchRolloutStatusFromFile drives a --in JSON file end to end: a routine
// item currently on the frontier seat WOULD choose the cheapest, and is a canary
// candidate — but nothing is applied.
func TestDispatchRolloutStatusFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "items.json")
	body := `[
	  {"id": "watchdog", "class": "routine", "current_model_tier": 1,
	   "labels": ["tier/T2-required", "tier/T2-optimal"],
	   "accounts": [
	     {"account": "frontier", "model_tier": 1, "available": true},
	     {"account": "small", "model_tier": 3, "available": true}
	   ]}
	]`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var out, errBuf bytes.Buffer
	if code := runDispatchRolloutStatus(&out, &errBuf, []string{"--in", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	var rep struct {
		AnyApplied     bool `json:"any_applied"`
		CanaryEligible int  `json:"canary_eligible"`
		Rows           []struct {
			ID              string `json:"id"`
			CurrentTier     int    `json:"current_model_tier"`
			WouldChooseTier int    `json:"would_choose_model_tier"`
			Delta           string `json:"delta"`
			InCanaryScope   bool   `json:"in_canary_scope"`
			Applied         bool   `json:"applied"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out.String())
	}
	if rep.AnyApplied {
		t.Fatalf("shadow readout applied something")
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(rep.Rows))
	}
	r := rep.Rows[0]
	if r.CurrentTier != 1 || r.WouldChooseTier != 3 || r.Delta != "cheaper" {
		t.Fatalf("routine work should shadow current=T1 would-choose=T3 cheaper, got %+v", r)
	}
	if !r.InCanaryScope || r.Applied {
		t.Fatalf("routine cheaper item is a canary candidate but must not be applied, got %+v", r)
	}
	if rep.CanaryEligible != 1 {
		t.Fatalf("canary-eligible = %d, want 1", rep.CanaryEligible)
	}
}

// TestDispatchRolloutStatusBadClass checks a typo'd work class fails loud rather
// than silently folding as the unknown-class conservative frontier route.
func TestDispatchRolloutStatusBadClass(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`[{"id":"x","class":"routiney"}]`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	var out, errBuf bytes.Buffer
	if code := runDispatchRolloutStatus(&out, &errBuf, []string{"--in", path}); code != 1 {
		t.Fatalf("exit = %d, want 1 for an unknown class (stderr: %s)", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "unknown class") {
		t.Fatalf("error should name the unknown class, got: %s", errBuf.String())
	}
}
