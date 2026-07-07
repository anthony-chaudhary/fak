package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDispatchTierStatusDemoJSON drives the demo fixture through the JSON surface
// and asserts the readout carries the schema, the five decisions, and both an
// over-tier waste and an under-tier refusal — the two asymmetric verdicts the
// surface exists to make visible — plus the tag-flag on the contradictory issue.
func TestDispatchTierStatusDemoJSON(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runDispatchTierStatus(&out, &errBuf, []string{"--demo", "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	var rep struct {
		Schema           string `json:"schema"`
		Decisions        int    `json:"decisions"`
		OverTier         int    `json:"over_tier"`
		UnderTierRefused int    `json:"under_tier_refused"`
		Rows             []struct {
			Issue    int      `json:"issue"`
			TagFlags []string `json:"tag_flags"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal report: %v\n%s", err, out.String())
	}
	if rep.Schema == "" {
		t.Fatalf("report missing schema:\n%s", out.String())
	}
	if rep.Decisions != 5 {
		t.Fatalf("decisions = %d, want 5", rep.Decisions)
	}
	if rep.OverTier < 1 {
		t.Fatalf("demo must show at least one over-tier waste, got %d", rep.OverTier)
	}
	if rep.UnderTierRefused != 1 {
		t.Fatalf("demo must show exactly one under-tier refusal, got %d", rep.UnderTierRefused)
	}
	var sawContradiction bool
	for _, row := range rep.Rows {
		if row.Issue == 4104 {
			for _, f := range row.TagFlags {
				if f == "model_tier_contradiction" {
					sawContradiction = true
				}
			}
		}
	}
	if !sawContradiction {
		t.Fatalf("issue 4104 (contradictory labels) must carry the contradiction tag flag:\n%s", out.String())
	}
}

// TestDispatchTierStatusDemoRender checks the human readout names the verdicts and
// the modeled-cost caveat.
func TestDispatchTierStatusDemoRender(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runDispatchTierStatus(&out, &errBuf, []string{"--demo"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	for _, want := range []string{"tier decisions", "OVER-TIER", "REFUSED", "modeled cost points", "note:", "tags=[model_tier_contradiction]"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, out.String())
		}
	}
}

// TestDispatchTierStatusFromFile drives a --in JSON file end to end: a routine issue
// with clean labels routes to the cheapest available seat.
func TestDispatchTierStatusFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "issues.json")
	body := `[
	  {"issue": 900, "lane": "docs",
	   "labels": ["tier/T2-required", "tier/T2-optimal"],
	   "accounts": [
	     {"account": "frontier", "model_tier": 1, "available": true},
	     {"account": "small", "model_tier": 3, "available": true}
	   ],
	   "outcome": "shipped"}
	]`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var out, errBuf bytes.Buffer
	if code := runDispatchTierStatus(&out, &errBuf, []string{"--in", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errBuf.String())
	}
	var rep struct {
		Rows []struct {
			Issue           int    `json:"issue"`
			Account         string `json:"account"`
			ChosenModelTier int    `json:"chosen_model_tier"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out.String())
	}
	if len(rep.Rows) != 1 || rep.Rows[0].Account != "small" || rep.Rows[0].ChosenModelTier != 3 {
		t.Fatalf("routine work should route to the cheapest (tier-3 'small') seat, got %+v", rep.Rows)
	}
}

// TestDispatchTierStatusBadOutcome checks a typo'd witnessed outcome fails loud
// rather than silently rendering as pending.
func TestDispatchTierStatusBadOutcome(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`[{"issue":1,"lane":"x","outcome":"finished"}]`), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	var out, errBuf bytes.Buffer
	if code := runDispatchTierStatus(&out, &errBuf, []string{"--in", path}); code != 1 {
		t.Fatalf("exit = %d, want 1 for an unknown outcome (stderr: %s)", code, errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "unknown outcome") {
		t.Fatalf("error should name the unknown outcome, got: %s", errBuf.String())
	}
}
