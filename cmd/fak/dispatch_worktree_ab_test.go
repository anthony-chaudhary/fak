package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

func TestDispatchWorktreeABProducesScoreboardRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ab.json")
	raw := `{"baseline":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":1200,"poison_incidents":1,"peak_concurrency":5},"isolated":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":600,"poison_incidents":0,"peak_concurrency":8}}`
	if err := os.WriteFile(path, []byte(raw), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runDispatchWorktreeAB(&out, &errb, []string{"--in", path}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"baseline: 6.00 issues/h", "isolated: 12.00 issues/h", "ISOLATION_POISON_FREE"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestDispatchWorktreeABRefusesDifferentWaves(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ab.json")
	if err := os.WriteFile(path, []byte(`{"baseline":{"wave_id":"fixed-wave","resolved":2,"duration_seconds":1},"isolated":{"wave_id":"fixed-wave","resolved":3,"duration_seconds":1}}`), 0644); err != nil {
		t.Fatal(err)
	}
	if code := runDispatchWorktreeAB(&bytes.Buffer{}, &bytes.Buffer{}, []string{"--in", path}); code != 3 {
		t.Fatalf("code=%d want 3", code)
	}
}

func TestDispatchWorktreeAB_AcceptedDeliveryLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ab_lifecycle.json")
	raw := `{
		"baseline": {
			"wave_id": "fixed-wave",
			"duration_seconds": 1200,
			"poison_incidents": 1,
			"peak_concurrency": 5,
			"delivery_records": [
				{"issue_id": 101, "outcome": "ACCEPTED", "setup_duration": 10, "execution_duration": 100, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 130, "spend": 1.0},
				{"issue_id": 102, "outcome": "ACCEPTED", "setup_duration": 10, "execution_duration": 100, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 130, "spend": 1.0},
				{"issue_id": 101, "outcome": "ACCEPTED", "setup_duration": 10, "execution_duration": 50, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 80, "spend": 0.5},
				{"issue_id": 103, "outcome": "REJECTED", "setup_duration": 10, "execution_duration": 50, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 80, "spend": 0.5}
			]
		},
		"isolated": {
			"wave_id": "fixed-wave",
			"duration_seconds": 600,
			"poison_incidents": 0,
			"peak_concurrency": 8,
			"delivery_records": [
				{"issue_id": 201, "outcome": "ACCEPTED", "setup_duration": 10, "execution_duration": 100, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 130, "spend": 1.0},
				{"issue_id": 202, "outcome": "ACCEPTED", "setup_duration": 10, "execution_duration": 100, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 130, "spend": 1.0},
				{"issue_id": 203, "outcome": "UNVERIFIED", "setup_duration": 10, "execution_duration": 50, "landing_duration": 10, "verification_duration": 10, "total_elapsed": 80, "spend": 0.5}
			]
		}
	}`
	if err := os.WriteFile(path, []byte(raw), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Human-readable text report surfaces accepted deliveries and verified status
	var out, errb bytes.Buffer
	if code := runDispatchWorktreeAB(&out, &errb, []string{"--in", path}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	text := out.String()
	for _, want := range []string{
		"baseline: 6.00 issues/h (2 accepted, COMPLETE)",
		"isolated: 12.00 issues/h (2 accepted, COMPLETE)",
		"ISOLATION_POISON_FREE",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("human report missing %q:\n%s", want, text)
		}
	}

	// 2. Machine-readable JSON report surfaces accepted_deliveries, status, and verified
	var jsonOut, jsonErrb bytes.Buffer
	if code := runDispatchWorktreeAB(&jsonOut, &jsonErrb, []string{"--in", path, "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, jsonErrb.String())
	}
	rawJSON := jsonOut.String()
	for _, want := range []string{
		`"accepted_deliveries": 2`,
		`"status": "COMPLETE"`,
		`"verified": true`,
		`"trunk_accounting"`,
		`"worktree_accounting"`,
	} {
		if !strings.Contains(rawJSON, want) {
			t.Fatalf("json report missing %q:\n%s", want, rawJSON)
		}
	}

	var rep scoreboard.WorktreeABReport
	if err := json.Unmarshal(jsonOut.Bytes(), &rep); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
	if rep.Baseline.Accounting.AcceptedDeliveries != 2 || !rep.Baseline.Accounting.Verified || rep.Baseline.Accounting.Status != "COMPLETE" {
		t.Fatalf("unexpected baseline accounting: %+v", rep.Baseline.Accounting)
	}
	if rep.Isolated.Accounting.AcceptedDeliveries != 2 || !rep.Isolated.Accounting.Verified || rep.Isolated.Accounting.Status != "COMPLETE" {
		t.Fatalf("unexpected isolated accounting: %+v", rep.Isolated.Accounting)
	}
	if rep.TrunkAccounting.AcceptedDeliveries != 2 || !rep.TrunkAccounting.Verified || rep.TrunkAccounting.Status != "COMPLETE" {
		t.Fatalf("unexpected trunk accounting: %+v", rep.TrunkAccounting)
	}
	if rep.WorktreeAccounting.AcceptedDeliveries != 2 || !rep.WorktreeAccounting.Verified || rep.WorktreeAccounting.Status != "COMPLETE" {
		t.Fatalf("unexpected worktree accounting: %+v", rep.WorktreeAccounting)
	}

	// 3. LifecycleRecords alias in JSON fixture
	pathAlias := filepath.Join(t.TempDir(), "ab_lifecycle_alias.json")
	rawAlias := `{
		"baseline": {
			"wave_id": "alias-wave",
			"duration_seconds": 1200,
			"lifecycle_records": [
				{"issue_id": 1, "outcome": "ACCEPTED", "total_elapsed": 100},
				{"issue_id": 2, "outcome": "ACCEPTED", "total_elapsed": 100}
			]
		},
		"isolated": {
			"wave_id": "alias-wave",
			"duration_seconds": 600,
			"lifecycle_records": [
				{"issue_id": 1, "outcome": "ACCEPTED", "total_elapsed": 100},
				{"issue_id": 2, "outcome": "ACCEPTED", "total_elapsed": 100}
			]
		}
	}`
	if err := os.WriteFile(pathAlias, []byte(rawAlias), 0644); err != nil {
		t.Fatal(err)
	}
	var aliasOut, aliasErr bytes.Buffer
	if code := runDispatchWorktreeAB(&aliasOut, &aliasErr, []string{"--in", pathAlias, "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, aliasErr.String())
	}
	if !strings.Contains(aliasOut.String(), `"accepted_deliveries": 2`) {
		t.Fatalf("lifecycle_records alias missing accepted_deliveries:\n%s", aliasOut.String())
	}
	var aliasTextOut, aliasTextErr bytes.Buffer
	if code := runDispatchWorktreeAB(&aliasTextOut, &aliasTextErr, []string{"--in", pathAlias}); code != 0 {
		t.Fatalf("code=%d err=%s", code, aliasTextErr.String())
	}
	if !strings.Contains(aliasTextOut.String(), "2 accepted, COMPLETE") {
		t.Fatalf("lifecycle_records alias text missing accepted/status:\n%s", aliasTextOut.String())
	}

	// 4. Trunk / worktree field alias in JSON fixture
	pathTrunk := filepath.Join(t.TempDir(), "ab_trunk_alias.json")
	rawTrunk := `{
		"trunk": {
			"wave_id": "trunk-alias-wave",
			"duration_seconds": 1200,
			"lifecycle_records": [
				{"issue_id": 1, "outcome": "ACCEPTED", "total_elapsed": 100}
			]
		},
		"worktree": {
			"wave_id": "trunk-alias-wave",
			"duration_seconds": 600,
			"lifecycle_records": [
				{"issue_id": 1, "outcome": "ACCEPTED", "total_elapsed": 100}
			]
		}
	}`
	if err := os.WriteFile(pathTrunk, []byte(rawTrunk), 0644); err != nil {
		t.Fatal(err)
	}
	var trunkOut, trunkErr bytes.Buffer
	if code := runDispatchWorktreeAB(&trunkOut, &trunkErr, []string{"--in", pathTrunk, "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, trunkErr.String())
	}
	if !strings.Contains(trunkOut.String(), `"accepted_deliveries": 1`) {
		t.Fatalf("trunk alias missing accepted_deliveries:\n%s", trunkOut.String())
	}
}
