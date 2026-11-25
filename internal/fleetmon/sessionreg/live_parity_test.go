package sessionreg

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestLiveParityWithPython compares the Python oracle and the Go port on THIS host's real
// transcript tree, back to back, each writing into its own copy of the live registry dir
// (the shared registry is never touched). Opt-in (FAK_SESSIONREG_LIVE=1) because it reads
// the operator's home. FAK_SESSIONREG_LIVE_NO_ACCOUNTS_REGISTRY=1 neutralizes the
// config-home registry for both sides, so every .claude* dir is scanned.
//
// A session whose transcript changed or appeared between the two runs is an expected
// difference and is reported, not failed; any other session, account or roster difference fails.
func TestLiveParityWithPython(t *testing.T) {
	if os.Getenv("FAK_SESSIONREG_LIVE") != "1" {
		t.Skip("set FAK_SESSIONREG_LIVE=1 to compare against the live host")
	}
	py, pyArgs := pythonExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	repo, _ := filepath.Abs(filepath.Join("..", "..", ".."))
	liveReg := FromEnv(repo, time.Now()).RegDir
	base := t.TempDir()
	pyReg, goReg := filepath.Join(base, "py"), filepath.Join(base, "go")
	for _, d := range []string{pyReg, goReg} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		for _, f := range []string{"sessions.json", "resume_ledger.jsonl", "_prev_disp.json"} {
			if data, err := os.ReadFile(filepath.Join(liveReg, f)); err == nil {
				if werr := os.WriteFile(filepath.Join(d, f), data, 0o644); werr != nil {
					t.Fatal(werr)
				}
			}
		}
	}
	if os.Getenv("FAK_SESSIONREG_LIVE_NO_ACCOUNTS_REGISTRY") == "1" {
		t.Setenv("FAK_ACCOUNTS_REGISTRY", filepath.Join(base, "no-accounts-registry.json"))
	}

	t.Setenv("FLEET_REG_DIR", pyReg)
	cmd := exec.Command(py, append(pyArgs, filepath.Join(repo, "tools", "fleet_sessions.py"), ModeRegistry)...)
	cmd.Dir = repo
	pyStart := time.Now()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("python registry: %v\n%s", err, out)
	}
	pyWall := time.Since(pyStart)
	pyDoc := readJSONFile(t, filepath.Join(pyReg, "sessions.json"))
	now := parseUTC(pyDoc["generated_utc"])
	if now == nil {
		t.Fatalf("python sessions.json has no generated_utc")
	}

	t.Setenv("FLEET_REG_DIR", goReg)
	opts := FromEnv(repo, *now)
	opts.LiveSIDs = ProcessLiveSIDs
	goStart := time.Now()
	pass, err := Evaluate(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pass.WriteRegistry(context.Background()); err != nil {
		t.Fatal(err)
	}
	goWall := time.Since(goStart)
	goDoc := readJSONFile(t, filepath.Join(goReg, "sessions.json"))
	t.Logf("wall-clock: python registry %v, go registry %v (go evaluated at python's now %s)", pyWall.Round(time.Millisecond), goWall.Round(time.Millisecond), isoUTC(*now))

	key := func(s map[string]any) string {
		return fmt.Sprint(s["account"], "/", s["project"], "/", s["session"])
	}
	index := func(doc map[string]any) map[string]map[string]any {
		out := map[string]map[string]any{}
		rows, _ := doc["sessions"].([]any)
		for _, r := range rows {
			if m, ok := r.(map[string]any); ok {
				out[key(m)] = m
			}
		}
		return out
	}
	pyRows, goRows := index(pyDoc), index(goDoc)
	var churn, oneSided, unexplained []string
	churned := map[string]bool{}
	// A Go-only row is churn when its transcript was written after the oracle's scan began;
	// any other one-sided row is a roster or classification divergence.
	appearedAfterPython := func(r map[string]any) bool {
		seen := parseUTC(r["seen_utc"])
		return seen != nil && !seen.Before(pyStart.Add(-time.Second))
	}
	keys := map[string]bool{}
	for k := range pyRows {
		keys[k] = true
	}
	for k := range goRows {
		keys[k] = true
	}
	sorted := make([]string, 0, len(keys))
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	for _, k := range sorted {
		a, b := pyRows[k], goRows[k]
		switch {
		case a == nil && appearedAfterPython(b):
			churn = append(churn, fmt.Sprintf("%s transcript appeared after the python scan began (%v)", k, b["seen_utc"]))
			churned[fmt.Sprint(b["account"])] = true
		case a == nil || b == nil:
			oneSided = append(oneSided, fmt.Sprintf("%s only in %s", k, map[bool]string{true: "go", false: "python"}[a == nil]))
		case a["seen_utc"] != b["seen_utc"]:
			churn = append(churn, fmt.Sprintf("%s transcript changed between runs (%v -> %v)", k, a["seen_utc"], b["seen_utc"]))
			churned[fmt.Sprint(a["account"])] = true
		default:
			if d := valueDiff(k, a, b); d != "" {
				unexplained = append(unexplained, d)
			}
		}
	}
	t.Logf("sessions: python %d, go %d; changed-between-runs %d; one-sided %d", len(pyRows), len(goRows), len(churn), len(oneSided))
	for _, c := range churn {
		t.Logf("  churn: %s", c)
	}
	if len(oneSided) > 0 {
		t.Errorf("%d session rows exist on one side only:\n  %s", len(oneSided), strings.Join(oneSided, "\n  "))
	}
	for _, f := range []string{"throttle", "auth"} {
		if d := valueDiff("$."+f, pyDoc[f], goDoc[f]); d != "" {
			if len(churn) > 0 {
				t.Logf("  %s differs alongside transcript churn: %s", f, d)
			} else {
				t.Errorf("%s differs with no transcript churn: %s", f, d)
			}
		}
	}
	accountsOf := func(doc map[string]any) (order []string, by map[string]any) {
		by = map[string]any{}
		rows, _ := doc["accounts"].([]any)
		for _, r := range rows {
			if m, ok := r.(map[string]any); ok {
				name := fmt.Sprint(m["account"])
				order = append(order, name)
				by[name] = m
			}
		}
		return order, by
	}
	pyOrder, pyAcc := accountsOf(pyDoc)
	goOrder, goAcc := accountsOf(goDoc)
	t.Logf("accounts: python %d, go %d", len(pyOrder), len(goOrder))
	for _, name := range pyOrder {
		if goAcc[name] == nil {
			t.Errorf("account only in python: %s", name)
		} else if d := valueDiff(name, pyAcc[name], goAcc[name]); d != "" {
			if churned[name] {
				t.Logf("  account differs alongside its transcript churn: %s", d)
			} else {
				t.Errorf("account differs with no transcript churn: %s", d)
			}
		}
	}
	for _, name := range goOrder {
		if pyAcc[name] == nil {
			t.Errorf("account only in go: %s", name)
		}
	}
	if strings.Join(pyOrder, ",") != strings.Join(goOrder, ",") {
		t.Errorf("account order differs:\n    python %v\n    go     %v", pyOrder, goOrder)
	}
	for _, f := range []string{"schema", "app_version", "window_h", "summary"} {
		if d := valueDiff("$."+f, pyDoc[f], goDoc[f]); d != "" && len(churn) == 0 {
			t.Errorf("%s differs with no transcript churn: %s", f, d)
		}
	}
	if len(unexplained) > 0 {
		t.Errorf("%d session rows differ with an unchanged transcript:\n  %s", len(unexplained), strings.Join(unexplained, "\n  "))
	}
}

func readJSONFile(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return doc
}
