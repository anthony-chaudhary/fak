package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetsearch"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

func TestSearchCapturedHumanAndJSONRenders(t *testing.T) {
	paths := writeSearchCommandFixtures(t)
	base := []string{
		"--lifecycle", paths.lifecycle,
		"--registrations", paths.registration,
		"--tool-processes", paths.tool,
		"--now", "1787421600",
		"--stale-after", "30m",
	}

	var human, stderr bytes.Buffer
	if code := runSearch(&human, &stderr, append(base, "confluence is:active")); code != 0 {
		t.Fatalf("human code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"VERDICT SOLE_MATCH\n",
		"COVERAGE lifecycle=COMPLETE registration=COMPLETE tool_process=COMPLETE\n",
		"1. session-1 ACTIVE",
		"tools: mcp__confluence__search",
		"evidence: lifecycle, registration, tool_process",
	} {
		if !strings.Contains(human.String(), want) {
			t.Fatalf("human render missing %q:\n%s", want, human.String())
		}
	}

	var machine bytes.Buffer
	stderr.Reset()
	if code := runSearch(&machine, &stderr, append(base, "--json", "confluence is:active")); code != 0 {
		t.Fatalf("json code=%d stderr=%s", code, stderr.String())
	}
	var report fleetsearch.Report
	if err := json.Unmarshal(machine.Bytes(), &report); err != nil {
		t.Fatalf("decode JSON render: %v\n%s", err, machine.String())
	}
	if report.Schema != fleetsearch.Schema || report.Verdict != fleetsearch.VerdictSoleMatch || len(report.Hits) != 1 {
		t.Fatalf("JSON render = %+v", report)
	}

	stderr.Reset()
	machine.Reset()
	partialArgs := append([]string(nil), base...)
	for i := range partialArgs {
		if partialArgs[i] == paths.registration {
			partialArgs[i] = t.TempDir()
		}
	}
	if code := runSearch(&machine, &stderr, append(partialArgs, "--json", "confluence is:active")); code != 0 {
		t.Fatalf("partial code=%d stderr=%s", code, stderr.String())
	}
	if err := json.Unmarshal(machine.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Verdict != fleetsearch.VerdictPartialCoverage || report.TotalMatches != 1 {
		t.Fatalf("partial JSON render = %+v", report)
	}
}

func TestSearchHelpSurface(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runSearch(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("help code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"usage: fak search", "is:active", "-lifecycle", "-tool-processes", fleetsearch.Schema} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("help missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestSearchTopLevelRoute(t *testing.T) {
	if os.Getenv("FAK_TEST_SEARCH_ROUTE") == "1" {
		os.Args = []string{
			"fak", "search", "--json",
			"--lifecycle", os.Getenv("FAK_TEST_SEARCH_LIFECYCLE"),
			"--registrations", os.Getenv("FAK_TEST_SEARCH_REGISTRATIONS"),
			"--tool-processes", os.Getenv("FAK_TEST_SEARCH_TOOLS"),
			"--now", "1787421600", "confluence is:active",
		}
		main()
		return
	}
	paths := writeSearchCommandFixtures(t)
	cmd := exec.Command(os.Args[0], "-test.run=^TestSearchTopLevelRoute$")
	cmd.Env = append(os.Environ(),
		"FAK_TEST_SEARCH_ROUTE=1",
		"FAK_TEST_SEARCH_LIFECYCLE="+paths.lifecycle,
		"FAK_TEST_SEARCH_REGISTRATIONS="+paths.registration,
		"FAK_TEST_SEARCH_TOOLS="+paths.tool,
	)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("top-level fak search route: %v", err)
	}
	var report fleetsearch.Report
	if err := json.Unmarshal(out, &report); err != nil {
		t.Fatalf("route JSON: %v\n%s", err, out)
	}
	if report.Verdict != fleetsearch.VerdictSoleMatch || len(report.Hits) != 1 || report.Hits[0].SessionID != "session-1" {
		t.Fatalf("route report = %+v", report)
	}
}

type searchFixturePaths struct{ lifecycle, registration, tool string }

func writeSearchCommandFixtures(t *testing.T) searchFixturePaths {
	t.Helper()
	now := time.Unix(1787421600, 0).UTC()
	dir := t.TempDir()
	paths := searchFixturePaths{
		lifecycle:    filepath.Join(dir, "lifecycle.jsonl"),
		registration: filepath.Join(dir, "registrations.jsonl"),
		tool:         filepath.Join(dir, "tool-processes.jsonl"),
	}
	writeSearchJSONL(t, paths.lifecycle, sessionjournal.Event{
		Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "life-1",
		TS: now.Add(-2 * time.Minute).Format(time.RFC3339), CWD: "/work/confluence-export",
		Registration: &sessionjournal.RegistrationCarry{RegistrationID: "reg-1", SessionID: "session-1", AttemptID: "attempt-1", TaskID: "confluence migration", State: "active"},
	})
	writeSearchJSONL(t, paths.registration, sessionregistry.Event{
		Schema: sessionregistry.Schema, At: now.Add(-2 * time.Minute),
		Record: sessionregistry.Record{
			Schema: sessionregistry.Schema, RegistrationID: "reg-1", RootRegistrationID: "reg-1", AttemptID: "attempt-1",
			LaunchKind: "guard", Scope: []string{"docs/confluence/**"}, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "session-1"},
			State: sessionregistry.StateActive, CreatedAt: now.Add(-2 * time.Minute),
		},
	})
	writeSearchJSONL(t, paths.tool, toolproc.Event{
		Kind: toolproc.EvSpawn, CallID: "tool-1", Session: "session-1", Tool: "mcp__confluence__search",
		AtMS: now.Add(-time.Minute).UnixMilli(), HeartbeatEveryMS: 120_000,
	})
	return paths
}

func writeSearchJSONL(t *testing.T, path string, rows ...any) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}
