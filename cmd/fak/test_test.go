package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// planTest is pure, so the routing decision -- the whole point of the verb -- is
// pinned here: tier -> go test args, and Windows -> WSL via test.ps1.
func TestPlanTest_Tiers(t *testing.T) {
	cases := []struct {
		name   string
		args   []string
		tier   string
		goArgs []string
	}{
		{"default is fast", nil, "fast", []string{"-short", "./..."}},
		{"explicit fast", []string{"fast"}, "fast", []string{"-short", "./..."}},
		{"full", []string{"full"}, "full", []string{"./..."}},
		{"all is full", []string{"all"}, "full", []string{"./..."}},
		{"race", []string{"race"}, "race", []string{"-short", "-race", "./..."}},
		{"package target", []string{"./internal/ctxmmu/"}, "package", []string{"./internal/ctxmmu/"}},
		{"passthrough after --", []string{"fast", "--", "-run", "TestX", "-count=1"}, "fast", []string{"-short", "./...", "-run", "TestX", "-count=1"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, err := planTest("linux", c.args)
			if err != nil {
				t.Fatalf("planTest(%v) error: %v", c.args, err)
			}
			if p.Tier != c.tier {
				t.Errorf("tier = %q, want %q", p.Tier, c.tier)
			}
			if !reflect.DeepEqual(p.GoArgs, c.goArgs) {
				t.Errorf("goArgs = %v, want %v", p.GoArgs, c.goArgs)
			}
		})
	}
}

func TestPlanTest_WindowsRoutesToWSL(t *testing.T) {
	p, err := planTest("windows", []string{"fast"})
	if err != nil {
		t.Fatalf("planTest error: %v", err)
	}
	if !p.ViaWSL {
		t.Fatalf("windows host must route via WSL, got ViaWSL=false")
	}
	joined := strings.Join(p.Argv, " ")
	if !strings.Contains(joined, "test.ps1") {
		t.Errorf("windows Argv must invoke test.ps1, got %q", joined)
	}
	// The go test args must still be forwarded to the wrapper verbatim.
	if p.Argv[len(p.Argv)-1] != "./..." {
		t.Errorf("windows Argv must forward go test args, got %q", joined)
	}
}

func TestPlanTest_NonWindowsRunsGoTestDirectly(t *testing.T) {
	p, err := planTest("darwin", []string{"full"})
	if err != nil {
		t.Fatalf("planTest error: %v", err)
	}
	if p.ViaWSL {
		t.Fatalf("non-windows host must not route via WSL")
	}
	if len(p.Argv) < 2 || p.Argv[0] != "go" || p.Argv[1] != "test" {
		t.Errorf("non-windows Argv must start with `go test`, got %v", p.Argv)
	}
}

func TestPlanTest_UnknownTierFailsLoudly(t *testing.T) {
	if _, err := planTest("linux", []string{"fastt"}); err == nil {
		t.Fatalf("a typo'd tier must error, not be handed to go test as a package")
	}
}

// The dry-run shell prints the resolved command and runs nothing -- the safe path
// to exercise the verb end-to-end without launching the suite.
func TestRunTest_DryRunPrintsCommand(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"-n", "race"}); rc != 0 {
		t.Fatalf("dry run rc = %d, stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "tier=race") || !strings.Contains(out.String(), "-race") {
		t.Errorf("dry run output missing resolved race command: %q", out.String())
	}
}

func TestRunTestJSONDryRunEmitsRepairPacket(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "-n", "race"}); rc != 0 {
		t.Fatalf("json dry run rc = %d, stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.Schema != testRepairPacketSchema || !packet.OK || packet.Verdict != "READY" || packet.Tier != "race" {
		t.Fatalf("packet header = %+v", packet)
	}
	if !reflect.DeepEqual(packet.GoArgs, []string{"-short", "-race", "./..."}) {
		t.Fatalf("go args = %v", packet.GoArgs)
	}
	if len(packet.Findings) != 1 || packet.Findings[0].Class != "resolved_command" {
		t.Fatalf("findings = %+v", packet.Findings)
	}
}

func TestRunTestJSONUnknownTierEmitsUsageFinding(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "fastt"}); rc != 2 {
		t.Fatalf("json unknown tier rc = %d, want 2; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Verdict != "USAGE" || packet.ExitCode != 2 {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Findings) != 1 || packet.Findings[0].Class != "usage" {
		t.Fatalf("findings = %+v", packet.Findings)
	}
}

func TestRunTestJSONFailureClassifiesGoTestOutput(t *testing.T) {
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() { runTestCommand = oldRunTestCommand })
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		if !strings.Contains(strings.Join(argv, " "), "./internal/foo") {
			t.Fatalf("unexpected command: %v", argv)
		}
		_, _ = io.WriteString(stdout, "--- FAIL: TestBroken (0.00s)\nFAIL\texample.com/fak/internal/foo\t0.01s\n")
		return testCommandResult{ExitCode: 1}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "./internal/foo"}); rc != 1 {
		t.Fatalf("json failed test rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Verdict != "FAIL" || packet.ExitCode != 1 {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Findings) != 1 || packet.Findings[0].Class != "go_test_failure" {
		t.Fatalf("findings = %+v", packet.Findings)
	}
	if !strings.Contains(packet.StdoutTail, "TestBroken") {
		t.Fatalf("stdout tail = %q", packet.StdoutTail)
	}
}

func TestRunTestJSONBuildFailureClassified(t *testing.T) {
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() { runTestCommand = oldRunTestCommand })
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		if !reflect.DeepEqual(argv, []string{"go", "build", "./..."}) {
			t.Fatalf("unexpected command: %v", argv)
		}
		_, _ = io.WriteString(stderr, "cmd/fak/broken.go:12:2: syntax error: unexpected name\n")
		return testCommandResult{ExitCode: 1}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "build"}); rc != 1 {
		t.Fatalf("json build rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "build" || packet.Findings[0].Class != "go_build_failure" {
		t.Fatalf("packet = %+v", packet)
	}
	if !strings.Contains(packet.StderrTail, "syntax error") {
		t.Fatalf("stderr tail = %q", packet.StderrTail)
	}
}

func TestRunTestJSONVetFailureClassified(t *testing.T) {
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() { runTestCommand = oldRunTestCommand })
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		if !reflect.DeepEqual(argv, []string{"go", "vet", "./cmd/fak"}) {
			t.Fatalf("unexpected command: %v", argv)
		}
		_, _ = io.WriteString(stderr, "cmd/fak/test.go:1: unreachable code\n")
		return testCommandResult{ExitCode: 1}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "vet", "./cmd/fak"}); rc != 1 {
		t.Fatalf("json vet rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "vet" || packet.Findings[0].Class != "go_vet_failure" {
		t.Fatalf("packet = %+v", packet)
	}
}

func TestRunTestJSONGofmtOutputFails(t *testing.T) {
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() { runTestCommand = oldRunTestCommand })
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		if !reflect.DeepEqual(argv, []string{"gofmt", "-l", "cmd/fak/test.go"}) {
			t.Fatalf("unexpected command: %v", argv)
		}
		_, _ = io.WriteString(stdout, "cmd/fak/test.go\n")
		return testCommandResult{ExitCode: 0}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "gofmt", "cmd/fak/test.go"}); rc != 1 {
		t.Fatalf("json gofmt rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "gofmt" || packet.Findings[0].Class != "gofmt_failure" {
		t.Fatalf("packet = %+v", packet)
	}
	if packet.Reason != "command produced output" || !strings.Contains(packet.StdoutTail, "cmd/fak/test.go") {
		t.Fatalf("packet reason/tail = reason %q stdout %q", packet.Reason, packet.StdoutTail)
	}
}

func TestRunTestJSONCodelintCleanPasses(t *testing.T) {
	dir := t.TempDir()
	ok := filepath.Join(dir, "ok.go")
	if err := os.WriteFile(ok, []byte("package x\n\nfunc F() int { return 1 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "codelint", ok}); rc != 0 {
		t.Fatalf("json codelint clean rc = %d, stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if !packet.OK || packet.Tier != "codelint" || packet.Findings[0].Class != "codelint_clean" {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Diagnostics) != 0 {
		t.Fatalf("diagnostics = %+v, want none", packet.Diagnostics)
	}
}

func TestRunTestJSONCodelintFailureCarriesDiagnostics(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.go")
	if err := os.WriteFile(bad, []byte("package x\nfunc ("), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "codelint", bad}); rc != 1 {
		t.Fatalf("json codelint bad rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "codelint" || packet.Findings[0].Class != "codelint_failure" {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Diagnostics) == 0 {
		t.Fatalf("diagnostics = %+v, want at least one", packet.Diagnostics)
	}
	if packet.Diagnostics[0].Tool != "go" || packet.Diagnostics[0].Code != "GO_PARSE" || packet.Diagnostics[0].File != bad {
		t.Fatalf("diagnostic = %+v", packet.Diagnostics[0])
	}
}

func TestRunTestJSONCodelintUsageCarriesNextAction(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "codelint"}); rc != 2 {
		t.Fatalf("json codelint usage rc = %d, want 2; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Verdict != "USAGE" || packet.Findings[0].Class != "codelint_usage" {
		t.Fatalf("packet = %+v", packet)
	}
	if !strings.Contains(packet.NextAction, "fak test --json codelint") {
		t.Fatalf("next action = %q", packet.NextAction)
	}
}

func TestRunTestJSONRuffUnavailableIsExplicitSkip(t *testing.T) {
	oldLookPath := ruffLookPath
	t.Cleanup(func() { ruffLookPath = oldLookPath })
	ruffLookPath = func(string) (string, error) { return "", errors.New("missing ruff") }

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "ruff", "tools"}); rc != 0 {
		t.Fatalf("json ruff missing rc = %d, stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if !packet.OK || packet.Verdict != "SKIP" || packet.Tier != "ruff" || packet.Findings[0].Class != "ruff_unavailable" {
		t.Fatalf("packet = %+v", packet)
	}
	if !strings.Contains(packet.NextAction, "explicitly skipped") {
		t.Fatalf("next action = %q", packet.NextAction)
	}
}

func TestRunTestJSONRuffCleanPasses(t *testing.T) {
	oldLookPath := ruffLookPath
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() {
		ruffLookPath = oldLookPath
		runTestCommand = oldRunTestCommand
	})
	ruffLookPath = func(string) (string, error) { return "/usr/bin/ruff", nil }
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		if !reflect.DeepEqual(argv, []string{"ruff", "check", "--output-format", "json", "tools"}) {
			t.Fatalf("unexpected command: %v", argv)
		}
		_, _ = io.WriteString(stdout, "[]\n")
		return testCommandResult{ExitCode: 0}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "ruff", "tools"}); rc != 0 {
		t.Fatalf("json ruff clean rc = %d, stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if !packet.OK || packet.Verdict != "PASS" || packet.Findings[0].Class != "ruff_clean" {
		t.Fatalf("packet = %+v", packet)
	}
}

func TestRunTestJSONRuffFailureCarriesDiagnostics(t *testing.T) {
	oldLookPath := ruffLookPath
	oldRunTestCommand := runTestCommand
	t.Cleanup(func() {
		ruffLookPath = oldLookPath
		runTestCommand = oldRunTestCommand
	})
	ruffLookPath = func(string) (string, error) { return "/usr/bin/ruff", nil }
	runTestCommand = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) testCommandResult {
		_, _ = io.WriteString(stdout, `[{"code":"F401","filename":"tools/bad.py","message":"imported but unused","location":{"row":3,"column":1}}]`)
		return testCommandResult{ExitCode: 1}
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "ruff", "tools/bad.py"}); rc != 1 {
		t.Fatalf("json ruff bad rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "ruff" || packet.Findings[0].Class != "ruff_failure" {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Diagnostics) != 1 {
		t.Fatalf("diagnostics = %+v, want one", packet.Diagnostics)
	}
	if packet.Diagnostics[0].Tool != "ruff" || packet.Diagnostics[0].Code != "F401" || packet.Diagnostics[0].File != "tools/bad.py" {
		t.Fatalf("diagnostic = %+v", packet.Diagnostics[0])
	}
}

func TestRunTestJSONScorecardCleanPasses(t *testing.T) {
	oldRunScorecard := runScorecardControlPaneForTest
	t.Cleanup(func() { runScorecardControlPaneForTest = oldRunScorecard })
	runScorecardControlPaneForTest = func(stdout, stderr io.Writer, argv []string) int {
		if !reflect.DeepEqual(argv, []string{"--check", "--json", "--timeout", "5"}) {
			t.Fatalf("unexpected scorecard args: %v", argv)
		}
		_, _ = io.WriteString(stdout, `{
			"schema":"fak-scorecard-control-pane/1",
			"ok":true,
			"verdict":"OK",
			"finding":"all_clear",
			"reason":"all scorecards clear",
			"next_action":"hold the line",
			"total_debt":0,
			"grade_debt":0,
			"errored":0,
			"metrics":[],
			"trend":{"direction":"flat","summary":"flat +0 vs @base","worsened":[]},
			"gate_exit":0,
			"gate_message":"RATCHET OK: flat +0 vs @base"
		}`)
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "scorecard", "--timeout", "5"}); rc != 0 {
		t.Fatalf("json scorecard clean rc = %d, stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if !packet.OK || packet.Tier != "scorecard" || packet.Findings[0].Class != "scorecard_clean" {
		t.Fatalf("packet = %+v", packet)
	}
	wantCommand := []string{"fak", "scorecard", "control-pane", "--check", "--json", "--timeout", "5"}
	if !reflect.DeepEqual(packet.Command, wantCommand) {
		t.Fatalf("command = %v, want %v", packet.Command, wantCommand)
	}
}

func TestRunTestJSONScorecardRegressionCarriesDiagnostics(t *testing.T) {
	oldRunScorecard := runScorecardControlPaneForTest
	t.Cleanup(func() { runScorecardControlPaneForTest = oldRunScorecard })
	runScorecardControlPaneForTest = func(stdout, stderr io.Writer, argv []string) int {
		_, _ = io.WriteString(stdout, `{
			"schema":"fak-scorecard-control-pane/1",
			"ok":false,
			"verdict":"ACTION",
			"finding":"scorecard_regressed",
			"reason":"scorecard debt regressed",
			"next_action":"retire the regressed metric",
			"total_debt":12,
			"grade_debt":4,
			"errored":0,
			"metrics":[{"key":"docs","label":"docs","debt_key":"doc_debt","debt":12,"ok":false,"verdict":"ACTION"}],
			"trend":{"direction":"regressed","summary":"+5 vs @base","worsened":["docs"]},
			"gate_exit":1,
			"gate_message":"RATCHET FAIL: +5 vs @base; worsened: docs"
		}`)
		return 1
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "scorecard"}); rc != 1 {
		t.Fatalf("json scorecard regression rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Tier != "scorecard" || packet.Findings[0].Class != "scorecard_regression" {
		t.Fatalf("packet = %+v", packet)
	}
	if !strings.Contains(packet.Reason, "RATCHET FAIL") {
		t.Fatalf("reason = %q", packet.Reason)
	}
	if len(packet.Diagnostics) != 1 || packet.Diagnostics[0].Code != "SCORECARD_REGRESSED" {
		t.Fatalf("diagnostics = %+v", packet.Diagnostics)
	}
}

func TestRunTestJSONScorecardUnmeasuredCarriesMetricError(t *testing.T) {
	oldRunScorecard := runScorecardControlPaneForTest
	t.Cleanup(func() { runScorecardControlPaneForTest = oldRunScorecard })
	runScorecardControlPaneForTest = func(stdout, stderr io.Writer, argv []string) int {
		_, _ = io.WriteString(stdout, `{
			"schema":"fak-scorecard-control-pane/1",
			"ok":false,
			"verdict":"ACTION",
			"finding":"scorecard_unmeasured",
			"reason":"1 scorecard(s) unmeasured",
			"next_action":"repair the failing scorecard",
			"total_debt":0,
			"grade_debt":0,
			"errored":1,
			"metrics":[{"key":"tooling","label":"tooling-quality","debt_key":"py_debt","debt":null,"ok":false,"verdict":"ERROR","error":"missing py_debt in payload"}],
			"trend":{"direction":"flat","summary":"flat +0 vs @base","worsened":[]},
			"gate_exit":1,
			"gate_message":"RATCHET FAIL: 1 scorecard(s) unmeasured"
		}`)
		return 1
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "scorecard"}); rc != 1 {
		t.Fatalf("json scorecard unmeasured rc = %d, want 1; stderr=%s", rc, errb.String())
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || packet.Findings[0].Class != "scorecard_unmeasured" {
		t.Fatalf("packet = %+v", packet)
	}
	if len(packet.Diagnostics) != 1 || packet.Diagnostics[0].Code != "SCORECARD_UNMEASURED" ||
		!strings.Contains(packet.Diagnostics[0].Detail, "missing py_debt") {
		t.Fatalf("diagnostics = %+v", packet.Diagnostics)
	}
}

func TestRunTestScorecardRejectsPin(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"scorecard", "--pin"}); rc != 2 {
		t.Fatalf("scorecard --pin rc = %d, want 2", rc)
	}
	if !strings.Contains(errb.String(), "read-only") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

func TestRunTestScorecardRejectsCheckFalse(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"scorecard", "--check=false"}); rc != 2 {
		t.Fatalf("scorecard --check=false rc = %d, want 2", rc)
	}
	if !strings.Contains(errb.String(), "always runs the ratchet check") {
		t.Fatalf("stderr = %q", errb.String())
	}
}

func TestRunTestJSONScorecardRejectsJSONFalse(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--json", "scorecard", "--json=false"}); rc != 2 {
		t.Fatalf("json scorecard --json=false rc = %d, want 2", rc)
	}
	packet := decodeTestRepairPacket(t, out.Bytes())
	if packet.OK || len(packet.Findings) != 1 || packet.Findings[0].Class != "usage" ||
		!strings.Contains(packet.Reason, "requires scorecard JSON output") {
		t.Fatalf("packet = %+v", packet)
	}
}

func TestRunTest_AffectedDelegatesToAffectedPlanner(t *testing.T) {
	oldListGraph := affectedListGraph
	oldRunGoTest := affectedRunGoTest
	t.Cleanup(func() {
		affectedListGraph = oldListGraph
		affectedRunGoTest = oldRunGoTest
	})

	affectedListGraph = func(root string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{
				"internal/foo/foo.go": "example.com/fak/internal/foo",
			}, map[string][]string{
				"example.com/fak/cmd/fak": {"example.com/fak/internal/foo"},
			}, 2, nil
	}
	affectedRunGoTest = func(root string, args []string, stdout, stderr io.Writer) (int, error) {
		t.Fatalf("fak test affected --json must not run go test; args=%v", args)
		return 1, nil
	}

	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{
		"affected",
		"--json",
		"--file", "internal/foo/foo.go",
	}); rc != 0 {
		t.Fatalf("affected json rc = %d, stderr=%s", rc, errb.String())
	}
	var plan affectedPlan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("bad affected json: %v\n%s", err, out.String())
	}
	want := []string{"example.com/fak/cmd/fak", "example.com/fak/internal/foo"}
	if !reflect.DeepEqual(plan.SelectedPackages, want) {
		t.Fatalf("selected = %v, want %v", plan.SelectedPackages, want)
	}
}

func TestRunTest_ListExitsZero(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runTest(&out, &errb, []string{"--list"}); rc != 0 {
		t.Fatalf("--list rc = %d", rc)
	}
	if !strings.Contains(out.String(), "fast") || !strings.Contains(out.String(), "full") ||
		!strings.Contains(out.String(), "affected") || !strings.Contains(out.String(), "durations") ||
		!strings.Contains(out.String(), "shards") || !strings.Contains(out.String(), "build") ||
		!strings.Contains(out.String(), "vet") || !strings.Contains(out.String(), "gofmt") ||
		!strings.Contains(out.String(), "codelint") || !strings.Contains(out.String(), "ruff") ||
		!strings.Contains(out.String(), "scorecard") {
		t.Errorf("--list output missing tiers: %q", out.String())
	}
}

func decodeTestRepairPacket(t *testing.T, raw []byte) testRepairPacket {
	t.Helper()
	var packet testRepairPacket
	if err := json.Unmarshal(raw, &packet); err != nil {
		t.Fatalf("bad repair packet json: %v\n%s", err, string(raw))
	}
	return packet
}
