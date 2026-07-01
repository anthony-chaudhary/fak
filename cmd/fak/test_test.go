package main

import (
	"bytes"
	"encoding/json"
	"io"
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
		!strings.Contains(out.String(), "shards") {
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
