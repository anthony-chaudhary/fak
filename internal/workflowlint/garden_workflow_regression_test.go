package workflowlint

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestGardenWorkflowUsesSingleCapturedPayload(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("executes the ubuntu garden workflow's Bash steps")
	}
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("Bash is required to execute the garden workflow")
	}
	if _, err := exec.LookPath("jq"); err != nil {
		t.Skip("jq is required to validate the garden capture")
	}
	_, file, _, _ := runtime.Caller(0)
	workflow, err := os.ReadFile(filepath.Join(filepath.Dir(file), "..", "..", ".github", "workflows", "garden.yml"))
	if err != nil {
		t.Fatal(err)
	}
	source := string(workflow)
	start, end := strings.Index(source, "\non:\n"), strings.Index(source, "\nconcurrency:\n")
	if start < 0 || end <= start {
		t.Fatal("garden event/concurrency boundary changed")
	}
	var events []string
	for _, line := range strings.Split(source[start:end], "\n") {
		if strings.TrimSpace(line) != "" && !strings.HasPrefix(strings.TrimSpace(line), "#") {
			events = append(events, line)
		}
	}
	if strings.Join(events, "\n") != "on:\n  push:\n    branches: [main, master]\n  pull_request:\n  schedule:\n    - cron: \"23 6 * * *\"" {
		t.Errorf("garden event contract changed: %q", events)
	}
	for _, want := range []string{"\nname: garden\n", "\n  push:\n    branches: [main, master]\n", "\n  pull_request:\n", "    - cron: \"23 6 * * *\"", "\n  garden:\n    name: garden bundle gate + payload\n", "run: go test ./internal/gardenbundle/"} {
		if !strings.Contains(source, want) {
			t.Errorf("garden event/check/self-test contract changed: missing %q", want)
		}
	}
	// Read the actual step bodies, not copies of the intended implementation.
	// The workflow uses named, block-mapping steps with inline or literal run values.
	step := func(name string) (string, string) {
		t.Helper()
		lines := strings.Split(source, "\n")
		for i, line := range lines {
			if line != "      - name: "+name {
				continue
			}
			var metadata []string
			for j := i + 1; j < len(lines) && !strings.HasPrefix(lines[j], "      - "); j++ {
				if !strings.HasPrefix(lines[j], "        run: ") {
					metadata = append(metadata, lines[j])
					continue
				}
				value := strings.TrimPrefix(lines[j], "        run: ")
				if value != "|" && value != "|-" {
					return value, strings.Join(metadata, "\n")
				}
				var body []string
				for j++; j < len(lines); j++ {
					if strings.TrimSpace(lines[j]) == "" {
						body = append(body, "")
						continue
					}
					if !strings.HasPrefix(lines[j], "          ") {
						break
					}
					body = append(body, strings.TrimPrefix(lines[j], "          "))
				}
				return strings.Join(body, "\n"), strings.Join(metadata, "\n")
			}
		}
		t.Fatalf("missing executable workflow step %q", name)
		return "", ""
	}
	gate, gateMeta := step("garden bundle gate (--check)")
	summary, summaryMeta := step("garden bundle payload -> step summary")
	if !strings.Contains(summaryMeta, "if: always()") {
		t.Fatal("payload summary must still run after a failed gate")
	}
	if strings.Contains(source, "continue-on-error:") {
		t.Fatal("garden failures must not be converted into successful steps")
	}
	// Only these real interpreters/utilities are reachable. Collection and Git are
	// intercepted shell functions; no helper scripts, real collector, or Git run.
	bin := t.TempDir()
	for _, name := range []string{"bash", "jq", "tee", "cat"} {
		path, err := exec.LookPath(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(path, filepath.Join(bin, name)); err != nil {
			t.Fatal(err)
		}
	}
	const fakeTools = `go() {
  printf '%s\n' "$*" >> "$GARDEN_INVOCATIONS"
  if [ "$GARDEN_PHASE" = gate ]; then
    cat "$GARDEN_FIXTURE"
    return "$GARDEN_STATUS"
  fi
  printf '%s\n' '{"schema":"unexpected-recollection","commit":"wrong-later-source"}'
  return 0
}
git() { printf '%s\n' 'unexpected Git invocation' >&2; return 99; }
`
	for _, tc := range []struct {
		name       string
		exit       int
		payload    string
		removeFile bool
	}{
		{name: "success"},
		{name: "failed_gate", exit: 1},
		{name: "timeout", exit: 124, payload: "empty"},
		{name: "missing_capture", removeFile: true},
		{name: "empty_json", payload: "empty"},
		{name: "truncated_json", payload: "truncated"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			capture := filepath.Join(dir, "garden-payload.json")
			fixture := filepath.Join(dir, "fixture.json")
			log := filepath.Join(dir, "invocations")
			report := filepath.Join(dir, "summary")
			verdict := "OK"
			if tc.exit != 0 {
				verdict = "RED"
			}
			payload, err := json.MarshalIndent(map[string]any{
				"schema": "fak-garden-bundle/1", "ok": tc.exit == 0, "verdict": verdict,
				"commit": "captured-source-" + tc.name, "workspace": ".", "finding": "fixture",
				"reason": "captured reason", "next_action": "inspect captured state",
				"gate_exit": tc.exit, "gate_message": "captured gate " + tc.name,
				"member_count": 1, "gating": []string{"scorecard"},
				"members": []any{map[string]any{"key": "scorecard", "label": "captured member", "gates": true, "state": strings.ToLower(verdict), "ok": tc.exit == 0, "verdict": verdict, "detail": "captured member detail", "exit_code": tc.exit}},
			}, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			payload = append(payload, '\n')
			if tc.payload == "empty" {
				payload = nil
			} else if tc.payload == "truncated" {
				payload = []byte(`{"schema":"fak-garden-bundle/1","ok":true`)
			}
			if err := os.WriteFile(fixture, payload, 0600); err != nil {
				t.Fatal(err)
			}
			run := func(phase, script, metadata string) (int, string) {
				t.Helper()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				args := []string{"--noprofile", "--norc", "-e"}
				if strings.Contains(metadata, "shell: bash") {
					args = append(args, "-o", "pipefail")
				}
				args = append(args, "-c", fakeTools+script)
				cmd := exec.CommandContext(ctx, bash, args...)
				cmd.Dir = dir
				cmd.Env = []string{"PATH=" + bin, "HOME=" + dir, "RUNNER_TEMP=" + dir, "GITHUB_STEP_SUMMARY=" + report, "GARDEN_FIXTURE=" + fixture, "GARDEN_INVOCATIONS=" + log, "GARDEN_STATUS=" + strconv.Itoa(tc.exit), "GARDEN_PHASE=" + phase}
				out, err := cmd.CombinedOutput()
				if ctx.Err() != nil {
					t.Fatalf("workflow %s exceeded its test bound: %s", phase, out)
				}
				if err == nil {
					return 0, string(out)
				}
				if exit, ok := err.(*exec.ExitError); ok {
					return exit.ExitCode(), string(out)
				}
				t.Fatalf("execute %s: %v", phase, err)
				return -1, string(out)
			}
			if code, out := run("gate", gate, gateMeta); code != tc.exit {
				t.Errorf("gate exit = %d, want original process status %d; %s", code, tc.exit, out)
			}
			captured, err := os.ReadFile(capture)
			if err != nil || string(captured) != string(payload) {
				t.Errorf("gate did not preserve exact captured source payload: read=%v got=%q", err, captured)
			}
			if tc.removeFile {
				if err := os.Remove(capture); err != nil && !os.IsNotExist(err) {
					t.Fatal(err)
				}
			}
			code, out := run("summary", summary, summaryMeta)
			valid := tc.payload == "" && !tc.removeFile
			if valid && code != 0 {
				t.Errorf("summary rejected valid captured payload: exit=%d %s", code, out)
			} else if !valid && code == 0 {
				t.Error("missing/invalid captured payload silently became a successful summary")
			}
			calls, err := os.ReadFile(log)
			if err != nil {
				t.Fatal(err)
			}
			if string(calls) != "run ./cmd/fak garden --check --json\n" {
				t.Errorf("want exactly one checked JSON collection; invocations=%q", calls)
			}
			shown, _ := os.ReadFile(report)
			if valid && !strings.Contains(string(shown), string(payload)) {
				t.Errorf("summary did not retain the exact captured JSON: %s", shown)
			}
			if !valid && strings.TrimSpace(out+string(shown)) == "" {
				t.Error("missing/invalid payload failure had no diagnostic")
			}
			if strings.Contains(string(shown), "wrong-later-source") {
				t.Error("summary recollected a later source instead of consuming the gate capture")
			}
		})
	}
}
