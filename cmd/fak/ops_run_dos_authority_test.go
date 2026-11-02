package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The current Go test executable stands in for DOS only in the hermetic
// environment-projection check; installed DOS is exercised separately below.
func init() {
	if capture := os.Getenv("FAK_OPS_DOS_CAPTURE_FILE"); capture != "" && len(os.Args) >= 3 && os.Args[1] == "lease-lane" && os.Args[2] == "live" {
		observation := map[string]any{"appdata": os.Getenv("APPDATA"), "home": os.Getenv("HOME"), "userprofile": os.Getenv("USERPROFILE"), "host_secret_present": os.Getenv("AWS_SECRET_ACCESS_KEY") != ""}
		raw, _ := json.Marshal(observation)
		if err := os.WriteFile(capture, raw, 0600); err != nil {
			os.Exit(2)
		}
		_, _ = fmt.Fprintln(os.Stdout, "[]")
		os.Exit(0)
	}
}

func opsDOSAuthorityValues(env []string) map[string]string {
	values := map[string]string{}
	for _, entry := range env {
		if k, v, ok := strings.Cut(entry, "="); ok {
			values[strings.ToUpper(k)] = v
		}
	}
	return values
}

// fak-test:runtime medium est=1s lane=default
func TestOpsRunDOSAuthorityEnvironment(t *testing.T) {
	trustedAppData := os.Getenv("APPDATA")
	t.Setenv("APPDATA", "untrusted-parent-appdata")
	t.Setenv("PYTHONPATH", "untrusted-parent-pythonpath")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "unselected-fixture-secret")
	const selected = "FAK_OPS_DOS_TEST_SELECTED"
	env, cleanup, err := opsRunChildEnvironmentValue("{}", selected, "selected-fixture-token")
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	values := opsDOSAuthorityValues(env)
	for _, name := range []string{"APPDATA", "PYTHONPATH", "AWS_SECRET_ACCESS_KEY", "FAK_LEASEPLANE_DIR"} {
		if _, present := values[name]; present {
			t.Errorf("Pi child inherited forbidden ambient %s", name)
		}
	}
	if values[selected] != "selected-fixture-token" {
		t.Error("named selected credential missing")
	}
	home := values["HOME"]
	if home == "" || home == os.Getenv("HOME") || home == os.Getenv("USERPROFILE") || values["USERPROFILE"] != home {
		t.Error("Pi HOME/USERPROFILE isolation lost")
	}
	for _, name := range []string{"HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"} {
		if info, err := os.Stat(values[name]); err != nil || !info.IsDir() {
			t.Errorf("isolated %s directory unavailable", name)
		}
	}
	if os.Getenv("APPDATA") != "untrusted-parent-appdata" {
		t.Error("child environment construction changed parent APPDATA")
	}
	root := t.TempDir()
	helper := filepath.Join(root, "dos")
	if runtime.GOOS == "windows" {
		helper += ".exe"
	}
	source, err := os.Open(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	destination, err := os.OpenFile(helper, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		_ = source.Close()
		t.Fatal(err)
	}
	_, copyErr := io.Copy(destination, source)
	sourceErr, destinationErr := source.Close(), destination.Close()
	if copyErr != nil || sourceErr != nil || destinationErr != nil {
		t.Fatalf("copy isolated DOS helper: copy=%v source=%v destination=%v", copyErr, sourceErr, destinationErr)
	}
	capture := filepath.Join(root, "dos-env.json")
	childEnv := append([]string(nil), env...)
	for i, entry := range childEnv {
		if strings.HasPrefix(strings.ToUpper(entry), "PATH=") {
			childEnv[i] = "PATH=" + root
		}
	}
	childEnv = append(childEnv, "FAK_OPS_DOS_AUTHORITY_CHILD=1", "FAK_OPS_DOS_CAPTURE_FILE="+capture)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunDOSAuthorityNative$", "-test.count=1")
	cmd.Dir = root
	cmd.Env = childEnv
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("hermetic authority child: %v: %.1000s", err, output)
	}
	raw, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	var observed struct {
		AppData           string `json:"appdata"`
		Home              string `json:"home"`
		UserProfile       string `json:"userprofile"`
		HostSecretPresent bool   `json:"host_secret_present"`
	}
	if err := json.Unmarshal(raw, &observed); err != nil {
		t.Fatal(err)
	}
	if observed.Home != home || observed.UserProfile != home || observed.HostSecretPresent {
		t.Error("DOS-only environment changed isolated HOME or restored host secret")
	}
	if runtime.GOOS == "windows" {
		if observed.AppData == "" || !filepath.IsAbs(observed.AppData) || observed.AppData == "untrusted-parent-appdata" {
			t.Error("DOS subprocess lacks trusted OS-derived APPDATA")
		}
		if trustedAppData != "" && observed.AppData != trustedAppData {
			t.Error("DOS subprocess APPDATA differs from host known-folder baseline")
		}
	} else if observed.AppData != "" {
		t.Error("non-Windows DOS subprocess unexpectedly restored APPDATA")
	}
	if _, present := opsDOSAuthorityValues(env)["APPDATA"]; present {
		t.Error("DOS projection mutated original Pi child environment")
	}

}

type opsDOSAuthorityObservation struct {
	Available         bool   `json:"available"`
	MissingExecutable bool   `json:"missing_executable"`
	PiAppDataAbsent   bool   `json:"pi_appdata_absent"`
	Error             string `json:"error,omitempty"`
	CauseStderr       string `json:"cause_stderr,omitempty"`
	Rows              int    `json:"rows"`
}

// fak-test:runtime slow est=20s lane=default
func TestOpsRunDOSAuthorityNative(t *testing.T) {
	if os.Getenv("FAK_OPS_DOS_AUTHORITY_CHILD") == "1" {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		view, err := serveWorkspaceLeaseAdmission(ctx)
		observation := opsDOSAuthorityObservation{Available: err == nil, MissingExecutable: errors.Is(err, exec.ErrNotFound), PiAppDataAbsent: os.Getenv("APPDATA") == "", Rows: len(view.Leases)}
		if err != nil {
			observation.Error = err.Error()
			var exit *exec.ExitError
			if errors.As(err, &exit) {
				observation.CauseStderr = string(exit.Stderr)
				if len(observation.CauseStderr) > 2048 {
					observation.CauseStderr = observation.CauseStderr[:2048]
				}
			}
		}
		_ = json.NewEncoder(os.Stdout).Encode(observation)
		return
	}
	if os.Getenv("FAK_OPS_DOS_AUTHORITY_WITNESS") != "1" {
		t.Skip("set FAK_OPS_DOS_AUTHORITY_WITNESS=1 for installed DOS authority witness")
	}
	if runtime.GOOS != "windows" {
		t.Skip("Windows user-site packaging witness")
	}
	root := opsDOSAuthorityWorkspace(t)
	for _, missing := range []bool{false, true} {
		t.Run(fmt.Sprintf("missing-executable=%t", missing), func(t *testing.T) {
			env, cleanup, err := opsRunChildEnvironmentValue("{}", "", "")
			if err != nil {
				t.Fatal(err)
			}
			defer cleanup()
			if missing {
				empty := filepath.Join(root, "empty-path")
				if err := os.MkdirAll(empty, 0700); err != nil {
					t.Fatal(err)
				}
				for i, entry := range env {
					if strings.HasPrefix(strings.ToUpper(entry), "PATH=") {
						env[i] = "PATH=" + empty
					}
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunDOSAuthorityNative$", "-test.count=1")
			cmd.Dir = root
			cmd.Env = append(env, "FAK_OPS_DOS_AUTHORITY_CHILD=1")
			output, runErr := cmd.CombinedOutput()
			var observation opsDOSAuthorityObservation
			for _, line := range bytes.Split(output, []byte("\n")) {
				if len(line) > 0 && line[0] == '{' {
					_ = json.Unmarshal(line, &observation)
				}
			}
			t.Logf("native authority observation: %+v", observation)
			opsDOSAuthorityEvidence(t, fmt.Sprintf("native-missing-%t", missing), output)
			if runErr != nil {
				t.Fatalf("authority helper process failed: %v: %.3000s", runErr, output)
			}
			if !observation.PiAppDataAbsent {
				t.Error("Pi subprocess unexpectedly inherited APPDATA")
			}
			if missing {
				if observation.Available || !observation.MissingExecutable {
					t.Error("missing native authority did not fail closed with exec.ErrNotFound")
				}
			} else if !observation.Available || observation.Rows != 0 {
				t.Errorf("native DOS authority unavailable under actual Ops environment: %+v", observation)
			}
		})
	}
}

func opsDOSAuthorityWorkspace(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("dos"); err != nil {
		t.Skip("installed DOS unavailable")
	}
	root := t.TempDir()
	init := exec.Command("git", "init", "--quiet")
	init.Dir = root
	if out, err := init.CombinedOutput(); err != nil {
		t.Fatalf("isolated Git init: %v: %.1000s", err, out)
	}
	if out, err := exec.Command("dos", "init", root).CombinedOutput(); err != nil {
		t.Fatalf("isolated DOS init: %v: %.1000s", err, out)
	}
	warm := exec.Command("dos", "lease-lane", "live")
	warm.Dir = root
	if out, err := warm.CombinedOutput(); err != nil {
		t.Fatalf("native authority warmup: %v: %.1000s", err, out)
	}
	return root
}

func opsDOSAuthorityEvidence(t *testing.T, name string, output []byte) {
	t.Helper()
	if dir := os.Getenv("FAK_OPS_DOS_EVIDENCE_DIR"); dir != "" {
		if err := os.WriteFile(filepath.Join(dir, "ops-dos-"+name+".jsonl"), output, 0600); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime slow est=45s lane=default
func TestOpsRunPiDOSAuthorityEffect(t *testing.T) {
	if os.Getenv("FAK_OPS_DOS_AUTHORITY_WITNESS") != "1" {
		t.Skip("set FAK_OPS_DOS_AUTHORITY_WITNESS=1 for real Ops/Pi write witness")
	}
	if runtime.GOOS != "windows" {
		t.Skip("Windows user-site packaging witness")
	}
	pi, err := exec.LookPath("pi")
	if err != nil {
		t.Skip("installed Pi unavailable")
	}
	root := opsDOSAuthorityWorkspace(t)
	target := filepath.Join(root, "output.json")
	const payload = "{\"ops_authority_effect\":true}\n"
	var turns atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		var request struct {
			Model string `json:"model"`
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(raw, &request); err != nil {
			t.Error(err)
		}
		name, id, args, finish := "", "", "", "stop"
		for _, tool := range request.Tools {
			if tool.Function.Name == "fak_inference_preflight" {
				name, id, args, finish = "fak_inference_preflight", "probe", `{"ok":true}`, "tool_calls"
			}
		}
		if name == "" && turns.Add(1) == 1 {
			name, id, finish = "write", "ops-write-effect-1", "tool_calls"
			a, _ := json.Marshal(map[string]string{"path": filepath.ToSlash(target), "content": payload})
			args = string(a)
		}
		delta := map[string]any{"role": "assistant", "content": "done"}
		if name != "" {
			delta = map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"index": 0, "id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args}}}}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, choice := range []map[string]any{{"delta": delta, "finish_reason": nil}, {"delta": map[string]any{}, "finish_reason": finish}} {
			frame, _ := json.Marshal(map[string]any{"id": "ops-dos-fixture", "object": "chat.completion.chunk", "model": request.Model, "choices": []any{choice}})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", frame)
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	prompt, policy, audit, receipt := filepath.Join(root, "prompt.txt"), filepath.Join(root, "policy.json"), filepath.Join(root, "audit.jsonl"), filepath.Join(root, "receipt.json")
	if err := os.WriteFile(prompt, []byte("Write the exact scoped output."), 0600); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(map[string]any{"version": "fak-policy/v1", "allow": []string{"write"}, "posture": "fail_closed", "arg_rules": []any{map[string]any{"tool": "write", "arg": "path", "allow_exact": filepath.ToSlash(target), "reason": "POLICY_BLOCK"}}})
	if err := os.WriteFile(policy, data, 0600); err != nil {
		t.Fatal(err)
	}
	const key = "FAK_OPS_DOS_EFFECT_KEY"
	t.Setenv(key, "fixture-only")
	prior := opsPiRunExecute
	t.Cleanup(func() { opsPiRunExecute = prior })
	opsPiRunExecute = func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, prompt []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		values := opsDOSAuthorityValues(env)
		if values["APPDATA"] != "" || values["HOME"] == os.Getenv("HOME") {
			t.Error("actual Ops launch lost environment isolation")
		}
		env = append(env, guardE2EHelperEnv+"="+strings.Join(argv[2:], " "))
		return opsPiExecute(ctx, stdout, stderr, argv, env, prompt)
	}
	var stdout, stderr bytes.Buffer
	code := runOpsPi(&stdout, &stderr, []string{"--harness", "pi", "--provider", "fak", "--model", "ops-dos-fixture", "--base-url", upstream.URL + "/v1", "--api-key-env", key, "--workspace", root, "--prompt-file", prompt, "--policy", policy, "--audit", audit, "--receipt", receipt, "--pi-bin", pi, "--timeout", "60s"})
	opsDOSAuthorityEvidence(t, "actual-ops-pi-stdout", stdout.Bytes())
	opsDOSAuthorityEvidence(t, "actual-ops-pi-stderr", stderr.Bytes())
	got, readErr := os.ReadFile(target)
	matched := false
	for _, line := range bytes.Split(stdout.Bytes(), []byte("\n")) {
		var event struct {
			Type       string `json:"type"`
			ToolCallID string `json:"toolCallId"`
			IsError    bool   `json:"isError"`
		}
		if json.Unmarshal(line, &event) == nil && event.Type == "tool_execution_end" && event.ToolCallID == "ops-write-effect-1" && !event.IsError {
			matched = true
		}
	}
	t.Logf("actual Ops/Pi result: code=%d turns=%d matched_tool=%t file_read=%v", code, turns.Load(), matched, readErr)
	if code != 0 || readErr != nil || string(got) != payload || !matched {
		t.Fatalf("actual Ops isolated Pi write failed: code=%d read=%v matched=%t stderr=%.2000s", code, readErr, matched, stderr.Bytes())
	}
	receiptBytes, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Status string `json:"status"`
	}
	if json.Unmarshal(receiptBytes, &result) != nil || result.Status != "succeeded" {
		t.Errorf("actual Ops receipt status=%q", result.Status)
	}
}
