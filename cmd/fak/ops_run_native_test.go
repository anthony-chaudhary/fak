package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// The real subprocess executes the production chat entry point, including its
// file transport, HTTP planner and kernel tools, rather than a fake child receipt.
func init() {
	if mode := os.Getenv("FAK_OPS_NATIVE_RECEIPT_MODE"); mode != "" && len(os.Args) > 1 && os.Args[1] == "chat" {
		switch mode {
		case "complete":
			if writeOpsNativeFixtureReceipt(os.Args[2:], nativeAgentReceiptSchema, "completed", true, true) != nil {
				os.Exit(2)
			}
			os.Exit(0)
		case "incomplete":
			if writeOpsNativeFixtureReceipt(os.Args[2:], nativeAgentReceiptSchema, "running", false, false) != nil {
				os.Exit(2)
			}
			os.Exit(0)
		case "unsupported":
			if writeOpsNativeFixtureReceipt(os.Args[2:], "fak.agent.native.v999", "completed", true, false) != nil {
				os.Exit(2)
			}
			os.Exit(0)
		case "missing":
			os.Exit(0)
		default:
			os.Exit(2)
		}
	}
	if marker := os.Getenv("FAK_OPS_NATIVE_SELECTED"); marker != "" && len(os.Args) > 1 && os.Args[1] == "chat" {
		observed := make(map[string]string)
		for _, key := range []string{
			"FAK_OPS_NATIVE_SELECTED",
			"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AWS_SECRET_ACCESS_KEY",
			"NODE_OPTIONS", "OPENCODE_HOME", "OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT",
			"XDG_CONFIG_HOME", "XDG_DATA_HOME", "HOME", "USERPROFILE", "PATH", "TEMP", "TMP", "TMPDIR",
		} {
			if value, ok := os.LookupEnv(key); ok {
				observed[key] = value
			}
		}
		data, err := json.Marshal(observed)
		if err != nil || os.WriteFile(marker, data, 0600) != nil {
			os.Exit(2)
		}
		if writeOpsNativeFixtureReceipt(os.Args[2:], nativeAgentReceiptSchema, "completed", true, true) != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	if sentinel := os.Getenv("FAK_OPS_NATIVE_PREFLIGHT_CHILD_SENTINEL"); sentinel != "" && len(os.Args) > 1 && os.Args[1] == "chat" {
		if err := os.WriteFile(sentinel, []byte("launched"), 0600); err != nil {
			os.Exit(2)
		}
		if writeOpsNativeFixtureReceipt(os.Args[2:], nativeAgentReceiptSchema, "completed", true, true) != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	if os.Getenv("FAK_OPS_NATIVE_TEST_CHILD") == "1" && len(os.Args) > 1 && os.Args[1] == "chat" {
		cmdChat(os.Args[2:])
		os.Exit(0)
	}
}

func writeOpsNativeFixtureReceipt(args []string, schema, status string, taskCompleted, includeEnforcement bool) error {
	values := make(map[string]string)
	for i := 0; i+1 < len(args); i++ {
		switch args[i] {
		case "--receipt", "--policy", "--code-workspace":
			values[args[i]] = args[i+1]
			i++
		}
	}
	if values["--receipt"] == "" || values["--code-workspace"] == "" {
		return fmt.Errorf("fixture lacks receipt/workspace binding")
	}
	payload := map[string]any{
		"schema":  schema,
		"status":  status,
		"metrics": map[string]any{"arm": "fak", "task_completed": taskCompleted},
	}
	if includeEnforcement {
		policyBytes := guardDefaultPolicyJSON
		if path := values["--policy"]; path != "" {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			policyBytes = data
		}
		workspace, err := resolveOpsRunWorkspace(values["--code-workspace"])
		if err != nil {
			return err
		}
		payload["enforcement"] = map[string]any{
			"schema":           opsRunNativeEnforcementSchema,
			"guard_posture":    "fail_closed",
			"policy_digest":    guardPolicyDigest(policyBytes),
			"workspace_digest": opsRunDigest(workspace),
			"tools":            map[string]bool{"system": false, "mcp": false, "skills": false, "memory": false},
		}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return os.WriteFile(values["--receipt"], data, 0600)
}

func TestOpsNativeLaunchIdentity(t *testing.T) {
	run := func(t *testing.T, mode string) (int, opsRunNativeReceipt, []byte, string) {
		t.Helper()
		root := t.TempDir()
		prompt := filepath.Join(root, "prompt.txt")
		policy := filepath.Join(root, "policy.json")
		receiptPath := filepath.Join(root, "run.json")
		if err := os.WriteFile(prompt, []byte("private-native-launch-sentinel\n"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(policy, []byte(`{"posture":"fail_closed","allow":[]}`), 0600); err != nil {
			t.Fatal(err)
		}
		t.Setenv("FAK_OPS_NATIVE_RECEIPT_MODE", mode)
		gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !serveOpsNativeInferencePreflight(t, w, r) {
				t.Errorf("unexpected native provider request: %s %s", r.Method, r.URL.Path)
			}
		}))
		defer gateway.Close()

		var stderr bytes.Buffer
		code := runOpsNativeFixture(io.Discard, &stderr, "FAK_OPS_NATIVE_RECEIPT_MODE", []string{
			"--harness", "native", "--workspace", root,
			"--prompt-file", prompt, "--receipt", receiptPath,
			"--provider", "openai", "--model", "fixture", "--base-url", gateway.URL + "/v1",
			"--policy", policy, "--max-turns", "1", "--timeout", "5s",
		})
		data, err := os.ReadFile(receiptPath)
		if err != nil {
			t.Fatalf("read receipt: %v; stderr=%s", err, &stderr)
		}
		var receipt opsRunNativeReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			t.Fatal(err)
		}
		return code, receipt, data, receiptPath
	}

	t.Run("durable_redacted_child_receipt_binds_actual_launch", func(t *testing.T) {
		code, receipt, data, receiptPath := run(t, "complete")
		if code != 0 || receipt.Status != "succeeded" || receipt.LaunchIdentity == nil {
			t.Fatalf("native launch code=%d status=%q identity=%+v", code, receipt.Status, receipt.LaunchIdentity)
		}
		identity := receipt.LaunchIdentity
		if identity.Harness != "native" || identity.PolicySource != "flag" || identity.PolicyDigest == "" || identity.PolicyDigest == "unknown" {
			t.Errorf("native policy identity is not explicit: %+v", identity.opsRunLaunchIdentityReceipt)
		}
		if identity.WorkspaceDigest == "" || identity.WorkspaceDigest == "unknown" || identity.InferenceProbeRef == "" || identity.InferenceProbeRef == "unknown" {
			t.Errorf("native workspace/route evidence missing: %+v", identity.opsRunLaunchIdentityReceipt)
		}
		if identity.Tools.System || identity.Tools.MCP || identity.Tools.Skills || identity.Tools.Memory {
			t.Errorf("native receipt did not preserve actual disabled tool toggles: %+v", identity.Tools)
		}
		if !strings.HasPrefix(identity.ChildReceiptRef, "sha256:") || identity.ChildReceiptArtifact == "" || identity.ChildReceiptArtifact == "unknown" || filepath.IsAbs(identity.ChildReceiptArtifact) {
			t.Fatalf("child receipt is not durably content-addressed: ref=%q artifact=%q", identity.ChildReceiptRef, identity.ChildReceiptArtifact)
		}
		artifactPath := filepath.Join(filepath.Dir(receiptPath), filepath.FromSlash(identity.ChildReceiptArtifact))
		artifact, err := os.ReadFile(artifactPath)
		if err != nil {
			t.Fatalf("redacted artifact did not survive temporary receipt deletion: %v", err)
		}
		sum := sha256.Sum256(artifact)
		if got := fmt.Sprintf("sha256:%x", sum); got != identity.ChildReceiptRef {
			t.Fatalf("artifact digest=%q want=%q", got, identity.ChildReceiptRef)
		}
		combined := string(data) + string(artifact)
		for _, raw := range []string{"private-native-launch-sentinel", filepath.Dir(receiptPath)} {
			if strings.Contains(combined, raw) {
				t.Errorf("native receipt leaked raw private value %q", raw)
			}
		}
	})

	for _, mode := range []string{"missing", "incomplete"} {
		t.Run("zero_exit_"+mode+"_child_receipt_is_failure", func(t *testing.T) {
			code, receipt, _, _ := run(t, mode)
			if code == 0 || receipt.Status == "succeeded" {
				t.Fatalf("zero-exit %s child receipt was accepted: code=%d status=%q", mode, code, receipt.Status)
			}
		})
	}

	t.Run("unsupported_child_evidence_cannot_inherit_qualification", func(t *testing.T) {
		t.Setenv("OPENCODE_CONFIG_CONTENT", `{"permission":{"*":"allow"}}`)
		code, receipt, _, _ := run(t, "unsupported")
		if code == 0 || receipt.Status == "succeeded" || receipt.LaunchIdentity == nil {
			t.Fatalf("unsupported native evidence was accepted: code=%d receipt=%+v", code, receipt)
		}
		identity := receipt.LaunchIdentity
		if identity.ChildReceiptRef != "unknown" || identity.GuardEffective != "unknown" || identity.GuardEvidenceRef != "unknown" || identity.CapabilityEvidenceRef != "unknown" {
			t.Errorf("unsupported child evidence appeared qualified: %+v", identity.opsRunLaunchIdentityReceipt)
		}
		if receipt.ConfigPolicy == nil || receipt.ConfigPolicy.Status == "qualified" || receipt.ConfigPolicy.Source == "OPENCODE_CONFIG_CONTENT" {
			t.Errorf("unsupported native evidence inherited qualification: %+v", receipt.ConfigPolicy)
		}
	})
}

// runOpsNativeFixture explicitly admits the one marker that turns the test
// binary into the native child fixture. Production strips ambient startup
// controls; selecting the marker through the child-environment seam keeps these
// tests honest without weakening that boundary.
func runOpsNativeFixture(stdout, stderr io.Writer, marker string, args []string) int {
	fixtureArgs := append([]string(nil), args...)
	fixtureArgs = append(fixtureArgs, "--api-key-env", marker)
	return runOpsRun(stdout, stderr, fixtureArgs)
}

func TestOpsNativeEnvironment(t *testing.T) {
	run := func(t *testing.T) (map[string]string, []byte) {
		t.Helper()
		root := t.TempDir()
		prompt := filepath.Join(root, "prompt.txt")
		receipt := filepath.Join(root, "receipt.json")
		observation := filepath.Join(root, "environment.json")
		if err := os.WriteFile(prompt, []byte("inspect isolated native environment\n"), 0600); err != nil {
			t.Fatal(err)
		}
		t.Setenv("FAK_OPS_NATIVE_SELECTED", observation)

		gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !serveOpsNativeInferencePreflight(t, w, r) {
				t.Errorf("unexpected native provider request: %s %s", r.Method, r.URL.Path)
			}
		}))
		defer gateway.Close()

		var stderr bytes.Buffer
		code := runOpsRun(io.Discard, &stderr, []string{
			"--harness", "native", "--workspace", root,
			"--prompt-file", prompt, "--receipt", receipt,
			"--provider", "openai", "--model", "fixture", "--base-url", gateway.URL + "/v1",
			"--api-key-env", "FAK_OPS_NATIVE_SELECTED", "--max-turns", "1", "--timeout", "5s",
		})
		if code != 0 {
			t.Fatalf("native run exit=%d: %s", code, stderr.String())
		}
		data, err := os.ReadFile(observation)
		if err != nil {
			t.Fatalf("read child environment: %v", err)
		}
		var observed map[string]string
		if err := json.Unmarshal(data, &observed); err != nil {
			t.Fatal(err)
		}
		receiptData, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		return observed, receiptData
	}

	t.Run("retains_only_selected_provider_credential", func(t *testing.T) {
		for key, value := range map[string]string{
			"OPENAI_API_KEY":        "ambient-openai-secret",
			"ANTHROPIC_API_KEY":     "ambient-anthropic-secret",
			"GOOGLE_API_KEY":        "ambient-google-secret",
			"AWS_SECRET_ACCESS_KEY": "ambient-aws-secret",
		} {
			t.Setenv(key, value)
		}
		observed, receipt := run(t)
		if observed["FAK_OPS_NATIVE_SELECTED"] == "" {
			t.Fatal("native child lost explicitly selected credential")
		}
		for _, key := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AWS_SECRET_ACCESS_KEY"} {
			if value := observed[key]; value != "" {
				t.Errorf("native child inherited unrelated %s=%q", key, value)
			}
		}
		for _, secret := range []string{"ambient-openai-secret", "ambient-anthropic-secret", "ambient-google-secret", "ambient-aws-secret"} {
			if strings.Contains(string(receipt), secret) {
				t.Errorf("metadata receipt leaked ambient credential %q", secret)
			}
		}
	})

	t.Run("strips_startup_config_and_rehomes_runtime", func(t *testing.T) {
		ambientConfig := filepath.Join(t.TempDir(), "ambient-config")
		ambientData := filepath.Join(t.TempDir(), "ambient-data")
		for key, value := range map[string]string{
			"NODE_OPTIONS":            "--require=ambient-startup-hook.js",
			"OPENCODE_HOME":           filepath.Join(t.TempDir(), "ambient-opencode"),
			"OPENCODE_CONFIG":         filepath.Join(t.TempDir(), "ambient-opencode.json"),
			"OPENCODE_CONFIG_CONTENT": `{"provider":{"ambient":{}},"plugin":["ambient"]}`,
			"XDG_CONFIG_HOME":         ambientConfig,
			"XDG_DATA_HOME":           ambientData,
		} {
			t.Setenv(key, value)
		}
		observed, receipt := run(t)
		for _, key := range []string{"NODE_OPTIONS", "OPENCODE_HOME", "OPENCODE_CONFIG"} {
			if value := observed[key]; value != "" {
				t.Errorf("native child inherited startup control %s=%q", key, value)
			}
		}
		if strings.Contains(observed["OPENCODE_CONFIG_CONTENT"], "ambient") {
			t.Errorf("native child inherited executable provider config: %q", observed["OPENCODE_CONFIG_CONTENT"])
		}
		if observed["XDG_CONFIG_HOME"] == "" || observed["XDG_CONFIG_HOME"] == ambientConfig || observed["XDG_DATA_HOME"] == "" || observed["XDG_DATA_HOME"] == ambientData {
			t.Errorf("native child runtime roots are not isolated: config=%q data=%q", observed["XDG_CONFIG_HOME"], observed["XDG_DATA_HOME"])
		}
		if observed["PATH"] == "" || (observed["TEMP"] == "" && observed["TMP"] == "" && observed["TMPDIR"] == "") {
			t.Errorf("native child lost required platform runtime: PATH=%q TEMP=%q TMP=%q TMPDIR=%q", observed["PATH"], observed["TEMP"], observed["TMP"], observed["TMPDIR"])
		}
		if strings.Contains(string(receipt), "ambient-startup-hook") || strings.Contains(string(receipt), `"ambient"`) {
			t.Fatalf("metadata receipt leaked ambient startup config: %s", receipt)
		}
	})
}

func serveOpsNativeInferencePreflight(t *testing.T, w http.ResponseWriter, r *http.Request) bool {
	t.Helper()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("read provider request: %v", err)
		return false
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var request struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
		Tools  []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
	}
	if json.Unmarshal(body, &request) != nil || !request.Stream {
		return false
	}
	found := false
	for _, tool := range request.Tools {
		if tool.Function.Name == opsRunInferenceProbeTool {
			found = true
			break
		}
	}
	if !found {
		return false
	}
	if request.Model != "fixture" {
		t.Errorf("preflight model = %q, want fixture", request.Model)
	}
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "data: {\"model\":\"fixture\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"native-preflight\",\"type\":\"function\",\"function\":{\"name\":%q,\"arguments\":\"{\\\"ok\\\":true}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", opsRunInferenceProbeTool)
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	return true
}

func TestOpsNativeInferencePreflight(t *testing.T) {
	newGateway := func(t *testing.T, status int, probes *atomic.Int32) *httptest.Server {
		t.Helper()
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			probes.Add(1)
			if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
				t.Errorf("probe = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
			}
			var req struct {
				Model  string `json:"model"`
				Stream bool   `json:"stream"`
				Tools  []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tools"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode probe: %v", err)
			}
			if req.Model != "fixture" || !req.Stream || len(req.Tools) != 1 || req.Tools[0].Function.Name != opsRunInferenceProbeTool {
				t.Errorf("probe did not bind native route: %+v", req)
			}
			if status != http.StatusOK {
				http.Error(w, "refused", status)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: {\"model\":\"fixture\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"native-preflight\",\"type\":\"function\",\"function\":{\"name\":%q,\"arguments\":\"{\\\"ok\\\":true}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", opsRunInferenceProbeTool)
			_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		}))
	}

	run := func(t *testing.T, baseURL string) (int, opsRunReceipt, bool) {
		t.Helper()
		dir := t.TempDir()
		prompt := filepath.Join(dir, "prompt.txt")
		receiptPath := filepath.Join(dir, "receipt.json")
		sentinel := filepath.Join(dir, "child-launched")
		if err := os.WriteFile(prompt, []byte("native preflight witness"), 0600); err != nil {
			t.Fatal(err)
		}
		t.Setenv("FAK_OPS_NATIVE_PREFLIGHT_CHILD_SENTINEL", sentinel)
		code := runOpsNativeFixture(io.Discard, io.Discard, "FAK_OPS_NATIVE_PREFLIGHT_CHILD_SENTINEL", []string{
			"--harness", "native", "--prompt-file", prompt, "--receipt", receiptPath,
			"--provider", "openai", "--model", "fixture", "--base-url", baseURL,
			"--max-turns", "1", "--timeout", "5s",
		})
		data, err := os.ReadFile(receiptPath)
		if err != nil {
			t.Fatal(err)
		}
		var receipt opsRunReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			t.Fatal(err)
		}
		_, err = os.Stat(sentinel)
		return code, receipt, err == nil
	}

	t.Run("failed_probe_launches_no_child", func(t *testing.T) {
		var probes atomic.Int32
		gateway := newGateway(t, http.StatusUnauthorized, &probes)
		defer gateway.Close()
		code, receipt, launched := run(t, gateway.URL+"/v1")
		if code == 0 || launched {
			t.Fatalf("failed inference probe: exit=%d child_launched=%v", code, launched)
		}
		if probes.Load() != 1 || receipt.InferencePreflight == nil || receipt.InferencePreflight.Status != "failed" {
			t.Fatalf("failed inference receipt/probe mismatch: probes=%d receipt=%+v", probes.Load(), receipt.InferencePreflight)
		}
	})

	t.Run("qualified_route_is_bound_and_route_change_reprobes", func(t *testing.T) {
		var firstProbes, secondProbes atomic.Int32
		first := newGateway(t, http.StatusOK, &firstProbes)
		defer first.Close()
		second := newGateway(t, http.StatusOK, &secondProbes)
		defer second.Close()

		firstCode, firstReceipt, firstLaunched := run(t, first.URL+"/v1")
		secondCode, secondReceipt, secondLaunched := run(t, second.URL+"/v1")
		if firstCode != 0 || secondCode != 0 || !firstLaunched || !secondLaunched {
			t.Fatalf("qualified native launch: first=(%d,%v) second=(%d,%v)", firstCode, firstLaunched, secondCode, secondLaunched)
		}
		if firstProbes.Load() != 1 || secondProbes.Load() != 1 {
			t.Fatalf("route-specific probes = (%d,%d), want (1,1)", firstProbes.Load(), secondProbes.Load())
		}
		if firstReceipt.InferencePreflight == nil || secondReceipt.InferencePreflight == nil ||
			firstReceipt.InferencePreflight.Status != "qualified" || secondReceipt.InferencePreflight.Status != "qualified" ||
			firstReceipt.InferencePreflight.ReceiptRef == "" || firstReceipt.InferencePreflight.ReceiptRef == secondReceipt.InferencePreflight.ReceiptRef {
			t.Fatalf("native route receipts are not bound to distinct qualified probes: first=%+v second=%+v", firstReceipt.InferencePreflight, secondReceipt.InferencePreflight)
		}
	})
}

func TestOpsNativeRealExecution(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	for _, mode := range []string{"complete", "turn_cap", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			prompt := filepath.Join(root, "prompt.txt")
			artifact := filepath.Join(root, "witness.txt")
			policy := filepath.Join(root, "policy.json")
			receipt := filepath.Join(root, "run.json")
			for path, value := range map[string]string{prompt: "private native task sentinel", policy: `{"allow":["Write"]}`} {
				if err := os.WriteFile(path, []byte(value), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var requests atomic.Int32
			releaseHandler := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if serveOpsNativeInferencePreflight(t, w, r) {
					return
				}
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("provider request: %v", err)
					return
				}
				n := requests.Add(1)
				if mode == "timeout" {
					select {
					case <-r.Context().Done():
					case <-releaseHandler:
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if n > 1 && mode == "complete" {
					fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Witness written."},"finish_reason":"stop"}]}`)
					return
				}
				args, _ := json.Marshal(map[string]string{"file_path": artifact, "content": "native mediated witness", "mode": "create"})
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "write-witness", "type": "function", "function": map[string]any{"name": "Write", "arguments": string(args)}}}}, "finish_reason": "tool_calls"}}})
			}))
			defer server.Close()
			deadline := "10s"
			if mode == "timeout" {
				deadline = "2s"
			}
			var stdout, stderr bytes.Buffer
			code := runOpsNativeFixture(&stdout, &stderr, "FAK_OPS_NATIVE_TEST_CHILD", []string{"--harness", "native", "--prompt-file", prompt, "--receipt", receipt, "--provider", "openai", "--model", "fixture", "--base-url", server.URL + "/v1", "--workspace", root, "--policy", policy, "--max-turns", "3", "--timeout", deadline, "--effort", "low"})
			close(releaseHandler)
			data, err := os.ReadFile(receipt)
			if err != nil {
				t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
			}
			var got opsRunReceipt
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if got.Harness != "native" || got.Finished.IsZero() || strings.Contains(string(data), "private native") {
				t.Fatalf("invalid metadata receipt: %s", data)
			}
			if mode == "complete" {
				written, err := os.ReadFile(artifact)
				if code != 0 || got.Status != "succeeded" || err != nil || string(written) != "native mediated witness" || requests.Load() < 2 {
					t.Fatalf("code=%d receipt=%s artifact=%q err=%v stdout=%s stderr=%s", code, data, written, err, &stdout, &stderr)
				}
			} else if code == 0 || got.Status == "succeeded" || (mode == "timeout" && (code != 124 || got.Status != "timed_out")) {
				t.Fatalf("uncompleted run accepted: code=%d receipt=%s stderr=%s", code, data, &stderr)
			}
			if mode == "timeout" && requests.Load() == 0 {
				t.Fatal("timeout did not reach the native provider request")
			}
		})
	}
}

func TestOpsNativeRetainsExplicitPolicy(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	root := t.TempDir()
	prompt := filepath.Join(root, "prompt.txt")
	receipt := filepath.Join(root, "run.json")
	policyFile := filepath.Join(root, "policy.json")
	outOfScopeFile := filepath.Join(root, "evil.txt")
	permittedFile := filepath.Join(root, "report", "summary.txt")
	if err := os.MkdirAll(filepath.Join(root, "report"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(prompt, []byte("native policy retention task"), 0600); err != nil {
		t.Fatal(err)
	}
	policyJSON := `{
		"allow": ["Write", "Bash"],
		"arg_rules": [
			{
				"tool": "Write",
				"arg": "file_path",
				"allow_glob": "report/**",
				"reason": "POLICY_BLOCK"
			},
			{
				"tool": "Bash",
				"arg": "command",
				"allow_exact": "printf 'witness marker\\n'",
				"reason": "POLICY_BLOCK"
			}
		]
	}`
	if err := os.WriteFile(policyFile, []byte(policyJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveOpsNativeInferencePreflight(t, w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		n := requests.Add(1)
		switch n {
		case 1:
			// Attempt 1: Write outside permitted scope (should be denied).
			args, _ := json.Marshal(map[string]string{"file_path": "evil.txt", "content": "evil content", "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-evil",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 2:
			// Attempt 2: Bash command violating allow_exact (should be denied).
			args, _ := json.Marshal(map[string]string{"command": "printf 'near match\\n'"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-near-match",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 3:
			// Attempt 3: Write inside permitted scope (should succeed).
			args, _ := json.Marshal(map[string]string{"file_path": "report/summary.txt", "content": "report ok", "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-report",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		default:
			// Final completion.
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Done."},"finish_reason":"stop"}]}`)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := runOpsNativeFixture(&stdout, &stderr, "FAK_OPS_NATIVE_TEST_CHILD", []string{
		"--harness", "native",
		"--prompt-file", prompt,
		"--receipt", receipt,
		"--provider", "openai",
		"--model", "fixture",
		"--base-url", server.URL + "/v1",
		"--workspace", root,
		"--policy", policyFile,
		"--max-turns", "5",
		"--timeout", "10s",
		"--effort", "low",
	})

	if _, err := os.Stat(outOfScopeFile); !os.IsNotExist(err) {
		t.Fatalf("out-of-policy file %s must NOT exist, err=%v", outOfScopeFile, err)
	}
	written, err := os.ReadFile(permittedFile)
	if err != nil || string(written) != "report ok" {
		t.Fatalf("permitted report file %s not written: err=%v, content=%q\nstdout: %s\nstderr: %s", permittedFile, err, written, &stdout, &stderr)
	}

	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
	}
	var got opsRunReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || got.Status != "succeeded" {
		t.Fatalf("code=%d, status=%q, stderr=%s", code, got.Status, &stderr)
	}
}

func TestOpsNativePolicyExactCommand(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	root := t.TempDir()
	prompt := filepath.Join(root, "prompt.txt")
	receipt := filepath.Join(root, "run.json")
	policyFile := filepath.Join(root, "policy.json")
	permittedFile := filepath.Join(root, "report", "inventory.txt")
	if err := os.MkdirAll(filepath.Join(root, "report"), 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(prompt, []byte("native exact command inventory task"), 0600); err != nil {
		t.Fatal(err)
	}

	exactCmd := "echo exact-authorized-marker"
	policyJSON := `{
		"posture": "fail_closed",
		"allow": ["Write", "Bash"],
		"arg_rules": [
			{
				"tool": "Write",
				"arg": "file_path",
				"allow_glob": "report/**",
				"reason": "POLICY_BLOCK"
			},
			{
				"tool": "Bash",
				"arg": "command",
				"allow_exact": "echo exact-authorized-marker",
				"reason": "POLICY_BLOCK"
			}
		]
	}`
	if err := os.WriteFile(policyFile, []byte(policyJSON), 0600); err != nil {
		t.Fatal(err)
	}

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if serveOpsNativeInferencePreflight(t, w, r) {
			return
		}
		var request map[string]any
		_ = json.NewDecoder(r.Body).Decode(&request)
		w.Header().Set("Content-Type", "application/json")
		n := requests.Add(1)
		switch n {
		case 1:
			// Attempt 1: Near-match Bash command (denied by policy).
			args, _ := json.Marshal(map[string]string{"command": "echo near-match-marker"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-near-match",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 2:
			// Attempt 2: Exact authorized command but with wrong/escaping cwd (denied).
			args, _ := json.Marshal(map[string]string{"command": exactCmd, "cwd": "../escape"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-wrong-cwd",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 3:
			// Attempt 3: Exact authorized command in workspace cwd (should succeed).
			args, _ := json.Marshal(map[string]string{"command": exactCmd})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "bash-exact-authorized",
									"type": "function",
									"function": map[string]any{
										"name":      "Bash",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		case 4:
			// Verify that the previous tool execution for bash-exact-authorized returned the exact marker.
			msgs, _ := request["messages"].([]any)
			var markerWitnessed bool
			for _, m := range msgs {
				msgMap, _ := m.(map[string]any)
				if msgMap["role"] == "tool" && msgMap["tool_call_id"] == "bash-exact-authorized" {
					content, _ := msgMap["content"].(string)
					if strings.Contains(content, "exact-authorized-marker") {
						markerWitnessed = true
					}
				}
			}
			content := "failed: command denied"
			if markerWitnessed {
				content = "witness: exact-authorized-marker"
			}
			// Attempt 4: Confined Write with the witnessed marker (should succeed).
			args, _ := json.Marshal(map[string]string{"file_path": "report/inventory.txt", "content": content, "mode": "create"})
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"role": "assistant",
							"tool_calls": []any{
								map[string]any{
									"id":   "write-inventory",
									"type": "function",
									"function": map[string]any{
										"name":      "Write",
										"arguments": string(args),
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
			})
		default:
			// Final completion.
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Operational inventory complete."},"finish_reason":"stop"}]}`)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	code := runOpsNativeFixture(&stdout, &stderr, "FAK_OPS_NATIVE_TEST_CHILD", []string{
		"--harness", "native",
		"--prompt-file", prompt,
		"--receipt", receipt,
		"--provider", "openai",
		"--model", "fixture",
		"--base-url", server.URL + "/v1",
		"--workspace", root,
		"--policy", policyFile,
		"--max-turns", "6",
		"--timeout", "10s",
		"--effort", "low",
	})

	written, err := os.ReadFile(permittedFile)
	if err != nil || string(written) != "witness: exact-authorized-marker" {
		t.Fatalf("permitted report file %s not written: err=%v, content=%q\nstdout: %s\nstderr: %s", permittedFile, err, written, &stdout, &stderr)
	}

	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
	}
	var got opsRunReceipt
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if code != 0 || got.Status != "succeeded" {
		t.Fatalf("code=%d, status=%q, stderr=%s", code, got.Status, &stderr)
	}
}
