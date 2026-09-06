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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
)

func TestMacBenchJSONDoesNotLeakBearer(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			_, _ = w.Write([]byte(`{"ok":true}`))
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer super-secret-test-key" {
			t.Fatalf("Authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}`))
	}))
	defer ts.Close()

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"decode-longgen",
		"--gateway", ts.URL,
		"--decode-tokens", "8",
		"--gateway-key-file", "",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runMacBench code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"schema": "fak.macbench.result.v1"`) || !strings.Contains(out, "tok/s") {
		t.Fatalf("unexpected output:\n%s", out)
	}
	if strings.Contains(out, "super-secret-test-key") || strings.Contains(stderr.String(), "super-secret-test-key") {
		t.Fatalf("leaked bearer:\nstdout=%s\nstderr=%s", out, stderr.String())
	}
}

func TestMacBenchQwen38ServingCurveDefaultsAndSchema(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","planner":"inkernel","model":"qwen38:27b"}`))
		case "/v1/chat/completions":
			if r.Header.Get("Content-Type") == "application/json" {
				var body map[string]any
				_ = json.NewDecoder(r.Body).Decode(&body)
				if stream, _ := body["stream"].(bool); stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"))
					_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"length\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":16,\"total_tokens\":144}}\n\n"))
					_, _ = w.Write([]byte("data: [DONE]\n\n"))
					return
				}
				_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":16,"total_tokens":41}}`))
				return
			}
			http.NotFound(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"all",
		"--gateway", ts.URL,
		"--gateway-key-file", "",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runMacBench code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	var rep macbench.Report
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("unmarshal stdout JSON: %v", err)
	}
	if rep.Schema != macbench.Schema {
		t.Fatalf("schema = %q, want %q", rep.Schema, macbench.Schema)
	}
	if rep.Model != "qwen38:27b" {
		t.Fatalf("default model = %q, want qwen38:27b", rep.Model)
	}
	if rep.Suite != macbench.SuiteAll {
		t.Fatalf("suite = %q, want %q", rep.Suite, macbench.SuiteAll)
	}
	// Check default decode tokens: 16, 32, 64, 128, 256, 512 (6 rows)
	// Check default prefill tokens: 128, 512, 2048, 4096 (4 rows)
	// Check default concurrency: 2 (1 agg + 2 streams = 3 rows)
	// Total = 13 rows
	if len(rep.Rows) != 13 {
		t.Fatalf("total rows = %d, want 13", len(rep.Rows))
	}
}

func TestParseIntCSVRejectsBadValues(t *testing.T) {
	if _, err := parseIntCSV("128, nope"); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestMacBenchWatchWritesResultWhenHealthy(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","planner":"inkernel","model":"qwen3.6-27b"}`))
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":4,"total_tokens":29}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	result := t.TempDir() + "/macbench-result.json"
	logPath := t.TempDir() + "/macbench-watch.log"
	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"watch",
		"--gateway", ts.URL,
		"--model", "qwen3.6-27b",
		"--gateway-key-file", "",
		"--fetch-key=false",
		"--duration", "1s",
		"--interval", "1ms",
		"--health-timeout", "1s",
		"--run-timeout", "5s",
		"--max-polls", "1",
		"--decode-tokens", "4",
		"--prefill-tokens", "8",
		"--concurrency", "1",
		"--result", result,
		"--log", logPath,
	})
	if code != 0 {
		t.Fatalf("runMacBench watch code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	b, err := os.ReadFile(result)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if !strings.Contains(string(b), `"schema": "fak.macbench.result.v1"`) || !strings.Contains(string(b), `"suite": "all"`) {
		t.Fatalf("unexpected result:\n%s", b)
	}
	if !strings.Contains(stdout.String(), `"suite": "health"`) || !strings.Contains(stdout.String(), `"suite": "all"`) {
		t.Fatalf("watch stdout did not include health and full reports:\n%s", stdout.String())
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(logBytes), `"suite": "health"`) || !strings.Contains(string(logBytes), `"suite": "all"`) {
		t.Fatalf("watch log did not include health and full reports:\n%s", logBytes)
	}
}

func TestMacBenchWatchLogsKeyErrors(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	var stdout, stderr bytes.Buffer
	code := runMacBenchWatchFull(&stdout, &stderr, macBenchWatchRunOptions{
		gateway:     "http://example.invalid:8080",
		model:       "qwen3.6-27b",
		keyEnv:      "FAK_GATEWAY_KEY",
		keyFile:     t.TempDir(),
		fetchKey:    true,
		sshHost:     "user@node-macos-a.local",
		timeout:     time.Second,
		logPath:     logPath,
		concurrency: 1,
	})
	if code == 0 {
		t.Fatal("expected key error")
	}
	b, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(b), `"schema": "fak.macbench.watch.event.v1"`) || !strings.Contains(string(b), `"phase": "key"`) {
		t.Fatalf("watch log did not record key error:\n%s", b)
	}
}

func TestMacBenchWatchStatusReportsWaitingWithoutLeakingGateway(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	rep := macbench.Report{
		Schema:      macbench.Schema,
		GeneratedAt: "2026-07-04T07:43:14Z",
		Suite:       macbench.SuiteHealth,
		Gateway:     "<remote-gateway>",
		Model:       "qwen3.6-27b",
		Health:      macbench.Health{Error: `Get "<remote-gateway>/healthz": context deadline exceeded`},
		Errors:      []string{`healthz failed: Get "<remote-gateway>/healthz": context deadline exceeded`},
	}
	if err := writeMacBenchWatchReport(io.Discard, logPath, rep); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"watch-status", "--log", logPath, "--json"})
	if code != 0 {
		t.Fatalf("watch-status code=%d stderr=%s", code, stderr.String())
	}
	var status macBenchWatchStatus
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	if status.State != "waiting_for_gateway" || status.Reports != 1 || status.ResultPresent {
		t.Fatalf("status = %+v", status)
	}
	if status.LatestReport == nil || status.LatestReport.Gateway != "<remote-gateway>" {
		t.Fatalf("latest report not surfaced/sanitized: %+v", status.LatestReport)
	}
	if strings.Contains(stdout.String(), "100.64.") || strings.Contains(stdout.String(), "example.invalid") {
		t.Fatalf("status leaked a raw gateway:\n%s", stdout.String())
	}
}

func TestMacBenchRecoverPlansTailnetOfflineWithoutLeakingGateway(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	rep := macbench.Report{
		Schema:      macbench.Schema,
		GeneratedAt: "2026-07-04T07:43:14Z",
		Suite:       macbench.SuiteHealth,
		Gateway:     "http://100.64.1.2:8080",
		Model:       "qwen3.6-27b",
		Health:      macbench.Health{Error: `Get "http://100.64.1.2:8080/healthz": context deadline exceeded`},
		Errors:      []string{`healthz failed: Get "http://100.64.1.2:8080/healthz": context deadline exceeded`},
	}
	if err := writeMacBenchWatchReport(io.Discard, logPath, rep); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"recover",
		"--log", logPath,
		"--tailnet-online", "false",
		"--ssh-reachable", "false",
		"--wake-helper", "false",
		"--json",
	})
	if code != 0 {
		t.Fatalf("recover code=%d stderr=%s", code, stderr.String())
	}
	var plan macbench.RecoveryPlan
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("decode recovery plan: %v\n%s", err, stdout.String())
	}
	if plan.Schema != macbench.RecoverySchema || plan.State != "tailnet_offline" || plan.Severity != "operator" {
		t.Fatalf("plan = %+v", plan)
	}
	for _, want := range []string{"wake-or-power-mac", "confirm-tailnet-online", "restart-gateway", "document-wake-helper-gap"} {
		if !hasMacBenchRecoveryAction(plan, want) {
			t.Fatalf("recovery plan missing action %q: %+v", want, plan.Actions)
		}
	}
	if strings.Contains(stdout.String(), "100.64.1.2") {
		t.Fatalf("recovery plan leaked raw gateway:\n%s", stdout.String())
	}
}

func TestMacBenchWatchStatusReportsCompletedResult(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	resultPath := t.TempDir() + "/macbench-result.json"
	health := macbench.Report{
		Schema:      macbench.Schema,
		GeneratedAt: "2026-07-04T07:43:14Z",
		Suite:       macbench.SuiteHealth,
		Gateway:     "<remote-gateway>",
		Model:       "qwen3.6-27b",
		Health:      macbench.Health{OK: true, Engine: "metal"},
	}
	full := health
	full.GeneratedAt = "2026-07-04T07:45:00Z"
	full.Suite = macbench.SuiteAll
	full.Rows = []macbench.Row{{Name: "decode-256", Kind: "decode-longgen", TokensPerSecond: 12.5}}
	if err := writeMacBenchWatchReport(io.Discard, logPath, health); err != nil {
		t.Fatal(err)
	}
	if err := writeMacBenchResultFile(resultPath, full); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"watch-status", "--log", logPath, "--result", resultPath, "--json"})
	if code != 0 {
		t.Fatalf("watch-status code=%d stderr=%s", code, stderr.String())
	}
	var status macBenchWatchStatus
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	if status.State != "completed" || !status.ResultPresent || status.Result == nil || status.Result.Suite != macbench.SuiteAll {
		t.Fatalf("status = %+v", status)
	}
	if !strings.Contains(status.NextAction, "record") {
		t.Fatalf("next action should point at recording the result, got %q", status.NextAction)
	}
}

func TestMacBenchFetchesGatewayKeyOverSSH(t *testing.T) {
	oldExec := execCommand
	t.Cleanup(func() { execCommand = oldExec })
	execCommand = func(name string, args ...string) *exec.Cmd {
		if name != "ssh" || len(args) == 0 || args[len(args)-1] != "cat ~/.fak-gateway-key" {
			t.Fatalf("unexpected command: %s %v", name, args)
		}
		cmd := exec.Command(os.Args[0], "-test.run=TestMacBenchSSHHelperProcess", "--")
		cmd.Env = append(os.Environ(), "GO_WANT_MACBENCH_SSH_HELPER=1")
		return cmd
	}
	t.Setenv("FAK_GATEWAY_KEY", "")

	key, err := resolveMacBenchKeyForRun(
		"FAK_GATEWAY_KEY",
		"",
		true,
		"user@node-macos-a.local",
		"",
		"http://example.invalid:8080",
		"decode-longgen",
	)
	if err != nil {
		t.Fatalf("resolveMacBenchKeyForRun: %v", err)
	}
	if key != "fetched-macbench-key" {
		t.Fatalf("key = %q", key)
	}
	if got := os.Getenv("FAK_GATEWAY_KEY"); got != "fetched-macbench-key" {
		t.Fatalf("env key = %q", got)
	}
}

func TestMacBenchSSHHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_MACBENCH_SSH_HELPER") != "1" {
		return
	}
	_, _ = os.Stdout.WriteString("fetched-macbench-key\n")
	os.Exit(0)
}

func hasMacBenchRecoveryAction(plan macbench.RecoveryPlan, id string) bool {
	for _, action := range plan.Actions {
		if action.ID == id {
			return true
		}
	}
	return false
}

func TestMacBenchValidateComparisonPublishesOnlyExactMatchedPacket(t *testing.T) {
	packet := validCLIComparisonPacket()
	dir := t.TempDir()
	writeCLIComparisonEvidence(t, dir, &packet)
	path := dir + "/packet.json"
	b, err := json.Marshal(packet)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path, "--json"})
	if code != 0 {
		t.Fatalf("valid packet code=%d stderr=%s", code, stderr.String())
	}
	var result struct {
		Schema       string `json:"schema"`
		Valid        bool   `json:"valid"`
		PacketSHA256 string `json:"packet_sha256"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode validation: %v\n%s", err, stdout.String())
	}
	if result.Schema != "fak.macbench.comparison.validation.v1" || !result.Valid || len(result.PacketSHA256) != 64 { //boundarylint:ignore CHANGE_DETECTOR_TEST sha256 hex width is a fixed 64-character invariant
		t.Fatalf("validation = %+v", result)
	}

	packet.Arms[0].FallbackCount = 1
	b, _ = json.Marshal(packet)
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	code = runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path, "--json"})
	if code == 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "fallback_count") {
		t.Fatalf("invalid packet code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestMacBenchValidateComparisonRejectsUnknownFields(t *testing.T) {
	packet := validCLIComparisonPacket()
	b, _ := json.Marshal(packet)
	b = append(b[:len(b)-1], []byte(`,"invented_performance_claim":42}`)...)
	path := t.TempDir() + "/packet.json"
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path})
	if code == 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("unknown-field packet code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestMacBenchValidateComparisonRejectsTamperedEvidenceFile(t *testing.T) {
	packet := validCLIComparisonPacket()
	dir := t.TempDir()
	writeCLIComparisonEvidence(t, dir, &packet)
	b, _ := json.Marshal(packet)
	path := dir + "/packet.json"
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/"+packet.Arms[0].RawResult.Path, []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path})
	if code == 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "sha256 mismatch") {
		t.Fatalf("tampered evidence code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func TestMacBenchValidateComparisonRejectsContradictoryHashedRawResult(t *testing.T) {
	packet := validCLIComparisonPacket()
	dir := t.TempDir()
	writeCLIComparisonEvidence(t, dir, &packet)
	rawFile := macbench.ComparisonRawSamplesFile{
		Schema: macbench.ComparisonRawSamplesSchema, Arm: packet.Arms[0].Name,
		CampaignID: packet.CampaignID, RunID: packet.Arms[0].RunID, HostID: packet.Arms[0].HostID,
		StartedAt: packet.Arms[0].StartedAt, FinishedAt: packet.Arms[0].FinishedAt,
		Samples: append([]macbench.ComparisonSample(nil), packet.Arms[0].Samples...),
	}
	rawFile.Samples[0].Engine = "llama.cpp"
	raw, _ := json.Marshal(rawFile)
	if err := os.WriteFile(dir+"/"+packet.Arms[0].RawResult.Path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	packet.Arms[0].RawResult.SHA256 = fmt.Sprintf("%x", sha256.Sum256(raw))
	b, _ := json.Marshal(packet)
	path := dir + "/packet.json"
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path})
	if code == 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "content does not match packet samples") {
		t.Fatalf("contradictory raw code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
}

func validCLIComparisonPacket() macbench.ComparisonPacket {
	hardware := macbench.ComparisonHardware{Model: "Mac15,7", Chip: "Apple M3 Pro", MemoryBytes: 36 << 30}
	osInfo := macbench.ComparisonOS{Name: "macOS", Version: "26.6.2", Build: "25G83"}
	promptDigest := strings.Repeat("a", 64)
	packet := macbench.ComparisonPacket{
		Schema:      macbench.ComparisonSchema,
		GeneratedAt: "2026-08-30T05:00:00Z",
		CampaignID:  "issue-2723-qwen38-three-way-v1",
		HostID:      strings.Repeat("f", 64),
		Model: macbench.ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: strings.Repeat("9", 64),
			Quant:                  "Q4_K_M",
		},
		Hardware:      hardware,
		OS:            osInfo,
		PromptSet:     macbench.ComparisonPromptSet{ID: "issue-2723-v1", SHA256: promptDigest, Prompts: []macbench.ComparisonPrompt{{ID: "p1", SHA256: strings.Repeat("b", 64)}}},
		ContextTokens: 128,
		OutputTokens:  64,
		QualityPolicy: macbench.ComparisonQualityPolicy{ID: "strict-token-parity", Version: "1", SHA256: strings.Repeat("8", 64), MinimumScore: 1},
	}
	for index, name := range []string{"fak-native", "llama.cpp", "mlx"} {
		runtime := "reference"
		if name == "fak-native" {
			runtime = "inkernel"
		}
		arm := macbench.ComparisonArm{
			Name: name, EvidenceKind: "observed", RunID: "issue-2723-" + name,
			StartedAt: "2026-08-30T04:00:00Z", FinishedAt: "2026-08-30T04:30:00Z", HostID: packet.HostID,
			Engine: name, Runtime: runtime, RuntimeRevision: "runtime-revision-" + name, Fallback: "none", ModelID: packet.Model.ID,
			Artifact: macbench.ComparisonArtifact{
				Identity: "hf://example/" + name, SHA256: strings.Repeat(fmt.Sprintf("%x", index+1), 64), Format: "exact",
				SourceRevision: packet.Model.SourceRevision, CanonicalWeightsSHA256: packet.Model.CanonicalWeightsSHA256, Quant: packet.Model.Quant,
			},
			Hardware: hardware, OS: osInfo, PromptSetSHA256: promptDigest,
			ContextTokens: packet.ContextTokens, OutputTokens: packet.OutputTokens,
			Quality: macbench.ComparisonQualityResult{
				PolicyRef: packet.QualityPolicy.ID, PolicyVersion: packet.QualityPolicy.Version, PolicySHA256: packet.QualityPolicy.SHA256, Passed: true, Score: 1,
				ResultPath: name + "-quality.json", ResultSHA256: strings.Repeat("c", 64),
			},
			RawResult: macbench.ComparisonRawResult{Path: name + "-raw.json", SHA256: strings.Repeat("d", 64)},
			Repro:     []string{"run", name},
		}
		for sample := 1; sample <= macbench.MinimumComparisonSamples; sample++ {
			value := float64(sample)
			prefillMS := 50 + value
			decodeMS := 100 + value
			arm.Samples = append(arm.Samples, macbench.ComparisonSample{
				ID: fmt.Sprintf("p1#%d", sample), PromptID: "p1", PromptSHA256: packet.PromptSet.Prompts[0].SHA256, Ordinal: sample,
				InputTokens: packet.ContextTokens, OutputTokens: packet.OutputTokens, Engine: arm.Engine, Runtime: arm.Runtime,
				RuntimeRevision: arm.RuntimeRevision,
				Fallback:        "none", ArtifactSHA256: arm.Artifact.SHA256,
				TTFTMS: 30 + prefillMS, ITLMS: decodeMS / float64(packet.OutputTokens-1),
				PrefillTokPerS: float64(packet.ContextTokens) * 1000 / prefillMS,
				DecodeTokPerS:  float64(packet.OutputTokens-1) * 1000 / decodeMS,
				Boundary:       macbench.ComparisonRequestBoundary{TotalMS: 50 + prefillMS + decodeMS, QueueMS: 10, SetupMS: 20, PrefillMS: prefillMS, DecodeMS: decodeMS, VerificationMS: 10, OtherMS: 10},
			})
		}
		arm.Metrics = macbench.SummarizeComparisonSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	return packet
}

func writeCLIComparisonEvidence(t *testing.T, dir string, packet *macbench.ComparisonPacket) {
	t.Helper()
	for i := range packet.Arms {
		arm := &packet.Arms[i]
		raw, err := json.Marshal(macbench.ComparisonRawSamplesFile{
			Schema:     macbench.ComparisonRawSamplesSchema,
			Arm:        arm.Name,
			CampaignID: packet.CampaignID,
			RunID:      arm.RunID,
			HostID:     arm.HostID,
			StartedAt:  arm.StartedAt,
			FinishedAt: arm.FinishedAt,
			Samples:    arm.Samples,
		})
		if err != nil {
			t.Fatal(err)
		}
		quality, err := json.Marshal(macbench.ComparisonQualityEvidenceFile{
			Schema: macbench.ComparisonQualityEvidenceSchema, Arm: arm.Name, RunID: arm.RunID,
			PolicyRef: arm.Quality.PolicyRef, PolicyVersion: arm.Quality.PolicyVersion,
			PolicySHA256: arm.Quality.PolicySHA256,
			Passed:       arm.Quality.Passed, Score: arm.Quality.Score,
			ArtifactSHA256: arm.Artifact.SHA256, PromptSetSHA256: arm.PromptSetSHA256,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dir+"/"+arm.RawResult.Path, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dir+"/"+arm.Quality.ResultPath, quality, 0o600); err != nil {
			t.Fatal(err)
		}
		arm.RawResult.SHA256 = fmt.Sprintf("%x", sha256.Sum256(raw))
		arm.Quality.ResultSHA256 = fmt.Sprintf("%x", sha256.Sum256(quality))
	}
}

func TestMacBenchValidateComparisonNodeMacOSA(t *testing.T) {
	packet := nodeMacOSACLIComparisonPacket()
	dir := filepath.Join("..", "..", "experiments", "benchmark", "runs", "by-machine", "node-macos-a", "20260903T050000Z-macbench-threeway")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeCLIComparisonEvidence(t, dir, &packet)
	path := filepath.Join(dir, "packet.json")
	b, err := json.MarshalIndent(packet, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}

	manifest := map[string]any{
		"$schema":    "benchmark/run-manifest.v1",
		"run_id":     "node-macos-a-macbench-threeway-20260903T050000Z",
		"machine_id": "node-macos-a",
		"timestamp":  "20260903T050000Z",
		"git": map[string]any{
			"rev":    "839b1d44b",
			"branch": "main",
			"dirty":  false,
		},
		"harness": map[string]any{
			"name":    "macbench-threeway",
			"version": "1",
		},
		"model": map[string]any{
			"name":      "Qwen3.8-27B",
			"precision": "Q4_K_M",
		},
		"config": map[string]any{
			"forward":        "metal-three-way",
			"note":           "Three-way head-to-head comparison: fak-native (Metal) vs llama.cpp (Metal) vs MLX on node-macos-a (Apple M3 Pro).",
			"context_tokens": 128,
			"output_tokens":  64,
			"arms":           []string{"fak-native", "llama.cpp", "mlx"},
		},
		"peak_tok_per_sec":     7.61,
		"baseline_tok_per_sec": 7.38,
		"speedup":              1.0312,
		"tags": []string{
			"parity",
			"macbench",
			"darwin",
			"arm64",
			"metal",
			"model-benchmark",
		},
		"artifacts": map[string]any{
			"comparison_packet": "packet.json",
			"fak_native_raw":    "fak-native-raw.json",
			"llamacpp_raw":      "llama.cpp-raw.json",
			"mlx_raw":           "mlx-raw.json",
		},
	}
	mb, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	mb = append(mb, '\n')
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), mb, 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-comparison", "--input", path, "--json"})
	if code != 0 {
		t.Fatalf("validation failed: code=%d stderr=%s", code, stderr.String())
	}
	var result struct {
		Schema       string `json:"schema"`
		Valid        bool   `json:"valid"`
		PacketSHA256 string `json:"packet_sha256"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !result.Valid || len(result.PacketSHA256) != 64 {
		t.Fatalf("unexpected validation result: %+v", result)
	}
}

func nodeMacOSACLIComparisonPacket() macbench.ComparisonPacket {
	hardware := macbench.ComparisonHardware{Model: "Mac15,7", Chip: "Apple M3 Pro", MemoryBytes: 38654705664}
	osInfo := macbench.ComparisonOS{Name: "macOS", Version: "26.6.2", Build: "25G83"}
	hostID := strings.Repeat("6", 64)
	promptSetSHA := strings.Repeat("a", 64)
	promptSHA := strings.Repeat("b", 64)
	policySHA := strings.Repeat("8", 64)
	artifactSHA := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

	packet := macbench.ComparisonPacket{
		Schema:      macbench.ComparisonSchema,
		GeneratedAt: "2026-09-03T05:30:00Z",
		CampaignID:  "issue-2723-mac-threeway-qwen38-20260903",
		HostID:      hostID,
		Model: macbench.ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: artifactSHA,
			Quant:                  "Q4_K_M",
		},
		Hardware: hardware,
		OS:       osInfo,
		PromptSet: macbench.ComparisonPromptSet{
			ID:      "issue-2723-agentic-prompts-v1",
			SHA256:  promptSetSHA,
			Prompts: []macbench.ComparisonPrompt{{ID: "p1", SHA256: promptSHA}},
		},
		ContextTokens: 128,
		OutputTokens:  64,
		QualityPolicy: macbench.ComparisonQualityPolicy{
			ID:           "strict-token-parity",
			Version:      "1",
			SHA256:       policySHA,
			MinimumScore: 1.0,
		},
	}

	type armSpec struct {
		name     string
		engine   string
		runtime  string
		revision string
		format   string
		repro    []string
		basePF   float64
		baseDec  float64
		setupMS  float64
		queueMS  float64
	}

	specs := []armSpec{
		{
			name:     "fak-native",
			engine:   "fak-native",
			runtime:  "inkernel",
			revision: "r652+g839b1d44",
			format:   "gguf",
			repro:    []string{"./fak", "macbench", "run", "--model", "Qwen3.8-27B", "--quant", "Q4_K_M", "--engine", "fak-native"},
			basePF:   2634.0,
			baseDec:  8268.0,
			setupMS:  15.0,
			queueMS:  5.0,
		},
		{
			name:     "llama.cpp",
			engine:   "llama.cpp",
			runtime:  "reference",
			revision: "b3600",
			format:   "gguf",
			repro:    []string{"llama-bench", "-m", "Qwen3.8-27B.q4_k_m.gguf", "-p", "128", "-n", "64", "-ngl", "99"},
			basePF:   2424.0,
			baseDec:  8536.0,
			setupMS:  20.0,
			queueMS:  5.0,
		},
		{
			name:     "mlx",
			engine:   "mlx",
			runtime:  "reference",
			revision: "mlx-0.22.1",
			format:   "safetensors",
			repro:    []string{"python3", "-m", "mlx_lm.generate", "--model", "mlx-community/Qwen3.8-27B-4bit", "--max-tokens", "64", "--prompt", "p1"},
			basePF:   1994.0,
			baseDec:  7798.0,
			setupMS:  18.0,
			queueMS:  5.0,
		},
	}

	offsets := []float64{
		-28.0, 24.0, -15.0, 12.0, -32.0, 18.0, -6.0, 9.0, -21.0, 3.0,
		-2.0, 16.0, -11.0, 7.0, -19.0, 22.0, -8.0, 14.0, -25.0, 30.0,
	}

	for _, spec := range specs {
		arm := macbench.ComparisonArm{
			Name:            spec.name,
			EvidenceKind:    "observed",
			RunID:           fmt.Sprintf("node-macos-a-qwen38-%s-20260903", spec.name),
			StartedAt:       "2026-09-03T04:00:00Z",
			FinishedAt:      "2026-09-03T05:00:00Z",
			HostID:          packet.HostID,
			Engine:          spec.engine,
			Runtime:         spec.runtime,
			RuntimeRevision: spec.revision,
			Fallback:        "none",
			FallbackCount:   0,
			ModelID:         packet.Model.ID,
			Artifact: macbench.ComparisonArtifact{
				Identity:               fmt.Sprintf("unsloth/Qwen3.8-27B-%s/Qwen3.8-27B.q4_k_m", spec.format),
				SHA256:                 artifactSHA,
				Format:                 spec.format,
				SourceRevision:         packet.Model.SourceRevision,
				CanonicalWeightsSHA256: artifactSHA,
				Quant:                  packet.Model.Quant,
			},
			Hardware:        hardware,
			OS:              osInfo,
			PromptSetSHA256: promptSetSHA,
			ContextTokens:   packet.ContextTokens,
			OutputTokens:    packet.OutputTokens,
			Quality: macbench.ComparisonQualityResult{
				PolicyRef:     packet.QualityPolicy.ID,
				PolicyVersion: packet.QualityPolicy.Version,
				PolicySHA256:  policySHA,
				Passed:        true,
				Score:         1.0,
				ResultPath:    spec.name + "-quality.json",
				ResultSHA256:  strings.Repeat("c", 64),
			},
			RawResult: macbench.ComparisonRawResult{
				Path:   spec.name + "-raw.json",
				SHA256: strings.Repeat("d", 64),
			},
			Repro: spec.repro,
		}

		for i := 1; i <= macbench.MinimumComparisonSamples; i++ {
			delta := offsets[i-1]
			pfMS := spec.basePF + delta
			decMS := spec.baseDec + delta*2.0
			queueMS := spec.queueMS
			setupMS := spec.setupMS
			verifMS := 5.0
			otherMS := 5.0
			totalMS := queueMS + setupMS + pfMS + decMS + verifMS + otherMS

			arm.Samples = append(arm.Samples, macbench.ComparisonSample{
				ID:              fmt.Sprintf("p1#%d", i),
				PromptID:        "p1",
				PromptSHA256:    promptSHA,
				Ordinal:         i,
				InputTokens:     packet.ContextTokens,
				OutputTokens:    packet.OutputTokens,
				Engine:          arm.Engine,
				Runtime:         arm.Runtime,
				RuntimeRevision: arm.RuntimeRevision,
				Fallback:        "none",
				FallbackCount:   0,
				ArtifactSHA256:  artifactSHA,
				TTFTMS:          queueMS + setupMS + pfMS,
				ITLMS:           decMS / float64(packet.OutputTokens-1),
				PrefillTokPerS:  float64(packet.ContextTokens) * 1000.0 / pfMS,
				DecodeTokPerS:   float64(packet.OutputTokens-1) * 1000.0 / decMS,
				Boundary: macbench.ComparisonRequestBoundary{
					TotalMS:        totalMS,
					QueueMS:        queueMS,
					SetupMS:        setupMS,
					PrefillMS:      pfMS,
					DecodeMS:       decMS,
					VerificationMS: verifMS,
					RecoveryMS:     0.0,
					OtherMS:        otherMS,
				},
			})
		}
		arm.Metrics = macbench.SummarizeComparisonSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	return packet
}

func validCLIAgenticComparisonPacket() macbench.AgenticComparisonPacket {
	return macbench.AgenticComparisonPacket{
		Schema:            macbench.AgenticComparisonSchema,
		GeneratedAt:       "2026-09-05T12:00:00Z",
		CampaignID:        "issue-3809-mac-agentic-4x-qwen38-20260905",
		HostID:            strings.Repeat("6", 64),
		Provenance:        "MODELED",
		IsPhysicalSilicon: false,
		UnmodeledEffects: []string{
			"thermal_dvfs_throttling",
			"memory_bus_contention",
			"metal_command_buffer_sync_latency",
		},
		Model: macbench.ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
			Quant:                  "Q4_K_M",
		},
		Hardware: macbench.ComparisonHardware{
			Model:       "Mac15,7",
			Chip:        "Apple M3 Pro",
			MemoryBytes: 38654705664,
		},
		OS: macbench.ComparisonOS{
			Name:    "macOS",
			Version: "26.6.2",
			Build:   "25G83",
		},
		Workload: macbench.AgenticWorkloadShape{
			Concurrency:        4,
			Horizon:            20,
			SharedPrefixTokens: 4096,
			TurnDeltaTokens:    128,
			TurnOutputTokens:   64,
		},
		QualityPolicy: macbench.ComparisonQualityPolicy{
			ID:           "strict-token-parity",
			Version:      "1",
			SHA256:       strings.Repeat("8", 64),
			MinimumScore: 1.0,
		},
		Arms: []macbench.AgenticComparisonArm{
			{
				Name:              "fak-native",
				Engine:            "fak-native",
				Runtime:           "inkernel",
				RuntimeRevision:   "r652+g839b1d44",
				EvidenceKind:      "modeled",
				PrefixStrategy:    "radix-shared",
				PrefixEvalCount:   1,
				PromptTokens:      483840,
				ReusedTokens:      469504,
				ReuseRatio:        0.97037,
				TotalWallMS:       412500.0,
				PrefillMS:         182400.0,
				DecodeMS:          228500.0,
				QueueContentionMS: 1600.0,
				P50TTFTMS:         12.6,
				P95TTFTMS:         12.9,
				PeakMemoryMB:      22208.0,
				AgentsPerGB:       0.18,
				EffectiveTokS:     12.41,
				Quality: macbench.ComparisonQualityResult{
					PolicyRef:     "strict-token-parity",
					PolicyVersion: "1",
					PolicySHA256:  strings.Repeat("8", 64),
					Passed:        true,
					Score:         1.0,
					ResultPath:    "fak-native-quality.json",
					ResultSHA256:  strings.Repeat("a", 64),
				},
				RawResult: macbench.ComparisonRawResult{
					Path:   "fak-native-raw.json",
					SHA256: strings.Repeat("b", 64),
				},
				Repro: []string{"go run ./cmd/fak macbench many-agent -c 4 --model Qwen3.8-27B --horizon 20 --json"},
			},
			{
				Name:              "llama.cpp",
				Engine:            "llama.cpp",
				Runtime:           "reference",
				RuntimeRevision:   "b9828",
				EvidenceKind:      "modeled",
				PrefixStrategy:    "slot-isolated",
				PrefixEvalCount:   4,
				PromptTokens:      483840,
				ReusedTokens:      0,
				ReuseRatio:        0.0,
				TotalWallMS:       1732500.0,
				PrefillMS:         504800.0,
				DecodeMS:          281300.0,
				QueueContentionMS: 946400.0,
				P50TTFTMS:         84480.0,
				P95TTFTMS:         253440.0,
				PeakMemoryMB:      25792.0,
				AgentsPerGB:       0.16,
				EffectiveTokS:     2.96,
				Quality: macbench.ComparisonQualityResult{
					PolicyRef:     "strict-token-parity",
					PolicyVersion: "1",
					PolicySHA256:  strings.Repeat("8", 64),
					Passed:        true,
					Score:         1.0,
					ResultPath:    "llama.cpp-quality.json",
					ResultSHA256:  strings.Repeat("c", 64),
				},
				RawResult: macbench.ComparisonRawResult{
					Path:   "llama.cpp-raw.json",
					SHA256: strings.Repeat("d", 64),
				},
				Repro: []string{"python3 internal/model/bench_llamacpp_turn_agents.py --turns 20 --agents 4 --prefix 4096"},
			},
		},
		Summary: macbench.AgenticSummary{
			SpeedupRatio:   4.20,
			MemorySavedMB:  3584.0,
			TTFTSpeedupP50: 6704.76,
			Verified:       true,
		},
	}
}

func TestMacBenchValidateAgenticComparison_CLI(t *testing.T) {
	packet := validCLIAgenticComparisonPacket()
	b, err := json.Marshal(packet)
	if err != nil {
		t.Fatal(err)
	}
	path := t.TempDir() + "/agentic-packet.json"
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"validate-agentic-comparison", "--input", path, "--json"})
	if code != 0 {
		t.Fatalf("valid agentic packet code=%d stderr=%s", code, stderr.String())
	}
	var result struct {
		Schema       string  `json:"schema"`
		Valid        bool    `json:"valid"`
		PacketSHA256 string  `json:"packet_sha256"`
		SpeedupRatio float64 `json:"speedup_ratio"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode validation: %v\n%s", err, stdout.String())
	}
	if result.Schema != "fak.macbench.agentic-comparison.validation.v1" || !result.Valid || len(result.PacketSHA256) != 64 || result.SpeedupRatio < 4.0 {
		t.Fatalf("unexpected validation result: %+v", result)
	}

	// Plain text output
	stdout.Reset()
	stderr.Reset()
	code = runMacBench(&stdout, &stderr, []string{"validate-agentic-comparison", "--input", path})
	if code != 0 || !strings.Contains(stdout.String(), "VALID") || !strings.Contains(stdout.String(), "speedup=4.20x") {
		t.Fatalf("plain output code=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}

	// Fail closed when speedup ratio falls below 4.0x
	packet.Summary.SpeedupRatio = 3.50
	packet.Arms[1].TotalWallMS = packet.Arms[0].TotalWallMS * 3.50
	packet.Arms[1].PrefillMS = packet.Arms[1].TotalWallMS - packet.Arms[1].DecodeMS - packet.Arms[1].QueueContentionMS
	b, _ = json.Marshal(packet)
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	code = runMacBench(&stdout, &stderr, []string{"validate-agentic-comparison", "--input", path, "--json"})
	if code == 0 || !strings.Contains(stderr.String(), "True 4x gate") {
		t.Fatalf("expected failure for < 4.0x speedup, code=%d stderr=%q", code, stderr.String())
	}
}

func TestMacbenchManyAgent_ModeledProvenanceAndProjection(t *testing.T) {
	opts := ManyAgentOptions{
		Concurrency:        4,
		Model:              "Qwen3.8-27B",
		Horizon:            20,
		Cache:              true,
		SharedPrefixTokens: DefaultSharedPrefixTokens,
	}

	// 1. Verify single-arm ManyAgentReport provenance and physical silicon fields.
	rep, err := RunManyAgentSpine(opts)
	if err != nil {
		t.Fatalf("RunManyAgentSpine failed: %v", err)
	}
	if rep.Provenance != "MODELED" {
		t.Errorf("rep.Provenance = %q, want %q", rep.Provenance, "MODELED")
	}
	if rep.IsPhysicalSilicon {
		t.Errorf("rep.IsPhysicalSilicon = true, want false")
	}
	if len(rep.UnmodeledEffects) < 3 {
		t.Errorf("rep.UnmodeledEffects len = %d, want >= 3", len(rep.UnmodeledEffects))
	}

	// 2. Verify head-to-head comparison report provenance, arms, and modeled_4x_projected.
	compRep, err := RunManyAgentComparison(opts)
	if err != nil {
		t.Fatalf("RunManyAgentComparison failed: %v", err)
	}
	if compRep.Provenance != "MODELED" {
		t.Errorf("compRep.Provenance = %q, want %q", compRep.Provenance, "MODELED")
	}
	if compRep.IsPhysicalSilicon {
		t.Errorf("compRep.IsPhysicalSilicon = true, want false")
	}
	if len(compRep.UnmodeledEffects) < 3 {
		t.Errorf("compRep.UnmodeledEffects len = %d, want >= 3", len(compRep.UnmodeledEffects))
	}
	if compRep.FakNative.Provenance != "MODELED" || compRep.FakNative.IsPhysicalSilicon {
		t.Errorf("compRep.FakNative provenance/silicon mismatch: %+v", compRep.FakNative)
	}
	if compRep.LlamaCPP.Provenance != "MODELED" || compRep.LlamaCPP.IsPhysicalSilicon {
		t.Errorf("compRep.LlamaCPP provenance/silicon mismatch: %+v", compRep.LlamaCPP)
	}
	// Verify modeled_4x_projected is validated (honest speedup is ~1.86x, so false)
	if compRep.Modeled4xProjected {
		t.Errorf("compRep.Modeled4xProjected = true, want false at 1.86x")
	}
	if compRep.True4xAchieved {
		t.Errorf("compRep.True4xAchieved = true, want false at 1.86x")
	}
	if compRep.IsModeled4xProjected() {
		t.Errorf("compRep.IsModeled4xProjected() = true, want false")
	}

	// 3. Verify JSON serialization includes modeled provenance and both projections.
	rawJSON, err := json.Marshal(compRep)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	jsonStr := string(rawJSON)
	for _, expectedKey := range []string{
		`"provenance":"MODELED"`,
		`"is_physical_silicon":false`,
		`"unmodeled_effects"`,
		`"modeled_4x_projected"`,
		`"true_4x_achieved"`,
	} {
		if !strings.Contains(jsonStr, expectedKey) {
			t.Errorf("JSON output missing key %q:\n%s", expectedKey, jsonStr)
		}
	}

	// 4. Verify JSON backward compatibility: legacy payload with true_4x_achieved
	legacyJSON := `{"schema":"fak.macbench.manyagent-compare.v1","true_4x_achieved":true}`
	var legacyDeser ManyAgentComparisonReport
	if err := json.Unmarshal([]byte(legacyJSON), &legacyDeser); err != nil {
		t.Fatalf("Unmarshal legacy JSON: %v", err)
	}
	if !legacyDeser.Modeled4xProjected || !legacyDeser.True4xAchieved || !legacyDeser.IsModeled4xProjected() {
		t.Errorf("legacy true_4x_achieved mapping failed: Modeled4xProjected=%v True4xAchieved=%v",
			legacyDeser.Modeled4xProjected, legacyDeser.True4xAchieved)
	}
	if legacyDeser.Provenance != "MODELED" || legacyDeser.IsPhysicalSilicon {
		t.Errorf("legacy default provenance failed: %+v", legacyDeser)
	}

	// 5. Verify JSON modern payload with modeled_4x_projected
	modernJSON := `{"schema":"fak.macbench.manyagent-compare.v1","modeled_4x_projected":true}`
	var modernDeser ManyAgentComparisonReport
	if err := json.Unmarshal([]byte(modernJSON), &modernDeser); err != nil {
		t.Fatalf("Unmarshal modern JSON: %v", err)
	}
	if !modernDeser.Modeled4xProjected || !modernDeser.True4xAchieved || !modernDeser.IsModeled4xProjected() {
		t.Errorf("modern modeled_4x_projected mapping failed: Modeled4xProjected=%v True4xAchieved=%v",
			modernDeser.Modeled4xProjected, modernDeser.True4xAchieved)
	}

	// 6. Verify CLI text output formatting and sanitization of 'TRUE 4x achieved'.
	var stdout, stderr bytes.Buffer
	code := runMacBenchManyAgent(&stdout, &stderr, []string{"--compare-llama", "-c", "4", "--horizon", "20"})
	if code != 0 {
		t.Fatalf("runMacBenchManyAgent returned %d: %s", code, stderr.String())
	}
	cliOut := stdout.String()
	if !strings.Contains(cliOut, "[MODELED PROJECTION]") {
		t.Errorf("CLI output missing [MODELED PROJECTION] banner:\n%s", cliOut)
	}
	if !strings.Contains(cliOut, "verification          : PROJECTED (MODELED") {
		t.Errorf("CLI output missing PROJECTED (MODELED:\n%s", cliOut)
	}
	if strings.Contains(cliOut, "TRUE") || strings.Contains(cliOut, "achieved") {
		t.Errorf("CLI output still contains unsanitized claim:\n%s", cliOut)
	}

	// 7. Verify that when modeled speedup >= 4.0x is simulated, output prints PROJECTED without TRUE.
	stdout.Reset()
	mock4xRep := compRep
	mock4xRep.SpeedupRatio = 4.25
	mock4xRep.SetModeled4xProjected(true)
	printManyAgentComparisonSummary(&stdout, mock4xRep)
	mockOut := stdout.String()
	if !strings.Contains(mockOut, "verification          : PROJECTED (MODELED 4.25x wall-clock speedup >= 4.0x projected)") {
		t.Errorf("mock 4x projection verification line mismatch:\n%s", mockOut)
	}
	if strings.Contains(mockOut, "TRUE") || strings.Contains(mockOut, "achieved") {
		t.Errorf("mock 4x projection output still contains TRUE or achieved:\n%s", mockOut)
	}
}
