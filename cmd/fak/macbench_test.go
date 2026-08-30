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
	if result.Schema != "fak.macbench.comparison.validation.v1" || !result.Valid || len(result.PacketSHA256) != 64 {
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
