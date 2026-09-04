package macbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRunDecodeLongUsesBearerAndReportsTokPerSecond(t *testing.T) {
	var sawAuth bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","planner":"inkernel","model":"qwen3.6-27b"}`))
		case "/v1/chat/completions":
			if r.Header.Get("Authorization") == "Bearer secret" {
				sawAuth = true
			}
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":31,"completion_tokens":64,"total_tokens":95}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	rep, err := Run(context.Background(), Options{
		Gateway:      ts.URL + "/v1",
		Model:        "qwen3.6-27b",
		Key:          "secret",
		Suite:        SuiteDecodeLong,
		DecodeTokens: []int{64},
		HTTPClient:   ts.Client(),
		Now:          func() time.Time { return time.Date(2026, 7, 4, 6, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !sawAuth {
		t.Fatal("chat request did not carry the bearer")
	}
	if rep.Schema != Schema || !strings.HasPrefix(rep.Gateway, "http://127.0.0.1:") || !rep.Health.OK {
		t.Fatalf("unexpected report header: %+v", rep)
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rep.Rows))
	}
	row := rep.Rows[0]
	if row.CompletionTokens != 64 || row.TokensPerSecond <= 0 || !strings.Contains(row.Headline, "tok/s") {
		t.Fatalf("bad row: %+v", row)
	}
	b, _ := json.Marshal(rep)
	if strings.Contains(string(b), "secret") {
		t.Fatalf("report leaked bearer: %s", b)
	}
}

func TestRunPrefillSweepParsesSSEUsageAndTTFT(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true}`))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"length\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":16,\"total_tokens\":144}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	rep, err := Run(context.Background(), Options{
		Gateway:       ts.URL,
		Suite:         SuitePrefillSweep,
		PrefillTokens: []int{128},
		HTTPClient:    ts.Client(),
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rep.Rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(rep.Rows))
	}
	row := rep.Rows[0]
	if row.PromptTokens != 128 || row.CompletionTokens != 16 || row.TTFTSeconds <= 0 || row.PrefillTokensPerSecond <= 0 {
		t.Fatalf("bad prefill row: %+v", row)
	}
}

func TestSanitizeGatewayForReportKeepsLoopbackOnly(t *testing.T) {
	if got := SanitizeGatewayForReport("http://127.0.0.1:8080"); got != "http://127.0.0.1:8080" {
		t.Fatalf("loopback sanitize = %q", got)
	}
	if got := SanitizeGatewayForReport("http://100.64.0.1:8080"); got != "<remote-gateway>" {
		t.Fatalf("remote sanitize = %q", got)
	}
}

func TestRemoteGatewayErrorTextIsSanitized(t *testing.T) {
	rep, err := Run(context.Background(), Options{
		Gateway: "http://example.invalid:8080",
		Suite:   SuiteHealth,
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New(`Get "http://example.invalid:8080/healthz": context deadline exceeded`)
		})},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	joined := rep.Health.Error + "\n" + strings.Join(rep.Errors, "\n")
	if strings.Contains(joined, "example.invalid") || !strings.Contains(joined, "<remote-gateway>/healthz") {
		t.Fatalf("gateway was not sanitized in errors: %q", joined)
	}
}

func TestPlanRecoveryTailnetOfflineStaysScrubbed(t *testing.T) {
	no := false
	rep := Report{
		Suite:   SuiteHealth,
		Gateway: "http://100.64.1.2:8080",
		Health:  Health{Error: `Get "http://100.64.1.2:8080/healthz": context deadline exceeded`},
		Errors:  []string{`healthz failed: Get "http://100.64.1.2:8080/healthz": context deadline exceeded`},
	}
	plan := PlanRecovery(RecoverySignals{
		WatcherRunning: true,
		LatestReport:   &rep,
		TailnetOnline:  &no,
		SSHReachable:   &no,
		WakeHelper:     &no,
	})
	if plan.State != "tailnet_offline" || plan.Severity != "operator" {
		t.Fatalf("plan state=%q severity=%q, want tailnet_offline/operator: %+v", plan.State, plan.Severity, plan)
	}
	joined, _ := json.Marshal(plan)
	if strings.Contains(string(joined), "100.64.1.2") {
		t.Fatalf("recovery plan leaked raw gateway: %s", joined)
	}
	for _, want := range []string{"wake-or-power-mac", "confirm-tailnet-online", "restart-gateway", "document-wake-helper-gap"} {
		if !hasRecoveryAction(plan, want) {
			t.Fatalf("recovery plan missing action %q: %+v", want, plan.Actions)
		}
	}
}

func TestPlanRecoveryGatewayReadyWaitsForResult(t *testing.T) {
	rep := Report{
		Suite:   SuiteHealth,
		Gateway: "<remote-gateway>",
		Health:  Health{OK: true, Engine: "metal"},
	}
	plan := PlanRecovery(RecoverySignals{WatcherRunning: true, LatestReport: &rep})
	if plan.State != "gateway_ready" {
		t.Fatalf("plan state=%q, want gateway_ready: %+v", plan.State, plan)
	}
	if !hasRecoveryAction(plan, "wait-full-suite") {
		t.Fatalf("recovery plan missing wait-full-suite action: %+v", plan.Actions)
	}
}

// A watch log that no longer exists must not read as "the watcher just has not
// polled yet". Nightrun artifacts are host-local and rotate (.gitignore keeps
// experiments/nightrun/*/ out of git), so once a run's log ages out there is
// nothing left to wait for -- telling an operator to keep waiting is why #2611
// sat for 16 days looking benign.
func TestPlanRecoveryMissingLogIsNotMistakenForFirstPoll(t *testing.T) {
	absent := false
	plan := PlanRecovery(RecoverySignals{WatcherRunning: true, LogPresent: &absent})
	if plan.State != "log_missing" {
		t.Fatalf("plan state=%q, want log_missing: %+v", plan.State, plan)
	}
	if plan.Severity == "info" {
		t.Fatalf("a missing watch log must not be info severity: %+v", plan)
	}
	if hasRecoveryAction(plan, "wait-first-poll") {
		t.Fatalf("missing log must not advise waiting for a first poll: %+v", plan.Actions)
	}
	for _, want := range []string{"confirm-log-path", "start-fresh-watch"} {
		if !hasRecoveryAction(plan, want) {
			t.Fatalf("recovery plan missing action %q: %+v", want, plan.Actions)
		}
	}
}

// An unknown (never-inspected) log path keeps the pre-existing verdict, so a
// --result-only recover call is unchanged by the log_missing branch.
func TestPlanRecoveryUnknownLogPresenceKeepsFirstPollVerdict(t *testing.T) {
	plan := PlanRecovery(RecoverySignals{WatcherRunning: true})
	if plan.State != "no_health_report" || plan.Severity != "info" {
		t.Fatalf("plan state=%q severity=%q, want no_health_report/info: %+v", plan.State, plan.Severity, plan)
	}
	if !hasRecoveryAction(plan, "wait-first-poll") {
		t.Fatalf("unknown log presence should still advise waiting: %+v", plan.Actions)
	}
}

func TestElapsedSecondsFloorsNonPositiveDurations(t *testing.T) {
	got := elapsedSeconds(time.Now().Add(time.Second))
	if got != 0.001 {
		t.Fatalf("elapsedSeconds future start = %v, want 0.001", got)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func hasRecoveryAction(plan RecoveryPlan, id string) bool {
	for _, action := range plan.Actions {
		if action.ID == id {
			return true
		}
	}
	return false
}

func TestValidateComparisonPacketAcceptsExactQwen38ThreeWayEvidence(t *testing.T) {
	packet := validComparisonPacket()
	if err := ValidateComparisonPacket(packet); err != nil {
		t.Fatalf("ValidateComparisonPacket: %v", err)
	}
}

func TestValidateComparisonPacketNodeMacOSA(t *testing.T) {
	packet := nodeMacOSAComparisonPacket()
	if err := ValidateComparisonPacket(packet); err != nil {
		t.Fatalf("ValidateComparisonPacket(nodeMacOSA): %v", err)
	}
	for _, arm := range packet.Arms {
		t.Logf("Arm %s:", arm.Name)
		t.Logf("  Prefill TTFT: p50=%.2f ms, p95=%.2f ms", arm.Metrics.Prefill.TTFTMS.P50, arm.Metrics.Prefill.TTFTMS.P95)
		t.Logf("  Prefill tok/s: p50=%.2f, p95=%.2f", arm.Metrics.Prefill.TokPerS.P50, arm.Metrics.Prefill.TokPerS.P95)
		t.Logf("  Decode ITL: p50=%.2f ms, p95=%.2f ms", arm.Metrics.Decode.ITLMS.P50, arm.Metrics.Decode.ITLMS.P95)
		t.Logf("  Decode tok/s: p50=%.2f, p95=%.2f", arm.Metrics.Decode.TokPerS.P50, arm.Metrics.Decode.TokPerS.P95)
	}
}

func TestValidateComparisonPacketFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ComparisonPacket)
		want string
	}{
		{name: "non Qwen3.8", edit: func(p *ComparisonPacket) { p.Model.Family = "Qwen3.6" }, want: "model.family"},
		{name: "missing artifact digest", edit: func(p *ComparisonPacket) { p.Arms[0].Artifact.SHA256 = "" }, want: "artifact.sha256"},
		{name: "quant mismatch", edit: func(p *ComparisonPacket) { p.Arms[1].Artifact.Quant = "Q8_0" }, want: "artifact.quant"},
		{name: "canonical weight mismatch", edit: func(p *ComparisonPacket) { p.Arms[1].Artifact.CanonicalWeightsSHA256 = strings.Repeat("0", 64) }, want: "canonical_weights_sha256"},
		{name: "hardware mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].Hardware.Chip = "Apple M4 Max" }, want: "hardware"},
		{name: "host mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].HostID = strings.Repeat("0", 64) }, want: "host_id"},
		{name: "OS mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].OS.Build = "different" }, want: "os"},
		{name: "prompt mismatch", edit: func(p *ComparisonPacket) { p.Arms[1].PromptSetSHA256 = strings.Repeat("b", 64) }, want: "prompt_set_sha256"},
		{name: "context mismatch", edit: func(p *ComparisonPacket) { p.Arms[1].ContextTokens++ }, want: "context_tokens"},
		{name: "output mismatch", edit: func(p *ComparisonPacket) { p.Arms[1].OutputTokens++ }, want: "output_tokens"},
		{name: "quality failed", edit: func(p *ComparisonPacket) { p.Arms[2].Quality.Passed = false }, want: "quality.passed"},
		{name: "quality version mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].Quality.PolicyVersion = "old" }, want: "quality.policy_version"},
		{name: "quality policy digest mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].Quality.PolicySHA256 = strings.Repeat("0", 64) }, want: "quality.policy_sha256"},
		{name: "fak wrong engine", edit: func(p *ComparisonPacket) { p.Arms[0].Engine = "llama.cpp" }, want: "engine"},
		{name: "fak wrong runtime", edit: func(p *ComparisonPacket) { p.Arms[0].Runtime = "gateway" }, want: "runtime"},
		{name: "fallback count", edit: func(p *ComparisonPacket) { p.Arms[0].FallbackCount = 1 }, want: "fallback_count"},
		{name: "fallback named", edit: func(p *ComparisonPacket) { p.Arms[1].Fallback = "fak-native" }, want: "fallback"},
		{name: "sample fallback", edit: func(p *ComparisonPacket) { p.Arms[0].Samples[0].FallbackCount = 1 }, want: "samples[0].fallback_count"},
		{name: "sample population mismatch", edit: func(p *ComparisonPacket) { p.Arms[2].Samples[0].Ordinal = 21; p.Arms[2].Samples[0].ID = "p1#21" }, want: "same prompt/ordinal"},
		{name: "phase rate mismatch", edit: func(p *ComparisonPacket) { p.Arms[0].Samples[0].DecodeTokPerS++ }, want: "decode_tok_s"},
		{name: "missing phase metrics", edit: func(p *ComparisonPacket) { p.Arms[2].Metrics.Decode.ITLMS.P95 = 0 }, want: "decode.itl_ms"},
		{name: "missing raw samples", edit: func(p *ComparisonPacket) { p.Arms[1].Samples = nil }, want: "samples"},
		{name: "metrics disagree with samples", edit: func(p *ComparisonPacket) { p.Arms[0].Metrics.Prefill.TTFTMS.P50++ }, want: "prefill.ttft_ms"},
		{name: "boundary not fully accounted", edit: func(p *ComparisonPacket) { p.Arms[0].Samples[0].Boundary.OtherMS = 0 }, want: "request_boundary"},
		{name: "missing raw result", edit: func(p *ComparisonPacket) { p.Arms[0].RawResult.SHA256 = "" }, want: "raw_result.sha256"},
		{name: "missing repro", edit: func(p *ComparisonPacket) { p.Arms[0].Repro = nil }, want: "repro"},
		{name: "duplicate arm", edit: func(p *ComparisonPacket) { p.Arms[2].Name = "llama.cpp" }, want: "exactly one"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			packet := validComparisonPacket()
			tt.edit(&packet)
			err := ValidateComparisonPacket(packet)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func validComparisonPacket() ComparisonPacket {
	hardware := ComparisonHardware{Model: "MacBookPro18,3", Chip: "Apple M3 Pro", MemoryBytes: 36 << 30}
	os := ComparisonOS{Name: "macOS", Version: "26.6.2", Build: "25G83"}
	promptDigest := strings.Repeat("a", 64)
	packet := ComparisonPacket{
		Schema:      ComparisonSchema,
		GeneratedAt: "2026-08-30T05:00:00Z",
		CampaignID:  "issue-2723-qwen38-three-way-v1",
		HostID:      strings.Repeat("f", 64),
		Model: ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: strings.Repeat("9", 64),
			Quant:                  "Q4_K_M",
		},
		Hardware: hardware,
		OS:       os,
		PromptSet: ComparisonPromptSet{
			ID:      "issue-2723-v1",
			SHA256:  promptDigest,
			Prompts: []ComparisonPrompt{{ID: "p1", SHA256: strings.Repeat("c", 64)}},
		},
		ContextTokens: 128,
		OutputTokens:  64,
		QualityPolicy: ComparisonQualityPolicy{ID: "strict-token-parity", Version: "1", SHA256: strings.Repeat("8", 64), MinimumScore: 1},
	}
	for _, name := range []string{"fak-native", "llama.cpp", "mlx"} {
		engine, runtime := name, "reference"
		if name == "fak-native" {
			runtime = "inkernel"
		}
		arm := ComparisonArm{
			Name:            name,
			EvidenceKind:    "observed",
			RunID:           "issue-2723-" + name,
			StartedAt:       "2026-08-30T04:00:00Z",
			FinishedAt:      "2026-08-30T04:30:00Z",
			HostID:          packet.HostID,
			Engine:          engine,
			Runtime:         runtime,
			RuntimeRevision: "runtime-revision-" + name,
			Fallback:        "none",
			ModelID:         packet.Model.ID,
			Hardware:        hardware,
			OS:              os,
			PromptSetSHA256: promptDigest,
			ContextTokens:   packet.ContextTokens,
			OutputTokens:    packet.OutputTokens,
			Artifact: ComparisonArtifact{
				Identity:               "hf://example/" + name,
				SHA256:                 strings.Repeat(fmt.Sprintf("%x", len(packet.Arms)+1), 64),
				Format:                 map[string]string{"fak-native": "gguf", "llama.cpp": "gguf", "mlx": "safetensors"}[name],
				SourceRevision:         packet.Model.SourceRevision,
				CanonicalWeightsSHA256: packet.Model.CanonicalWeightsSHA256,
				Quant:                  packet.Model.Quant,
			},
			Quality: ComparisonQualityResult{
				PolicyRef: packet.QualityPolicy.ID, PolicyVersion: packet.QualityPolicy.Version, PolicySHA256: packet.QualityPolicy.SHA256,
				Passed: true, Score: 1, ResultPath: "quality/" + name + ".json", ResultSHA256: strings.Repeat("d", 64),
			},
			RawResult: ComparisonRawResult{Path: "raw/" + name + ".json", SHA256: strings.Repeat("e", 64)},
			Repro:     []string{"run", name},
		}
		for i := 1; i <= MinimumComparisonSamples; i++ {
			f := float64(i)
			prefillMS := 50 + f
			decodeMS := 100 + f
			arm.Samples = append(arm.Samples, ComparisonSample{
				ID:              fmt.Sprintf("p1#%d", i),
				PromptID:        "p1",
				PromptSHA256:    packet.PromptSet.Prompts[0].SHA256,
				Ordinal:         i,
				InputTokens:     packet.ContextTokens,
				OutputTokens:    packet.OutputTokens,
				Engine:          arm.Engine,
				Runtime:         arm.Runtime,
				RuntimeRevision: arm.RuntimeRevision,
				Fallback:        "none",
				ArtifactSHA256:  arm.Artifact.SHA256,
				TTFTMS:          30 + prefillMS,
				ITLMS:           decodeMS / float64(packet.OutputTokens-1),
				PrefillTokPerS:  float64(packet.ContextTokens) * 1000 / prefillMS,
				DecodeTokPerS:   float64(packet.OutputTokens-1) * 1000 / decodeMS,
				Boundary: ComparisonRequestBoundary{
					TotalMS: 50 + prefillMS + decodeMS, QueueMS: 10, SetupMS: 20, PrefillMS: prefillMS,
					DecodeMS: decodeMS, VerificationMS: 10, RecoveryMS: 0, OtherMS: 10,
				},
			})
		}
		arm.Metrics = SummarizeComparisonSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	return packet
}

func nodeMacOSAComparisonPacket() ComparisonPacket {
	hardware := ComparisonHardware{Model: "Mac15,7", Chip: "Apple M3 Pro", MemoryBytes: 38654705664}
	osInfo := ComparisonOS{Name: "macOS", Version: "26.6.2", Build: "25G83"}
	hostID := strings.Repeat("6", 64)
	promptSetSHA := strings.Repeat("a", 64)
	promptSHA := strings.Repeat("b", 64)
	policySHA := strings.Repeat("8", 64)
	artifactSHA := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"

	packet := ComparisonPacket{
		Schema:      ComparisonSchema,
		GeneratedAt: "2026-09-03T05:30:00Z",
		CampaignID:  "issue-2723-mac-threeway-qwen38-20260903",
		HostID:      hostID,
		Model: ComparisonModel{
			Family:                 "Qwen3.8",
			ID:                     "Qwen3.8-27B",
			SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			CanonicalWeightsSHA256: artifactSHA,
			Quant:                  "Q4_K_M",
		},
		Hardware: hardware,
		OS:       osInfo,
		PromptSet: ComparisonPromptSet{
			ID:      "issue-2723-agentic-prompts-v1",
			SHA256:  promptSetSHA,
			Prompts: []ComparisonPrompt{{ID: "p1", SHA256: promptSHA}},
		},
		ContextTokens: 128,
		OutputTokens:  64,
		QualityPolicy: ComparisonQualityPolicy{
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
		arm := ComparisonArm{
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
			Artifact: ComparisonArtifact{
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
			Quality: ComparisonQualityResult{
				PolicyRef:     packet.QualityPolicy.ID,
				PolicyVersion: packet.QualityPolicy.Version,
				PolicySHA256:  policySHA,
				Passed:        true,
				Score:         1.0,
				ResultPath:    spec.name + "-quality.json",
				ResultSHA256:  strings.Repeat("c", 64),
			},
			RawResult: ComparisonRawResult{
				Path:   spec.name + "-raw.json",
				SHA256: strings.Repeat("d", 64),
			},
			Repro: spec.repro,
		}

		for i := 1; i <= MinimumComparisonSamples; i++ {
			delta := offsets[i-1]
			pfMS := spec.basePF + delta
			decMS := spec.baseDec + delta*2.0
			queueMS := spec.queueMS
			setupMS := spec.setupMS
			verifMS := 5.0
			otherMS := 5.0
			totalMS := queueMS + setupMS + pfMS + decMS + verifMS + otherMS

			arm.Samples = append(arm.Samples, ComparisonSample{
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
				Boundary: ComparisonRequestBoundary{
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
		arm.Metrics = SummarizeComparisonSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	return packet
}
