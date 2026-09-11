package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func TestResolveServeNativeContext(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	weights := compute.MemoryPlan{{
		Class: compute.MemoryWeights,
		Bytes: 1 << 20,
		Scope: compute.MemoryScopeDevice,
	}}
	fit := serveFitBudget{Base: 1 << 30}

	tests := []struct {
		name       string
		requested  int
		wantSource string
		want       int
	}{
		{name: "auto uses declared window when it fits", wantSource: "auto", want: 4096},
		{name: "explicit narrows declared window", requested: 2048, wantSource: "explicit", want: 2048},
		{name: "explicit declared boundary", requested: 4096, wantSource: "explicit", want: 4096},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, gotPlan, err := resolveServeNativeContext(ws, weights, fit, tt.requested)
			if err != nil {
				t.Fatal(err)
			}
			if got.RequestedTokens != tt.requested || got.ModelDeclaredTokens != 4096 || got.ResolvedTokens != tt.want || got.Source != tt.wantSource {
				t.Fatalf("resolution = %+v, want requested=%d declared=4096 resolved=%d source=%q", got, tt.requested, tt.want, tt.wantSource)
			}
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatal(err)
			}
			wantPlan := cfg.ContextSizeConfig().PerContextMemoryPlan(tt.want)
			if !reflect.DeepEqual(gotPlan, wantPlan) {
				t.Fatalf("context plan = %+v, want exact plan for resolved %d tokens %+v", gotPlan, tt.want, wantPlan)
			}
		})
	}
}

func TestResolveServeNativeContextRejectsInvalidExplicitLimits(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	weights := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 1 << 20}}
	fit := serveFitBudget{Base: 1 << 30}

	if _, _, err := resolveServeNativeContext(ws, weights, fit, -1); err == nil || !strings.Contains(err.Error(), "0 (auto) or positive") {
		t.Fatalf("negative native context error = %v, want flag validation refusal", err)
	}
	// The fixture's WeightSource has no tensor payload reader. A typed header
	// declaration is sufficient to reject the request before payload loading.
	if _, _, err := resolveServeNativeContext(ws, weights, fit, 4097); err == nil || !strings.Contains(err.Error(), "exceeds model-declared context window 4096") {
		t.Fatalf("over-declared native context error = %v, want header-only refusal", err)
	}
}

func TestResolveServeNativeContextUnknownMetadataStaysTruthful(t *testing.T) {
	got, plan, err := resolveServeNativeContext(nil, nil, serveFitBudget{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelDeclaredTokens != 0 || got.ResolvedTokens != 0 || got.Source != "auto" || len(plan) != 0 {
		t.Fatalf("unknown auto resolution fabricated capacity: resolution=%+v plan=%+v", got, plan)
	}

	got, plan, err = resolveServeNativeContext(nil, nil, serveFitBudget{}, 32768)
	if err != nil {
		t.Fatal(err)
	}
	if got.ModelDeclaredTokens != 0 || got.ResolvedTokens != 32768 || got.Source != "explicit" || len(plan) != 0 {
		t.Fatalf("explicit unknown-model resolution = %+v plan=%+v, want operator cap 32768 without fabricated model metadata", got, plan)
	}
}

func TestResolveServeNativeContextDirectoryUsesConfigOnly(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"max_position_embeddings":65536}`), 0o644); err != nil {
		t.Fatal(err)
	}

	auto, err := resolveServeNativeContextDirectory(dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	if auto.RequestedTokens != 0 || auto.ModelDeclaredTokens != 65536 || auto.ResolvedTokens != 65536 || auto.Source != "model-metadata" {
		t.Fatalf("directory auto resolution = %+v, want 0/65536/65536/model-metadata", auto)
	}
	explicit, err := resolveServeNativeContextDirectory(dir, 32768)
	if err != nil {
		t.Fatal(err)
	}
	if explicit.ModelDeclaredTokens != 65536 || explicit.ResolvedTokens != 32768 || explicit.Source != "explicit" {
		t.Fatalf("directory explicit resolution = %+v, want declared 65536 resolved 32768 explicit", explicit)
	}
	// The directory intentionally contains no weight payload. The declaration in
	// config.json must reject an impossible explicit cap before payload loading.
	if _, err := resolveServeNativeContextDirectory(dir, 65537); err == nil || !strings.Contains(err.Error(), "exceeds model-declared context window 65536") {
		t.Fatalf("directory over-declared error = %v, want config-only refusal", err)
	}
}

func TestResolveServeNativeContextAutoCanExceedSchedulerDefault(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	ws.File.Metadata["qwen2.context_length"] = ggufload.Value{Type: ggufload.TypeUint64, Value: uint64(65536)}
	weights := compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 1 << 20, Scope: compute.MemoryScopeDevice}}
	got, _, err := resolveServeNativeContext(ws, weights, serveFitBudget{Base: 1 << 30}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got.ResolvedTokens <= 8192 || got.ResolvedTokens > got.ModelDeclaredTokens {
		t.Fatalf("auto resolution = %+v, want a fitted model-aware window above scheduler default 8192", got)
	}

	_, sf := newServeFlagSet()
	if budget := effectiveNativeAdmissionTokenBudget(sf, false, got.ResolvedTokens); budget != got.ResolvedTokens {
		t.Fatalf("default scheduler budget = %d, want resolved context %d so valid long requests reach native admission", budget, got.ResolvedTokens)
	}
	if cfg := serveNativePlannerConfigWithContext(sf, got.ResolvedTokens); cfg.ContextTokens != got.ResolvedTokens {
		t.Fatalf("auto planner context = %d, want resolved long context %d", cfg.ContextTokens, got.ResolvedTokens)
	}
}

func TestServeNativeContextAutoAdmitsLongNativeHTTP(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	const declaredWindow = 65536

	ws := serveSynthWideWindowWeightSource(t)
	ws.File.Metadata["qwen2.context_length"] = ggufload.Value{Type: ggufload.TypeUint64, Value: uint64(declaredWindow)}
	resolution, _, err := resolveServeNativeContext(ws, nil, serveFitBudget{Base: 1 << 30}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ResolvedTokens != declaredWindow {
		t.Fatalf("auto resolved context = %d, want model-declared %d", resolution.ResolvedTokens, declaredWindow)
	}

	_, sf := newServeFlagSet()
	*sf.nativeAdmissionTokenBudget = effectiveNativeAdmissionTokenBudget(sf, false, resolution.ResolvedTokens)
	if *sf.nativeAdmissionTokenBudget != declaredWindow {
		t.Fatalf("auto scheduler budget = %d, want resolved context %d", *sf.nativeAdmissionTokenBudget, declaredWindow)
	}

	tok := testProbeTokenizer(t)
	prompt := strings.Repeat("a", 9000)
	promptIDs, err := tok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if len(promptIDs) <= 8192 {
		t.Fatalf("synthetic prompt tokens = %d, want a real long-context request above the former 8192 cap", len(promptIDs))
	}

	m := fakmodel.NewSynthetic(fakmodel.Config{
		ModelType:             "llama",
		HiddenSize:            32,
		NumLayers:             1,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             260,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		MaxPositionEmbeddings: declaredWindow,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
	})
	m.Quantize()
	srv, err := gateway.New(gateway.Config{
		Model:           "native-long-context",
		InKernelModel:   m,
		Tokenizer:       tok,
		InKernelPlanner: serveNativePlannerConfigWithContext(sf, resolution.ResolvedTokens),
	})
	if err != nil {
		t.Fatal(err)
	}
	controller, _, err := newServeNativeAdmissionController(sf)
	if err != nil {
		t.Fatal(err)
	}
	srv.SetAdmissionController(controller)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, err := json.Marshal(map[string]any{
		"model":       "native-long-context",
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  1,
		"temperature": 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("long native HTTP status = %d, want 200; body=%s", resp.StatusCode, raw)
	}
	var envelope struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatalf("decode long native response: %v; body=%s", err, raw)
	}
	if envelope.Usage.PromptTokens <= 8192 {
		t.Fatalf("served native prompt tokens = %d, want exact tokenizer usage above former 8192 cap", envelope.Usage.PromptTokens)
	}
}

type serveContextBudgetedBackend struct {
	serveCapBackend
	weightBudget int64
}

func (b *serveContextBudgetedBackend) DeviceWeightBudget() (int64, bool) {
	return b.weightBudget, b.weightBudget > 0
}

func TestServeNativeContextDeviceSizingMatchesWeightBudgetedLoadPlan(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	be := &serveContextBudgetedBackend{
		serveCapBackend: serveCapBackend{
			Backend:     compute.Default(),
			total:       2 << 20,
			free:        2 << 20,
			known:       true,
			uploadDtype: true,
		},
		weightBudget: 256 << 10,
	}

	arm := resolveDeviceServeLoadArm(ws, be, false)
	rawWeights, err := serveGGUFWeightMemoryPlanForArm(ws, arm)
	if err != nil {
		t.Fatal(err)
	}
	weights, fit, err := serveNativeContextSizingInputs(ws, be, false, false, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := weights.DeviceTotal(); got != be.weightBudget {
		t.Fatalf("budgeted device weights = %d, want explicit device weight budget %d", got, be.weightBudget)
	}
	if got, want := weights.HostTotal(), rawWeights.DeviceTotal()-be.weightBudget; got != want {
		t.Fatalf("host-visible weight spill = %d, want %d from the live load-plan split", got, want)
	}

	resolution, contextPlan, err := resolveServeNativeContext(ws, weights, fit, 0)
	if err != nil {
		t.Fatal(err)
	}
	rawResolution, _, err := resolveServeNativeContext(ws, rawWeights, fit, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resolution.ResolvedTokens <= rawResolution.ResolvedTokens {
		t.Fatalf("budgeted auto context = %d, want above raw-weight context %d", resolution.ResolvedTokens, rawResolution.ResolvedTokens)
	}

	artifact, err := buildServeSizingArtifact(ws, be, false, 0, "native-model.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.ResolvedNativeContextTokens != resolution.ResolvedTokens {
		t.Fatalf("artifact resolved context = %d, want load-plan resolution %d", artifact.ResolvedNativeContextTokens, resolution.ResolvedTokens)
	}
	if got, want := artifact.Tiers.VRAMBytes, weights.DeviceTotal()+contextPlan.DeviceTotal(); got != want {
		t.Fatalf("artifact VRAM tier = %d, want budgeted weights plus context %d", got, want)
	}
	if got, want := artifact.Tiers.RAMBytes, weights.HostTotal()+contextPlan.HostTotal(); got != want {
		t.Fatalf("artifact RAM tier = %d, want host-visible weight spill plus context %d", got, want)
	}
}

func TestServeNativeContextMetalHostFitUsesSelectedResidentArm(t *testing.T) {
	origMetalAvailable := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = origMetalAvailable })
	serveMetalAvailable = func() bool { return false }
	t.Setenv("FAK_Q4K", "")
	const contextTokens = 16
	path := createTestQ4KGGUF(t)

	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	arm := resolveMetalServeLoadArm(ws)
	if arm != serveLoadArmResidentQ4K {
		t.Fatalf("selected Metal arm = %q, want resident Q4_K", arm)
	}
	if cpuArm := resolveHostServeLoadArm(ws, false); cpuArm != serveLoadArmQuantProfileQ8 {
		t.Fatalf("legacy CPU arm = %q, want quantized Q8", cpuArm)
	}
	residentPlan, err := serveGGUFMemoryPlanForArm(ws, arm, contextTokens, serveFitBudget{})
	if err != nil {
		t.Fatal(err)
	}
	q8Plan, err := serveGGUFMemoryPlanForArm(ws, serveLoadArmQuantProfileQ8, contextTokens, serveFitBudget{})
	if err != nil {
		t.Fatal(err)
	}
	reportedTotal := q8Plan.Total()
	usable := compute.BudgetAfterHeadroom(reportedTotal, serveGGUFHostHeadroom)
	if residentPlan.Total() > usable || q8Plan.Total() <= usable {
		t.Fatalf("fixture does not distinguish resident and Q8 plans: resident=%d q8=%d usable=%d", residentPlan.Total(), q8Plan.Total(), usable)
	}
	if err := fitServeGGUFPathOnReportedHostForArm(path, arm, contextTokens, reportedTotal, reportedTotal, true, nil); err != nil {
		t.Fatalf("selected Metal resident plan should fit reported unified memory: %v", err)
	}
	if err := fitServeGGUFPathOnReportedHost(path, false, contextTokens, reportedTotal, reportedTotal, true, nil); err == nil {
		t.Fatal("legacy CPU Q8 plan unexpectedly fit the same reported host; test no longer distinguishes the Metal arm")
	}

	// An injected sizing snapshot must win over the separately reported values so the load-time
	// admission cannot disagree with the context resolution after host memory drifts.
	fit := &serveFitBudget{Base: reportedTotal, Headroom: serveGGUFHostHeadroom}
	if err := fitServeGGUFPathOnReportedHostForArm(path, arm, contextTokens, 1, 1, true, fit); err != nil {
		t.Fatalf("injected fitting host snapshot should override later tiny report: %v", err)
	}
	fit = &serveFitBudget{Base: 1, Headroom: serveGGUFHostHeadroom}
	if err := fitServeGGUFPathOnReportedHostForArm(path, arm, contextTokens, reportedTotal, reportedTotal, true, fit); err == nil {
		t.Fatal("injected undersized host snapshot should override later fitting report")
	}
}

func TestServeNativeContextKeepsThreeTokenControlsIndependent(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{
		"--native-context-tokens", "32768",
		"--native-admission-token-budget", "4096",
		"--context-budget-tokens", "2048",
	}); err != nil {
		t.Fatal(err)
	}
	if *sf.nativeContextTokens != 32768 || *sf.nativeAdmissionTokenBudget != 4096 || *sf.contextBudgetTokens != 2048 {
		t.Fatalf("token controls crossed during parse: native=%d admission=%d session=%d", *sf.nativeContextTokens, *sf.nativeAdmissionTokenBudget, *sf.contextBudgetTokens)
	}
	if budget := effectiveNativeAdmissionTokenBudget(sf, true, 32768); budget != 4096 {
		t.Fatalf("explicit scheduler budget = %d, want its independent 4096 cap", budget)
	}
	if cfg := serveNativePlannerConfigWithContext(sf, 32768); cfg.ContextTokens != 32768 {
		t.Fatalf("planner context = %d, want resolved native context 32768", cfg.ContextTokens)
	}
	report := effectiveServeConfig(sf, deploymanifest.Manifest{}, false, explicitFlagNames(fs))
	if got := report.Values["native_context_tokens"]; got.Value != 32768 || got.Source != "flag" {
		t.Fatalf("effective config native_context_tokens = %+v, want 32768 from flag", got)
	}
}

func TestServePrintEffectiveConfigKeepsNativeContextUnresolvedWithoutModelIO(t *testing.T) {
	const childEnv = "FAK_TEST_SERVE_NATIVE_CONTEXT_EFFECTIVE_CONFIG"
	if os.Getenv(childEnv) == "1" {
		cmdServe([]string{
			"--gguf", "must-not-load.gguf",
			"--native-context-tokens", "65536",
			"--print-effective-config",
		})
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestServePrintEffectiveConfigKeepsNativeContextUnresolvedWithoutModelIO$")
	cmd.Env = append(os.Environ(), childEnv+"=1")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("print-effective-config reached model I/O or failed: %v\n%s", err, output)
	}
	line := strings.TrimSpace(strings.SplitN(string(output), "\n", 2)[0])
	var report effectiveServeConfigReport
	if err := json.Unmarshal([]byte(line), &report); err != nil {
		t.Fatalf("decode effective config: %v\n%s", err, output)
	}
	if got := report.Values["native_context_tokens"]; got.Value != float64(65536) || got.Source != "flag" {
		t.Fatalf("effective config native_context_tokens = %+v, want requested 65536 from flag", got)
	}
	if got := report.Values["native_context_resolution"]; got.Value != "unresolved" || got.Source != "model-header" {
		t.Fatalf("effective config native_context_resolution = %+v, want unresolved until model-header inspection", got)
	}
}

func TestServeNativeContextUnknownAutoFallsBackToSchedulerDefault(t *testing.T) {
	_, sf := newServeFlagSet()
	if got := effectiveNativeAdmissionTokenBudget(sf, false, 0); got != 8192 {
		t.Fatalf("unknown auto scheduler budget = %d, want existing default 8192", got)
	}
}

func TestServeSizingArtifactReportsResolvedNativeContext(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	artifact, err := buildServeSizingArtifact(ws, nil, false, 2048, "native-model.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.NativeContextTokens != 2048 || artifact.ModelDeclaredContextTokens != 4096 || artifact.ResolvedNativeContextTokens != 2048 || artifact.NativeContextSource != "explicit" {
		t.Fatalf("sizing artifact context = requested=%d declared=%d resolved=%d source=%q, want 2048/4096/2048/explicit",
			artifact.NativeContextTokens, artifact.ModelDeclaredContextTokens, artifact.ResolvedNativeContextTokens, artifact.NativeContextSource)
	}
	if artifact.Version != serveSizingVersion || artifact.ContextBudgetTokens != 2048 {
		t.Fatalf("sizing artifact compatibility = version %q context_budget_tokens %d, want %q deprecated alias 2048", artifact.Version, artifact.ContextBudgetTokens, serveSizingVersion)
	}
	raw, err := json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]any{
		"version":                        serveSizingVersion,
		"native_context_tokens":          float64(2048),
		"context_budget_tokens":          float64(2048),
		"model_declared_context_tokens":  float64(4096),
		"resolved_native_context_tokens": float64(2048),
		"native_context_source":          "explicit",
	} {
		if got := wire[key]; got != want {
			t.Fatalf("serialized sizing %s = %v, want %v; json=%s", key, got, want, raw)
		}
	}
}
