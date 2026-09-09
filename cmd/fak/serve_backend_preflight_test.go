package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

type servePreflightBackend struct {
	compute.Backend
	name string
	tier string
}

func (b *servePreflightBackend) Name() string { return b.name }

func (b *servePreflightBackend) Tier() string {
	if b.tier != "" {
		return b.tier
	}
	if b.Backend != nil {
		return b.Backend.Tier()
	}
	return ""
}

type servePreflightMarkerBackend struct{ *servePreflightBackend }

func (*servePreflightMarkerBackend) Qwen35GDNPath() string { return fakmodel.Qwen35GDNCUDAPath }

type servePreflightGDNBackend struct {
	*servePreflightBackend
	path string
}

func (b *servePreflightGDNBackend) Qwen35GDNPath() string { return b.path }

func (*servePreflightGDNBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	return compute.Tensor{}, compute.Tensor{}, compute.Tensor{}, nil
}

func newServePreflightBackend(name string) *servePreflightBackend {
	return &servePreflightBackend{Backend: compute.Default(), name: name}
}

func newServePreflightBackendWithTier(name, tier string) *servePreflightBackend {
	return &servePreflightBackend{Backend: compute.Default(), name: name, tier: tier}
}

func servePreflightHeader(t *testing.T, arch string, r io.ReaderAt) *ggufload.WeightSource {
	t.Helper()
	p := arch + "."
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                 {Type: ggufload.TypeString, Value: arch},
			p + "embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			p + "block_count":                      {Type: ggufload.TypeUint64, Value: uint64(4)},
			p + "attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			p + "attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			p + "feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			p + "attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
		},
		// A sparse tensor directory entry points far beyond the header. Config derivation
		// may inspect its name, but any attempt to touch its payload hits the trapped reader.
		Tensors: []ggufload.TensorInfo{{
			Name:       "model.layers.0.linear_attn.in_proj_qkv.weight",
			Dims:       []uint64{32, 32},
			Type:       ggufload.TensorF32,
			FileOffset: 1 << 40,
		}},
	}
	if arch == "qwen35" {
		f.Metadata[p+"full_attention_interval"] = ggufload.Value{Type: ggufload.TypeUint64, Value: uint64(4)}
	}
	ws, err := ggufload.NewWeightSource(f, r, 1<<41)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func servePreflightOpen(t *testing.T, arch string, r io.ReaderAt) serveGGUFHeaderOpener {
	t.Helper()
	return func(string) (*ggufload.WeightSource, error) {
		return servePreflightHeader(t, arch, r), nil
	}
}

func TestServeBackendForwardPreflightSupportMatrix(t *testing.T) {
	missing := newServePreflightBackend("cuda")
	marker := &servePreflightMarkerBackend{servePreflightBackend: newServePreflightBackend("cuda")}
	exact := &servePreflightGDNBackend{
		servePreflightBackend: newServePreflightBackend("cuda"),
		path:                  fakmodel.Qwen35GDNCUDAPath,
	}
	wrong := &servePreflightGDNBackend{
		servePreflightBackend: newServePreflightBackend("cuda"),
		path:                  "cuda/qwen35-gdn-wrong-v0",
	}

	for name, tc := range map[string]struct {
		be             compute.Backend
		reasonContains string
	}{
		"missing-interface": {be: missing, reasonContains: "does not structurally implement model.Qwen35GDNBackend"},
		"marker-only":       {be: marker, reasonContains: "advertises marker path"},
		"wrong-path":        {be: wrong, reasonContains: "cuda/qwen35-gdn-wrong-v0"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := preflightServeBackendForwardWith("synthetic.gguf", tc.be, servePreflightOpen(t, "qwen35", nil))
			if got != (serveBackendForwardPreflight{}) {
				t.Fatalf("refusal result=%#v, want empty", got)
			}
			var unsupported *fakmodel.UnsupportedBackendForwardError
			if !errors.As(err, &unsupported) {
				t.Fatalf("error=%T %v, want *model.UnsupportedBackendForwardError", err, err)
			}
			if unsupported.Backend != "cuda" || unsupported.Forward != fakmodel.ForwardQwen35GDN ||
				unsupported.IntendedPath != fakmodel.Qwen35GDNCUDAPath+" or "+fakmodel.Qwen35GDNVulkanPath {
				t.Fatalf("wrong typed refusal: %#v", unsupported)
			}
			for _, want := range []string{fakmodel.Qwen35GDNCUDAPath, fakmodel.Qwen35GDNVulkanPath, "refusing generic QKV/CPU fallback"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("refusal missing %q: %v", want, err)
				}
			}
			if !strings.Contains(unsupported.Reason, tc.reasonContains) {
				t.Errorf("typed refusal reason=%q, want %q", unsupported.Reason, tc.reasonContains)
			}
		})
	}

	got, err := preflightServeBackendForwardWith("synthetic.gguf", exact, servePreflightOpen(t, "qwen35", nil))
	if err != nil {
		t.Fatalf("exact-path Qwen3.6 backend refused: %v", err)
	}
	want := serveBackendForwardPreflight{Backend: "cuda", Forward: fakmodel.ForwardQwen35GDN, Path: fakmodel.Qwen35GDNCUDAPath}
	if got != want {
		t.Fatalf("exact-path result=%#v, want %#v", got, want)
	}
	if got, err := preflightServeBackendForwardWith("synthetic.gguf", missing, servePreflightOpen(t, "qwen2", nil)); err != nil || got != (serveBackendForwardPreflight{}) {
		t.Fatalf("dense Qwen2.5 admission changed: result=%#v err=%v", got, err)
	}
	if got, err := preflightServeBackendForwardWith("synthetic.gguf", nil, servePreflightOpen(t, "qwen35", nil)); err != nil || got != (serveBackendForwardPreflight{}) {
		t.Fatalf("Qwen3.6 CPU/reference admission changed: result=%#v err=%v", got, err)
	}
	called := false
	if got, err := preflightServeBackendForwardWith("", missing, func(string) (*ggufload.WeightSource, error) {
		called = true
		return nil, errors.New("must not open")
	}); err != nil || got != (serveBackendForwardPreflight{}) || called {
		t.Fatalf("empty path result=%#v err=%v opener_called=%v, want empty no-op", got, err, called)
	}
}

type servePreflightTrapReader struct{ calls int }

func (r *servePreflightTrapReader) ReadAt([]byte, int64) (int, error) {
	r.calls++
	return 0, errors.New("trapped GGUF tensor payload read")
}

// runServePreflightRefusalHarness mirrors the production return boundary and makes
// downstream construction observable without invoking must, whose contract is os.Exit.
func runServePreflightRefusalHarness(
	be compute.Backend,
	open serveGGUFHeaderOpener,
	record func(gateway.StartupMessage),
	downstream func(string),
) error {
	pf, err := preflightServeBackendForwardWith("synthetic-sparse.gguf", be, open)
	if err != nil {
		return err
	}
	if message := serveBackendForwardPreflightMessage(pf); message.Text != "" && record != nil {
		record(message)
	}
	for _, event := range []string{"memory-plan", "model-load", "tokenizer", "planner", "listener"} {
		downstream(event)
	}
	return nil
}

func TestServeBackendForwardPreflightRefusesBeforeSparsePayloadAndDownstreamConstruction(t *testing.T) {
	trap := &servePreflightTrapReader{}
	var messages []gateway.StartupMessage
	var downstream []string
	err := runServePreflightRefusalHarness(
		newServePreflightBackend("cuda"),
		servePreflightOpen(t, "qwen35", trap),
		func(message gateway.StartupMessage) { messages = append(messages, message) },
		func(event string) { downstream = append(downstream, event) },
	)
	var unsupported *fakmodel.UnsupportedBackendForwardError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error=%T %v, want typed backend-forward refusal", err, err)
	}
	if trap.calls != 0 {
		t.Fatalf("GGUF payload ReadAt calls=%d, want zero", trap.calls)
	}
	if len(downstream) != 0 {
		t.Fatalf("downstream construction=%v, want zero memory-plan/model/tokenizer/planner/listener events", downstream)
	}
	if len(messages) != 0 {
		t.Fatalf("refusal retained success/readiness message: %+v", messages)
	}
}

func TestServeBackendForwardPreflightWritesExactCUDAAdmission(t *testing.T) {
	be := &servePreflightGDNBackend{
		servePreflightBackend: newServePreflightBackend("cuda"),
		path:                  fakmodel.Qwen35GDNCUDAPath,
	}
	result, err := preflightServeBackendForwardWith("synthetic.gguf", be, servePreflightOpen(t, "qwen35", nil))
	if err != nil {
		t.Fatal(err)
	}
	got := serveBackendForwardPreflightMessage(result)
	want := gateway.StartupMessage{Source: "model-load", Kind: "backend-forward", Level: "info", Text: "backend=cuda forward=qwen35-gdn path=cuda/qwen35-gdn-ssm-decode-v1"}
	if got != want {
		t.Fatalf("startup message=%+v, want %+v", got, want)
	}
}

func TestServeBackendForwardPreflightProductionOrder(t *testing.T) {
	root := repoRootFromTest(t)
	stages, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve_stages.go"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(stages)
	ordered := []string{
		"preflightServeBackendForward(*sf.ggufPath, rt.chatBackend)",
		"var expertShard *ggufload.ExpertShard",
		"loadServeInKernelModel(",
		"resolveServeTokenizer(",
	}
	last := -1
	for _, needle := range ordered {
		at := strings.Index(src, needle)
		if at < 0 {
			t.Fatalf("serve_stages.go missing %q", needle)
		}
		if at <= last {
			t.Fatalf("serve startup order does not place %q after prior stage", needle)
		}
		last = at
	}

	serve, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		t.Fatal(err)
	}
	frontDoor := string(serve)
	for _, pair := range [][2]string{{"rt.loadModel(sf)", "rt.buildGateway(sf)"}, {"rt.loadModel(sf)", "rt.run(sf)"}} {
		before, after := strings.Index(frontDoor, pair[0]), strings.Index(frontDoor, pair[1])
		if before < 0 || after < 0 || before >= after {
			t.Fatalf("serve.go must place %s before %s (planner/listener construction stays downstream)", pair[0], pair[1])
		}
	}
}

func TestServeBackendForwardPreflightLauncherLogsButDoesNotGrantReadiness(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "tools", "qwen36_a100_fak_serve.sh"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(body)
	for _, want := range []string{
		`FAK_Q4K=1 "$FAK_BIN" "${SERVE_ARGS[@]}" > "$QWEN_DIR/server.log" 2>&1 &`,
		`if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then`,
		`ph "QWEN36_A100_FAK_SERVE_READY`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("launcher missing %q", want)
		}
	}
	ready := strings.Index(src, `ph "QWEN36_A100_FAK_SERVE_READY`)
	smoke := strings.Index(src, `if printf '%s' "$smoke" | grep -q '"content"' && ! printf '%s' "$smoke" | grep -q '"error"'; then`)
	if smoke < 0 || ready <= smoke {
		t.Fatalf("readiness marker must remain inside the real chat-smoke success branch: smoke=%d ready=%d", smoke, ready)
	}
	if strings.Contains(src, "BACKEND_FORWARD_PREFLIGHT_OK") {
		t.Fatal("launcher must record the serve marker via server.log, not promote it into readiness logic")
	}
}

func TestServeBackendForwardPreflightOpenErrorRemainsGGUFError(t *testing.T) {
	want := fmt.Errorf("gguf: synthetic header refusal")
	_, got := preflightServeBackendForwardWith("bad.gguf", newServePreflightBackend("cuda"), func(string) (*ggufload.WeightSource, error) {
		return nil, want
	})
	if !errors.Is(got, want) || !strings.HasPrefix(got.Error(), "gguf:") {
		t.Fatalf("open error=%v, want unchanged GGUF error", got)
	}
}

func validServeLocalAdmissionRequest() modelreg.LocalAdmissionRequest {
	const gib = int64(1 << 30)
	sha := strings.Repeat("a", 64)
	return modelreg.LocalAdmissionRequest{
		Declaration: modelreg.LocalAdmissionDeclaration{
			ModelID: "tiny-gguf", ArtifactSHA256: sha, ArtifactBytes: gib,
			RuntimeID: "llama.cpp", RuntimeVersion: "b1234", RequiredRuntimeCapability: "gguf",
			Requested: modelreg.LocalDeviceTarget{DeviceKind: "cuda", DeviceID: "0", Resources: modelreg.LocalResourceRequirements{DiskBytes: gib, RAMBytes: gib, VRAMBytes: 2 * gib}},
		},
		Artifact: modelreg.LocalVerifiedArtifactFacts{Path: "/cache/tiny.gguf", SHA256: sha, Bytes: gib, Verified: true},
		Runtime:  modelreg.LocalRuntimeFacts{ID: "llama.cpp", Version: "b1234", Capabilities: []string{"gguf", "openai-http", "cpu", "cuda"}, Verified: true},
		Host: modelreg.LocalHostFacts{
			DiskKnown: true, FreeDiskBytes: 8 * gib, RAMKnown: true, FreeRAMBytes: 8 * gib,
			Devices: []modelreg.LocalDeviceFacts{{Kind: "cuda", ID: "0", Available: true, VRAMKnown: true, FreeVRAMBytes: 4 * gib}},
		},
	}
}

func TestServeLocalRuntimePreflightRefusesBeforeLauncherOrNetwork(t *testing.T) {
	req := validServeLocalAdmissionRequest()
	req.Host.Devices = nil // requested GPU is not measured on this host
	launcherCalls, networkCalls := 0, 0

	decision, err := preflightServeLocalRuntime(req, func(modelreg.LocalLaunchResourceReservation) error {
		launcherCalls++
		// A real external launcher may later contact its loopback health endpoint;
		// trapping it here proves admission refusal precedes that entire boundary.
		networkCalls++
		return nil
	})
	var refusal *modelreg.LocalAdmissionRefusalError
	if !errors.As(err, &refusal) || decision.Verdict != modelreg.LocalAdmissionRefuse || len(decision.Refusals) == 0 || decision.Refusals[0].Code != modelreg.LocalRefusalDeviceUnavailable {
		t.Fatalf("preflight decision=%+v err=%T %v, want typed unavailable-GPU refusal", decision, err, err)
	}
	if launcherCalls != 0 || networkCalls != 0 {
		t.Fatalf("refusal crossed effect boundary: launcher=%d network=%d, want both zero", launcherCalls, networkCalls)
	}
}

func TestServeLocalRuntimePreflightPassesAuditablePlanToLauncher(t *testing.T) {
	req := validServeLocalAdmissionRequest()
	launcherCalls := 0
	var launched modelreg.LocalLaunchResourceReservation
	decision, err := preflightServeLocalRuntime(req, func(plan modelreg.LocalLaunchResourceReservation) error {
		launcherCalls++
		launched = plan
		return nil
	})
	if err != nil {
		t.Fatalf("admitted local runtime refused: %v", err)
	}
	if launcherCalls != 1 || decision.Verdict != modelreg.LocalAdmissionAdmit || decision.Plan == nil {
		t.Fatalf("decision=%+v launcher_calls=%d, want one admitted launch", decision, launcherCalls)
	}
	if launched.ArtifactSHA256 != req.Declaration.ArtifactSHA256 || launched.DeviceKind != "cuda" || launched.DeviceID != "0" || launched.Required.VRAMBytes != 2*(1<<30) {
		t.Fatalf("launcher did not receive the admitted artifact/runtime/resource plan: %+v", launched)
	}
}

func TestServeBackendPreflight_StrixHaloActivation(t *testing.T) {
	defer os.Unsetenv("FAK_STRIX_GFX1151_OVERRIDE")

	// Case 1: Environment override activation (1 and true)
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "1")
	res1 := preflightServeStrixHalo(nil)
	if !res1.Detected {
		t.Fatal("expected Strix Halo detected=true via env override=1")
	}
	if res1.UMAPointerManager == nil {
		t.Fatal("expected initialized UMAPointerManager")
	}
	if res1.MALLTiler == nil {
		t.Fatal("expected initialized MALLTiler")
	}
	// MALL Infinity Cache tiling window capacity equals 8,192 tokens
	if capTokens := res1.MALLTiler.CapacityTokens(); capTokens != 8192 {
		t.Fatalf("expected 8192 MALL tokens, got %d", capTokens)
	}
	rootSpan := res1.MALLTiler.ClassifyTokenSpan(0, 8192)
	if rootSpan.PinnedTokens != 8192 || rootSpan.BypassTokens != 0 {
		t.Fatalf("expected 8192 pinned tokens in root context, got %+v", rootSpan)
	}
	bypassSpan := res1.MALLTiler.ClassifyTokenSpan(8192, 16384)
	if bypassSpan.PinnedTokens != 0 || bypassSpan.BypassTokens != 8192 {
		t.Fatalf("expected 8192 bypass tokens beyond MALL root context, got %+v", bypassSpan)
	}

	// UMA buffer allocation guarantees zero-copy pointer identity uintptr(HostPtr) == uintptr(DevPtr)
	buf, err := res1.UMAPointerManager.AllocateUMABuffer(1024, 64)
	if err != nil {
		t.Fatalf("failed to allocate UMA buffer: %v", err)
	}
	if !buf.IsZeroCopy() {
		t.Fatal("expected zero-copy UMA buffer on Strix Halo")
	}
	hp := buf.HostPtr()
	dp := buf.DevPtr()
	if hp == 0 || dp == 0 {
		t.Fatalf("expected non-zero pointers: hp=%x dp=%x", hp, dp)
	}
	if uintptr(hp) != uintptr(dp) {
		t.Fatalf("zero-copy pointer identity violation: uintptr(HostPtr)=0x%x != uintptr(DevPtr)=0x%x", hp, dp)
	}
	if !res1.UMAPointerManager.AssertPointerIdentity(buf.UnsafePointer(), dp) {
		t.Fatalf("AssertPointerIdentity failed: hp=0x%x dp=0x%x", hp, dp)
	}
	if err := res1.UMAPointerManager.FreeUMABuffer(buf); err != nil {
		t.Fatalf("failed to free UMA buffer: %v", err)
	}

	msg1 := serveStrixHaloPreflightMessage(res1)
	if msg1.Source != "strix-halo" || msg1.Kind != "apu-preflight" || msg1.Level != "info" {
		t.Fatalf("unexpected message metadata: %+v", msg1)
	}
	if !strings.Contains(msg1.Text, "detected=true") || !strings.Contains(msg1.Text, "uma_zero_copy=active") || !strings.Contains(msg1.Text, "mall_tiling_32mb=active") {
		t.Fatalf("unexpected message text: %q", msg1.Text)
	}

	// Case 1b: Environment override via "true"
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "true")
	res1b := preflightServeStrixHalo(nil)
	if !res1b.Detected || res1b.UMAPointerManager == nil || res1b.MALLTiler == nil {
		t.Fatal("expected Strix Halo detected=true via env override=true")
	}

	// Case 2: Environment override explicit disable (0 and false)
	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "0")
	res2 := preflightServeStrixHalo(nil)
	if res2.Detected {
		t.Fatal("expected detected=false when override=0")
	}
	if res2.UMAPointerManager != nil || res2.MALLTiler != nil {
		t.Fatal("expected nil subsystems when detected=false")
	}
	msg2 := serveStrixHaloPreflightMessage(res2)
	if msg2 != (gateway.StartupMessage{}) {
		t.Fatalf("expected empty startup message, got %+v", msg2)
	}

	os.Setenv("FAK_STRIX_GFX1151_OVERRIDE", "false")
	res2b := preflightServeStrixHalo(nil)
	if res2b.Detected || res2b.UMAPointerManager != nil || res2b.MALLTiler != nil {
		t.Fatal("expected detected=false when override=false")
	}

	// Case 3: Mock DRM sysfs detection (vendor 0x1002, device 0x1586)
	os.Unsetenv("FAK_STRIX_GFX1151_OVERRIDE")
	tmpSysfs := t.TempDir()
	card0Dev := filepath.Join(tmpSysfs, "card0", "device")
	if err := os.MkdirAll(card0Dev, 0755); err != nil {
		t.Fatalf("failed to create mock sysfs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Dev, "vendor"), []byte("0x1002\n"), 0644); err != nil {
		t.Fatalf("failed to write mock vendor: %v", err)
	}
	if err := os.WriteFile(filepath.Join(card0Dev, "device"), []byte("0x1586\n"), 0644); err != nil {
		t.Fatalf("failed to write mock device: %v", err)
	}

	res3 := preflightServeStrixHaloWithSysfs(nil, tmpSysfs)
	if !res3.Detected {
		t.Fatal("expected detected=true from mock DRM sysfs (0x1002:0x1586)")
	}
	if res3.UMAPointerManager == nil || res3.MALLTiler == nil {
		t.Fatal("expected active UMAPointerManager and MALLTiler on mock sysfs detection")
	}

	// Case 4: Backend device name detection
	beStrix := newServePreflightBackend("AMD Radeon 8060S Graphics (gfx1151)")
	res4 := preflightServeStrixHalo(beStrix)
	if !res4.Detected {
		t.Fatal("expected detected=true via backend name")
	}
	if res4.DeviceName != "AMD Radeon 8060S Graphics (gfx1151)" {
		t.Fatalf("got device name %q", res4.DeviceName)
	}

	// Case 4b: Vulkan backend with tier containing gfx1151 matches directly without DRM sysfs
	beVulkan := newServePreflightBackendWithTier("vulkan", "integrated:AMD Radeon 8060S Graphics (gfx1151)")
	resVulkan := preflightServeStrixHaloWithSysfs(beVulkan, t.TempDir())
	if !resVulkan.Detected {
		t.Fatal("expected detected=true via Vulkan backend tier without DRM sysfs")
	}
	if resVulkan.DeviceName != "integrated:AMD Radeon 8060S Graphics (gfx1151)" {
		t.Fatalf("got device name %q, want %q", resVulkan.DeviceName, "integrated:AMD Radeon 8060S Graphics (gfx1151)")
	}
	if resVulkan.UMAPointerManager == nil || resVulkan.MALLTiler == nil {
		t.Fatal("expected initialized UMAPointerManager and MALLTiler for Vulkan Strix Halo backend")
	}

	// Case 4c: CPU backend guards against APU acceleration even with mock DRM sysfs present
	var logBuf strings.Builder
	strixHaloPreflightLogWriter = &logBuf
	defer func() { strixHaloPreflightLogWriter = nil }()

	beCPU := newServePreflightBackend("cpu-ref")
	resCPU := preflightServeStrixHaloWithSysfs(beCPU, tmpSysfs)
	if resCPU.Detected {
		t.Fatal("expected detected=false for CPU backend despite DRM sysfs presence")
	}
	if resCPU.UMAPointerManager != nil || resCPU.MALLTiler != nil {
		t.Fatal("expected nil subsystems for CPU backend")
	}
	if !strings.Contains(logBuf.String(), "APU acceleration is skipped for CPU inference") {
		t.Fatalf("expected skip log message, got %q", logBuf.String())
	}
	strixHaloPreflightLogWriter = nil

	// Case 5: Graceful fallback on non-Strix hardware with zero errors and no crash
	beCUDA := newServePreflightBackend("cuda")
	res5 := preflightServeStrixHaloWithSysfs(beCUDA, t.TempDir())
	if res5.Detected {
		t.Fatal("expected detected=false for standard CUDA backend without GFX1151 hardware")
	}
	if res5.UMAPointerManager != nil || res5.MALLTiler != nil {
		t.Fatal("expected nil subsystems for non-Strix hardware")
	}
	msg5 := serveStrixHaloPreflightMessage(res5)
	if msg5.Text != "" {
		t.Fatalf("expected empty message on non-Strix device, got %q", msg5.Text)
	}

	// Sysfs with non-AMD vendor (Intel 0x8086 with device 0x1586)
	tmpIntelSysfs := t.TempDir()
	intelDev := filepath.Join(tmpIntelSysfs, "card0", "device")
	_ = os.MkdirAll(intelDev, 0755)
	_ = os.WriteFile(filepath.Join(intelDev, "vendor"), []byte("0x8086\n"), 0644)
	_ = os.WriteFile(filepath.Join(intelDev, "device"), []byte("0x1586\n"), 0644)
	resIntel := preflightServeStrixHaloWithSysfs(nil, tmpIntelSysfs)
	if resIntel.Detected {
		t.Fatal("expected detected=false for non-AMD vendor 0x8086")
	}

	// Sysfs with AMD vendor (0x1002) but discrete GPU device (0x1100 / Navi 31)
	tmpDiscreteSysfs := t.TempDir()
	discreteDev := filepath.Join(tmpDiscreteSysfs, "card0", "device")
	_ = os.MkdirAll(discreteDev, 0755)
	_ = os.WriteFile(filepath.Join(discreteDev, "vendor"), []byte("0x1002\n"), 0644)
	_ = os.WriteFile(filepath.Join(discreteDev, "device"), []byte("0x1100\n"), 0644)
	resDiscrete := preflightServeStrixHaloWithSysfs(nil, tmpDiscreteSysfs)
	if resDiscrete.Detected {
		t.Fatal("expected detected=false for AMD discrete GPU device 0x1100")
	}

	// Case 6: Concurrent preflight safety across multiple goroutines
	{
		var wg sync.WaitGroup
		concurrency := 16
		errCh := make(chan error, concurrency)
		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				res := preflightServeStrixHaloWithSysfs(beStrix, tmpSysfs)
				if !res.Detected {
					errCh <- fmt.Errorf("goroutine %d: expected detected=true", idx)
					return
				}
				if res.UMAPointerManager == nil || res.MALLTiler == nil {
					errCh <- fmt.Errorf("goroutine %d: subsystems nil", idx)
					return
				}
				if c := res.MALLTiler.CapacityTokens(); c != 8192 {
					errCh <- fmt.Errorf("goroutine %d: capacity tokens=%d, want 8192", idx, c)
					return
				}
				b, err := res.UMAPointerManager.AllocateUMABuffer(512, 64)
				if err != nil {
					errCh <- fmt.Errorf("goroutine %d: alloc error: %w", idx, err)
					return
				}
				if !b.IsZeroCopy() || uintptr(b.HostPtr()) != uintptr(b.DevPtr()) {
					errCh <- fmt.Errorf("goroutine %d: pointer identity mismatch", idx)
					return
				}
				if err := res.UMAPointerManager.FreeUMABuffer(b); err != nil {
					errCh <- fmt.Errorf("goroutine %d: free error: %w", idx, err)
					return
				}
			}(i)
		}
		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Errorf("concurrent preflight error: %v", err)
		}
	}
}
