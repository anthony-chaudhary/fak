package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestServeQ4KWeightAdmissionUsesTransformedPlan(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	path := writeServePackedEmbeddingFixture(t, "mixed-qwen.gguf", false)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	arm := resolveMetalServeLoadArm(ws)
	if arm != serveLoadArmResidentQ4K {
		t.Fatalf("actual header selected %q, want resident-Q4K", arm)
	}
	m, err := ggufload.LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	// Three 256x256 Q4_K projections, a three-row Q6_K head, and
	// an embedding expanded to F32 by the loader's default options.
	const want = 3*256*144 + 3*210 + 3*256*4
	if got := m.ResidentReport().TotalResidentBytes; got != want {
		t.Fatalf("actual stored weights=%d, independent count=%d", got, want)
	}
	plan, err := serveGGUFWeightMemoryPlanForArm(ws, arm)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Total() != want {
		t.Errorf("central weight admission=%d, actual transformed storage=%d", plan.Total(), want)
	}
	raw, err := ws.EstimateLoadBytes()
	if err != nil || raw >= want {
		t.Fatalf("fixture must distinguish raw payload from storage: raw=%d stored=%d error=%v", raw, want, err)
	}

	// Streaming is outside the estimator's qualified transformed-storage route.
	// Per the #11962 contract it retains the prior conservative raw-payload
	// estimate rather than an exactness claim, so the live streamed Metal serve
	// keeps working instead of failing the pre-load fit.
	t.Setenv("FAK_STREAM_Q4K", "1")
	err = fitServeGGUFPathOnReportedHostForArm(path, arm, 16, 0, 0, false, nil)
	if err != nil {
		t.Errorf("streamed resident admission must retain the conservative fallback, got %v", err)
	}
	streamedPlan, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm, serveFitBudget{})...)
	if err != nil {
		t.Fatalf("streamed conservative fallback: %v", err)
	}
	if streamedPlan.Total() != raw {
		t.Errorf("streamed fallback total=%d, want the prior raw payload estimate %d", streamedPlan.Total(), raw)
	}
}

func TestServeResidentQ4KSharedOptionsMatchStoredWeights(t *testing.T) {
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return true }
	t.Cleanup(func() { serveMetalAvailable = original })
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	for _, tc := range []struct {
		name   string
		q2     bool
		device bool
		want   int64
	}{
		{"packed-q2", true, false, 2*256*144 + 256*84 + 3*210 + 3*84},
		{"packed-q4", false, false, 3*256*144 + 3*210 + 3*144},
		{"explicit-backend-q8-fallback", true, true, 2*256*144 + 256*256/32*36 + 3*256/32*36 + 3*256*4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var path string
			if tc.q2 {
				path = writeServePackedQ2EmbeddingFixture(t, "Qwen3.8-27B-UD-Q2_K_XL.gguf", true)
			} else {
				path = writeServePackedEmbeddingFixture(t, "Qwen3.8-27B-Q4_K_M.gguf", true)
			}
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ws.Close() })
			var backend compute.Backend
			if tc.device {
				backend = compute.Default()
			}
			artifact := ggufload.ClassifyTensorQuant(ws.File.Tensors)
			loadOpts := serveResidentQ4KLoadOptions(backend, path, true, artifact)
			m, err := ggufload.LoadModelQ4KProfileOptions(path, nil, loadOpts...)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			fitOpts := serveQ4KFitOptions(path, ws, backend, serveLoadArmResidentQ4K, serveFitBudget{})
			plan, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmResidentQ4K, fitOpts...)
			if err != nil {
				t.Fatal(err)
			}
			if got := m.ResidentReport().TotalResidentBytes; got != tc.want || plan.Total() != tc.want {
				t.Fatalf("loader=%d admission=%d independent=%d", got, plan.Total(), tc.want)
			}
		})
	}
}

// The fixture is a physical, loadable storage slice with complete architecture
// metadata and payloads. It does not claim full-model generation qualification.
func TestServeResidentDenseStandardArchitectureAdmissionMatchesLoader(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	for _, arch := range []string{"llama", "qwen2"} {
		t.Run(arch, func(t *testing.T) {
			path := createTestQ4KGGUF(t)
			if arch == "qwen2" {
				b, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				// Both architecture names have five bytes. Replace the architecture and
				// its seven metadata prefixes without changing directory/payload offsets.
				if got := bytes.Count(b, []byte("llama")); got != 8 {
					t.Fatalf("fixture architecture occurrences=%d, want 8", got)
				}
				b = bytes.ReplaceAll(b, []byte("llama"), []byte("qwen2"))
				if err := os.WriteFile(path, b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ws.Close() })
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ModelType != arch || cfg.TieWordEmbeddings || cfg.IsMoE() {
				t.Fatalf("expected untied dense %s configuration, got %+v", arch, cfg)
			}
			arm := resolveMetalServeLoadArm(ws)
			if arm != serveLoadArmResidentQ4K {
				t.Fatalf("selected load arm=%q, want resident Q4_K", arm)
			}
			opts := serveResidentQ4KLoadOptions(nil, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
			m, err := ggufload.LoadModelQ4KProfileOptions(path, nil, opts...)
			if err != nil {
				t.Fatalf("actual supported loader failed before admission comparison: %v", err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			// Three matrices remain Q4_K; rotary-sensitive q_proj is normalized into
			// native Q8 with F32 scales (36 bytes per 32 values, not GGUF's 34).
			const wantStored int64 = 3*256*144 + 256*256/32*36
			report := m.ResidentReport()
			if report.Q4KTensors != 3 || report.Q8Tensors != 1 || !m.HasQ8("model.layers.0.self_attn.q_proj.weight") || report.TotalResidentBytes != wantStored {
				t.Fatalf("actual loader report=%+v, want three Q4/one normalized Q8 and %d bytes", report, wantStored)
			}
			raw, err := ws.EstimateLoadBytes()
			if err != nil || raw != 4*256*144 {
				t.Fatalf("raw bytes=%d error=%v, want147456", raw, err)
			}
			t.Logf("actual %s loader succeeded: raw=%d stored=%d Q4=%d Q8=%d", arch, raw, report.TotalResidentBytes, report.Q4KTensors, report.Q8Tensors)
			plan, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm, serveFitBudget{})...)
			if err != nil {
				t.Fatalf("admission rejected supported %s loader storage (%d bytes): %v", arch, wantStored, err)
			}
			if plan.Total() != wantStored {
				t.Fatalf("admission=%d actual stored=%d", plan.Total(), wantStored)
			}
		})
	}
}

// TestServeRefusalNamesViableOffloadArms is the GOAL-model-fit-tiers-doctrine witness at the
// serve admission seam: a refusal must name the viable offload arm(s) as STRUCTURED data
// (compute.FitError.SuggestedArms, a closed vocabulary) without weakening admission. The fixture
// is the documented physical-strix3 shape (the serveCPUOffloadStreamedResidentMargin comment,
// fak#13140): the full host-resident routed-expert charge misses the host window by a razor-thin
// 48.01-vs-47.99 GiB, and the bounded-resident streamed policy is the arm that admits it.
// Assertions target the structured field, never exact prose.
func TestServeRefusalNamesViableOffloadArms(t *testing.T) {
	const gib = int64(1 << 30)
	var gibF = float64(gib) // variable base keeps the 48.01/47.99 arithmetic out of constant folding
	// The witnessed strix3 arithmetic as runtime floats: plan needs 48.01 GiB, host has 47.99 GiB.
	want := int64(48.01 * gibF)
	avail := int64(47.99 * gibF)

	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 4 * gib, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: want, Detail: servePlanDetailHostExpertOffload},
	}

	// A) The documented fixture: a host-scope FitTooBig on the full host-resident routed set.
	// The structured field carries exactly the one arm whose premise the data proves.
	fe := &compute.FitError{Verdict: compute.FitTooBig, Want: want, Avail: avail, Demands: plan, Scope: compute.MemoryScopeHost}
	if got := serveSuggestedFitArms(fe, plan, nil); len(got) != 1 || got[0] != serveFitSuggestedArmStreamedExperts {
		t.Fatalf("fixture SuggestedArms = %v, want [%s]", got, serveFitSuggestedArmStreamedExperts)
	}
	err := serveAttachSuggestedArms(error(fe), plan, nil)
	var attached *compute.FitError
	if !errors.As(err, &attached) {
		t.Fatalf("attach lost the typed refusal: %T: %v", err, err)
	}
	if attached != fe {
		t.Fatalf("attach must populate the SAME *compute.FitError the funnel returns")
	}
	if len(attached.SuggestedArms) != 1 || attached.SuggestedArms[0] != serveFitSuggestedArmStreamedExperts {
		t.Fatalf("attached SuggestedArms = %v, want [%s]", attached.SuggestedArms, serveFitSuggestedArmStreamedExperts)
	}
	// Human-facing text: the original sentence stays intact; the arm names are appended after it.
	msg := err.Error()
	if !strings.Contains(msg, "(FitTooBig: request exceeds the reported host capacity") {
		t.Fatalf("original refusal sentence missing or altered: %q", msg)
	}
	if !strings.HasSuffix(msg, "; viable offload arm(s): "+serveFitSuggestedArmStreamedExperts) {
		t.Fatalf("arm names not appended after the original sentence: %q", msg)
	}

	// B) The production funnel: refuseIfTooBigOnDevice populates the same field on the typed
	// refusal it returns. A split-aperture integrated backend (distinct device heap beside the
	// system window, the strix3 shape) whose host window cannot hold the full routed set refuses
	// host-scoped, and the named arm is the streamed policy - while the verdict itself is
	// unchanged (still FitTooBig, still the same shortfall).
	be := serveCapBackend{
		Backend: compute.Default(), total: 60 * gib, free: 60 * gib, known: true,
		hostTotal: 56 * gib, hostFree: 56 * gib, hostKnown: true,
		tier: "integrated:test-apu",
	}
	_, funnelErr := refuseIfTooBigOnDevice(plan, nil, be, nil)
	if funnelErr == nil {
		t.Fatal("funnel admitted a plan whose host-resident routed set exceeds the host window")
	}
	var ffe *compute.FitError
	if !errors.As(funnelErr, &ffe) {
		t.Fatalf("funnel returned an untyped refusal: %T: %v", funnelErr, funnelErr)
	}
	if ffe.Verdict != compute.FitTooBig || ffe.Scope != compute.MemoryScopeHost {
		t.Fatalf("funnel verdict changed: %+v", ffe)
	}
	if len(ffe.SuggestedArms) != 1 || ffe.SuggestedArms[0] != serveFitSuggestedArmStreamedExperts {
		t.Fatalf("funnel SuggestedArms = %v, want [%s]", ffe.SuggestedArms, serveFitSuggestedArmStreamedExperts)
	}

	// C) Proven-no-arm cases: the list is EMPTY when no arm's premise is provable.
	// C1: a host-scope refusal whose failing charge is NOT a host-resident routed set (a dense
	// CPU plan) names no arm.
	densePlan := compute.MemoryPlan{{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: want, Detail: "gguf-q8-load"}}
	denseFe := &compute.FitError{Verdict: compute.FitTooBig, Want: want, Avail: avail, Demands: densePlan, Scope: compute.MemoryScopeHost}
	if arms := serveSuggestedFitArms(denseFe, densePlan, nil); len(arms) != 0 {
		t.Fatalf("dense host refusal SuggestedArms = %v, want empty (proven no arm)", arms)
	}
	// C2: a device-scope refusal whose host pool holds NO headroom beyond the plan's own host
	// charge names no arm (the double's window is deterministic: 0.85*56 GiB < 50 GiB charged).
	devPlan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 50 * gib, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: 50 * gib, Detail: "kv-cache"},
	}
	devBe := serveCapBackend{Backend: compute.Default(), total: 40 * gib, free: 40 * gib, known: true, hostTotal: 56 * gib, hostFree: 56 * gib, hostKnown: true}
	devFe := &compute.FitError{Verdict: compute.FitTooBig, Want: 50 * gib, Avail: compute.BudgetAfterHeadroom(40*gib, serveGGUFHostHeadroom), Demands: devPlan, Scope: compute.MemoryScopeDevice}
	if arms := serveSuggestedFitArms(devFe, devPlan, devBe); len(arms) != 0 {
		t.Fatalf("device refusal with no host headroom SuggestedArms = %v, want empty (proven no arm)", arms)
	}
	// C3: a plan that ALREADY carries the bounded streamed row has taken the streamed arm; the
	// refusal names it again nowhere.
	streamedPlan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 4 * gib, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: 40 * gib, Detail: servePlanDetailHostExpertOffloadStreamed},
	}
	streamedFe := &compute.FitError{Verdict: compute.FitTooBig, Want: 40 * gib, Avail: 30 * gib, Demands: streamedPlan, Scope: compute.MemoryScopeHost}
	if arms := serveSuggestedFitArms(streamedFe, streamedPlan, nil); len(arms) != 0 {
		t.Fatalf("already-streamed plan SuggestedArms = %v, want empty (arm in use)", arms)
	}

	// D) The device-scope positive: a device overflow with PROVABLE host headroom beyond the
	// plan's own host charge names host-pool-headroom (deterministic via the double's window).
	devPlan2 := compute.MemoryPlan{{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 50 * gib, Detail: "gguf-device-dense-load"}}
	devFe2 := &compute.FitError{Verdict: compute.FitTooBig, Want: 50 * gib, Avail: compute.BudgetAfterHeadroom(40*gib, serveGGUFHostHeadroom), Demands: devPlan2, Scope: compute.MemoryScopeDevice}
	if arms := serveSuggestedFitArms(devFe2, devPlan2, devBe); len(arms) != 1 || arms[0] != serveFitSuggestedArmHostPoolHeadroom {
		t.Fatalf("device refusal with host headroom SuggestedArms = %v, want [%s]", arms, serveFitSuggestedArmHostPoolHeadroom)
	}

	// E) The reserved token is part of the closed vocabulary but NEVER emitted: no input shapes
	// the streamed-dense route into a suggestion (fak#13175 open - a refusal must not claim it
	// works). Assert the guard on every branch above by scanning the outputs.
	for _, out := range [][]string{
		serveSuggestedFitArms(fe, plan, nil),
		serveSuggestedFitArms(devFe2, devPlan2, devBe),
		serveSuggestedFitArms(denseFe, densePlan, nil),
	} {
		for _, tok := range out {
			if tok == serveFitSuggestedArmStreamedDense {
				t.Fatalf("reserved token %q was emitted; the streamed-dense route is not built (fak#13175)", tok)
			}
		}
	}
}

// writeServePackedQ2EmbeddingFixture writes the smallest dense Qwen hybrid GGUF
// whose token_embd.weight is Q2_K (matching the UD-Q2_K_XL artifact) plus a Q4_K
// tensor so the artifact stays on the resident-Q4_K serve arm. It optionally
// truncates the file to the exact witnessed UD-Q2_K_XL byte size so the
// file-name/size gate admits it, without allocating or reading a 27B checkpoint.
func writeServePackedQ2EmbeddingFixture(t *testing.T, base string, witnessedSize bool) string {
	t.Helper()
	const (
		dim          = 256
		vocab        = 3
		q2BlockBytes = 84
		q4BlockBytes = 144
		q6BlockBytes = 210
	)
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 5, 10)
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", dim)
	writeKVUint32ForTest(&b, "qwen35.block_count", 2)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", dim)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", dim)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)

	offset := uint64(0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ2_K), offset)
	offset = align32(offset + vocab*q2BlockBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ6_K), offset)
	offset = align32(offset + vocab*q6BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + dim*q4BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ2_K), offset)
	offset = align32(offset + dim*q2BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + dim*q4BlockBytes)
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))

	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if witnessedSize {
		if err := os.Truncate(path, qwen38UDQ2KXLArtifactBytes); err != nil {
			t.Fatal(err)
		}
	}
	return path
}
