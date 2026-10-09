package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime medium est=8s lane=default
// Estimate is unmeasured. Fixtures exercise identity/structure only, not a
// compiler, executable SPIR-V, source correspondence, or hardware qualification.
func TestVulkanReceiptV5Current62Boundary(t *testing.T) {
	want := strings.Fields(`matmul matmul_add matmul_argmax matmul_argmax_blocks
matmul2 matmul3 rmsnorm rmsnorm_matmul rmsnorm_matmul2 rmsnorm_matmul3
rmsnorm_matmul_argmax_blocks rope swiglu swiglu_matmul_add add add_bias attention
argmax argmax_pairs q8_matmul q8_matmul2 q8_matmul3 rmsnorm_q8_matmul2 rmsnorm_q8_matmul3
swiglu_q8_matmul_add qwen35_gdn_q8_in_proj qwen35_gdn_conv qwen35_gdn_recurrent
q4k_matmul q4k_matmul_wave32 q4k_matmul_coopmat q6k_matmul q5k_matmul q3k_matmul q2k_matmul
qwen35_split_qg_panel qwen35_partial_rope_panel qwen35_causal_attention_panel sigmoid_mul
q8_matmul_decode glm_kda_recurrent_reread glm_kda_recurrent_wave32 flash_attn_dequant
qwen35_gdn_tiled_transpose coopmat_wave32_wmma rmsnorm_q4k_matmul2 swiglu_q4k_matmul_add
qwen35_gdn_prefill_tiled qwen35_gdn_prefill_norm qwen35_gdn_verify_tiled q2k_matvec
rmsnorm_q8_matmul2_coop iq4xs_matvec iq3xxs_matvec iq2s_matvec iq3s_matvec iq2xxs_matvec
iq2xs_matvec iq1s_matvec v41_tail_rope_qk v41_shared_attention v41_indexer_score`)
	if len(want) != 62 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV5(), want) || !reflect.DeepEqual(VulkanShaders, want) {
		t.Fatal("current builder does not use the independent exact62 registry")
	}
	if !reflect.DeepEqual(historicalVulkanV2Shaders(), want[:59]) || !reflect.DeepEqual(CurrentVulkanShaderRegistry(), want[:60]) || !reflect.DeepEqual(CurrentVulkanShaderRegistryV4(), want[:61]) {
		t.Fatal("historical V2/59, V3/60, or V4/61 membership changed")
	}
	owned := CurrentVulkanShaderRegistryV5()
	owned[0] = "forged"
	if !reflect.DeepEqual(CurrentVulkanShaderRegistryV5(), want) {
		t.Fatal("mutable trusted registry")
	}
	registry, err := currentVulkanRegistryV5Identity()
	if err != nil {
		t.Fatal(err)
	}
	if registry.ID != "fak.vulkan-shader-registry.v5.current62" || registry.ModuleCount != 62 || registry.SHA256 != "0ed07ae3f7bf45e6a65fd01e2062e070677b9929609bdf98422e86ca78b7c18f" {
		t.Fatalf("V5 registry identity differs from independent JSON vector: %+v", registry)
	}
	t.Run("stable framing vector", func(t *testing.T) {
		source := BuildSourceProvenance{GitCommit: strings.Repeat("1", 40), GitTree: strings.Repeat("2", 40), Clean: true, SourceArchiveSHA256: strings.Repeat("3", 64)}
		archive := BuildArtifact{Path: "internal/compute/libfakvulkan.a", SizeBytes: 17, SHA256: strings.Repeat("8", 64)}
		score := VulkanIndexerScoreContract{PolicyID: "fak.v41-indexer-score-f32-structure.v1", ModuleSHA256: strings.Repeat("9", 64), EntryPoint: "main", LocalSize: [3]uint32{1, 1, 1}, FloatWidth: 32, DenormPreserve: true, SignedZeroInfNanPreserve: true, RoundingModeRTE: true, NoContractionFAddCount: 2, NoContractionFMulCount: 2}
		got, err := vulkanStableIdentityV5SHA(source, strings.Repeat("4", 64), 62, strings.Repeat("5", 64), strings.Repeat("6", 64), strings.Repeat("7", 64), archive, score)
		if err != nil || got != "306c858944042cd77f08a1af65d7a3d20bd8c28ff4f3aa587fd273b6379b00fc" {
			t.Fatalf("V5 framing = %s, %v", got, err)
		}
	})

	ctx := context.Background()
	fixture := newVulkanVerifierFixture(t)
	if _, err := verifyVulkanFixture(ctx, fixture, fixture.evidence); err != nil {
		t.Fatalf("historical V2 control: %v", err)
	}
	evidence := fixture.evidence
	for _, stem := range want[59:61] {
		writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stem+".spv"), "inert historical shader\n")
	}
	writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv"), string(vulkanV5StructuralFixture()))
	archivePath := filepath.Join(evidence.SourceRoot, "internal", "compute", "libfakvulkan.a")
	writeFixtureFile(t, archivePath, "inert native archive identity\n")
	runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
	runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", "V5 exact62 inert fixture")
	evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
	evidence.ExpectedSPIRVModules = append([]string(nil), want...)
	evidence.ReceiptPath = filepath.Join(fixture.root, "v5-receipt.json")
	observed, err := observeVulkanReceiptEvidenceV5(ctx, evidence, fixture.gitRun)
	if err != nil {
		t.Fatal(err)
	}
	receipt := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV5, Backend: "vulkan", Command: "binary", Outcome: "success", ReceiptPath: evidence.ReceiptPath,
		Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
		Vulkan:         &VulkanBuildProvenance{Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount, Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256, NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256, StableIdentitySHA256: observed.StableIdentitySHA256},
		VulkanRegistry: &registry, VulkanNativeArchive: &observed.NativeArchive, VulkanIndexerScore: &observed.IndexerScore, Reproducibility: &BuildReproducibility{Status: "baseline"},
	}
	raw := writeVerifierReceipt(t, evidence.ReceiptPath, receipt)
	verify := func(e VulkanBinaryReceiptEvidence) error {
		_, err := verifyVulkanBinaryReceiptV5IdentityWithRunners(ctx, e, fixture.gitRun, fixture.gitRun)
		return err
	}
	verified, err := verifyVulkanBinaryReceiptV5IdentityWithRunners(ctx, evidence, fixture.gitRun, nil)
	if err != nil {
		t.Fatal(err)
	}
	if verified.HistoricalBuildCausality != "unavailable" || len(verified.UnavailableClaims) != len(unavailableVulkanV2Causality)+5 {
		t.Fatal("static proof overstated qualification")
	}
	clone := func(t *testing.T) ComputeBuildReceipt {
		t.Helper()
		r, err := decodeStrictVulkanReceiptV5(raw)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	repair := func(t *testing.T, r *ComputeBuildReceipt) {
		t.Helper()
		forgeVerifierReceiptDownstream(t, r, evidence)
		r.Vulkan.StableIdentitySHA256, err = vulkanStableIdentityV5SHA(r.Vulkan.Source, r.Vulkan.SPIRVBundleSHA256, r.Vulkan.SPIRVModuleCount, r.Vulkan.ToolchainSHA256, r.Vulkan.BuildCommandSHA256, r.Artifact.SHA256, *r.VulkanNativeArchive, *r.VulkanIndexerScore)
		if err != nil {
			t.Fatal(err)
		}
	}
	reset := func(t *testing.T) {
		t.Helper()
		if err := os.WriteFile(evidence.ReceiptPath, raw, 0644); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("strict wire and historical isolation", func(t *testing.T) {
		for _, decoder := range []func([]byte) (ComputeBuildReceipt, error){decodeStrictVulkanReceipt, decodeStrictVulkanReceiptV3, decodeStrictVulkanReceiptV4} {
			if _, err := decoder(raw); err == nil {
				t.Fatal("historical decoder admitted V5-only fields")
			}
		}
		for _, schema := range []string{VulkanBuildReceiptSchema, VulkanBuildReceiptSchemaV3, VulkanBuildReceiptSchemaV4} {
			r := clone(t)
			r.Schema = schema
			old, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(old, []byte("native_archive")) || bytes.Contains(old, []byte("indexer_score_contract")) {
				t.Fatal("V5 extras leaked into historical wire")
			}
			if _, err := decodeStrictVulkanReceiptV5(old); err == nil {
				t.Fatal("V5 decoder admitted historical wire")
			}
			if CompareReceiptProvenance(&r, &receipt) == nil || CompareReceiptProvenance(&receipt, &r) == nil {
				t.Fatal("cross-schema identity accepted")
			}
		}
		for _, invalid := range [][]byte{append(append([]byte(nil), raw...), []byte("{}")...), bytes.Replace(raw, []byte(`"indexer_score_contract": {`), []byte(`"indexer_score_contract": {"forged":true,`), 1)} {
			if bytes.Equal(invalid, raw) {
				t.Fatal("invalid wire fixture unchanged")
			}
			if _, err := decodeStrictVulkanReceiptV5(invalid); err == nil {
				t.Fatal("unknown nested field or trailing JSON accepted")
			}
		}
		wrapped, err := json.Marshal(VulkanBinaryReceiptIdentityVerification{Receipt: receipt})
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(wrapped, []byte(`"native_archive":`)) || !bytes.Contains(wrapped, []byte(`"indexer_score_contract":`)) {
			t.Fatal("nested serialization lost V5 identities")
		}
	})
	for _, tc := range []struct {
		name string
		edit func(*ComputeBuildReceipt)
	}{
		{"archive", func(r *ComputeBuildReceipt) { r.VulkanNativeArchive.SHA256 = strings.Repeat("0", 64) }},
		{"score module", func(r *ComputeBuildReceipt) { r.VulkanIndexerScore.ModuleSHA256 = strings.Repeat("0", 64) }},
		{"source", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitTree = strings.Repeat("0", 40) }},
		{"compiler", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[4].SHA256 = strings.Repeat("0", 64) }},
		{"bundle", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64) }},
		{"binary", func(r *ComputeBuildReceipt) { r.Artifact.SHA256 = strings.Repeat("0", 64) }},
	} {
		t.Run("self-consistent forged "+tc.name, func(t *testing.T) {
			r := clone(t)
			tc.edit(&r)
			repair(t, &r)
			if err := validateVulkanReceiptV5Success(&r); err != nil {
				t.Fatalf("forged envelope must be self-consistent: %v", err)
			}
			if CompareReceiptProvenance(&receipt, &r) == nil {
				t.Fatal("changed identity compared equal")
			}
			writeVerifierReceipt(t, evidence.ReceiptPath, r)
			defer reset(t)
			if err := verify(evidence); err == nil {
				t.Fatal("forged receipt authenticated")
			}
		})
	}
	t.Run("invalid extras and failure cleanup", func(t *testing.T) {
		for _, edit := range []func(*ComputeBuildReceipt){
			func(r *ComputeBuildReceipt) { r.VulkanNativeArchive = nil },
			func(r *ComputeBuildReceipt) {
				r.VulkanNativeArchive.Path = "internal/compute/libfakvulkan_v41_indexer_witness.a"
			},
			func(r *ComputeBuildReceipt) {
				r.Vulkan.NormalizedBuildCommand = append(r.Vulkan.NormalizedBuildCommand, "cxx|-DFAK_V41_INDEXER_SCORE_WITNESS")
			},
			func(r *ComputeBuildReceipt) {
				r.Vulkan.NormalizedBuildCommand = append(r.Vulkan.NormalizedBuildCommand, "go|build|-tags|vulkan,v41_indexer_witness")
			}, func(r *ComputeBuildReceipt) { r.VulkanNativeArchive.SizeBytes = 0 }, func(r *ComputeBuildReceipt) { r.VulkanNativeArchive.Signed = true }, func(r *ComputeBuildReceipt) { r.VulkanIndexerScore = nil }, func(r *ComputeBuildReceipt) { r.VulkanIndexerScore.DenormPreserve = false }, func(r *ComputeBuildReceipt) { r.VulkanIndexerScore.NoContractionFAddCount = 0 },
		} {
			r := clone(t)
			edit(&r)
			if CompareReceiptProvenance(&r, &r) == nil {
				t.Fatal("invalid extras accepted")
			}
		}
		r := clone(t)
		tracker := receiptTracker{receipt: &r}
		tracker.fail(context.Canceled, 1)
		if r.VulkanNativeArchive != nil || r.VulkanIndexerScore != nil || r.VulkanRegistry != nil || r.Vulkan != nil || r.Artifact != nil {
			t.Fatal("failed receipt retained V5 success evidence")
		}
	})
	t.Run("exact census", func(t *testing.T) {
		for _, stems := range [][]string{want[:61], append(append([]string(nil), want...), "extra"), append(append([]string(nil), want[:61]...), want[0])} {
			if validateCurrentVulkanRegistryV5(stems) == nil {
				t.Fatal("caller census replaced exact62 policy")
			}
		}
		path := filepath.Join(evidence.SPIRVRoot, "v41_indexer_score.spv")
		original, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		defer os.WriteFile(path, original, 0644)
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := hashCurrentSPIRVBundleV5(evidence.SPIRVRoot); err == nil {
			t.Fatal("absent score module accepted")
		}
		if err := os.WriteFile(path, []byte("stale non-SPIR-V shader"), 0644); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := hashCurrentSPIRVBundleV5(evidence.SPIRVRoot); err == nil {
			t.Fatal("stale score module accepted")
		}
	})
	t.Run("authenticated prior", func(t *testing.T) {
		prior, current := clone(t), clone(t)
		prior.ReceiptPath = filepath.Join(fixture.root, "v5-prior.json")
		current.ReceiptPath = filepath.Join(fixture.root, "v5-current.json")
		priorEvidence, currentEvidence := evidence, evidence
		priorEvidence.ReceiptPath = prior.ReceiptPath
		currentEvidence.ReceiptPath = current.ReceiptPath
		currentEvidence.Comparison = &priorEvidence
		writePair := func() {
			previous := writeVerifierReceipt(t, prior.ReceiptPath, prior)
			digest := sha256.Sum256(previous)
			current.Reproducibility = &BuildReproducibility{Status: "match", ComparedReceiptSHA256: hex.EncodeToString(digest[:])}
			writeVerifierReceipt(t, current.ReceiptPath, current)
		}
		writePair()
		if err := verify(currentEvidence); err != nil {
			t.Fatalf("prior acceptance control: %v", err)
		}
		prior.VulkanNativeArchive.SHA256 = strings.Repeat("0", 64)
		repair(t, &prior)
		writePair()
		if err := verify(currentEvidence); err == nil {
			t.Fatal("self-consistent forged authenticated prior admitted")
		}
	})
	t.Run("before after archive replacement", func(t *testing.T) {
		// A deterministic runner seam mutates the tracked build artifact after the
		// second source snapshot, so source dirtiness cannot mask reobservation.
		original, err := os.ReadFile(archivePath)
		if err != nil {
			t.Fatal(err)
		}
		defer os.WriteFile(archivePath, original, 0644)
		archives := 0
		race := func(ctx context.Context, root string, args ...string) ([]byte, error) {
			out, err := fixture.gitRun(ctx, root, args...)
			// Change only after the second source observation has completed all
			// its Git commands, just before native-archive reobservation.
			if err == nil && len(args) > 0 && args[0] == "archive" {
				archives++
			}
			if err == nil && archives == 2 && len(args) > 0 && args[0] == "status" {
				writeFixtureFile(t, archivePath, "replaced native archive\n")
			}
			return out, err
		}
		_, err = verifyVulkanBinaryReceiptV5IdentityWithRunners(ctx, evidence, race, nil)
		if archives != 2 || err == nil || !strings.Contains(err.Error(), "changed during verification") {
			t.Fatalf("archive revalidation: count=%d error=%v", archives, err)
		}
	})
}
