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
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// fak-test:runtime medium est=12s lane=default
// The estimate is unmeasured. This bounded corpus uses inert identity fixtures,
// not shader compilation, Vulkan execution, or evidence of historical causality.
func TestVulkanReceiptV4Current61Boundary(t *testing.T) {
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
iq2xs_matvec iq1s_matvec v41_tail_rope_qk v41_shared_attention`)
	t.Run("exact names and copy isolation", func(t *testing.T) {
		if len(want) != 61 || !reflect.DeepEqual(CurrentVulkanShaderRegistryV4(), want) {
			t.Fatal("historical V4 registry differs from the independent exact61 names")
		}
		if !reflect.DeepEqual(historicalVulkanV2Shaders(), want[:59]) || !reflect.DeepEqual(CurrentVulkanShaderRegistry(), want[:60]) {
			t.Fatal("historical V2/59 or V3/60 membership changed")
		}
		raw, err := os.ReadFile(filepath.Join("..", "compute", "build_vulkan.ps1"))
		if err != nil {
			t.Fatal(err)
		}
		parts := strings.SplitN(string(raw), "function Build-Shaders {", 2)
		if len(parts) != 2 {
			t.Fatal("Windows Build-Shaders registry not found")
		}
		block := strings.SplitN(parts[1], "foreach ($s in $shaders)", 2)[0]
		var windows []string
		for _, match := range regexp.MustCompile(`"([a-z0-9_]+)"`).FindAllStringSubmatch(block, -1) {
			windows = append(windows, match[1])
		}
		// Windows retains its own historical build order; membership is exact.
		sortedWant := append(append([]string(nil), want...), "v41_indexer_score")
		sort.Strings(sortedWant)
		sort.Strings(windows)
		if !reflect.DeepEqual(windows, sortedWant) {
			t.Fatalf("Windows current shader names = %v, want current exact62 names", windows)
		}
		copy := CurrentVulkanShaderRegistryV4()
		copy[0] = "forged"
		old := VulkanShaders[0]
		VulkanShaders[0] = "forged"
		defer func() { VulkanShaders[0] = old }()
		if !reflect.DeepEqual(CurrentVulkanShaderRegistryV4(), want) || !reflect.DeepEqual(CurrentVulkanShaderRegistry(), want[:60]) {
			t.Fatal("a caller or compatibility snapshot mutated trusted registry storage")
		}
	})
	t.Run("frozen registry and stable digest vectors", func(t *testing.T) {
		v3, err := currentVulkanRegistryIdentity()
		if err != nil {
			t.Fatal(err)
		}
		v4, err := currentVulkanRegistryV4Identity()
		if err != nil {
			t.Fatal(err)
		}
		if v3.ID != "fak.vulkan-shader-registry.v3.current60" || v3.ModuleCount != 60 || v3.SHA256 != "3f1453b85d7428c27c09bbc1edb300f1cc15a339b31ffa72be160e5aae1ceab6" {
			t.Fatalf("historical V3 registry identity changed: %+v", v3)
		}
		if v4.ID != "fak.vulkan-shader-registry.v4.current61" || v4.ModuleCount != 61 || v4.SHA256 != "ce11e837237012d3175dad8bb97bdd5ff5cf13d64cac478d6c8ea23fe06aec4a" {
			t.Fatalf("V4 registry identity differs from the independent sorted-name JSON digest: %+v", v4)
		}
		// Fixed ASCII JSON vectors pin historical framing independently of the
		// production observer and prevent a V4 cutover from relabeling old hashes.
		source := BuildSourceProvenance{GitCommit: strings.Repeat("1", 40), GitTree: strings.Repeat("2", 40), Clean: true, SourceArchiveSHA256: strings.Repeat("3", 64)}
		for _, tc := range []struct {
			count int
			hash  func(BuildSourceProvenance, string, int, string, string, string) (string, error)
			want  string
		}{
			{59, vulkanStableIdentitySHA, "f246ec010ea809d6348299dd95f2c9690a4566bc87310f2ac68e56a34c1fe6b4"},
			{60, vulkanStableIdentityV3SHA, "2269c247238da552aa614a5c1db9227867e9b9dbfa9729dcbefe194e191f6476"},
			{61, vulkanStableIdentityV4SHA, "dad8de2d1e7314a51af0e0acf46e1dd4f3adaf6b9430a51c21f85b18f96a9435"},
		} {
			got, err := tc.hash(source, strings.Repeat("4", 64), tc.count, strings.Repeat("5", 64), strings.Repeat("6", 64), strings.Repeat("7", 64))
			if err != nil || got != tc.want {
				t.Fatalf("%d-module stable vector = %q, %v; want %s", tc.count, got, err, tc.want)
			}
		}
	})

	ctx := context.Background()
	fixture := newVulkanVerifierFixture(t)
	if _, err := verifyVulkanFixture(ctx, fixture, fixture.evidence); err != nil {
		t.Fatalf("independent historical V2/59 fixture: %v", err)
	}
	versions := []ComputeBuildReceipt{fixture.receipt}
	evidence := fixture.evidence
	// Extend this test's private fixture only. Historical constructors and
	// historical verifiers retain their own frozen membership and wire shapes.
	for _, stage := range []struct {
		stem, schema string
		stems        []string
		registry     func() (VulkanShaderRegistryIdentity, error)
		observe      func(context.Context, VulkanBinaryReceiptEvidence, vulkanGitRunner) (observedVulkanReceiptIdentity, error)
		verify       func(context.Context, VulkanBinaryReceiptEvidence, vulkanGitRunner, vulkanGitRunner) (*VulkanBinaryReceiptIdentityVerification, error)
	}{
		{"v41_tail_rope_qk", VulkanBuildReceiptSchemaV3, want[:60], currentVulkanRegistryIdentity, observeVulkanReceiptEvidenceV3, verifyVulkanBinaryReceiptV3IdentityWithRunners},
		{"v41_shared_attention", VulkanBuildReceiptSchemaV4, want, currentVulkanRegistryV4Identity, observeVulkanReceiptEvidenceV4, verifyVulkanBinaryReceiptV4IdentityWithRunners},
	} {
		writeFixtureFile(t, filepath.Join(evidence.SPIRVRoot, stage.stem+".spv"), "inert identity fixture:"+stage.stem+"\n")
		runFixtureGit(t, evidence.SourceRoot, "add", "-f", ".")
		runFixtureGit(t, evidence.SourceRoot, "commit", "-q", "-m", stage.stem+" fixture")
		evidence.ExpectedCommit = strings.TrimSpace(runFixtureGit(t, evidence.SourceRoot, "rev-parse", "HEAD"))
		evidence.ExpectedSPIRVModules = append([]string(nil), stage.stems...)
		evidence.ReceiptPath = filepath.Join(fixture.root, stage.stem+"-receipt.json")
		observed, err := stage.observe(ctx, evidence, fixture.gitRun)
		if err != nil {
			t.Fatal(err)
		}
		registry, err := stage.registry()
		if err != nil {
			t.Fatal(err)
		}
		r := ComputeBuildReceipt{
			Schema: stage.schema, Backend: "vulkan", Command: "binary", Outcome: "success",
			ReceiptPath: evidence.ReceiptPath, Phases: []ComputeBuildPhase{}, Artifact: &observed.Artifact,
			Vulkan: &VulkanBuildProvenance{
				Source: observed.Source, SPIRVBundleSHA256: observed.SPIRVBundleSHA256, SPIRVModuleCount: observed.SPIRVModuleCount,
				Toolchain: observed.Toolchain, ToolchainSHA256: observed.ToolchainSHA256,
				NormalizedBuildCommand: observed.NormalizedBuildCommand, BuildCommandSHA256: observed.BuildCommandSHA256,
				StableIdentitySHA256: observed.StableIdentitySHA256,
			},
			VulkanRegistry: &registry, Reproducibility: &BuildReproducibility{Status: "baseline"},
		}
		if err := WriteReceiptAtomic(evidence.ReceiptPath, &r); err != nil {
			t.Fatal(err)
		}
		if runtime.GOOS == "linux" {
			_, err = VerifyVulkanBinaryReceiptIdentityForSchema(ctx, stage.schema, evidence)
		} else {
			_, err = stage.verify(ctx, evidence, fixture.gitRun, fixture.gitRun)
		}
		if err != nil {
			t.Fatalf("%s independently observed acceptance: %v", stage.schema, err)
		}
		versions = append(versions, r)
	}
	receipt := versions[2]
	raw, err := os.ReadFile(evidence.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	clone := func(t *testing.T) ComputeBuildReceipt {
		t.Helper()
		r, err := decodeStrictVulkanReceiptV4(raw)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	verify := func(e VulkanBinaryReceiptEvidence) error {
		_, err := verifyVulkanBinaryReceiptV4IdentityWithRunners(ctx, e, fixture.gitRun, fixture.gitRun)
		return err
	}
	reset := func(t *testing.T) {
		t.Helper()
		if err := os.WriteFile(evidence.ReceiptPath, raw, 0644); err != nil {
			t.Fatal(err)
		}
	}
	repairClaims := func(t *testing.T, r *ComputeBuildReceipt) {
		t.Helper()
		forgeVerifierReceiptDownstream(t, r, evidence)
		r.Vulkan.StableIdentitySHA256, err = vulkanStableIdentityV4SHA(r.Vulkan.Source,
			r.Vulkan.SPIRVBundleSHA256, r.Vulkan.SPIRVModuleCount,
			r.Vulkan.ToolchainSHA256, r.Vulkan.BuildCommandSHA256, r.Artifact.SHA256)
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Run("generic nested and historical wire", func(t *testing.T) {
		for _, r := range versions {
			var wire any = computeBuildReceiptV2Wire(r)
			if r.Schema == VulkanBuildReceiptSchemaV3 {
				wire = vulkanBuildReceiptV3Wire{computeBuildReceiptV2Wire: computeBuildReceiptV2Wire(r), ShaderRegistry: r.VulkanRegistry}
			} else if r.Schema == VulkanBuildReceiptSchemaV4 {
				wire = vulkanBuildReceiptV4Wire{computeBuildReceiptV2Wire: computeBuildReceiptV2Wire(r), ShaderRegistry: r.VulkanRegistry}
			}
			wantRaw, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			for _, value := range []any{r, &r, VulkanBinaryReceiptIdentityVerification{Receipt: r}} {
				got, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				if _, nested := value.(VulkanBinaryReceiptIdentityVerification); nested {
					var envelope struct {
						Receipt json.RawMessage `json:"receipt"`
					}
					if err := json.Unmarshal(got, &envelope); err != nil {
						t.Fatal(err)
					}
					got = envelope.Receipt
				}
				if !bytes.Equal(got, wantRaw) {
					t.Fatalf("%s generic/nested serialization changed frozen wire", r.Schema)
				}
			}
		}
	})
	t.Run("strict decoder and cross schema refusal", func(t *testing.T) {
		unknown := bytes.TrimSpace(append([]byte(nil), raw...))
		unknown = append(unknown[:len(unknown)-1], []byte(`,"modules":["forged"]}`)...)
		nested := bytes.Replace(raw, []byte(`"shader_registry": {`), []byte(`"shader_registry": {"modules":[],`), 1)
		if bytes.Equal(nested, raw) {
			t.Fatal("strict nested-field fixture did not change the wire")
		}
		for _, invalid := range [][]byte{unknown, nested, append(append([]byte(nil), raw...), []byte("{}\n")...)} {
			if _, err := decodeStrictVulkanReceiptV4(invalid); err == nil {
				t.Fatal("V4 decoder accepted unknown fields or trailing JSON")
			}
		}
		if _, err := decodeStrictVulkanReceipt(raw); err == nil {
			t.Fatal("V2 decoder accepted V4")
		}
		if _, err := decodeStrictVulkanReceiptV3(raw); err == nil {
			t.Fatal("V3 decoder accepted V4")
		}
		for _, old := range versions[:2] {
			oldRaw, err := os.ReadFile(old.ReceiptPath)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := decodeStrictVulkanReceiptV4(oldRaw); err == nil {
				t.Fatalf("V4 decoder accepted %s", old.Schema)
			}
			if _, err := compareVulkanBuildReceiptV4(old.ReceiptPath, receipt.Artifact, receipt.Vulkan, receipt.VulkanRegistry); err == nil {
				t.Fatalf("V4 builder compared equal to %s", old.Schema)
			}
			forged := clone(t)
			forged.Schema = old.Schema // Keep the copied V4 stable digest deliberately.
			if CompareReceiptProvenance(&receipt, &forged) == nil || CompareReceiptProvenance(&forged, &receipt) == nil {
				t.Fatalf("cross-schema equality accepted copied V4 hashes under %s", old.Schema)
			}
		}
		if _, err := VerifyVulkanBinaryReceiptIdentityForSchema(ctx, "forged-schema", evidence); err == nil {
			t.Fatal("untrusted schema selected verification policy")
		}
	})
	t.Run("complete success admission", func(t *testing.T) {
		if err := CompareReceiptProvenance(&receipt, &receipt); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name string
			edit func(*ComputeBuildReceipt)
		}{
			{"missing provenance", func(r *ComputeBuildReceipt) { r.Vulkan = nil }},
			{"missing registry", func(r *ComputeBuildReceipt) { r.VulkanRegistry = nil }},
			{"wrong registry", func(r *ComputeBuildReceipt) { r.VulkanRegistry.ID += ".forged" }},
			{"wrong registry digest", func(r *ComputeBuildReceipt) { r.VulkanRegistry.SHA256 = strings.Repeat("0", 64) }},
			{"wrong registry count", func(r *ComputeBuildReceipt) { r.VulkanRegistry.ModuleCount-- }},
			{"wrong module count", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVModuleCount-- }},
			{"malformed source", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitCommit = "not-a-git-id" }},
			{"malformed bundle digest", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVBundleSHA256 = "not-a-digest" }},
			{"stale stable digest", func(r *ComputeBuildReceipt) { r.Vulkan.StableIdentitySHA256 = strings.Repeat("0", 64) }},
			{"missing binary", func(r *ComputeBuildReceipt) { r.Artifact = nil }},
			{"empty binary", func(r *ComputeBuildReceipt) { r.Artifact.SizeBytes = 0 }},
			{"signed binary", func(r *ComputeBuildReceipt) { r.Artifact.Signed = true }},
			{"failed outcome", func(r *ComputeBuildReceipt) { r.Outcome = "failed" }},
			{"nonzero exit", func(r *ComputeBuildReceipt) { r.ExitCode = 1 }},
			{"error text", func(r *ComputeBuildReceipt) { r.Error = "failed" }},
			{"wrong backend", func(r *ComputeBuildReceipt) { r.Backend = "cuda" }},
			{"wrong command", func(r *ComputeBuildReceipt) { r.Command = "shaders" }},
			{"ambiguous legacy source", func(r *ComputeBuildReceipt) { r.GitCommit = evidence.ExpectedCommit }},
		} {
			t.Run(tc.name, func(t *testing.T) {
				invalid := clone(t)
				tc.edit(&invalid)
				if err := CompareReceiptProvenance(&invalid, &invalid); err == nil {
					t.Fatal("identical copied hashes concealed an invalid success envelope")
				}
				writeVerifierReceipt(t, evidence.ReceiptPath, invalid)
				defer reset(t)
				if _, err := compareVulkanBuildReceiptV4(evidence.ReceiptPath, receipt.Artifact, receipt.Vulkan, receipt.VulkanRegistry); err == nil {
					t.Fatal("builder comparison accepted invalid prior success envelope")
				}
			})
		}
	})
	for _, tc := range []struct {
		name string
		edit func(*ComputeBuildReceipt)
	}{
		{"source", func(r *ComputeBuildReceipt) { r.Vulkan.Source.GitTree = strings.Repeat("0", 40) }},
		{"tool", func(r *ComputeBuildReceipt) { r.Vulkan.Toolchain[0].SHA256 = strings.Repeat("0", 64) }},
		{"command", func(r *ComputeBuildReceipt) { r.Vulkan.NormalizedBuildCommand[0] += "|forged" }},
		{"bundle", func(r *ComputeBuildReceipt) { r.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64) }},
		{"binary", func(r *ComputeBuildReceipt) { r.Artifact.SHA256 = strings.Repeat("0", 64) }},
	} {
		t.Run("forged "+tc.name, func(t *testing.T) {
			forged := clone(t)
			tc.edit(&forged)
			if err := CompareReceiptProvenance(&receipt, &forged); err == nil {
				t.Fatal("copied digest concealed changed declared identity")
			}
			// Recompute attacker-controlled claims as well; trusted observation must
			// reject this self-consistent envelope rather than only stale hashes.
			repairClaims(t, &forged)
			writeVerifierReceipt(t, evidence.ReceiptPath, forged)
			defer reset(t)
			if err := verify(evidence); err == nil {
				t.Fatal("independent verification accepted self-consistent forged identity")
			}
		})
	}
	t.Run("authenticated prior comparison", func(t *testing.T) {
		prior, current := clone(t), clone(t)
		prior.ReceiptPath = filepath.Join(fixture.root, "v4-prior.json")
		current.ReceiptPath = filepath.Join(fixture.root, "v4-compared.json")
		priorEvidence, comparedEvidence := evidence, evidence
		priorEvidence.ReceiptPath = prior.ReceiptPath
		comparedEvidence.ReceiptPath = current.ReceiptPath
		comparedEvidence.Comparison = &priorEvidence
		writePair := func() {
			priorRaw := writeVerifierReceipt(t, prior.ReceiptPath, prior)
			digest := sha256.Sum256(priorRaw)
			current.Reproducibility = &BuildReproducibility{Status: "match", ComparedReceiptSHA256: hex.EncodeToString(digest[:])}
			writeVerifierReceipt(t, current.ReceiptPath, current)
		}
		writePair()
		if err := verify(comparedEvidence); err != nil {
			t.Fatalf("authenticated V4 comparison control: %v", err)
		}
		prior.Vulkan.SPIRVBundleSHA256 = strings.Repeat("0", 64)
		repairClaims(t, &prior)
		writePair() // Even the current receipt now names the exact forged prior bytes.
		if err := verify(comparedEvidence); err == nil {
			t.Fatal("authenticated comparison accepted a self-consistent forged prior")
		}
	})
	t.Run("trusted census", func(t *testing.T) {
		for _, edit := range []func([]string) []string{
			func(s []string) []string { return s[:60] },
			func(s []string) []string { return append(s, "forged") },
			func(s []string) []string { s[60] = s[0]; return s },
			func(s []string) []string { s[60] = "forged"; return s },
		} {
			changed := evidence
			changed.ExpectedSPIRVModules = edit(CurrentVulkanShaderRegistryV4())
			if _, err := observeVulkanReceiptEvidenceV4(ctx, changed, fixture.gitRun); err == nil {
				t.Fatal("caller-defined census replaced trusted current61 policy")
			}
		}
		for _, mode := range []string{"missing", "extra", "symlink"} {
			t.Run(mode, func(t *testing.T) {
				path := filepath.Join(evidence.SPIRVRoot, "v41_shared_attention.spv")
				if mode == "extra" {
					path = filepath.Join(evidence.SPIRVRoot, "forged.spv")
					writeFixtureFile(t, path, "inert extra fixture\n")
					defer os.Remove(path)
				} else {
					if err := os.Rename(path, path+".parked"); err != nil {
						t.Fatal(err)
					}
					defer os.Rename(path+".parked", path)
					if mode == "symlink" {
						if err := os.Symlink(path+".parked", path); err != nil {
							t.Skipf("symlink creation unavailable: %v", err)
						}
						defer os.Remove(path)
					}
				}
				if _, _, err := hashCurrentSPIRVBundleV4(evidence.SPIRVRoot); err == nil {
					t.Fatal("invalid filesystem census accepted")
				}
			})
		}
	})
	t.Run("revalidation catches evidence replacement", func(t *testing.T) {
		// Keep this tool outside the source tree so Git cleanliness cannot mask
		// the before/after tool-identity check. The runner seam makes the change
		// deterministic without sleeps, goroutines, or a production race hook.
		tool := filepath.Join(fixture.root, "external-tools", "inert-tool")
		original, err := os.ReadFile(evidence.Tools.Go)
		if err != nil {
			t.Fatal(err)
		}
		writeFixtureFile(t, tool, string(original))
		changed := evidence
		changed.Tools.Go, changed.Tools.CC, changed.Tools.CXX, changed.Tools.AR, changed.Tools.GLSLC = tool, tool, tool, tool, tool
		if err := verify(changed); err != nil {
			t.Fatalf("relocated same-name same-byte tool control: %v", err)
		}
		archives := 0
		race := func(ctx context.Context, root string, args ...string) ([]byte, error) {
			out, err := fixture.gitRun(ctx, root, args...)
			if err == nil && len(args) != 0 && args[0] == "archive" {
				archives++
				if archives == 2 {
					writeFixtureFile(t, tool, "replaced between observations\n")
				}
			}
			return out, err
		}
		_, err = verifyVulkanBinaryReceiptV4IdentityWithRunners(ctx, changed, race, nil)
		if archives != 2 || err == nil || !strings.Contains(err.Error(), "changed during verification") {
			t.Fatalf("deterministic evidence race: archives=%d, error=%v", archives, err)
		}
	})
}
