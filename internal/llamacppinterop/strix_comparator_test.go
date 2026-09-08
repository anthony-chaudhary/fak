package llamacppinterop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func TestStrixComparatorManifest(t *testing.T) {
	valid := validStrixComparatorManifest()
	got := ValidateStrixComparatorPlan(valid)
	if got.Schema != StrixComparatorPlanValidationSchema || got.Outcome != OutcomeAbstain || !got.PlanValid || got.PerformanceCredit {
		t.Fatalf("valid manifest result = %+v", got)
	}
	if len(got.Reasons) != 1 || got.Reasons[0] != "device-free-plan-valid-observation-required" {
		t.Fatalf("valid manifest reasons = %v", got.Reasons)
	}

	tests := []struct {
		name   string
		mutate func(*StrixComparatorManifest)
		reason string
	}{
		{"schema", func(m *StrixComparatorManifest) { m.Schema = "" }, "manifest-schema-mismatch"},
		{"source repository", func(m *StrixComparatorManifest) { m.SourceRepository = "fork/llama.cpp" }, "source-repository-mismatch"},
		{"source revision", func(m *StrixComparatorManifest) { m.SourceRevision = strings.Repeat("0", 40) }, "source-revision-mismatch"},
		{"source tree", func(m *StrixComparatorManifest) { m.SourceTree = strings.Repeat("0", 40) }, "source-tree-mismatch"},
		{"source archive hash missing", func(m *StrixComparatorManifest) { m.SourceArchiveSHA256 = "" }, "source-archive-sha256-invalid"},
		{"source archive hash malformed", func(m *StrixComparatorManifest) { m.SourceArchiveSHA256 = strings.Repeat("z", 64) }, "source-archive-sha256-invalid"},
		{"source archive hash all zero", func(m *StrixComparatorManifest) { m.SourceArchiveSHA256 = strings.Repeat("0", 64) }, "source-archive-sha256-invalid"},
		{"source archive hash uppercase", func(m *StrixComparatorManifest) { m.SourceArchiveSHA256 = strings.Repeat("A", 64) }, "source-archive-sha256-invalid"},
		{"source archive hash whitespace", func(m *StrixComparatorManifest) { m.SourceArchiveSHA256 = " " + strings.Repeat("a", 63) }, "source-archive-sha256-invalid"},
		{"upstream build", func(m *StrixComparatorManifest) { m.UpstreamBuild = "b10589" }, "upstream-build-mismatch"},
		{"build manifest hash", func(m *StrixComparatorManifest) { m.BuildManifestSHA256 = "short" }, "build-manifest-sha256-invalid"},
		{"vulkan flag missing", func(m *StrixComparatorManifest) { m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release"} }, "build-flags-frozen-set-invalid"},
		{"duplicate vulkan flag", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release", "GGML_VULKAN=ON", "GGML_VULKAN=ON"}
		}, "build-flags-frozen-set-invalid"},
		{"conflicting vulkan flag", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release", "GGML_VULKAN=ON", "GGML_VULKAN=OFF"}
		}, "build-flags-frozen-set-invalid"},
		{"hidden conflicting vulkan flag", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release", "GGML_VULKAN=ON", " -Dggml_vulkan=off "}
		}, "build-flags-frozen-set-invalid"},
		{"typed vulkan override", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release", "GGML_VULKAN=ON", "-DGGML_VULKAN:BOOL=OFF"}
		}, "build-flags-frozen-set-invalid"},
		{"alternate vulkan spelling", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"CMAKE_BUILD_TYPE=Release", "-DGGML_VULKAN=ON"}
		}, "build-flags-frozen-set-invalid"},
		{"release flag missing", func(m *StrixComparatorManifest) { m.BuildFlags = []string{"GGML_VULKAN=ON"} }, "build-flags-frozen-set-invalid"},
		{"debug only", func(m *StrixComparatorManifest) { m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Debug"} }, "build-flags-frozen-set-invalid"},
		{"release and debug", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release", "CMAKE_BUILD_TYPE=Debug"}
		}, "build-flags-frozen-set-invalid"},
		{"typed build type override", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release", "-DCMAKE_BUILD_TYPE:STRING=Debug"}
		}, "build-flags-frozen-set-invalid"},
		{"duplicate release", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release", "CMAKE_BUILD_TYPE=Release"}
		}, "build-flags-frozen-set-invalid"},
		{"empty flag", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release", ""}
		}, "build-flags-frozen-set-invalid"},
		{"whitespace flag", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release", " "}
		}, "build-flags-frozen-set-invalid"},
		{"padded release", func(m *StrixComparatorManifest) {
			m.BuildFlags = []string{"GGML_VULKAN=ON", " CMAKE_BUILD_TYPE=Release"}
		}, "build-flags-frozen-set-invalid"},
		{"server target missing", func(m *StrixComparatorManifest) { m.BuildTargets = []string{"llama-bench"} }, "build-target-set-invalid"},
		{"bench target missing", func(m *StrixComparatorManifest) { m.BuildTargets = []string{"llama-server"} }, "build-target-set-invalid"},
		{"duplicate target", func(m *StrixComparatorManifest) { m.BuildTargets = []string{"llama-server", "llama-server"} }, "build-target-set-invalid"},
		{"extra target", func(m *StrixComparatorManifest) {
			m.BuildTargets = []string{"llama-server", "llama-bench", "llama-cli"}
		}, "build-target-set-invalid"},
		{"server binary hash", func(m *StrixComparatorManifest) { m.ServerBinarySHA256 = "" }, "llama-server-sha256-invalid"},
		{"bench binary hash", func(m *StrixComparatorManifest) { m.BenchBinarySHA256 = strings.Repeat("g", 64) }, "llama-bench-sha256-invalid"},
		{"reused archive and build hash", func(m *StrixComparatorManifest) { m.BuildManifestSHA256 = m.SourceArchiveSHA256 }, "sha256-claims-reused"},
		{"reused binary hashes", func(m *StrixComparatorManifest) { m.BenchBinarySHA256 = m.ServerBinarySHA256 }, "sha256-claims-reused"},
		{"reused packet and model hashes", func(m *StrixComparatorManifest) { m.PromptPacketDigest = m.ModelSHA256 }, "sha256-claims-reused"},
		{"backend", func(m *StrixComparatorManifest) { m.Backend = "cpu" }, "backend-not-vulkan"},
		{"backend mixed case", func(m *StrixComparatorManifest) { m.Backend = "VuLkAn" }, "backend-not-vulkan"},
		{"backend whitespace", func(m *StrixComparatorManifest) { m.Backend = " vulkan " }, "backend-not-vulkan"},
		{"driver", func(m *StrixComparatorManifest) { m.Driver = "AMDVLK" }, "driver-not-radv"},
		{"device", func(m *StrixComparatorManifest) { m.DeviceName = "AMD Radeon 8060S Graphics" }, "device-name-mismatch"},
		{"target ISA", func(m *StrixComparatorManifest) { m.TargetISA = "gfx1100" }, "target-isa-not-gfx1151"},
		{"compute units", func(m *StrixComparatorManifest) { m.ComputeUnits = 39 }, "compute-units-not-40"},
		{"memory", func(m *StrixComparatorManifest) { m.MemoryGiB = 128 }, "memory-not-64-gib"},
		{"model hash", func(m *StrixComparatorManifest) { m.ModelSHA256 = strings.Repeat("d", 64) }, "model-sha256-mismatch"},
		{"packet schema", func(m *StrixComparatorManifest) { m.PromptPacketSchema = "fak.qwen38.prompt-token-packet.v1" }, "prompt-packet-schema-not-v2"},
		{"packet digest", func(m *StrixComparatorManifest) { m.PromptPacketDigest = "" }, "prompt-packet-digest-invalid"},
		{"context", func(m *StrixComparatorManifest) { m.ContextTokens = 16384 }, "context-not-32768"},
		{"temperature", func(m *StrixComparatorManifest) { m.Temperature = 0.1 }, "temperature-not-zero"},
		{"temperature NaN", func(m *StrixComparatorManifest) { m.Temperature = math.NaN() }, "temperature-not-zero"},
		{"temperature positive infinity", func(m *StrixComparatorManifest) { m.Temperature = math.Inf(1) }, "temperature-not-zero"},
		{"temperature negative infinity", func(m *StrixComparatorManifest) { m.Temperature = math.Inf(-1) }, "temperature-not-zero"},
		{"reasoning", func(m *StrixComparatorManifest) { m.ReasoningOff = false }, "reasoning-not-off"},
		{"cache", func(m *StrixComparatorManifest) { m.CacheMode = "warm" }, "cache-mode-not-cold"},
		{"prefix reuse", func(m *StrixComparatorManifest) { m.PrefixReuse = true }, "prefix-reuse-enabled"},
		{"ngram", func(m *StrixComparatorManifest) { m.NGramSpeculation = true }, "ngram-speculation-enabled"},
		{"copy style", func(m *StrixComparatorManifest) { m.CopyStyleSpeculation = true }, "copy-style-speculation-enabled"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			manifest := valid
			manifest.BuildFlags = append([]string(nil), valid.BuildFlags...)
			manifest.BuildTargets = append([]string(nil), valid.BuildTargets...)
			tc.mutate(&manifest)
			result := ValidateStrixComparatorPlan(manifest)
			if result.Schema != StrixComparatorPlanValidationSchema || result.Outcome != OutcomeRefuse || result.PlanValid || result.PerformanceCredit {
				t.Fatalf("mutated manifest received credit: %+v", result)
			}
			if !containsReason(result.Reasons, tc.reason) {
				t.Fatalf("reasons = %v, want %q", result.Reasons, tc.reason)
			}
		})
	}
}

func TestStrixComparatorManifestEvidenceBundle(t *testing.T) {
	// This intentionally arbitrary, self-consistent fixture proves only claim-
	// bundle consistency. It is not artifact identity or physical evidence.
	valid := validStrixComparatorEvidenceBundle(t)
	got := ValidateStrixComparatorEvidenceBundle(valid)
	if got.Schema != StrixComparatorEvidenceBundleValidationSchema || got.Outcome != OutcomeAbstain || !got.PlanValid || !got.ClaimBundleConsistent {
		t.Fatalf("valid evidence bundle result = %+v", got)
	}
	if got.IdentityVerified || got.PhysicalExecutionVerified || got.ComparisonCredit || got.PerformanceCredit {
		t.Fatalf("self-consistent claim bundle received authoritative credit: %+v", got)
	}
	if len(got.Reasons) != 1 || got.Reasons[0] != "claim-bundle-consistent-authoritative-observation-required" {
		t.Fatalf("valid evidence bundle reasons = %v", got.Reasons)
	}

	raw, err := json.Marshal(valid)
	if err != nil {
		t.Fatal(err)
	}
	var shape map[string]json.RawMessage
	if err := json.Unmarshal(raw, &shape); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"claimed_lease_path", "claimed_lease_mode", "claimed_lease_held", "claimed_bind_address", "claimed_service_before", "claimed_service_after"} {
		if _, ok := shape[key]; !ok {
			t.Fatalf("evidence bundle JSON missing %q: %s", key, raw)
		}
	}
	for _, key := range []string{"lease_path", "lease_mode", "lease_held", "bind_address", "service_before", "service_after"} {
		if _, ok := shape[key]; ok {
			t.Fatalf("evidence bundle JSON exposes unqualified physical claim %q: %s", key, raw)
		}
	}

	tests := []struct {
		name   string
		mutate func(*StrixComparatorEvidenceBundle)
		reason string
	}{
		{"invalid plan", func(o *StrixComparatorEvidenceBundle) { o.Manifest.SourceTree = "wrong" }, "plan-invalid"},
		{"source archive missing", func(o *StrixComparatorEvidenceBundle) { o.SourceArchiveBytes = nil }, "source-archive-bytes-missing-or-mismatch"},
		{"source archive mismatch", func(o *StrixComparatorEvidenceBundle) { o.SourceArchiveBytes[0] ^= 1 }, "source-archive-bytes-missing-or-mismatch"},
		{"build manifest missing", func(o *StrixComparatorEvidenceBundle) { o.BuildManifestBytes = nil }, "build-manifest-not-canonical-json"},
		{"build manifest noncanonical", func(o *StrixComparatorEvidenceBundle) {
			o.BuildManifestBytes = []byte(" {\"build_type\":\"Release\"}")
			o.Manifest.BuildManifestSHA256 = sha256String(o.BuildManifestBytes)
		}, "build-manifest-not-canonical-json"},
		{"build manifest mismatch", func(o *StrixComparatorEvidenceBundle) { o.BuildManifestBytes[2] ^= 1 }, "build-manifest-bytes-missing-or-mismatch"},
		{"server missing", func(o *StrixComparatorEvidenceBundle) { o.ServerBinaryBytes = nil }, "llama-server-bytes-missing-or-mismatch"},
		{"server mismatch", func(o *StrixComparatorEvidenceBundle) { o.ServerBinaryBytes[0] ^= 1 }, "llama-server-bytes-missing-or-mismatch"},
		{"bench missing", func(o *StrixComparatorEvidenceBundle) { o.BenchBinaryBytes = nil }, "llama-bench-bytes-missing-or-mismatch"},
		{"bench mismatch", func(o *StrixComparatorEvidenceBundle) { o.BenchBinaryBytes[0] ^= 1 }, "llama-bench-bytes-missing-or-mismatch"},
		{"packet missing", func(o *StrixComparatorEvidenceBundle) { o.PromptPacketBytes = nil }, "prompt-packet-invalid-or-tampered"},
		{"packet tampered", func(o *StrixComparatorEvidenceBundle) {
			o.PromptPacketBytes = append([]byte(nil), o.PromptPacketBytes...)
			o.PromptPacketBytes[len(o.PromptPacketBytes)-2] ^= 1
		}, "prompt-packet-invalid-or-tampered"},
		{"packet digest outer mismatch", func(o *StrixComparatorEvidenceBundle) { o.Manifest.PromptPacketDigest = strings.Repeat("f", 64) }, "prompt-packet-digest-mismatch"},
		{"packet model mismatch", func(o *StrixComparatorEvidenceBundle) {
			o.PromptPacketBytes = frozenStrixPacketBytes(t, strings.Repeat("9", 64), 32768, 0)
		}, "prompt-packet-model-mismatch"},
		{"packet context mismatch", func(o *StrixComparatorEvidenceBundle) {
			o.PromptPacketBytes = frozenStrixPacketBytes(t, StrixComparatorModelSHA256, 16384, 0)
		}, "prompt-packet-context-mismatch"},
		{"packet temperature mismatch", func(o *StrixComparatorEvidenceBundle) {
			o.PromptPacketBytes = frozenStrixPacketBytes(t, StrixComparatorModelSHA256, 32768, 0.1)
		}, "prompt-packet-temperature-mismatch"},
		{"claimed backend", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.Backend = "Vulkan" }, "claimed-backend-mismatch"},
		{"claimed driver", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.Driver = "AMDVLK" }, "claimed-driver-mismatch"},
		{"claimed device", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.DeviceName = "other" }, "claimed-device-name-mismatch"},
		{"claimed ISA", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.TargetISA = "gfx1100" }, "claimed-target-isa-mismatch"},
		{"claimed CUs", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.ComputeUnits = 39 }, "claimed-compute-units-mismatch"},
		{"claimed memory", func(o *StrixComparatorEvidenceBundle) { o.ClaimedDevice.MemoryGiB = 128 }, "claimed-memory-mismatch"},
		{"lease path", func(o *StrixComparatorEvidenceBundle) { o.ClaimedLeasePath = "/tmp/other.lease" }, "claimed-exclusive-canonical-lease-path-required"},
		{"lease mode", func(o *StrixComparatorEvidenceBundle) { o.ClaimedLeaseMode = "shared" }, "claimed-exclusive-lease-mode-required"},
		{"lease not held", func(o *StrixComparatorEvidenceBundle) { o.ClaimedLeaseHeld = false }, "claimed-exclusive-lease-not-held"},
		{"non-loopback bind", func(o *StrixComparatorEvidenceBundle) { o.ClaimedBindAddress = "0.0.0.0" }, "claimed-loopback-binding-required"},
		{"service before missing", func(o *StrixComparatorEvidenceBundle) { o.ClaimedServiceBefore.InvocationID = "" }, "claimed-service-before-identity-incomplete"},
		{"service after missing", func(o *StrixComparatorEvidenceBundle) { o.ClaimedServiceAfter.PID = 0 }, "claimed-service-after-identity-incomplete"},
		{"service unhealthy throughout", func(o *StrixComparatorEvidenceBundle) {
			o.ClaimedServiceBefore.Health = "unhealthy"
			o.ClaimedServiceAfter.Health = "unhealthy"
		}, "claimed-service-before-identity-incomplete"},
		{"service PID changed", func(o *StrixComparatorEvidenceBundle) { o.ClaimedServiceAfter.PID++ }, "claimed-protected-service-identity-or-health-changed"},
		{"service invocation changed", func(o *StrixComparatorEvidenceBundle) { o.ClaimedServiceAfter.InvocationID = "invocation-2" }, "claimed-protected-service-identity-or-health-changed"},
		{"service health changed", func(o *StrixComparatorEvidenceBundle) { o.ClaimedServiceAfter.Health = "unhealthy" }, "claimed-protected-service-identity-or-health-changed"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bundle := cloneStrixEvidenceBundle(valid)
			tc.mutate(&bundle)
			result := ValidateStrixComparatorEvidenceBundle(bundle)
			if result.Schema != StrixComparatorEvidenceBundleValidationSchema || result.Outcome != OutcomeRefuse || result.ClaimBundleConsistent || result.IdentityVerified || result.PhysicalExecutionVerified || result.ComparisonCredit || result.PerformanceCredit {
				t.Fatalf("mutated evidence bundle received credit: %+v", result)
			}
			if !containsReason(result.Reasons, tc.reason) {
				t.Fatalf("reasons = %v, want %q", result.Reasons, tc.reason)
			}
		})
	}
}

func validStrixComparatorEvidenceBundle(t *testing.T) StrixComparatorEvidenceBundle {
	t.Helper()
	sourceArchive := []byte("pinned source archive bytes")
	buildManifest := []byte(`{"build_type":"Release","flags":["GGML_VULKAN=ON"],"targets":["llama-server","llama-bench"]}`)
	serverBinary := []byte("llama-server executable bytes")
	benchBinary := []byte("llama-bench executable bytes")
	packetBytes := frozenStrixPacketBytes(t, StrixComparatorModelSHA256, 32768, 0)
	packet, err := qwen38quantrun.ImportPromptPacket(packetBytes)
	if err != nil {
		t.Fatal(err)
	}

	manifest := validStrixComparatorManifest()
	manifest.SourceArchiveSHA256 = sha256String(sourceArchive)
	manifest.BuildManifestSHA256 = sha256String(buildManifest)
	manifest.ServerBinarySHA256 = sha256String(serverBinary)
	manifest.BenchBinarySHA256 = sha256String(benchBinary)
	manifest.PromptPacketDigest = packet.PacketDigest

	service := StrixComparatorClaimedServiceIdentity{PID: 4242, InvocationID: "invocation-1", Health: "healthy"}
	return StrixComparatorEvidenceBundle{
		Manifest:             manifest,
		SourceArchiveBytes:   sourceArchive,
		BuildManifestBytes:   buildManifest,
		ServerBinaryBytes:    serverBinary,
		BenchBinaryBytes:     benchBinary,
		PromptPacketBytes:    packetBytes,
		ClaimedDevice:        StrixComparatorClaimedDevice{Backend: "vulkan", Driver: "RADV", DeviceName: StrixComparatorDeviceName, TargetISA: "gfx1151", ComputeUnits: 40, MemoryGiB: 64},
		ClaimedLeasePath:     StrixComparatorLeasePath,
		ClaimedLeaseMode:     "exclusive",
		ClaimedLeaseHeld:     true,
		ClaimedBindAddress:   StrixComparatorLoopbackAddress,
		ClaimedServiceBefore: service,
		ClaimedServiceAfter:  service,
	}
}

func frozenStrixPacketBytes(t *testing.T, artifact string, contextTokens int, temperature float64) []byte {
	t.Helper()
	packet, err := qwen38quantrun.FreezePromptPacket(qwen38quantrun.PromptTokenPacket{
		Schema:            qwen38quantrun.PromptTokenPacketSchema,
		PacketID:          "strix-comparator-observation",
		ArtifactSHA256:    artifact,
		TokenizerIdentity: "qwen-test-tokenizer",
		TokenizerDigest:   strings.Repeat("6", 64),
		TemplateDigest:    strings.Repeat("7", 64),
		PromptTokenIDs:    []int{151644, 872, 198},
		StopTokens:        []string{"<|im_end|>"},
		StopTokenIDs:      []int{151645},
		ContextBudget: qwen38quantrun.ContextBudget{
			ContextTokens:      contextTokens,
			ContextBudgetBytes: 1 << 30,
		},
		GenerationControls: qwen38quantrun.GenerationControls{
			Temperature:     temperature,
			TopP:            1,
			TopK:            1,
			MaxOutputTokens: 8,
			StopTokens:      []string{"<|im_end|>"},
			StopTokenIDs:    []int{151645},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := qwen38quantrun.ExportPromptPacket(packet)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func cloneStrixEvidenceBundle(in StrixComparatorEvidenceBundle) StrixComparatorEvidenceBundle {
	out := in
	out.Manifest.BuildFlags = append([]string(nil), in.Manifest.BuildFlags...)
	out.Manifest.BuildTargets = append([]string(nil), in.Manifest.BuildTargets...)
	out.SourceArchiveBytes = append([]byte(nil), in.SourceArchiveBytes...)
	out.BuildManifestBytes = append([]byte(nil), in.BuildManifestBytes...)
	out.ServerBinaryBytes = append([]byte(nil), in.ServerBinaryBytes...)
	out.BenchBinaryBytes = append([]byte(nil), in.BenchBinaryBytes...)
	out.PromptPacketBytes = append([]byte(nil), in.PromptPacketBytes...)
	return out
}

func sha256String(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func containsReason(reasons []string, want string) bool {
	for _, reason := range reasons {
		if reason == want {
			return true
		}
	}
	return false
}

func validStrixComparatorManifest() StrixComparatorManifest {
	return StrixComparatorManifest{
		Schema:               StrixComparatorManifestSchema,
		SourceRepository:     StrixComparatorSourceRepository,
		SourceRevision:       StrixComparatorSourceRevision,
		SourceTree:           StrixComparatorSourceTree,
		SourceArchiveSHA256:  strings.Repeat("a", 64),
		UpstreamBuild:        StrixComparatorUpstreamBuild,
		BuildManifestSHA256:  strings.Repeat("b", 64),
		BuildFlags:           []string{"GGML_VULKAN=ON", "CMAKE_BUILD_TYPE=Release"},
		BuildTargets:         []string{"llama-server", "llama-bench"},
		ServerBinarySHA256:   strings.Repeat("c", 64),
		BenchBinarySHA256:    strings.Repeat("d", 64),
		Backend:              "vulkan",
		Driver:               "RADV",
		DeviceName:           StrixComparatorDeviceName,
		TargetISA:            "gfx1151",
		ComputeUnits:         40,
		MemoryGiB:            64,
		ModelSHA256:          StrixComparatorModelSHA256,
		PromptPacketSchema:   StrixComparatorPacketSchema,
		PromptPacketDigest:   strings.Repeat("e", 64),
		ContextTokens:        32768,
		Temperature:          0,
		ReasoningOff:         true,
		CacheMode:            "cold",
		PrefixReuse:          false,
		NGramSpeculation:     false,
		CopyStyleSpeculation: false,
	}
}
