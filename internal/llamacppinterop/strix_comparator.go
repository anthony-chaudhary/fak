package llamacppinterop

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

const (
	// StrixComparatorManifestSchema identifies the device-free comparator
	// provenance and envelope contract used by issue #12148.
	StrixComparatorManifestSchema = "fak.llamacppinterop.strix-comparator-manifest/1"
	// StrixComparatorPlanValidationSchema identifies device-free validation of
	// a proposed manifest, not the manifest or a physical receipt itself.
	StrixComparatorPlanValidationSchema = "fak.llamacppinterop.strix-comparator-plan-validation/1"
	// StrixComparatorEvidenceBundleValidationSchema identifies the result of
	// checking a caller-supplied claim bundle. It is intentionally distinct
	// from an authoritative artifact or physical-execution receipt.
	StrixComparatorEvidenceBundleValidationSchema = "fak.llamacppinterop.strix-comparator-evidence-bundle-validation/1"

	StrixComparatorSourceRepository = "ggml-org/llama.cpp"
	StrixComparatorSourceRevision   = "70adb1b4cea5ee39f867792c78dc59320921eda7"
	StrixComparatorSourceTree       = "174f4186cb52cad7783040246a7f40b7a7cd5329"
	StrixComparatorUpstreamBuild    = "b10588"
	StrixComparatorModelSHA256      = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	StrixComparatorPacketSchema     = "fak.qwen38.prompt-token-packet.v2"
	StrixComparatorDeviceName       = "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
	StrixComparatorLeasePath        = "/tmp/fak-gpu.lease"
	StrixComparatorLoopbackAddress  = "127.0.0.1"
)

// StrixComparatorManifest describes the exact source, build, device, model,
// packet, and workload envelope planned by #12148. Device-free validation can
// verify only the shape and consistency of these claims; it does not observe
// the named bytes, device, service, lease, or packet consumption.
type StrixComparatorManifest struct {
	Schema              string   `json:"schema"`
	SourceRepository    string   `json:"source_repository"`
	SourceRevision      string   `json:"source_revision"`
	SourceTree          string   `json:"source_tree"`
	SourceArchiveSHA256 string   `json:"source_archive_sha256"`
	UpstreamBuild       string   `json:"upstream_build"`
	BuildManifestSHA256 string   `json:"build_manifest_sha256"`
	BuildFlags          []string `json:"build_flags"`
	BuildTargets        []string `json:"build_targets"`
	ServerBinarySHA256  string   `json:"llama_server_sha256"`
	BenchBinarySHA256   string   `json:"llama_bench_sha256"`

	Backend      string `json:"backend"`
	Driver       string `json:"driver"`
	DeviceName   string `json:"device_name"`
	TargetISA    string `json:"target_isa"`
	ComputeUnits int    `json:"compute_units"`
	MemoryGiB    int    `json:"memory_gib"`

	ModelSHA256        string `json:"model_sha256"`
	PromptPacketSchema string `json:"prompt_packet_schema"`
	PromptPacketDigest string `json:"prompt_packet_digest"`

	ContextTokens        int     `json:"context_tokens"`
	Temperature          float64 `json:"temperature"`
	ReasoningOff         bool    `json:"reasoning_off"`
	CacheMode            string  `json:"cache_mode"`
	PrefixReuse          bool    `json:"prefix_reuse"`
	NGramSpeculation     bool    `json:"ngram_speculation"`
	CopyStyleSpeculation bool    `json:"copy_style_speculation"`
}

// StrixComparatorPlanValidation is the fail-closed device-free plan result.
// PlanValid means the manifest is internally complete enough to take to a
// physical verifier. Outcome remains abstain until that separate verifier
// observes the bytes and packet; this type never grants readiness or credit.
type StrixComparatorPlanValidation struct {
	Schema            string   `json:"schema"`
	Outcome           Outcome  `json:"outcome"`
	Reasons           []string `json:"reasons,omitempty"`
	PlanValid         bool     `json:"plan_valid"`
	PerformanceCredit bool     `json:"performance_credit"`
}

// StrixComparatorClaimedDevice is device identity supplied in a device-free
// claim bundle. Equality here proves only internal consistency; it is not proof
// that the caller queried the hardware.
type StrixComparatorClaimedDevice struct {
	Backend      string `json:"backend"`
	Driver       string `json:"driver"`
	DeviceName   string `json:"device_name"`
	TargetISA    string `json:"target_isa"`
	ComputeUnits int    `json:"compute_units"`
	MemoryGiB    int    `json:"memory_gib"`
}

// StrixComparatorClaimedServiceIdentity captures caller-claimed protected-
// service continuity fields. A physical verifier must obtain them authoritatively.
type StrixComparatorClaimedServiceIdentity struct {
	PID          int    `json:"pid"`
	InvocationID string `json:"invocation_id"`
	Health       string `json:"health"`
}

// StrixComparatorEvidenceBundle carries caller-supplied bytes and claims into
// a pure consistency checker. These are not filesystem-streamed or host-sourced
// observations and therefore cannot establish artifact identity or execution.
type StrixComparatorEvidenceBundle struct {
	Manifest StrixComparatorManifest `json:"manifest"`

	SourceArchiveBytes []byte `json:"-"`
	BuildManifestBytes []byte `json:"-"`
	ServerBinaryBytes  []byte `json:"-"`
	BenchBinaryBytes   []byte `json:"-"`
	PromptPacketBytes  []byte `json:"-"`

	ClaimedDevice        StrixComparatorClaimedDevice          `json:"claimed_device"`
	ClaimedLeasePath     string                                `json:"claimed_lease_path"`
	ClaimedLeaseMode     string                                `json:"claimed_lease_mode"`
	ClaimedLeaseHeld     bool                                  `json:"claimed_lease_held"`
	ClaimedBindAddress   string                                `json:"claimed_bind_address"`
	ClaimedServiceBefore StrixComparatorClaimedServiceIdentity `json:"claimed_service_before"`
	ClaimedServiceAfter  StrixComparatorClaimedServiceIdentity `json:"claimed_service_after"`
}

// StrixComparatorBundleValidation reports only whether a caller-supplied claim
// bundle is internally consistent with a valid plan. All authoritative identity,
// physical execution, comparison, and performance fields are hard-false.
type StrixComparatorBundleValidation struct {
	Schema                    string   `json:"schema"`
	Outcome                   Outcome  `json:"outcome"`
	Reasons                   []string `json:"reasons,omitempty"`
	PlanValid                 bool     `json:"plan_valid"`
	ClaimBundleConsistent     bool     `json:"claim_bundle_consistent"`
	IdentityVerified          bool     `json:"identity_verified"`
	PhysicalExecutionVerified bool     `json:"physical_execution_verified"`
	ComparisonCredit          bool     `json:"comparison_credit"`
	PerformanceCredit         bool     `json:"performance_credit"`
}

// ValidateStrixComparatorPlan validates only the proposed immutable identity
// and frozen benchmark envelope. It does not inspect a host, hash bytes,
// verify packet consumption, or execute a binary.
func ValidateStrixComparatorPlan(m StrixComparatorManifest) StrixComparatorPlanValidation {
	reasons := make([]string, 0)
	require := func(ok bool, reason string) {
		if !ok {
			reasons = append(reasons, reason)
		}
	}

	require(m.Schema == StrixComparatorManifestSchema, "manifest-schema-mismatch")
	require(m.SourceRepository == StrixComparatorSourceRepository, "source-repository-mismatch")
	require(m.SourceRevision == StrixComparatorSourceRevision, "source-revision-mismatch")
	require(m.SourceTree == StrixComparatorSourceTree, "source-tree-mismatch")
	require(validStrixSHA256(m.SourceArchiveSHA256), "source-archive-sha256-invalid")
	require(m.UpstreamBuild == StrixComparatorUpstreamBuild, "upstream-build-mismatch")
	require(validStrixSHA256(m.BuildManifestSHA256), "build-manifest-sha256-invalid")
	require(validStrixBuildFlags(m.BuildFlags), "build-flags-frozen-set-invalid")
	require(validStrixBuildTargets(m.BuildTargets), "build-target-set-invalid")
	require(validStrixSHA256(m.ServerBinarySHA256), "llama-server-sha256-invalid")
	require(validStrixSHA256(m.BenchBinarySHA256), "llama-bench-sha256-invalid")

	require(m.Backend == "vulkan", "backend-not-vulkan")
	require(m.Driver == "RADV", "driver-not-radv")
	require(m.DeviceName == StrixComparatorDeviceName, "device-name-mismatch")
	require(m.TargetISA == "gfx1151", "target-isa-not-gfx1151")
	require(m.ComputeUnits == 40, "compute-units-not-40")
	require(m.MemoryGiB == 64, "memory-not-64-gib")

	require(m.ModelSHA256 == StrixComparatorModelSHA256, "model-sha256-mismatch")
	require(m.PromptPacketSchema == StrixComparatorPacketSchema, "prompt-packet-schema-not-v2")
	require(validStrixSHA256(m.PromptPacketDigest), "prompt-packet-digest-invalid")
	require(uniqueStrixSHA256Claims(
		m.SourceArchiveSHA256,
		m.BuildManifestSHA256,
		m.ServerBinarySHA256,
		m.BenchBinarySHA256,
		m.ModelSHA256,
		m.PromptPacketDigest,
	), "sha256-claims-reused")

	require(m.ContextTokens == 32768, "context-not-32768")
	require(m.Temperature == 0, "temperature-not-zero")
	require(m.ReasoningOff, "reasoning-not-off")
	require(m.CacheMode == "cold", "cache-mode-not-cold")
	require(!m.PrefixReuse, "prefix-reuse-enabled")
	require(!m.NGramSpeculation, "ngram-speculation-enabled")
	require(!m.CopyStyleSpeculation, "copy-style-speculation-enabled")

	result := StrixComparatorPlanValidation{
		Schema:            StrixComparatorPlanValidationSchema,
		Outcome:           OutcomeRefuse,
		Reasons:           reasons,
		PerformanceCredit: false,
	}
	if len(reasons) == 0 {
		result.Outcome = OutcomeAbstain
		result.PlanValid = true
		result.Reasons = []string{"device-free-plan-valid-observation-required"}
	}
	return result
}

// ValidateStrixComparatorEvidenceBundle checks supplied bytes, a frozen prompt
// packet, and declared safety claims for internal consistency after plan
// validation. It performs no host access and grants no authoritative credit.
func ValidateStrixComparatorEvidenceBundle(o StrixComparatorEvidenceBundle) StrixComparatorBundleValidation {
	plan := ValidateStrixComparatorPlan(o.Manifest)
	result := StrixComparatorBundleValidation{
		Schema:                    StrixComparatorEvidenceBundleValidationSchema,
		Outcome:                   OutcomeRefuse,
		PlanValid:                 plan.PlanValid,
		IdentityVerified:          false,
		PhysicalExecutionVerified: false,
		ComparisonCredit:          false,
		PerformanceCredit:         false,
	}
	if !plan.PlanValid {
		result.Reasons = append([]string{"plan-invalid"}, plan.Reasons...)
		return result
	}

	reasons := make([]string, 0)
	require := func(ok bool, reason string) {
		if !ok {
			reasons = append(reasons, reason)
		}
	}

	require(bytesMatchSHA256(o.SourceArchiveBytes, o.Manifest.SourceArchiveSHA256), "source-archive-bytes-missing-or-mismatch")
	require(canonicalJSON(o.BuildManifestBytes), "build-manifest-not-canonical-json")
	require(bytesMatchSHA256(o.BuildManifestBytes, o.Manifest.BuildManifestSHA256), "build-manifest-bytes-missing-or-mismatch")
	require(bytesMatchSHA256(o.ServerBinaryBytes, o.Manifest.ServerBinarySHA256), "llama-server-bytes-missing-or-mismatch")
	require(bytesMatchSHA256(o.BenchBinaryBytes, o.Manifest.BenchBinarySHA256), "llama-bench-bytes-missing-or-mismatch")

	packet, packetErr := qwen38quantrun.ImportPromptPacket(o.PromptPacketBytes)
	require(packetErr == nil, "prompt-packet-invalid-or-tampered")
	if packetErr == nil {
		require(packet.Schema == qwen38quantrun.PromptTokenPacketSchema, "prompt-packet-schema-not-v2")
		require(packet.PacketDigest == o.Manifest.PromptPacketDigest, "prompt-packet-digest-mismatch")
		require(packet.ArtifactSHA256 == o.Manifest.ModelSHA256, "prompt-packet-model-mismatch")
		require(packet.ContextBudget.ContextTokens == o.Manifest.ContextTokens, "prompt-packet-context-mismatch")
		require(packet.GenerationControls.Temperature == o.Manifest.Temperature, "prompt-packet-temperature-mismatch")
	}

	require(o.ClaimedDevice.Backend == o.Manifest.Backend, "claimed-backend-mismatch")
	require(o.ClaimedDevice.Driver == o.Manifest.Driver, "claimed-driver-mismatch")
	require(o.ClaimedDevice.DeviceName == o.Manifest.DeviceName, "claimed-device-name-mismatch")
	require(o.ClaimedDevice.TargetISA == o.Manifest.TargetISA, "claimed-target-isa-mismatch")
	require(o.ClaimedDevice.ComputeUnits == o.Manifest.ComputeUnits, "claimed-compute-units-mismatch")
	require(o.ClaimedDevice.MemoryGiB == o.Manifest.MemoryGiB, "claimed-memory-mismatch")

	require(o.ClaimedLeasePath == StrixComparatorLeasePath, "claimed-exclusive-canonical-lease-path-required")
	require(o.ClaimedLeaseMode == "exclusive", "claimed-exclusive-lease-mode-required")
	require(o.ClaimedLeaseHeld, "claimed-exclusive-lease-not-held")
	require(o.ClaimedBindAddress == StrixComparatorLoopbackAddress, "claimed-loopback-binding-required")
	require(validStrixServiceIdentity(o.ClaimedServiceBefore), "claimed-service-before-identity-incomplete")
	require(validStrixServiceIdentity(o.ClaimedServiceAfter), "claimed-service-after-identity-incomplete")
	require(o.ClaimedServiceBefore == o.ClaimedServiceAfter, "claimed-protected-service-identity-or-health-changed")

	result.Reasons = reasons
	if len(reasons) == 0 {
		result.Outcome = OutcomeAbstain
		result.ClaimBundleConsistent = true
		result.Reasons = []string{"claim-bundle-consistent-authoritative-observation-required"}
	}
	return result
}

func bytesMatchSHA256(data []byte, want string) bool {
	if len(data) == 0 {
		return false
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]) == want
}

func canonicalJSON(data []byte) bool {
	if len(data) == 0 || !json.Valid(data) {
		return false
	}
	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		return false
	}
	encoded, err := json.Marshal(value)
	return err == nil && bytes.Equal(data, encoded)
}

func validStrixServiceIdentity(identity StrixComparatorClaimedServiceIdentity) bool {
	return identity.PID > 0 && identity.InvocationID != "" && identity.Health == "healthy"
}

func validStrixSHA256(value string) bool {
	if len(value) != 64 || value != strings.ToLower(value) || value != strings.TrimSpace(value) || value == strings.Repeat("0", 64) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func validStrixBuildFlags(flags []string) bool {
	seenVulkan, seenRelease := 0, 0
	for _, flag := range flags {
		if flag == "" || flag != strings.TrimSpace(flag) {
			return false
		}
		normalized := strings.TrimPrefix(flag, "-D")
		switch strixCMakeCacheKey(normalized) {
		case "GGML_VULKAN":
			if flag != "GGML_VULKAN=ON" {
				return false
			}
			seenVulkan++
		case "CMAKE_BUILD_TYPE":
			if flag != "CMAKE_BUILD_TYPE=Release" {
				return false
			}
			seenRelease++
		}
	}
	return seenVulkan == 1 && seenRelease == 1
}

func strixCMakeCacheKey(flag string) string {
	separator := strings.IndexAny(flag, ":=")
	if separator <= 0 {
		return ""
	}
	return strings.ToUpper(flag[:separator])
}

func validStrixBuildTargets(targets []string) bool {
	if len(targets) != 2 {
		return false
	}
	seenServer, seenBench := false, false
	for _, target := range targets {
		switch target {
		case "llama-server":
			if seenServer {
				return false
			}
			seenServer = true
		case "llama-bench":
			if seenBench {
				return false
			}
			seenBench = true
		default:
			return false
		}
	}
	return seenServer && seenBench
}

func uniqueStrixSHA256Claims(values ...string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !validStrixSHA256(value) {
			continue
		}
		if _, exists := seen[value]; exists {
			return false
		}
		seen[value] = struct{}{}
	}
	return true
}
