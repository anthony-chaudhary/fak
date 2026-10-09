package computebuild

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"
)

// V5 has its own strict envelope; the common alias deliberately does not gain
// an UnmarshalJSON method or expose successor fields to historical decoders.
type vulkanBuildReceiptV5Wire struct {
	computeBuildReceiptV2Wire
	ShaderRegistry *VulkanShaderRegistryIdentity `json:"shader_registry,omitempty"`
	NativeArchive  *BuildArtifact                `json:"native_archive,omitempty"`
	IndexerScore   *VulkanIndexerScoreContract   `json:"indexer_score_contract,omitempty"`
}

func decodeStrictVulkanReceiptV5(raw []byte) (ComputeBuildReceipt, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var wire vulkanBuildReceiptV5Wire
	if err := dec.Decode(&wire); err != nil {
		return ComputeBuildReceipt{}, fmt.Errorf("decode Vulkan V5 receipt: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return ComputeBuildReceipt{}, fmt.Errorf("decode Vulkan V5 receipt: trailing JSON data")
	}
	receipt := ComputeBuildReceipt(wire.computeBuildReceiptV2Wire)
	receipt.VulkanRegistry = wire.ShaderRegistry
	receipt.VulkanNativeArchive = wire.NativeArchive
	receipt.VulkanIndexerScore = wire.IndexerScore
	if receipt.Schema != VulkanBuildReceiptSchemaV5 {
		return ComputeBuildReceipt{}, fmt.Errorf("Vulkan receipt schema mismatch: want %s", VulkanBuildReceiptSchemaV5)
	}
	return receipt, nil
}

func validateVulkanReceiptV5Registry(receipt ComputeBuildReceipt) error {
	expected, err := currentVulkanRegistryV5Identity()
	if err != nil {
		return err
	}
	if receipt.Schema != VulkanBuildReceiptSchemaV5 || receipt.VulkanRegistry == nil || *receipt.VulkanRegistry != expected {
		return fmt.Errorf("Vulkan V5 receipt trusted registry identity mismatch")
	}
	if receipt.Vulkan == nil || receipt.Vulkan.SPIRVModuleCount != vulkanV5ModuleCount {
		return fmt.Errorf("Vulkan V5 receipt requires exactly %d modules", vulkanV5ModuleCount)
	}
	return nil
}

// Validate declared success before comparing identities. This is framing and
// completeness validation, not authentication; the evidence verifier still
// observes source, tools, bundle, build plan, and binary independently.
func validateVulkanReceiptV5Success(receipt *ComputeBuildReceipt) error {
	if receipt == nil || receipt.Artifact == nil || receipt.Backend != "vulkan" || receipt.Command != "binary" || receipt.Outcome != "success" || receipt.ExitCode != 0 || receipt.Error != "" || receipt.Artifact.Signed {
		return fmt.Errorf("Vulkan V5 requires a complete successful binary receipt")
	}
	if receipt.GitCommit != "" || receipt.GitRef != "" || receipt.Clean != nil || receipt.SourceArchiveSHA256 != "" || receipt.ShaderBundleSHA256 != "" || len(receipt.BuildArgs) != 0 || receipt.Toolchain != nil {
		return fmt.Errorf("Vulkan V5 rejects ambiguous legacy provenance fields")
	}
	if err := validateVulkanReceiptV5Registry(*receipt); err != nil {
		return err
	}
	if err := validateVulkanV5Extras(receipt.VulkanNativeArchive, receipt.VulkanIndexerScore); err != nil {
		return err
	}
	p := receipt.Vulkan
	if !p.Source.Clean || !validGitObjectID(p.Source.GitCommit) || !validGitObjectID(p.Source.GitTree) || !validLowerSHA256(p.Source.SourceArchiveSHA256) {
		return fmt.Errorf("Vulkan V5 source identity is incomplete")
	}
	if receipt.Artifact.SizeBytes <= 0 || receipt.Artifact.Path == "" || !validLowerSHA256(receipt.Artifact.SHA256) || !validLowerSHA256(p.SPIRVBundleSHA256) || !validLowerSHA256(p.ToolchainSHA256) || !validLowerSHA256(p.BuildCommandSHA256) || !validLowerSHA256(p.StableIdentitySHA256) {
		return fmt.Errorf("Vulkan V5 artifact or provenance digest is incomplete")
	}
	roles := []string{"go", "cc", "cxx", "ar", "glslc"}
	if len(p.Toolchain) != len(roles) {
		return fmt.Errorf("Vulkan V5 toolchain identity is incomplete")
	}
	for i, tool := range p.Toolchain {
		if tool.Role != roles[i] || tool.Executable == "" || tool.Executable == "." || tool.Executable == ".." || strings.ContainsAny(tool.Executable, "/\\") || !validLowerSHA256(tool.SHA256) {
			return fmt.Errorf("Vulkan V5 %s tool identity is invalid", roles[i])
		}
	}
	if len(p.NormalizedBuildCommand) == 0 {
		return fmt.Errorf("Vulkan V5 build command is missing")
	}
	for _, command := range p.NormalizedBuildCommand {
		if strings.TrimSpace(command) == "" || strings.ContainsRune(command, '\x00') {
			return fmt.Errorf("Vulkan V5 build command is invalid")
		}
		// Witness archives and binaries are a separate test profile. V5 binds
		// only the disarmed production archive and its ordinary compiler plan.
		if strings.Contains(command, "FAK_V41_INDEXER_SCORE_WITNESS") || strings.Contains(command, "v41_indexer_witness") {
			return fmt.Errorf("Vulkan V5 rejects the test-only indexer witness build profile")
		}
	}
	commandSHA, err := hashJSON(p.NormalizedBuildCommand)
	if err != nil || commandSHA != p.BuildCommandSHA256 {
		return fmt.Errorf("Vulkan V5 build command digest mismatch")
	}
	stableSHA, err := vulkanStableIdentityV5SHA(p.Source, p.SPIRVBundleSHA256, p.SPIRVModuleCount, p.ToolchainSHA256, p.BuildCommandSHA256, receipt.Artifact.SHA256, *receipt.VulkanNativeArchive, *receipt.VulkanIndexerScore)
	if err != nil || stableSHA != p.StableIdentitySHA256 {
		return fmt.Errorf("Vulkan V5 stable identity mismatch")
	}
	return nil
}

// Equality is not authentication. Compare every declared identity field and
// its known digest framing; the evidence verifier independently observes bytes.
func compareVulkanReceiptProvenanceV5(a, b *ComputeBuildReceipt) error {
	for _, receipt := range []*ComputeBuildReceipt{a, b} {
		if err := validateVulkanReceiptV5Success(receipt); err != nil {
			return err
		}
	}
	left, err := hashJSON(a.Vulkan)
	if err != nil {
		return err
	}
	right, err := hashJSON(b.Vulkan)
	if err != nil {
		return err
	}
	if *a.VulkanNativeArchive != *b.VulkanNativeArchive || *a.VulkanIndexerScore != *b.VulkanIndexerScore || left != right || a.Artifact.SHA256 != b.Artifact.SHA256 || a.Artifact.SizeBytes != b.Artifact.SizeBytes || a.Artifact.Signed != b.Artifact.Signed {
		return fmt.Errorf("Vulkan V5 provenance mismatch")
	}
	return nil
}

func vulkanStableIdentityV5SHA(source BuildSourceProvenance, spirvSHA string, spirvCount int, toolchainSHA, commandSHA, binarySHA string, archive BuildArtifact, score VulkanIndexerScoreContract) (string, error) {
	registry, err := currentVulkanRegistryV5Identity()
	if err != nil {
		return "", err
	}
	if spirvCount != vulkanV5ModuleCount {
		return "", fmt.Errorf("Vulkan V5 stable identity requires %d modules", vulkanV5ModuleCount)
	}
	return hashJSON(struct {
		Schema             string                       `json:"schema"`
		Registry           VulkanShaderRegistryIdentity `json:"shader_registry"`
		Source             BuildSourceProvenance        `json:"source"`
		SPIRVBundleSHA256  string                       `json:"spirv_bundle_sha256"`
		SPIRVModuleCount   int                          `json:"spirv_module_count"`
		ToolchainSHA256    string                       `json:"toolchain_sha256"`
		BuildCommandSHA256 string                       `json:"build_command_sha256"`
		BinarySHA256       string                       `json:"binary_sha256"`
		NativeArchive      BuildArtifact                `json:"native_archive"`
		IndexerScore       VulkanIndexerScoreContract   `json:"indexer_score_contract"`
	}{VulkanBuildReceiptSchemaV5, registry, source, spirvSHA, spirvCount, toolchainSHA, commandSHA, binarySHA, archive, score})
}

func verifyVulkanBinaryReceiptV5Identity(ctx context.Context, evidence VulkanBinaryReceiptEvidence) (*VulkanBinaryReceiptIdentityVerification, error) {
	if ctx == nil {
		return nil, fmt.Errorf("verify Vulkan receipt identity: nil context")
	}
	if runtime.GOOS != "linux" {
		return nil, fmt.Errorf("verify Vulkan receipt identity: fd-bound Git execution is unsupported on %s", runtime.GOOS)
	}
	gitRun, closeGit, err := pinnedLinuxGitRunner(evidence.GitExecutable, evidence.ExpectedGitSHA256)
	if err != nil {
		return nil, err
	}
	defer closeGit()
	var comparisonGitRun vulkanGitRunner
	var closeComparisonGit func() error
	if evidence.Comparison != nil {
		comparisonGitRun, closeComparisonGit, err = pinnedLinuxGitRunner(evidence.Comparison.GitExecutable, evidence.Comparison.ExpectedGitSHA256)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		defer closeComparisonGit()
	}
	return verifyVulkanBinaryReceiptV5IdentityWithRunners(ctx, evidence, gitRun, comparisonGitRun)
}

func verifyVulkanBinaryReceiptV5IdentityWithRunners(ctx context.Context, evidence VulkanBinaryReceiptEvidence, gitRun, comparisonGitRun vulkanGitRunner) (*VulkanBinaryReceiptIdentityVerification, error) {
	if evidence.Comparison != nil && evidence.Comparison.Comparison != nil {
		return nil, fmt.Errorf("verify Vulkan receipt identity: nested comparison evidence is not supported")
	}
	if gitRun == nil || (evidence.Comparison != nil && comparisonGitRun == nil) {
		return nil, fmt.Errorf("verify Vulkan receipt identity: pinned Git runner is required")
	}
	raw, err := readStableRegularFile(evidence.ReceiptPath, "Vulkan receipt")
	if err != nil {
		return nil, err
	}
	var comparisonRaw []byte
	if evidence.Comparison != nil {
		comparisonRaw, err = readStableRegularFile(evidence.Comparison.ReceiptPath, "comparison Vulkan receipt")
		if err != nil {
			return nil, err
		}
	}

	before, err := observeVulkanReceiptEvidenceV5(ctx, evidence, gitRun)
	if err != nil {
		return nil, err
	}
	receipt, err := decodeStrictVulkanReceiptV5(raw)
	if err != nil {
		return nil, err
	}

	var comparisonReceipt *ComputeBuildReceipt
	var comparisonObservation *observedVulkanReceiptV5Identity
	if evidence.Comparison != nil {
		observed, err := observeVulkanReceiptEvidenceV5(ctx, *evidence.Comparison, comparisonGitRun)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		decoded, err := decodeStrictVulkanReceiptV5(comparisonRaw)
		if err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		if err := compareReceiptV5ToObservation(decoded, observed); err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		if err := verifyBaselineReproducibility(decoded); err != nil {
			return nil, fmt.Errorf("verify comparison Vulkan receipt identity: %w", err)
		}
		comparisonReceipt = &decoded
		comparisonObservation = &observed
	}

	if err := compareReceiptV5ToObservation(receipt, before); err != nil {
		return nil, err
	}
	if evidence.Comparison == nil {
		if err := verifyBaselineReproducibility(receipt); err != nil {
			return nil, err
		}
	} else if err := verifyMatchedReproducibility(receipt, comparisonRaw, *comparisonReceipt, before.observedVulkanReceiptIdentity, comparisonObservation.observedVulkanReceiptIdentity); err != nil {
		return nil, err
	}

	after, err := observeVulkanReceiptEvidenceV5(ctx, evidence, gitRun)
	if err != nil {
		return nil, fmt.Errorf("revalidate Vulkan receipt evidence: %w", err)
	}
	if !sameObservedVulkanV5Identity(before, after) {
		return nil, fmt.Errorf("Vulkan receipt evidence changed during verification")
	}
	if evidence.Comparison != nil {
		comparisonAfter, err := observeVulkanReceiptEvidenceV5(ctx, *evidence.Comparison, comparisonGitRun)
		if err != nil {
			return nil, fmt.Errorf("revalidate comparison Vulkan receipt evidence: %w", err)
		}
		if !sameObservedVulkanV5Identity(*comparisonObservation, comparisonAfter) {
			return nil, fmt.Errorf("comparison Vulkan receipt evidence changed during verification")
		}
	}
	digest := sha256.Sum256(raw)
	return &VulkanBinaryReceiptIdentityVerification{
		Receipt:                  receipt,
		ReceiptSHA256:            hex.EncodeToString(digest[:]),
		HistoricalBuildCausality: "unavailable",
		UnavailableClaims:        append(append([]string(nil), unavailableVulkanV2Causality...), "indexer_score_source_to_module_correspondence", "indexer_score_ordered_arithmetic_semantics", "device_f32_qualification", "native_archive_to_binary_linkage", "production_build_profile_causality"),
	}, nil
}

func observeVulkanReceiptEvidenceV5(ctx context.Context, evidence VulkanBinaryReceiptEvidence, gitRun vulkanGitRunner) (observedVulkanReceiptV5Identity, error) {
	if evidence.Comparison != nil && evidence.Comparison.ReceiptPath == evidence.ReceiptPath {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("comparison receipt must be distinct")
	}
	if err := validateCurrentVulkanRegistryV5(evidence.ExpectedSPIRVModules); err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	gitSHA, _, err := stableRegularFileSHA256(evidence.GitExecutable, "Git executable")
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	if !validLowerSHA256(evidence.ExpectedGitSHA256) || gitSHA != evidence.ExpectedGitSHA256 {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("policy-pinned Git executable identity mismatch")
	}
	source, err := prepareVulkanSourceWithGit(ctx, gitRun, evidence.SourceRoot, evidence.ExpectedCommit)
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	artifactSHA, artifactSize, err := stableRegularFileSHA256(evidence.BinaryPath, "sealed mapped executable")
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	if !validLowerSHA256(evidence.ExpectedBinarySHA256) || artifactSHA != evidence.ExpectedBinarySHA256 || artifactSize != evidence.ExpectedBinarySize || artifactSize <= 0 {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("sealed mapped executable identity mismatch")
	}
	spirvSHA, spirvCount, score, err := hashCurrentSPIRVBundleV5(evidence.SPIRVRoot)
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	toolchain, toolchainSHA, err := strictVulkanToolchainIdentity(evidence.Tools)
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	if evidence.BuildPlan.OutPackage != "./cmd/fak" {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("trusted Vulkan build plan must target ./cmd/fak")
	}
	repoRoot, err := strictAbsoluteDirectory(evidence.SourceRoot, "Vulkan source root")
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	expectedPkg := filepath.Join(repoRoot, "internal", "compute")
	if !samePath(evidence.BuildPlan.PackageDir, expectedPkg) {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("trusted Vulkan package directory must be source-root/internal/compute")
	}
	tc := evidence.Tools.toolchain()
	commands, commandSHA, err := normalizedVulkanBuildCommand(&VulkanConfig{
		RepoRoot:  repoRoot,
		PkgDir:    evidence.BuildPlan.PackageDir,
		OutPkg:    evidence.BuildPlan.OutPackage,
		OutBin:    evidence.BinaryPath,
		Smoke:     evidence.BuildPlan.Smoke,
		Toolchain: tc,
	})
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	archive, err := observeVulkanNativeArchiveV5(repoRoot)
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	stableSHA, err := vulkanStableIdentityV5SHA(source, spirvSHA, spirvCount, toolchainSHA, commandSHA, artifactSHA, archive, score)
	if err != nil {
		return observedVulkanReceiptV5Identity{}, err
	}
	gitAfterSHA, _, err := stableRegularFileSHA256(evidence.GitExecutable, "Git executable")
	if err != nil || gitAfterSHA != gitSHA {
		return observedVulkanReceiptV5Identity{}, fmt.Errorf("Git executable changed during verification")
	}
	return observedVulkanReceiptV5Identity{
		NativeArchive: archive, IndexerScore: score,
		observedVulkanReceiptIdentity: observedVulkanReceiptIdentity{
			ReceiptPath:       filepath.Clean(evidence.ReceiptPath),
			Source:            source,
			Artifact:          BuildArtifact{Path: filepath.Clean(evidence.BinaryPath), SizeBytes: artifactSize, SHA256: artifactSHA},
			SPIRVBundleSHA256: spirvSHA, SPIRVModuleCount: spirvCount,
			Toolchain: toolchain, ToolchainSHA256: toolchainSHA,
			NormalizedBuildCommand: commands, BuildCommandSHA256: commandSHA,
			StableIdentitySHA256: stableSHA, GitExecutableSHA256: gitSHA,
		}}, nil
}

func hashCurrentSPIRVBundleV5(root string) (string, int, VulkanIndexerScoreContract, error) {
	root, err := strictAbsoluteDirectory(root, "runtime SPIR-V root")
	if err != nil {
		return "", 0, VulkanIndexerScoreContract{}, err
	}
	expectedStems := CurrentVulkanShaderRegistryV5()
	expected := make(map[string]struct{}, len(expectedStems))
	for _, stem := range expectedStems {
		if stem == "" || filepath.Base(stem) != stem || filepath.Ext(stem) != "" {
			return "", 0, VulkanIndexerScoreContract{}, fmt.Errorf("invalid expected SPIR-V module stem %q", stem)
		}
		name := stem + ".spv"
		if _, exists := expected[name]; exists {
			return "", 0, VulkanIndexerScoreContract{}, fmt.Errorf("duplicate expected SPIR-V module %q", stem)
		}
		expected[name] = struct{}{}
	}
	entriesBefore, err := exactSPIRVEntries(root, expected)
	if err != nil {
		return "", 0, VulkanIndexerScoreContract{}, err
	}
	var score VulkanIndexerScoreContract
	h := sha256.New()
	for _, name := range entriesBefore {
		data, err := readStableRegularFile(filepath.Join(root, name), "SPIR-V module "+name)
		if err != nil {
			return "", 0, VulkanIndexerScoreContract{}, err
		}
		if name == "v41_indexer_score.spv" {
			score, err = inspectVulkanIndexerScoreSPIRV(data)
			if err != nil {
				return "", 0, VulkanIndexerScoreContract{}, err
			}
		}
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(filepath.Join("internal", "compute", "spirv", name)), len(data))
		_, _ = h.Write(data)
	}
	entriesAfter, err := exactSPIRVEntries(root, expected)
	if err != nil || strings.Join(entriesBefore, "\x00") != strings.Join(entriesAfter, "\x00") {
		return "", 0, VulkanIndexerScoreContract{}, fmt.Errorf("runtime SPIR-V directory changed during hashing")
	}
	return hex.EncodeToString(h.Sum(nil)), len(entriesBefore), score, nil
}

func compareReceiptV5ToObservation(receipt ComputeBuildReceipt, observed observedVulkanReceiptV5Identity) error {
	if err := validateVulkanReceiptV5Success(&receipt); err != nil {
		return err
	}
	if *receipt.VulkanNativeArchive != observed.NativeArchive || *receipt.VulkanIndexerScore != observed.IndexerScore {
		return fmt.Errorf("Vulkan V5 native archive or indexer score contract mismatch")
	}
	if !samePath(receipt.ReceiptPath, observed.ReceiptPath) {
		return fmt.Errorf("Vulkan receipt snapshot path mismatch")
	}
	if receipt.Vulkan.Source != observed.Source {
		return fmt.Errorf("Vulkan receipt source identity mismatch")
	}
	if receipt.Vulkan.SPIRVBundleSHA256 != observed.SPIRVBundleSHA256 || receipt.Vulkan.SPIRVModuleCount != observed.SPIRVModuleCount {
		return fmt.Errorf("Vulkan receipt SPIR-V identity mismatch")
	}
	if !sameToolIdentities(receipt.Vulkan.Toolchain, observed.Toolchain) || receipt.Vulkan.ToolchainSHA256 != observed.ToolchainSHA256 {
		return fmt.Errorf("Vulkan receipt toolchain identity mismatch")
	}
	if strings.Join(receipt.Vulkan.NormalizedBuildCommand, "\x00") != strings.Join(observed.NormalizedBuildCommand, "\x00") || receipt.Vulkan.BuildCommandSHA256 != observed.BuildCommandSHA256 {
		return fmt.Errorf("Vulkan receipt build-plan identity mismatch")
	}
	if !samePath(receipt.Artifact.Path, observed.Artifact.Path) || receipt.Artifact.SizeBytes != observed.Artifact.SizeBytes || receipt.Artifact.SHA256 != observed.Artifact.SHA256 || receipt.Artifact.Signed {
		return fmt.Errorf("Vulkan receipt binary identity mismatch")
	}
	if receipt.Vulkan.StableIdentitySHA256 != observed.StableIdentitySHA256 {
		return fmt.Errorf("Vulkan receipt stable identity mismatch")
	}
	return nil
}

// compareVulkanBuildReceiptV5 compares a current build only within V5. This
// comparison is build reproducibility metadata, not independent authentication
// of a historical receipt; that requires the evidence verifier above.
func compareVulkanBuildReceiptV5(path string, artifact *BuildArtifact, provenance *VulkanBuildProvenance, registry *VulkanShaderRegistryIdentity, archive *BuildArtifact, score *VulkanIndexerScoreContract) (*BuildReproducibility, error) {
	current := ComputeBuildReceipt{
		Schema: VulkanBuildReceiptSchemaV5, Backend: "vulkan", Command: "binary", Outcome: "success",
		Artifact: artifact, Vulkan: provenance, VulkanRegistry: registry, VulkanNativeArchive: archive, VulkanIndexerScore: score,
	}
	if err := validateVulkanReceiptV5Success(&current); err != nil {
		return &BuildReproducibility{Status: "invalid"}, fmt.Errorf("current Vulkan V5 build evidence: %w", err)
	}
	if path == "" {
		return &BuildReproducibility{Status: "baseline"}, nil
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return &BuildReproducibility{Status: "invalid"}, err
	}
	raw, err := readStableRegularFile(path, "comparison Vulkan V5 receipt")
	if err != nil {
		return &BuildReproducibility{Status: "invalid"}, err
	}
	digest := sha256.Sum256(raw)
	rep := &BuildReproducibility{Status: "mismatch", ComparedReceiptSHA256: hex.EncodeToString(digest[:])}
	prior, err := decodeStrictVulkanReceiptV5(raw)
	if err != nil {
		rep.Status = "invalid"
		return rep, err
	}
	if err := validateVulkanReceiptV5Success(&prior); err != nil {
		rep.Status = "invalid"
		return rep, err
	}
	checks := []struct {
		name string
		same bool
	}{
		{"registry", *registry == *prior.VulkanRegistry},
		{"native_archive", *archive == *prior.VulkanNativeArchive},
		{"indexer_score_contract", *score == *prior.VulkanIndexerScore},
		{"source", provenance.Source == prior.Vulkan.Source},
		{"spirv_bundle", provenance.SPIRVBundleSHA256 == prior.Vulkan.SPIRVBundleSHA256},
		{"spirv_module_count", provenance.SPIRVModuleCount == prior.Vulkan.SPIRVModuleCount},
		{"toolchain", provenance.ToolchainSHA256 == prior.Vulkan.ToolchainSHA256 && sameToolIdentities(provenance.Toolchain, prior.Vulkan.Toolchain)},
		{"build_command", provenance.BuildCommandSHA256 == prior.Vulkan.BuildCommandSHA256 && strings.Join(provenance.NormalizedBuildCommand, "\x00") == strings.Join(prior.Vulkan.NormalizedBuildCommand, "\x00")},
		{"binary", artifact.SHA256 == prior.Artifact.SHA256 && artifact.SizeBytes == prior.Artifact.SizeBytes},
		{"stable_identity", provenance.StableIdentitySHA256 == prior.Vulkan.StableIdentitySHA256},
		{"legacy_provenance", prior.GitCommit == "" && prior.GitRef == "" && prior.Clean == nil && prior.SourceArchiveSHA256 == "" && prior.ShaderBundleSHA256 == "" && len(prior.BuildArgs) == 0 && prior.Toolchain == nil},
	}
	for _, check := range checks {
		if !check.same {
			rep.MismatchedFields = append(rep.MismatchedFields, check.name)
		}
	}
	if len(rep.MismatchedFields) != 0 {
		return rep, fmt.Errorf("Vulkan build reproducibility mismatch: %s", strings.Join(rep.MismatchedFields, ", "))
	}
	rep.Status = "match"
	return rep, nil
}

type observedVulkanReceiptV5Identity struct {
	observedVulkanReceiptIdentity
	NativeArchive BuildArtifact
	IndexerScore  VulkanIndexerScoreContract
}

func sameObservedVulkanV5Identity(a, b observedVulkanReceiptV5Identity) bool {
	return a.NativeArchive == b.NativeArchive && a.IndexerScore == b.IndexerScore && sameObservedVulkanIdentity(a.observedVulkanReceiptIdentity, b.observedVulkanReceiptIdentity)
}

func finalizeVulkanBinaryV5(cfg *VulkanConfig, source BuildSourceProvenance, outBinPath string, expectedTools []BuildToolIdentity, expectedToolchainSHA string) (*BuildArtifact, *VulkanBuildProvenance, *BuildArtifact, *VulkanIndexerScoreContract, error) {
	binarySHA, binarySize, err := stableRegularFileSHA256(outBinPath, "Vulkan output binary")
	if err != nil {
		return nil, nil, nil, nil, err
	}
	spirvSHA, spirvCount, score, err := hashCurrentSPIRVBundleV5(filepath.Join(cfg.RepoRoot, "internal", "compute", "spirv"))
	if err != nil {
		return nil, nil, nil, nil, err
	}
	archive, err := observeVulkanNativeArchiveV5(cfg.RepoRoot)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	tools, toolchainSHA, err := vulkanToolchainIdentity(cfg.Toolchain)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if toolchainSHA != expectedToolchainSHA || !sameToolIdentities(tools, expectedTools) {
		return nil, nil, nil, nil, fmt.Errorf("Vulkan toolchain changed during build")
	}
	commands, commandSHA, err := normalizedVulkanBuildCommand(cfg)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	stableSHA, err := vulkanStableIdentityV5SHA(source, spirvSHA, spirvCount, toolchainSHA, commandSHA, binarySHA, archive, score)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	artifact := &BuildArtifact{Path: outBinPath, SizeBytes: binarySize, SHA256: binarySHA}
	provenance := &VulkanBuildProvenance{
		Source: source, SPIRVBundleSHA256: spirvSHA, SPIRVModuleCount: spirvCount,
		Toolchain: expectedTools, ToolchainSHA256: toolchainSHA,
		NormalizedBuildCommand: commands, BuildCommandSHA256: commandSHA, StableIdentitySHA256: stableSHA,
	}
	return artifact, provenance, &archive, &score, nil
}
