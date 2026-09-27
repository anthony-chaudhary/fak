package main

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/agent"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

const turnkeyMTPEvidenceCatalogSchema = "fak.up.mtp-evidence-catalog/1"

// Code-owned identities of the turnkey Metal MTP execution contract. A reviewed
// record binds these exact strings; bump a version whenever the named contract
// changes so older evidence stops matching (fail closed) instead of silently
// qualifying different code. They are emitted only after the live facts they
// describe are observed (Metal live, a resident Qwen3.8 hybrid MTP head).
const (
	// turnkeyMTPMetalCompatibility names the Metal P4 target-verify panel the
	// coordinator executes (model.Qwen35MetalMTPVerifyPanelPath).
	turnkeyMTPMetalCompatibility = "fak-metal/qwen38-mtp-p4-target-verify/v1"
	// turnkeyMTPForwardCompatibility names the fak-native Qwen3.8 hybrid Session
	// forward the planner decodes with on Metal ("metal/qwen35-hybrid-session-v1").
	turnkeyMTPForwardCompatibility = "fak-native/qwen38-hybrid-session/v1"
	turnkeyMTPNativeConfigSchema   = "fak.up.mtp-native-config/1"
	turnkeyMTPFamily               = "Qwen3.8"
)

//go:embed up_mtp_evidence_catalog.json
var turnkeyMTPEvidenceCatalogJSON []byte

type turnkeyMTPWorkloadEnvelope struct {
	Greedy        bool `json:"greedy"`
	ContextTokens int  `json:"context_tokens"`
	DraftDepth    int  `json:"draft_depth"`
}

// turnkeyMTPQualificationKey is the exact compatibility identity shared by a
// reviewed record and a runtime request. Measured headroom is kept separate.
type turnkeyMTPQualificationKey struct {
	ModelArtifactSHA256 string                         `json:"model_artifact_sha256"`
	ModelFamily         string                         `json:"model_family"`
	MTPFormat           fakmodel.Qwen38MTPTensorFormat `json:"mtp_format"`
	DeviceCompatibility string                         `json:"device_compatibility"`
	MetalCompatibility  string                         `json:"metal_compatibility"`
	EngineCompatibility string                         `json:"engine_compatibility"`
	NativeConfigSHA256  string                         `json:"native_config_sha256"`
	Workload            turnkeyMTPWorkloadEnvelope     `json:"workload"`
}

type turnkeyMTPQualificationContext struct {
	Qualification          turnkeyMTPQualificationKey `json:"qualification"`
	AvailableHeadroomBytes uint64                     `json:"available_headroom_bytes"`
}

type turnkeyMTPQualificationRefusal string

const (
	turnkeyMTPQualificationAvailable turnkeyMTPQualificationRefusal = ""
	turnkeyMTPNoEligibleContext      turnkeyMTPQualificationRefusal = "NO_ELIGIBLE_QUALIFICATION_CONTEXT"
	turnkeyMTPMalformedCatalog       turnkeyMTPQualificationRefusal = "MALFORMED_TRUSTED_CATALOG"
	turnkeyMTPAmbiguousQualification turnkeyMTPQualificationRefusal = "AMBIGUOUS_QUALIFICATION"
)

type turnkeyMTPQualificationSelection struct {
	Evidence              fakmodel.Qwen38MTPCanaryEvidence `json:"evidence"`
	ReceiptID             string                           `json:"receipt_id"`
	EvidenceSHA256        string                           `json:"evidence_sha256"`
	SourceURL             string                           `json:"source_url"`
	SourceRevision        string                           `json:"source_revision"`
	TestedSourceRevision  string                           `json:"tested_source_revision"`
	Qualification         turnkeyMTPQualificationKey       `json:"qualification"`
	RequiredHeadroomBytes uint64                           `json:"required_headroom_bytes"`
	RuntimeContext        turnkeyMTPQualificationContext   `json:"runtime_context"`
}

type turnkeyMTPQualificationResult struct {
	Selection *turnkeyMTPQualificationSelection `json:"selection,omitempty"`
	Refusal   turnkeyMTPQualificationRefusal    `json:"refusal,omitempty"`
	Detail    string                            `json:"detail,omitempty"`
}

type turnkeyMTPEvidenceCatalog struct {
	Schema      string                         `json:"schema"`
	Records     []turnkeyMTPEvidenceRecord     `json:"records"`
	Revocations []turnkeyMTPEvidenceRevocation `json:"revocations"`
}

type turnkeyMTPEvidenceRecord struct {
	Evidence             json.RawMessage            `json:"evidence"`
	EvidenceSHA256       string                     `json:"evidence_sha256"`
	SourceURL            string                     `json:"source_url"`
	SourceRevision       string                     `json:"source_revision"`
	TestedSourceRevision string                     `json:"tested_source_revision"`
	Qualification        turnkeyMTPQualificationKey `json:"qualification"`
}

type turnkeyMTPEvidenceRevocation struct {
	ReceiptID      string `json:"receipt_id"`
	EvidenceSHA256 string `json:"evidence_sha256"`
}

func embeddedTurnkeyMTPQualification(now time.Time, runtime turnkeyMTPQualificationContext) turnkeyMTPQualificationResult {
	return selectTurnkeyMTPQualificationCatalog(turnkeyMTPEvidenceCatalogJSON, now, runtime)
}

// selectTurnkeyMTPQualificationCatalog validates the complete reviewed catalog
// before selecting anything. It returns evidence bound to the exact runtime
// context; registration and activation are deliberately left to the caller.
func selectTurnkeyMTPQualificationCatalog(raw []byte, now time.Time, runtime turnkeyMTPQualificationContext) turnkeyMTPQualificationResult {
	if err := validateTurnkeyMTPQualificationContext(runtime); err != nil {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, err)
	}
	records, revoked, err := decodeTurnkeyMTPEvidenceCatalog(raw)
	if err != nil {
		return turnkeyMTPRefusal(turnkeyMTPMalformedCatalog, err)
	}

	var selected []*turnkeyMTPQualificationSelection
	for i := range records {
		record := records[i]
		if !turnkeyMTPRecordLive(record, revoked, now) {
			continue
		}
		if !turnkeyMTPQualificationMatches(record, runtime) {
			continue
		}
		selection := record
		selection.RuntimeContext = runtime
		selected = append(selected, &selection)
	}
	if len(selected) == 0 {
		return turnkeyMTPQualificationResult{Refusal: turnkeyMTPNoEligibleContext}
	}
	if len(selected) != 1 {
		return turnkeyMTPQualificationResult{Refusal: turnkeyMTPAmbiguousQualification, Detail: "multiple reviewed records match the exact runtime context"}
	}
	return turnkeyMTPQualificationResult{Selection: selected[0]}
}

func decodeTurnkeyMTPEvidenceCatalog(raw []byte) ([]turnkeyMTPQualificationSelection, map[string]bool, error) {
	var catalog turnkeyMTPEvidenceCatalog
	if err := decodeStrictJSON(raw, &catalog); err != nil {
		return nil, nil, fmt.Errorf("decode catalog: %w", err)
	}
	if catalog.Schema != turnkeyMTPEvidenceCatalogSchema {
		return nil, nil, fmt.Errorf("catalog schema %q, want %q", catalog.Schema, turnkeyMTPEvidenceCatalogSchema)
	}
	revoked := make(map[string]bool, len(catalog.Revocations))
	for i, tombstone := range catalog.Revocations {
		if tombstone.ReceiptID == "" || tombstone.ReceiptID != strings.TrimSpace(tombstone.ReceiptID) || !turnkeyMTPIsLowerSHA256(tombstone.EvidenceSHA256) {
			return nil, nil, fmt.Errorf("revocation %d has an invalid identity", i)
		}
		key := tombstone.ReceiptID + "\x00" + tombstone.EvidenceSHA256
		if revoked[key] {
			return nil, nil, fmt.Errorf("duplicate revocation for receipt %q", tombstone.ReceiptID)
		}
		revoked[key] = true
	}

	receiptIDs := make(map[string]bool, len(catalog.Records))
	bindings := make(map[string]bool, len(catalog.Records))
	records := make([]turnkeyMTPQualificationSelection, 0, len(catalog.Records))
	for i, record := range catalog.Records {
		var evidence fakmodel.Qwen38MTPCanaryEvidence
		if err := decodeStrictJSON(record.Evidence, &evidence); err != nil {
			return nil, nil, fmt.Errorf("record %d evidence: %w", i, err)
		}
		if err := evidence.Validate(); err != nil {
			return nil, nil, fmt.Errorf("record %d evidence: %w", i, err)
		}
		actual := sha256.Sum256(record.Evidence)
		// EvidenceSHA256 is the publication binding: it hashes the exact embedded
		// evidence JSON value bytes that this loader validates and returns.
		if !turnkeyMTPIsLowerSHA256(record.EvidenceSHA256) || record.EvidenceSHA256 != hex.EncodeToString(actual[:]) {
			return nil, nil, fmt.Errorf("record %d evidence digest mismatch", i)
		}
		if err := validateTurnkeyMTPQualificationKey(record.Qualification); err != nil {
			return nil, nil, fmt.Errorf("record %d context: %w", i, err)
		}
		if record.Qualification.Workload.DraftDepth != evidence.Receipt.Envelope.DraftDepth {
			return nil, nil, fmt.Errorf("record %d draft depth does not match evidence", i)
		}
		if normalizedSHA256(evidence.Receipt.Envelope.ArtifactHash) != record.Qualification.ModelArtifactSHA256 ||
			evidence.Receipt.Envelope.ModelFamily != record.Qualification.ModelFamily ||
			evidence.Receipt.Envelope.Format != record.Qualification.MTPFormat || evidence.Receipt.Envelope.Backend != fakmodel.Qwen38MTPBackendMetal {
			return nil, nil, fmt.Errorf("record %d evidence envelope does not match qualification context", i)
		}
		if err := validateImmutableTurnkeyMTPSource(record.SourceURL, record.SourceRevision); err != nil {
			return nil, nil, fmt.Errorf("record %d source: %w", i, err)
		}
		if !turnkeyMTPIsLowerHex(record.TestedSourceRevision, 40) {
			return nil, nil, fmt.Errorf("record %d tested source revision must be a full lowercase commit", i)
		}
		receiptID := strings.TrimSpace(evidence.Receipt.ReceiptID)
		if receiptID == "" || evidence.Receipt.ReceiptID != receiptID {
			return nil, nil, fmt.Errorf("record %d receipt id is not canonical", i)
		}
		if receiptIDs[receiptID] {
			return nil, nil, fmt.Errorf("duplicate receipt id %q", receiptID)
		}
		receiptIDs[receiptID] = true
		binding := record.EvidenceSHA256
		if bindings[binding] {
			return nil, nil, fmt.Errorf("duplicate evidence binding for receipt %q", receiptID)
		}
		bindings[binding] = true
		records = append(records, turnkeyMTPQualificationSelection{
			Evidence: evidence, ReceiptID: receiptID, EvidenceSHA256: record.EvidenceSHA256,
			SourceURL: record.SourceURL, SourceRevision: record.SourceRevision,
			TestedSourceRevision: record.TestedSourceRevision, Qualification: record.Qualification,
			RequiredHeadroomBytes: evidence.Receipt.Envelope.HeadroomBytes,
		})
	}
	return records, revoked, nil
}

// turnkeyMTPRecordLive reports whether a decoded record is unrevoked and inside
// its observation/expiry window at now.
func turnkeyMTPRecordLive(record turnkeyMTPQualificationSelection, revoked map[string]bool, now time.Time) bool {
	if revoked[record.ReceiptID+"\x00"+record.EvidenceSHA256] {
		return false
	}
	return !now.IsZero() && !record.Evidence.ObservedAt.After(now) && now.Before(record.Evidence.ValidUntil)
}

// turnkeyMTPCatalogHasLiveRecordFor reports whether a well-formed catalog carries
// a live record for this exact device identity. Startup retains the MTP head only
// then, so the empty shipped catalog (and every other host) keeps the historical
// head-dropping load byte-for-byte.
func turnkeyMTPCatalogHasLiveRecordFor(raw []byte, now time.Time, device string) bool {
	if device == "" {
		return false
	}
	records, revoked, err := decodeTurnkeyMTPEvidenceCatalog(raw)
	if err != nil {
		return false
	}
	for _, record := range records {
		if record.Qualification.DeviceCompatibility == device && turnkeyMTPRecordLive(record, revoked, now) {
			return true
		}
	}
	return false
}

// turnkeyMTPSplitShard matches a multi-file GGUF shard name at any digit width,
// exactly as the loader (ggufload shardSuffixRe) opens a split set. Only the named
// shard would be hashed, leaving the others unbound, so split artifacts are not
// qualified.
var turnkeyMTPSplitShard = regexp.MustCompile(`-\d+-of-\d+\.gguf$`)

// turnkeyMTPRuntimeFacts are the live startup observations a qualification
// context is derived from: the loaded model and artifact path, the resolved
// Metal device, the host, the built planner's fixed settings, and the Metal MTP
// coordinator config that admission would install.
type turnkeyMTPRuntimeFacts struct {
	Model        *fakmodel.Model
	ArtifactPath string
	// ArtifactBefore is the artifact's file identity observed before the load; the
	// digest must be of that same file, not one swapped in afterwards.
	ArtifactBefore os.FileInfo
	// MemoryOverride is set when an operator memory override (FAK_UP_MEMORY_BYTES
	// or FAK_UP_AVAILABLE_BYTES) replaced the host probe: the device identity and
	// headroom are then not live facts, so qualification refuses.
	MemoryOverride     bool
	MetalLive          bool
	MetalDevice        string
	HostTotalBytes     int64
	HostAvailableBytes int64
	ResidentQ4K        bool
	Planner            agent.InKernelPlannerConfig
	MTP                fakmodel.MetalMTPConfig
}

// turnkeyMTPNativeConfig is the canonical execution-affecting native config a
// reviewed record binds through NativeConfigSHA256. Prompt-shaping and cache
// budget knobs are deliberately excluded: they do not change decode kernels.
type turnkeyMTPNativeConfig struct {
	Schema                    string                    `json:"schema"`
	Backend                   fakmodel.Qwen38MTPBackend `json:"backend"`
	ResidentQ4K               bool                      `json:"resident_q4k"`
	ContextTokens             int                       `json:"context_tokens"`
	KVPrecision               fakmodel.KVPrecision      `json:"kv_precision"`
	QwenQ4KPrefillChunkTokens int                       `json:"qwen_q4k_prefill_chunk_tokens"`
	Qwen35MetalGDNSequence    bool                      `json:"qwen35_metal_gdn_sequence"`
	Q4KGateUpOutputSlab       bool                      `json:"q4k_gate_up_output_slab"`
	DenseGPULayers            int                       `json:"dense_gpu_layers"`
	MTP                       fakmodel.MetalMTPConfig   `json:"mtp"`
}

func turnkeyMTPNativeConfigSHA256(facts turnkeyMTPRuntimeFacts) (string, error) {
	kvPrecision := facts.Planner.KVPrecision
	if kvPrecision == "" {
		// The zero value is the f32 cache; bind one spelling for one behavior.
		kvPrecision = fakmodel.KVPrecisionFP32
	}
	raw, err := json.Marshal(turnkeyMTPNativeConfig{
		Schema: turnkeyMTPNativeConfigSchema, Backend: fakmodel.Qwen38MTPBackendMetal,
		ResidentQ4K: facts.ResidentQ4K, ContextTokens: facts.Planner.ContextTokens,
		KVPrecision: kvPrecision, QwenQ4KPrefillChunkTokens: facts.Planner.QwenQ4KPrefillChunkTokens,
		Qwen35MetalGDNSequence: facts.Planner.Qwen35MetalGDNSequence, Q4KGateUpOutputSlab: facts.Planner.Q4KGateUpOutputSlab,
		DenseGPULayers: facts.Planner.DenseGPULayers, MTP: facts.MTP,
	})
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

// turnkeyMTPDeviceCompatibility derives the device identity from the live Metal
// device name and physical memory, e.g. "Apple M3 Pro" + 36 GiB ->
// "apple/m3-pro-36gb/v1". It returns "" (fail closed) for a non-Apple or unknown
// device, or memory that is not a positive whole number of GiB.
func turnkeyMTPDeviceCompatibility(metalDevice string, totalBytes int64) string {
	name, ok := strings.CutPrefix(strings.TrimSpace(metalDevice), "Apple ")
	if !ok || totalBytes <= 0 || totalBytes%(1<<30) != 0 {
		return ""
	}
	slug := strings.Join(strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}), "-")
	if slug == "" {
		return ""
	}
	return fmt.Sprintf("apple/%s-%dgb/v1", slug, totalBytes>>30)
}

// turnkeyMTPLiveQualification derives every qualification field except the artifact
// digest from live facts. An underivable fact is a refusal, never a default.
func turnkeyMTPLiveQualification(facts turnkeyMTPRuntimeFacts) (turnkeyMTPQualificationContext, error) {
	var runtime turnkeyMTPQualificationContext
	if !facts.MetalLive {
		return runtime, errors.New("the Metal backend is not live")
	}
	if facts.Model == nil {
		return runtime, errors.New("no loaded model")
	}
	if facts.MemoryOverride {
		return runtime, errors.New("a host memory override (FAK_UP_MEMORY_BYTES/FAK_UP_AVAILABLE_BYTES) replaces the live device and headroom facts")
	}
	if turnkeyMTPSplitShard.MatchString(facts.ArtifactPath) {
		return runtime, fmt.Errorf("split GGUF artifact %q cannot be bound by one digest", facts.ArtifactPath)
	}
	if facts.Planner.ContextTokens <= 0 {
		return runtime, errors.New("the planner context bound is unknown")
	}
	cfg := facts.Model.Cfg
	if !cfg.IsQwen35Hybrid() || !strings.Contains(strings.ToLower(cfg.Name), "qwen3.8") {
		return runtime, fmt.Errorf("loaded model %q is not a Qwen3.8 hybrid", cfg.Name)
	}
	layout, err := facts.Model.Qwen38MTPTensorLayout()
	if err != nil {
		return runtime, fmt.Errorf("MTP head is not resident: %w", err)
	}
	device := turnkeyMTPDeviceCompatibility(facts.MetalDevice, facts.HostTotalBytes)
	if device == "" {
		return runtime, fmt.Errorf("device identity unavailable (metal device %q, host memory %d bytes)", facts.MetalDevice, facts.HostTotalBytes)
	}
	nativeConfig, err := turnkeyMTPNativeConfigSHA256(facts)
	if err != nil {
		return runtime, fmt.Errorf("native config identity: %w", err)
	}
	runtime.Qualification = turnkeyMTPQualificationKey{
		ModelFamily: turnkeyMTPFamily, MTPFormat: layout.Format,
		DeviceCompatibility: device, MetalCompatibility: turnkeyMTPMetalCompatibility,
		EngineCompatibility: turnkeyMTPForwardCompatibility, NativeConfigSHA256: nativeConfig,
		Workload: turnkeyMTPWorkloadEnvelope{
			Greedy: facts.MTP.EnforceGreedyTripwire, ContextTokens: facts.Planner.ContextTokens, DraftDepth: facts.MTP.DraftDepth,
		},
	}
	if facts.HostAvailableBytes > 0 {
		runtime.AvailableHeadroomBytes = uint64(facts.HostAvailableBytes)
	}
	return runtime, nil
}

// qualifyTurnkeyMTP derives the runtime context from live startup facts and
// selects the reviewed record for it. The loaded artifact is hashed (a full
// streaming SHA-256) only after a live record matches every other key field, so
// an empty or non-matching catalog costs startup no artifact read. The derived
// context is returned alongside the result so startup can report it.
func qualifyTurnkeyMTP(raw []byte, now time.Time, facts turnkeyMTPRuntimeFacts, hashArtifact func(string) (string, error)) (turnkeyMTPQualificationResult, turnkeyMTPQualificationContext) {
	// The catalog is validated first so a malformed catalog is always reported as
	// such, even when the head was (correctly) not retained for it.
	records, revoked, catalogErr := decodeTurnkeyMTPEvidenceCatalog(raw)
	runtime, err := turnkeyMTPLiveQualification(facts)
	if catalogErr != nil {
		return turnkeyMTPRefusal(turnkeyMTPMalformedCatalog, catalogErr), runtime
	}
	anyLive := false
	for _, record := range records {
		anyLive = anyLive || turnkeyMTPRecordLive(record, revoked, now)
	}
	if !anyLive {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, errors.New("the reviewed catalog has no live record")), runtime
	}
	if err != nil {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, err), runtime
	}
	candidate := false
	for _, record := range records {
		key := record.Qualification
		key.ModelArtifactSHA256 = ""
		if key == runtime.Qualification && turnkeyMTPRecordLive(record, revoked, now) {
			candidate = true
			break
		}
	}
	if !candidate {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, fmt.Errorf("no live reviewed record for %s %s on %s (native config %s, context %d, depth %d)",
			runtime.Qualification.ModelFamily, runtime.Qualification.MTPFormat, runtime.Qualification.DeviceCompatibility,
			runtime.Qualification.NativeConfigSHA256[:12], runtime.Qualification.Workload.ContextTokens, runtime.Qualification.Workload.DraftDepth)), runtime
	}
	if hashArtifact == nil {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, errors.New("artifact digest is unavailable")), runtime
	}
	digest, err := hashArtifact(facts.ArtifactPath)
	if err != nil {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, fmt.Errorf("artifact digest: %w", err)), runtime
	}
	if after, statErr := os.Stat(facts.ArtifactPath); facts.ArtifactBefore == nil || statErr != nil || !os.SameFile(facts.ArtifactBefore, after) ||
		after.Size() != facts.ArtifactBefore.Size() || !after.ModTime().Equal(facts.ArtifactBefore.ModTime()) {
		return turnkeyMTPRefusal(turnkeyMTPNoEligibleContext, fmt.Errorf("artifact digest: %q is not the file observed before the load", facts.ArtifactPath)), runtime
	}
	runtime.Qualification.ModelArtifactSHA256 = digest
	result := selectTurnkeyMTPQualificationCatalog(raw, now, runtime)
	if result.Refusal == turnkeyMTPNoEligibleContext && result.Detail == "" {
		result.Detail = fmt.Sprintf("no live reviewed record binds artifact sha256:%s with %d bytes of available headroom", digest[:12], runtime.AvailableHeadroomBytes)
	}
	return result, runtime
}

// turnkeyMTPArtifactSHA256 streams the loaded artifact through SHA-256 and fails
// closed when it is not a regular file or changes while being hashed.
func turnkeyMTPArtifactSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !before.Mode().IsRegular() {
		return "", fmt.Errorf("artifact %q is not a regular file", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	after, err := os.Stat(path)
	if err != nil || !os.SameFile(before, after) || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		return "", fmt.Errorf("artifact %q changed while hashing", path)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// admitTurnkeyMTPSelection binds a selected reviewed record to the planner's
// witnessed canary admission. The request envelope carries live facts (measured
// headroom, the observed digest and MTP format) and OperatorOptIn stays false, so
// admission rests on the reviewed evidence alone. The runtime context only exists
// because the MTP head is resident in the qualified format, so the model is ready.
// HeadroomBytes is the post-load measurement: the canary re-checks that startup
// value per request, it does not re-measure live memory. The planner keeps the
// resulting admission (read back through MetalMTPAdmitted/Qwen38MTPCanaryResult).
func admitTurnkeyMTPSelection(planner *agent.InKernelPlanner, selection *turnkeyMTPQualificationSelection, cfg fakmodel.MetalMTPConfig) error {
	manager := fakmodel.NewQwen38MTPCanaryManager()
	if err := manager.RegisterCanaryEvidence(selection.Evidence); err != nil {
		return err
	}
	key := selection.RuntimeContext.Qualification
	_, err := planner.ConfigureQwen38MTPCanary(manager, fakmodel.Qwen38CanaryRequest{
		Envelope: fakmodel.Qwen38CanaryEnvelope{
			ModelFamily: key.ModelFamily, Format: key.MTPFormat, Backend: fakmodel.Qwen38MTPBackendMetal,
			HeadroomBytes: selection.RuntimeContext.AvailableHeadroomBytes,
			ArtifactHash:  key.ModelArtifactSHA256, DraftDepth: key.Workload.DraftDepth,
		},
		EvidenceReceiptID: selection.ReceiptID,
		ModelReady:        true,
	}, nil, cfg)
	return err
}

func turnkeyMTPQualificationMatches(record turnkeyMTPQualificationSelection, runtime turnkeyMTPQualificationContext) bool {
	return record.Qualification == runtime.Qualification &&
		runtime.AvailableHeadroomBytes >= record.RequiredHeadroomBytes
}

func validateTurnkeyMTPQualificationContext(context turnkeyMTPQualificationContext) error {
	return validateTurnkeyMTPQualificationKey(context.Qualification)
}

func validateTurnkeyMTPQualificationKey(key turnkeyMTPQualificationKey) error {
	if !turnkeyMTPIsLowerSHA256(key.ModelArtifactSHA256) || !turnkeyMTPIsLowerSHA256(key.NativeConfigSHA256) {
		return fmt.Errorf("qualification context requires lowercase SHA-256 artifact and native-config identities")
	}
	if key.ModelFamily != "Qwen3.8" || key.MTPFormat == "" {
		return fmt.Errorf("qualification context requires model family and MTP format")
	}
	for name, value := range map[string]string{
		"device": key.DeviceCompatibility, "metal": key.MetalCompatibility, "engine": key.EngineCompatibility,
	} {
		if value != strings.TrimSpace(value) || !isVersionedCompatibility(value) {
			return fmt.Errorf("qualification context %s compatibility %q is not an exact versioned identity", name, value)
		}
	}
	if !key.Workload.Greedy || key.Workload.ContextTokens <= 0 || key.Workload.DraftDepth <= 0 || key.Workload.DraftDepth > fakmodel.Qwen35MTPMaxDraftDepth {
		return fmt.Errorf("qualification context requires greedy decode and supported positive context/depth")
	}
	return nil
}

func isVersionedCompatibility(value string) bool {
	value = strings.TrimSpace(value)
	return value != "" && !strings.ContainsAny(value, "*?") && strings.Contains(value, "/v")
}

func validateImmutableTurnkeyMTPSource(rawURL, revision string) error {
	if !turnkeyMTPIsLowerHex(revision, 40) {
		return fmt.Errorf("source revision must be a full lowercase commit")
	}
	u, err := url.Parse(rawURL)
	if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Host != u.Hostname() {
		return fmt.Errorf("source URL is not immutable HTTPS provenance")
	}
	parts := strings.Split(strings.Trim(u.EscapedPath(), "/"), "/")
	switch strings.ToLower(u.Hostname()) {
	case "github.com":
		if len(parts) < 5 || parts[2] != "blob" || parts[3] != revision {
			return fmt.Errorf("GitHub source URL must contain /owner/repo/blob/<full-commit>/<path>")
		}
	case "raw.githubusercontent.com":
		if len(parts) < 4 || parts[2] != revision {
			return fmt.Errorf("raw GitHub source URL must contain /owner/repo/<full-commit>/<path>")
		}
	default:
		return fmt.Errorf("source host is not in the immutable-public-source allowlist")
	}
	return nil
}

func decodeStrictJSON(raw []byte, dst any) error {
	if err := rejectTurnkeyMTPDuplicateJSONNames(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing JSON value")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

// rejectTurnkeyMTPDuplicateJSONNames walks every nested object before typed
// decoding so encoding/json's last-key-wins behavior cannot rewrite authority.
func rejectTurnkeyMTPDuplicateJSONNames(raw []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	var walk func(json.Token) error
	var next func() error
	next = func() error {
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		return walk(token)
	}
	walk = func(token json.Token) error {
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]struct{})
			for decoder.More() {
				nameToken, err := decoder.Token()
				if err != nil {
					return err
				}
				name, ok := nameToken.(string)
				if !ok {
					return fmt.Errorf("JSON object name is not a string")
				}
				canonicalName := strings.ToLower(name)
				if name != canonicalName || strings.Trim(name, "abcdefghijklmnopqrstuvwxyz0123456789_") != "" {
					return fmt.Errorf("JSON object name %q is not canonical lowercase", name)
				}
				if _, exists := seen[canonicalName]; exists {
					return fmt.Errorf("duplicate JSON object name %q", name)
				}
				seen[canonicalName] = struct{}{}
				if err := next(); err != nil {
					return err
				}
			}
			end, err := decoder.Token()
			if err != nil || end != json.Delim('}') {
				return fmt.Errorf("malformed JSON object")
			}
		case '[':
			for decoder.More() {
				if err := next(); err != nil {
					return err
				}
			}
			end, err := decoder.Token()
			if err != nil || end != json.Delim(']') {
				return fmt.Errorf("malformed JSON array")
			}
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
		return nil
	}
	return next()
}

func turnkeyMTPIsLowerSHA256(value string) bool { return turnkeyMTPIsLowerHex(value, sha256.Size*2) }

func turnkeyMTPIsLowerHex(value string, size int) bool {
	if len(value) != size || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func normalizedSHA256(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	return strings.TrimPrefix(value, "sha256:")
}

func turnkeyMTPRefusal(code turnkeyMTPQualificationRefusal, err error) turnkeyMTPQualificationResult {
	return turnkeyMTPQualificationResult{Refusal: code, Detail: turnkeyMTPBoundedDetail(err.Error())}
}

// turnkeyMTPBoundedDetail caps a refusal detail at 256 bytes without splitting a
// UTF-8 sequence (details carry GGUF names and paths).
func turnkeyMTPBoundedDetail(detail string) string {
	const limit = 256
	if len(detail) <= limit {
		return detail
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(detail[cut]) {
		cut--
	}
	return detail[:cut]
}
