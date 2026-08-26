package modelaccept

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/qwen38ladder"
)

// LadderEvidenceReasonCode is the closed vocabulary for checksum-admission
// failures. Callers can route on the code without parsing operator prose.
type LadderEvidenceReasonCode string

const (
	LadderEvidenceInvalidManifest    LadderEvidenceReasonCode = "LADDER_EVIDENCE_INVALID_MANIFEST"
	LadderEvidenceDuplicatePath      LadderEvidenceReasonCode = "LADDER_EVIDENCE_DUPLICATE_PATH"
	LadderEvidencePathTraversal      LadderEvidenceReasonCode = "LADDER_EVIDENCE_PATH_TRAVERSAL"
	LadderEvidenceMissingArtifact    LadderEvidenceReasonCode = "LADDER_EVIDENCE_MISSING_ARTIFACT"
	LadderEvidenceExtraArtifact      LadderEvidenceReasonCode = "LADDER_EVIDENCE_EXTRA_ARTIFACT"
	LadderEvidenceUnreadableArtifact LadderEvidenceReasonCode = "LADDER_EVIDENCE_UNREADABLE_ARTIFACT"
	LadderEvidenceChecksumMismatch   LadderEvidenceReasonCode = "LADDER_EVIDENCE_CHECKSUM_MISMATCH"
	LadderEvidenceIdentityMismatch   LadderEvidenceReasonCode = "LADDER_EVIDENCE_IDENTITY_MISMATCH"
)

const (
	qwen38ExactModel             = "Qwen/Qwen3.8-27B"
	qwen38ExactRevision          = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
	qwen38ExactCorpusID          = "qwen38-27b-semantic-answer-quality-v2"
	qwen38ExactCorpusSHA         = "4c1292921a94403cb0c1d03e7c7a359b230bf9145e44d211e6156d7dee552468"
	qwen38ExactEnvironmentSHA    = "7b8ee9c0fb632d0eb66e6b5c0cb0df3e7585be76e3b0c89ede2a5c904c94d60a"
	qwen38ExactBaselineRuntime   = "764328d54289e5685ac1ca12878c0c39d00e9c76"
	qwen38ExactCandidateRuntime  = "b0ce51b599718cb1a08a886ac61af928a5209b78"
	qwen38ExactBaselineP95MS     = 3378.019733
	qwen38ExactCandidateP95MS    = 376.181809
	qwen38ExactImprovementPct    = 88.86383624923602
	qwen38LadderEvidenceFilename = "evidence-complete.json"
)

var qwen38ExactArtifactHashes = map[string]string{
	"README.md":                  "f46b04533fd9dc3eecd2fa1b0ff1665b26fb35135cd106c3576de55e0463f9af",
	"corpus.json":                qwen38ExactCorpusSHA,
	"environment.json":           "10b860cc29e30ebe4edd2062b2ef1a8a7c6a59d11523d9ea64bc19dbad0dfe17",
	"evaluator-output.json":      "cb93e29fb55e61a1f30be99690d8cfedef0c6947be596b788502693f15955965",
	qwen38LadderEvidenceFilename: "b0fbcc8e042b09326c51ae6210ce5ce93d960415a30760ae6610b3d3d898face",
	"raw-run.json":               "c1794dde8635fabf1aed2ad9209e7617c25c80f977ceee92ca8c59812caf9ecf",
}

type LadderEvidenceOptions struct {
	Directory string
	Manifest  string
}

// LadderEvidenceReason is typed HOLD evidence. ExpectedSHA256 and
// ActualSHA256 are both populated for a checksum mismatch so the operator can
// distinguish stale evidence from a missing or unreadable artifact.
type LadderEvidenceReason struct {
	Code           LadderEvidenceReasonCode `json:"code"`
	Path           string                   `json:"path,omitempty"`
	ExpectedSHA256 string                   `json:"expected_sha256,omitempty"`
	ActualSHA256   string                   `json:"actual_sha256,omitempty"`
	Detail         string                   `json:"detail,omitempty"`
}

func (r LadderEvidenceReason) String() string {
	if r.Code == "" {
		return ""
	}
	parts := []string{string(r.Code)}
	if r.Path != "" {
		parts = append(parts, "path="+strconv.Quote(r.Path))
	}
	if r.ExpectedSHA256 != "" {
		parts = append(parts, "expected_sha256="+r.ExpectedSHA256)
	}
	if r.ActualSHA256 != "" {
		parts = append(parts, "actual_sha256="+r.ActualSHA256)
	}
	if r.Detail != "" {
		parts = append(parts, "detail="+strconv.Quote(r.Detail))
	}
	return strings.Join(parts, " ")
}

type LadderEvidenceAdmission struct {
	Verdict Verdict              `json:"verdict"`
	Reason  LadderEvidenceReason `json:"reason,omitempty"`
}

type ladderChecksumEntry struct {
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

// BuildInventoryWithLadderEvidence is the fail-closed readiness seam. It runs
// the existing inventory evaluator only after every declared artifact has been
// admitted byte-for-byte. A HOLD is constructed from declarations alone, so
// sample counts, observed values, and witnessed tiers cannot leak from evidence
// whose identity failed admission.
func BuildInventoryWithLadderEvidence(in Input, inventoryOpts InventoryOptions, evidenceOpts LadderEvidenceOptions) (Inventory, LadderEvidenceAdmission) {
	admission := VerifyLadderEvidence(evidenceOpts)
	if admission.Verdict == Pass {
		return BuildInventory(in, inventoryOpts), admission
	}
	return holdInventoryForLadderEvidence(in, inventoryOpts, admission.Reason), admission
}

// BuildQwen38LadderReadinessInventory admits the immutable #8623 packet and
// projects its one witnessed arithmetic-latency cell into readiness. The row
// and aggregate stay HOLD because the packet is not broad model acceptance.
func BuildQwen38LadderReadinessInventory(inventoryOpts InventoryOptions, evidenceOpts LadderEvidenceOptions) (Inventory, LadderEvidenceAdmission) {
	admission := VerifyLadderEvidence(evidenceOpts)
	if admission.Verdict != Pass {
		return holdQwen38LadderReadiness(inventoryOpts, evidenceOpts, admission.Reason), admission
	}
	evidence, reason := readQwen38LadderReadinessEvidence(inventoryOpts, evidenceOpts)
	if reason.Code != "" {
		admission = ladderEvidenceHold(reason)
		return holdQwen38LadderReadiness(inventoryOpts, evidenceOpts, reason), admission
	}
	artifact := strings.TrimSpace(inventoryOpts.Artifact)
	if artifact == "" {
		artifact = filepath.Join(evidenceOpts.Directory, qwen38LadderEvidenceFilename)
	}
	row := InventoryRow{
		Model:            evidence.Model,
		Family:           "Qwen3.8",
		Generation:       evidence.Revision,
		Lifecycle:        LifecycleLatest,
		CapabilityGate:   Hold,
		CorpusID:         evidence.CorpusID,
		DeclaredAt:       evidence.CapturedAt,
		Samples:          evidence.Correctness.Trials,
		Artifact:         artifact,
		ArtifactRevision: inventoryOpts.ArtifactRevision,
		Reasons:          []string{"only BF16 TP2 arithmetic latency is witnessed; production readiness remains HOLD"},
		ReadinessCells:   qwen38ReadinessCells(ReadinessCellPass),
		LadderEvidence:   &evidence,
	}
	return Inventory{
		Schema:   Qwen38LadderInventorySchema,
		Verdict:  Hold,
		CorpusID: evidence.CorpusID,
		Rows:     []InventoryRow{row},
		Semantics: &InventorySemantics{
			Default:     "generic readiness inventory remains the default; this exact ladder join is emitted only when explicitly selected",
			Replacement: "replace this row only with a reviewed immutable packet whose code-pinned identity and exact artifact hashes pass admission",
			Rollback:    "remove the ladder selector to return to generic inventory; this join changes no serving, model, or production default",
		},
	}, admission
}

func readQwen38LadderReadinessEvidence(inventoryOpts InventoryOptions, evidenceOpts LadderEvidenceOptions) (LadderReadinessEvidence, LadderEvidenceReason) {
	if strings.TrimSpace(inventoryOpts.ArtifactRevision) == "" {
		return LadderReadinessEvidence{}, qwen38IdentityReason("", "immutable artifact revision is required")
	}
	entries, reason := decodeLadderChecksumManifest(evidenceOpts.Manifest)
	if reason.Code != "" {
		return LadderReadinessEvidence{}, reason
	}
	if len(entries) != len(qwen38ExactArtifactHashes) {
		return LadderReadinessEvidence{}, qwen38IdentityReason("checksums.json", "manifest is not the six-file #8623 profile")
	}
	hashes := make([]ArtifactHash, 0, len(entries))
	seen := make(map[string]bool, len(entries))
	for _, entry := range entries {
		want, ok := qwen38ExactArtifactHashes[entry.File]
		if !ok || !strings.EqualFold(entry.SHA256, want) || seen[entry.File] {
			return LadderReadinessEvidence{}, qwen38IdentityReason("checksums.json", "manifest does not match the immutable #8623 artifact profile")
		}
		seen[entry.File] = true
		hashes = append(hashes, ArtifactHash{Path: entry.File, SHA256: strings.ToLower(entry.SHA256)})
	}
	sort.Slice(hashes, func(i, j int) bool { return hashes[i].Path < hashes[j].Path })

	evidencePath := filepath.Join(evidenceOpts.Directory, qwen38LadderEvidenceFilename)
	f, err := os.Open(evidencePath)
	if err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, err.Error())
	}
	ladder, err := qwen38ladder.Decode(f)
	closeErr := f.Close()
	if err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, err.Error())
	}
	if closeErr != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, closeErr.Error())
	}
	decision, err := qwen38ladder.Evaluate(ladder)
	if err != nil || decision.Verdict != "PASS" {
		detail := "exact ladder evaluator did not return PASS"
		if err != nil {
			detail = err.Error()
		}
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, detail)
	}
	if ladder.Concept != "no-thinking-campaign-requests-v1" || ladder.Metric != "p95_ms_to_first_correct_answer" || ladder.Direction != "lower" || ladder.BaselineRuntimeSHA != qwen38ExactBaselineRuntime || ladder.CandidateRuntimeSHA != qwen38ExactCandidateRuntime || len(ladder.Results) != 6 {
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, "ladder experiment identity does not match #8623")
	}
	target := ladder.Results[len(ladder.Results)-1]
	if target.StageID != "target" || target.Model != qwen38ExactModel || target.Revision != qwen38ExactRevision || target.CorpusSHA != qwen38ExactCorpusSHA || target.EnvironmentSHA != qwen38ExactEnvironmentSHA || target.Trials != 3 || target.BaselinePassed != 3 || target.CandidatePassed != 3 || target.BaselineMetric != qwen38ExactBaselineP95MS || target.CandidateMetric != qwen38ExactCandidateP95MS || decision.ImprovementPct != qwen38ExactImprovementPct {
		return LadderReadinessEvidence{}, qwen38IdentityReason(qwen38LadderEvidenceFilename, "exact target result does not match #8623")
	}

	var environment struct {
		Schema             string `json:"schema"`
		CapturedAt         string `json:"captured_at"`
		Model              string `json:"model"`
		Revision           string `json:"revision"`
		DType              string `json:"dtype"`
		TensorParallelSize int    `json:"tensor_parallel_size"`
		VLLM               string `json:"vllm"`
		EnvironmentSHA256  string `json:"environment_sha256"`
		Arms               struct {
			Baseline struct {
				RuntimeSHA string `json:"runtime_sha"`
			} `json:"baseline"`
			Candidate struct {
				RuntimeSHA string `json:"runtime_sha"`
			} `json:"candidate"`
		} `json:"arms"`
	}
	if err := readJSONFile(filepath.Join(evidenceOpts.Directory, "environment.json"), &environment); err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason("environment.json", err.Error())
	}
	if environment.Schema != "fak.qwen38-ladder-environment/1" || environment.Model != target.Model || environment.Revision != target.Revision || environment.DType != "bfloat16" || environment.TensorParallelSize != 2 || environment.EnvironmentSHA256 != target.EnvironmentSHA || environment.Arms.Baseline.RuntimeSHA != ladder.BaselineRuntimeSHA || environment.Arms.Candidate.RuntimeSHA != ladder.CandidateRuntimeSHA {
		return LadderReadinessEvidence{}, qwen38IdentityReason("environment.json", "environment does not bind the exact target result")
	}

	var corpus struct {
		Schema string `json:"schema"`
		ID     string `json:"id"`
	}
	corpusPath := filepath.Join(evidenceOpts.Directory, "corpus.json")
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason("corpus.json", err.Error())
	}
	if err := json.Unmarshal(corpusBytes, &corpus); err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason("corpus.json", err.Error())
	}
	corpusDigest := sha256.Sum256(corpusBytes)
	if corpus.Schema != "fak.qwen38-ladder-corpus/1" || corpus.ID != qwen38ExactCorpusID || hex.EncodeToString(corpusDigest[:]) != target.CorpusSHA {
		return LadderReadinessEvidence{}, qwen38IdentityReason("corpus.json", "corpus does not bind the exact target result")
	}
	if want := strings.TrimSpace(inventoryOpts.ExpectedCorpusID); want != "" && want != corpus.ID {
		return LadderReadinessEvidence{}, qwen38IdentityReason("corpus.json", "corpus ID does not match inventory expectation")
	}

	var raw struct {
		Schema            string `json:"schema"`
		EnvironmentSHA256 string `json:"environment_sha256"`
		CorpusSHA256      string `json:"corpus_sha256"`
		Summary           struct {
			Baseline struct {
				Passed, Trials int
				P95MS          float64 `json:"p95_ms"`
			} `json:"baseline"`
			Candidate struct {
				Passed, Trials int
				P95MS          float64 `json:"p95_ms"`
			} `json:"candidate"`
		} `json:"summary"`
	}
	if err := readJSONFile(filepath.Join(evidenceOpts.Directory, "raw-run.json"), &raw); err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason("raw-run.json", err.Error())
	}
	if raw.Schema != "fak.qwen38-ladder-raw-run/1" || raw.EnvironmentSHA256 != target.EnvironmentSHA || raw.CorpusSHA256 != target.CorpusSHA || raw.Summary.Baseline.Passed != target.BaselinePassed || raw.Summary.Baseline.Trials != target.Trials || raw.Summary.Baseline.P95MS != target.BaselineMetric || raw.Summary.Candidate.Passed != target.CandidatePassed || raw.Summary.Candidate.Trials != target.Trials || raw.Summary.Candidate.P95MS != target.CandidateMetric {
		return LadderReadinessEvidence{}, qwen38IdentityReason("raw-run.json", "raw run does not bind the exact target result")
	}

	var evaluator struct {
		Verdict        string  `json:"verdict"`
		ImprovementPct float64 `json:"improvement_pct"`
		Reason         string  `json:"reason"`
	}
	if err := readJSONFile(filepath.Join(evidenceOpts.Directory, "evaluator-output.json"), &evaluator); err != nil {
		return LadderReadinessEvidence{}, qwen38IdentityReason("evaluator-output.json", err.Error())
	}
	if evaluator.Verdict != decision.Verdict || evaluator.ImprovementPct != decision.ImprovementPct || evaluator.Reason != decision.Reason {
		return LadderReadinessEvidence{}, qwen38IdentityReason("evaluator-output.json", "recorded evaluator output does not match readback")
	}

	return LadderReadinessEvidence{
		Issue:             "https://github.com/anthony-chaudhary/fak/issues/8623",
		Model:             target.Model,
		Revision:          target.Revision,
		Precision:         "BF16",
		Topology:          "TP2",
		Runtime:           "vLLM " + environment.VLLM,
		CapturedAt:        environment.CapturedAt,
		RuntimePair:       RuntimePair{BaselineSHA: ladder.BaselineRuntimeSHA, CandidateSHA: ladder.CandidateRuntimeSHA},
		CorpusID:          corpus.ID,
		CorpusSHA256:      target.CorpusSHA,
		EnvironmentSHA256: target.EnvironmentSHA,
		Correctness:       CorrectnessPair{BaselinePassed: target.BaselinePassed, CandidatePassed: target.CandidatePassed, Trials: target.Trials},
		P95:               P95Pair{Metric: ladder.Metric, BaselineMetric: target.BaselineMetric, CandidateMetric: target.CandidateMetric, Improvement: decision.ImprovementPct},
		ArtifactHashes:    hashes,
	}, LadderEvidenceReason{}
}

func readJSONFile(name string, dst any) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func qwen38IdentityReason(path, detail string) LadderEvidenceReason {
	return LadderEvidenceReason{Code: LadderEvidenceIdentityMismatch, Path: path, Detail: detail}
}

func qwen38ReadinessCells(arithmeticStatus ReadinessCellStatus) []ReadinessCell {
	return []ReadinessCell{
		{ID: "arithmetic_latency", Status: arithmeticStatus, Envelope: "Qwen3.8-27B BF16 TP2 arithmetic p95 time-to-first-correct-answer only", Owner: "https://github.com/anthony-chaudhary/fak/issues/8623"},
		{ID: "broad_quality", Status: ReadinessCellUnwitnessed, Envelope: "repeated agent workloads and broad quality", Owner: "https://github.com/anthony-chaudhary/fak/issues/8309"},
		{ID: "fp8", Status: ReadinessCellHold, Envelope: "official FP8 TP2", Owner: "https://github.com/anthony-chaudhary/fak/issues/8309"},
		{ID: "gguf", Status: ReadinessCellHold, Envelope: "GGUF Q4_K_M", Owner: "https://github.com/anthony-chaudhary/fak/issues/8313"},
		{ID: "metal", Status: ReadinessCellUnwitnessed, Envelope: "Apple Metal", Owner: "https://github.com/anthony-chaudhary/fak/issues/8011"},
		{ID: "tool_call", Status: ReadinessCellHold, Envelope: "tool-call parsing and execution", Owner: "https://github.com/anthony-chaudhary/fak/issues/8382"},
		{ID: "structured_output", Status: ReadinessCellHold, Envelope: "JSON and structured output", Owner: "https://github.com/anthony-chaudhary/fak/issues/8382"},
		{ID: "production", Status: ReadinessCellHold, Envelope: "workflow cache and production readiness", Owner: "https://github.com/anthony-chaudhary/fak/issues/8189"},
	}
}

func holdQwen38LadderReadiness(inventoryOpts InventoryOptions, evidenceOpts LadderEvidenceOptions, reason LadderEvidenceReason) Inventory {
	artifact := strings.TrimSpace(inventoryOpts.Artifact)
	if artifact == "" && strings.TrimSpace(evidenceOpts.Directory) != "" {
		artifact = filepath.Join(evidenceOpts.Directory, qwen38LadderEvidenceFilename)
	}
	return Inventory{
		Schema:  Qwen38LadderInventorySchema,
		Verdict: Hold,
		Rows: []InventoryRow{{
			Model:            qwen38ExactModel,
			Family:           "Qwen3.8",
			Generation:       qwen38ExactRevision,
			Lifecycle:        LifecycleLatest,
			CapabilityGate:   Hold,
			Artifact:         artifact,
			ArtifactRevision: inventoryOpts.ArtifactRevision,
			Reasons:          []string{reason.String()},
			ReadinessCells:   qwen38ReadinessCells(ReadinessCellUnwitnessed),
		}},
		Reasons: []string{reason.String()},
		Semantics: &InventorySemantics{
			Default:     "generic readiness inventory remains the default; failed ladder admission publishes no measured cell",
			Replacement: "replace only after a reviewed immutable packet passes code-pinned identity and exact artifact hashes",
			Rollback:    "remove the ladder selector to return to generic inventory; no serving default changed",
		},
	}
}

// VerifyLadderEvidence proves that the manifest names exactly the regular
// evidence files in Directory and that every named file has the declared
// SHA-256. The manifest itself is metadata and is excluded from the exact-set
// comparison when it resides inside Directory.
func VerifyLadderEvidence(opts LadderEvidenceOptions) LadderEvidenceAdmission {
	directory := strings.TrimSpace(opts.Directory)
	manifest := strings.TrimSpace(opts.Manifest)
	if directory == "" || manifest == "" {
		return ladderEvidenceHold(LadderEvidenceReason{
			Code:   LadderEvidenceInvalidManifest,
			Detail: "evidence directory and checksum manifest are required",
		})
	}

	root, err := filepath.Abs(directory)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: err.Error()})
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: err.Error()})
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		detail := "evidence directory is not a directory"
		if err != nil {
			detail = err.Error()
		}
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: detail})
	}

	manifestPath, err := filepath.Abs(manifest)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()})
	}
	entries, reason := decodeLadderChecksumManifest(manifestPath)
	if reason.Code != "" {
		return ladderEvidenceHold(reason)
	}
	// root is already symlink-canonical. Canonicalize the manifest identity to
	// the same coordinate system before WalkDir compares it with candidates.
	// This matters on macOS, where /var/... and /private/var/... name the same
	// file, and for an explicit symlink alias on every platform. A manifest that
	// really lives outside root stays outside and is not excluded from the walk.
	manifestIdentity, err := filepath.EvalSymlinks(manifestPath)
	if err != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()})
	}
	manifestIdentity = filepath.Clean(manifestIdentity)

	declared := make(map[string]struct{}, len(entries))
	for i := range entries {
		rel, pathReason := canonicalLadderEvidencePath(entries[i].File)
		if pathReason.Code != "" {
			return ladderEvidenceHold(pathReason)
		}
		entries[i].File = rel
		if _, exists := declared[rel]; exists {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceDuplicatePath, Path: rel})
		}
		declared[rel] = struct{}{}
		decoded, decodeErr := hex.DecodeString(entries[i].SHA256)
		if decodeErr != nil || len(decoded) != sha256.Size {
			return ladderEvidenceHold(LadderEvidenceReason{
				Code:   LadderEvidenceInvalidManifest,
				Path:   rel,
				Detail: "sha256 must be exactly 64 hexadecimal characters",
			})
		}
	}

	for _, entry := range entries {
		artifactPath := filepath.Join(root, filepath.FromSlash(entry.File))
		artifactInfo, statErr := os.Lstat(artifactPath)
		if statErr != nil {
			if errors.Is(statErr, os.ErrNotExist) {
				return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceMissingArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256})
			}
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: statErr.Error()})
		}
		if artifactInfo.IsDir() {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: "declared artifact is a directory"})
		}
		resolved, resolveErr := filepath.EvalSymlinks(artifactPath)
		if resolveErr != nil {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: resolveErr.Error()})
		}
		if !pathWithin(root, resolved) {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: "resolved artifact escapes evidence directory"})
		}
		resolvedInfo, statErr := os.Stat(resolved)
		if statErr != nil || !resolvedInfo.Mode().IsRegular() {
			detail := "declared artifact is not a regular file"
			if statErr != nil {
				detail = statErr.Error()
			}
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: detail})
		}
		content, readErr := os.ReadFile(resolved)
		if readErr != nil {
			return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Path: entry.File, ExpectedSHA256: entry.SHA256, Detail: readErr.Error()})
		}
		actual := sha256.Sum256(content)
		actualText := hex.EncodeToString(actual[:])
		if !strings.EqualFold(entry.SHA256, actualText) {
			return ladderEvidenceHold(LadderEvidenceReason{
				Code:           LadderEvidenceChecksumMismatch,
				Path:           entry.File,
				ExpectedSHA256: entry.SHA256,
				ActualSHA256:   actualText,
			})
		}
	}

	walkErr := filepath.WalkDir(root, func(candidate string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if candidate == root || entry.IsDir() || filepath.Clean(candidate) == manifestIdentity {
			return nil
		}
		rel, relErr := filepath.Rel(root, candidate)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)
		if _, exists := declared[rel]; !exists {
			reason = LadderEvidenceReason{Code: LadderEvidenceExtraArtifact, Path: rel}
			return io.EOF
		}
		return nil
	})
	if reason.Code != "" {
		return ladderEvidenceHold(reason)
	}
	if walkErr != nil {
		return ladderEvidenceHold(LadderEvidenceReason{Code: LadderEvidenceUnreadableArtifact, Detail: walkErr.Error()})
	}
	return LadderEvidenceAdmission{Verdict: Pass}
}

func decodeLadderChecksumManifest(manifestPath string) ([]ladderChecksumEntry, LadderEvidenceReason) {
	f, err := os.Open(manifestPath)
	if err != nil {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var entries []ladderChecksumEntry
	if err := dec.Decode(&entries); err != nil {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	if len(entries) == 0 {
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: "checksum manifest contains no artifacts"}
	}
	var trailing any
	if err := dec.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return nil, LadderEvidenceReason{Code: LadderEvidenceInvalidManifest, Detail: err.Error()}
	}
	return entries, LadderEvidenceReason{}
}

func canonicalLadderEvidencePath(raw string) (string, LadderEvidenceReason) {
	if raw == "" || strings.Contains(raw, `\`) || path.IsAbs(raw) || filepath.IsAbs(raw) {
		return "", LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: raw, Detail: "artifact path must be a non-empty relative slash path"}
	}
	clean := path.Clean(raw)
	if clean != raw || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", LadderEvidenceReason{Code: LadderEvidencePathTraversal, Path: raw, Detail: "artifact path is not canonical within the evidence directory"}
	}
	return clean, LadderEvidenceReason{}
}

func pathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func ladderEvidenceHold(reason LadderEvidenceReason) LadderEvidenceAdmission {
	return LadderEvidenceAdmission{Verdict: Hold, Reason: reason}
}

func holdInventoryForLadderEvidence(in Input, opts InventoryOptions, reason LadderEvidenceReason) Inventory {
	reasonText := reason.String()
	out := Inventory{
		Schema:   InventorySchema,
		Verdict:  Hold,
		CorpusID: in.Corpus.ID,
		Rows:     []InventoryRow{},
	}
	models := append([]ModelRequest(nil), in.Models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Model < models[j].Model })
	for _, req := range models {
		out.Rows = append(out.Rows, InventoryRow{
			Model:            req.Model,
			Family:           req.Family,
			Generation:       req.Generation,
			Lifecycle:        req.Lifecycle,
			CapabilityGate:   Hold,
			RequestedTier:    req.RequestedTier,
			CorpusID:         in.Corpus.ID,
			DeclaredAt:       in.Corpus.DeclaredAt,
			Artifact:         opts.Artifact,
			ArtifactRevision: opts.ArtifactRevision,
			Reasons:          []string{reasonText},
		})
	}
	if len(out.Rows) == 0 {
		out.Reasons = append(out.Reasons, reasonText)
	}
	return out
}
