package newmodel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const (
	ReplayCorpusSchema       = "fak.new-model-replay-corpus/1"
	ReplayLedgerSchema       = "fak.new-model-replay-ledger/1"
	ReplayLicenseDisposition = "ADAPT:minimal derived config facts with repository, revision, digest, and source-license attribution; no expressive source copied"

	// ReplayEvidenceBoundary is intentionally stronger than an ordinary support disclaimer:
	// replay never acquires a runnable artifact or reaches an execution seam.
	ReplayEvidenceBoundary = "offline metadata compilation only; config facts were observed at immutable public revisions; model weights, tokenizers, and templates were not acquired; no model was allocated, executed, served, supported, or performance-tested"
)

type ReplayCorpus struct {
	Schema     string       `json:"schema"`
	ObservedAt string       `json:"observed_at"`
	Cases      []ReplayCase `json:"cases"`
}

type ReplayCase struct {
	ID                     string          `json:"id"`
	Repository             string          `json:"repository"`
	Revision               string          `json:"revision"`
	SourceAnchor           string          `json:"source_anchor"`
	SourceConfigSHA256     string          `json:"source_config_sha256"`
	SourceEventAt          string          `json:"source_event_at"`
	License                string          `json:"license"`
	LicenseDisposition     string          `json:"license_disposition"`
	Architecture           string          `json:"architecture"`
	ArchitectureFamily     string          `json:"architecture_family"`
	ModelType              string          `json:"model_type"`
	CompatibilityException string          `json:"compatibility_exception,omitempty"`
	ManualCorrections      int             `json:"manual_corrections"`
	ExpectedOutcome        string          `json:"expected_outcome"`
	ExpectedReason         RefusalReason   `json:"expected_reason,omitempty"`
	Manifest               json.RawMessage `json:"manifest"`
}

type ReplayLedger struct {
	Schema           string        `json:"schema"`
	ObservedAt       string        `json:"observed_at"`
	EvidenceBoundary string        `json:"evidence_boundary"`
	Rows             []ReplayRow   `json:"rows"`
	Summary          ReplaySummary `json:"summary"`
}

type ReplayRow struct {
	ID                     string        `json:"id"`
	Repository             string        `json:"repository"`
	Revision               string        `json:"revision"`
	SourceAnchor           string        `json:"source_anchor"`
	SourceConfigSHA256     string        `json:"source_config_sha256"`
	ObservedAt             string        `json:"observed_at"`
	SourceEventAt          string        `json:"source_event_at"`
	License                string        `json:"license"`
	LicenseDisposition     string        `json:"license_disposition"`
	Architecture           string        `json:"architecture"`
	ArchitectureFamily     string        `json:"architecture_family"`
	ModelType              string        `json:"model_type"`
	CompatibilityException string        `json:"compatibility_exception,omitempty"`
	Outcome                string        `json:"outcome"`
	Reason                 RefusalReason `json:"reason,omitempty"`
	Axis                   string        `json:"axis,omitempty"`
	OutcomeSHA256          string        `json:"outcome_sha256"`
	ByteIdentical          bool          `json:"byte_identical"`
	SemanticGaps           []string      `json:"semantic_gaps"`
	Obligations            []string      `json:"obligations"`
	ManualCorrections      int           `json:"manual_corrections"`
	Execution              string        `json:"execution"`
	ModelExecuted          bool          `json:"model_executed"`
	SupportClaim           bool          `json:"support_claim"`
	PerformanceClaim       bool          `json:"performance_claim"`
}

type ReplaySummary struct {
	Manifests            int  `json:"manifests"`
	ArchitectureFamilies int  `json:"architecture_families"`
	Packets              int  `json:"packets"`
	Refusals             int  `json:"refusals"`
	ManualCorrections    int  `json:"manual_corrections"`
	ByteIdentical        bool `json:"byte_identical"`
}

func ParseReplayCorpus(raw []byte) (ReplayCorpus, error) {
	var corpus ReplayCorpus
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&corpus); err != nil {
		return ReplayCorpus{}, fmt.Errorf("newmodel replay corpus: %w", err)
	}
	if corpus.Schema != ReplayCorpusSchema || corpus.ObservedAt == "" {
		return ReplayCorpus{}, errors.New("newmodel replay corpus: invalid schema or observation time")
	}
	return corpus, nil
}

// Replay compiles each committed manifest twice. A typed refusal is an observed
// compatibility result, not a failed replay and never becomes a support claim.
func Replay(corpus ReplayCorpus) (ReplayLedger, error) {
	cases := append([]ReplayCase(nil), corpus.Cases...)
	sort.Slice(cases, func(i, j int) bool { return cases[i].ID < cases[j].ID })
	ledger := ReplayLedger{
		Schema: ReplayLedgerSchema, ObservedAt: corpus.ObservedAt,
		EvidenceBoundary: ReplayEvidenceBoundary,
		Summary:          ReplaySummary{Manifests: len(cases), ByteIdentical: true},
	}
	seenIDs := map[string]bool{}
	seenPins := map[string]bool{}
	families := map[string]bool{}
	for _, replayCase := range cases {
		if replayCase.ID == "" || seenIDs[replayCase.ID] {
			return ReplayLedger{}, fmt.Errorf("newmodel replay: duplicate or empty case id %q", replayCase.ID)
		}
		seenIDs[replayCase.ID] = true
		pin := replayCase.Repository + "@" + replayCase.Revision
		if seenPins[pin] {
			return ReplayLedger{}, fmt.Errorf("newmodel replay: duplicate source pin %s", pin)
		}
		seenPins[pin] = true
		families[replayCase.ArchitectureFamily] = true

		var manifest ReleaseManifest
		if err := json.Unmarshal(replayCase.Manifest, &manifest); err != nil {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s manifest: %w", replayCase.ID, err)
		}
		if err := validateReplayProvenance(replayCase, manifest); err != nil {
			return ReplayLedger{}, err
		}
		normalizeManifest(&manifest)
		first, outcome, refusal, err := replayCompile(replayCase.Manifest)
		if err != nil {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s: %w", replayCase.ID, err)
		}
		second, secondOutcome, secondRefusal, err := replayCompile(replayCase.Manifest)
		if err != nil {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s second pass: %w", replayCase.ID, err)
		}
		if outcome != secondOutcome || !bytes.Equal(first, second) || refusalIdentity(refusal) != refusalIdentity(secondRefusal) {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s: compiler result is not byte-identical", replayCase.ID)
		}
		if outcome != replayCase.ExpectedOutcome {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s: outcome %s, want %s", replayCase.ID, outcome, replayCase.ExpectedOutcome)
		}
		if outcome == "refusal" && refusal.Reason != replayCase.ExpectedReason {
			return ReplayLedger{}, fmt.Errorf("newmodel replay %s: refusal %s, want %s", replayCase.ID, refusal.Reason, replayCase.ExpectedReason)
		}
		sum := sha256.Sum256(first)
		row := ReplayRow{
			ID: replayCase.ID, Repository: replayCase.Repository, Revision: replayCase.Revision,
			SourceAnchor: replayCase.SourceAnchor, SourceConfigSHA256: replayCase.SourceConfigSHA256,
			ObservedAt: corpus.ObservedAt, SourceEventAt: replayCase.SourceEventAt,
			License: replayCase.License, LicenseDisposition: replayCase.LicenseDisposition,
			Architecture: replayCase.Architecture, ArchitectureFamily: replayCase.ArchitectureFamily,
			ModelType: replayCase.ModelType, CompatibilityException: replayCase.CompatibilityException,
			Outcome: outcome, OutcomeSHA256: hex.EncodeToString(sum[:]), ByteIdentical: true,
			SemanticGaps: semanticReplayGaps(manifest, refusal), Obligations: replayObligations(manifest),
			ManualCorrections: replayCase.ManualCorrections, Execution: "not-run",
		}
		if refusal != nil {
			row.Reason, row.Axis = refusal.Reason, refusal.Axis
			ledger.Summary.Refusals++
		} else {
			ledger.Summary.Packets++
		}
		ledger.Summary.ManualCorrections += replayCase.ManualCorrections
		ledger.Rows = append(ledger.Rows, row)
	}
	ledger.Summary.ArchitectureFamilies = len(families)
	return ledger, nil
}

func MarshalReplayLedger(ledger ReplayLedger) ([]byte, error) {
	raw, err := json.MarshalIndent(ledger, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(raw, '\n'), nil
}

func validateReplayProvenance(replayCase ReplayCase, manifest ReleaseManifest) error {
	if replayCase.Revision != manifest.Source.Revision || replayCase.SourceConfigSHA256 != manifest.Source.ManifestSHA256 {
		return fmt.Errorf("newmodel replay %s: manifest source pin disagrees with corpus", replayCase.ID)
	}
	if !strings.Contains(replayCase.SourceAnchor, replayCase.Repository) || !strings.Contains(replayCase.SourceAnchor, replayCase.Revision) || !strings.HasSuffix(replayCase.SourceAnchor, "/config.json") {
		return fmt.Errorf("newmodel replay %s: source anchor is not immutable", replayCase.ID)
	}
	if len(replayCase.SourceConfigSHA256) != 64 || replayCase.SourceEventAt == "" || replayCase.License == "" || replayCase.LicenseDisposition != ReplayLicenseDisposition {
		return fmt.Errorf("newmodel replay %s: incomplete provenance or license disposition", replayCase.ID)
	}
	if !strings.HasPrefix(manifest.Artifact.URI, "fence://not-acquired/") {
		return fmt.Errorf("newmodel replay %s: artifact URI crosses the no-execution fence", replayCase.ID)
	}
	return nil
}

func replayCompile(raw []byte) ([]byte, string, *Refusal, error) {
	packet, err := CompileManifestJSON(raw)
	if err == nil {
		return packet, "packet", nil, nil
	}
	var refusal *Refusal
	if !errors.As(err, &refusal) {
		return nil, "", nil, err
	}
	encoded, marshalErr := json.MarshalIndent(refusal, "", "  ")
	if marshalErr != nil {
		return nil, "", nil, marshalErr
	}
	return append(encoded, '\n'), "refusal", refusal, nil
}

func refusalIdentity(refusal *Refusal) string {
	if refusal == nil {
		return ""
	}
	return string(refusal.Reason) + "\x00" + refusal.Axis + "\x00" + refusal.Detail
}

func replayObligations(manifest ReleaseManifest) []string {
	obligations := make([]string, 0, len(manifest.Obligations))
	for _, obligation := range manifest.Obligations {
		obligations = append(obligations, obligation.Kind+":"+obligation.ID)
	}
	sort.Strings(obligations)
	return obligations
}

func semanticReplayGaps(manifest ReleaseManifest, refusal *Refusal) []string {
	gaps := make([]string, 0)
	for _, obligation := range manifest.Obligations {
		if obligation.Kind == "semantic" {
			gaps = append(gaps, "obligation:"+obligation.ID)
		}
	}
	if refusal != nil && (refusal.Reason == RefusalUnknownSemanticDelta || refusal.Reason == RefusalContradictorySemantic) {
		for _, delta := range manifest.SemanticDeltas {
			if delta.Axis == refusal.Axis {
				gaps = append(gaps, delta.Axis+":"+delta.Value)
			}
		}
	}
	sort.Strings(gaps)
	return uniqueSorted(gaps)
}
