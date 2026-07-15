package modelroute

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const AccidentalCorpusManifestSchema = "fak-crossaudit-accidental-corpus/v1"

// AccidentalFailureClass names ordinary engineering failures that an independent
// issue audit must distinguish from style disagreement.
type AccidentalFailureClass string

const (
	AccidentalIncompleteDoneCondition AccidentalFailureClass = "incomplete_done_condition"
	AccidentalWrongEdgeCase           AccidentalFailureClass = "wrong_edge_case"
	AccidentalSwallowedError          AccidentalFailureClass = "swallowed_error"
	AccidentalStaleConsumer           AccidentalFailureClass = "stale_consumer"
	AccidentalMissingFailureTest      AccidentalFailureClass = "missing_failure_test"
	AccidentalRaceLostUpdate          AccidentalFailureClass = "race_lost_update"
	AccidentalPartialRename           AccidentalFailureClass = "partial_rename"
	AccidentalBuildPoison             AccidentalFailureClass = "build_poison"
	AccidentalDocsCLIDrift            AccidentalFailureClass = "docs_cli_drift"
	AccidentalOverBroadRewrite        AccidentalFailureClass = "over_broad_rewrite"
	AccidentalRevertedSafetyCheck     AccidentalFailureClass = "reverted_safety_check"
	AccidentalCleanHardRefactor       AccidentalFailureClass = "clean_hard_refactor"
)

func (c AccidentalFailureClass) Valid() bool {
	switch c {
	case AccidentalIncompleteDoneCondition, AccidentalWrongEdgeCase, AccidentalSwallowedError,
		AccidentalStaleConsumer, AccidentalMissingFailureTest, AccidentalRaceLostUpdate,
		AccidentalPartialRename, AccidentalBuildPoison, AccidentalDocsCLIDrift,
		AccidentalOverBroadRewrite, AccidentalRevertedSafetyCheck, AccidentalCleanHardRefactor:
		return true
	default:
		return false
	}
}

// AccidentalWitness is deterministic ground truth authored outside the auditor.
// The regexp is applied to the bounded issue bundle and Expected is the result
// required for the fixture's declared clean/corrupt label.
type AccidentalWitness struct {
	Command  string `json:"command"`
	Pattern  string `json:"pattern"`
	Expected bool   `json:"expected"`
}

// AccidentalCorpusFixture is one member of a clean/corrupt pair.
type AccidentalCorpusFixture struct {
	ID       string                 `json:"id"`
	Pair     string                 `json:"pair"`
	Class    AccidentalFailureClass `json:"error_class"`
	Corrupt  bool                   `json:"corrupt"`
	Witness  AccidentalWitness      `json:"ground_truth"`
	Evidence IssueAuditEvidence     `json:"-"`
	Author   AuditIdentity          `json:"author"`
	Auditor  AuditIdentity          `json:"auditor"`
}

// AccidentalCorpusManifest is the machine-readable inventory folded by
// calibration/dogfood. Digests bind each row to the same bundle selfchecked here.
type AccidentalCorpusManifest struct {
	Schema     string                         `json:"schema"`
	Pairs      int                            `json:"pairs"`
	Fixtures   int                            `json:"fixtures"`
	Corrupt    int                            `json:"corrupt"`
	Clean      int                            `json:"clean"`
	ClassSizes map[AccidentalFailureClass]int `json:"class_sizes"`
	Rows       []AccidentalCorpusManifestRow  `json:"rows"`
}

type AccidentalCorpusManifestRow struct {
	ID           string                 `json:"id"`
	Pair         string                 `json:"pair"`
	Class        AccidentalFailureClass `json:"error_class"`
	Corrupt      bool                   `json:"corrupt"`
	BundleDigest string                 `json:"bundle_digest"`
	Witness      AccidentalWitness      `json:"ground_truth"`
	Author       AuditIdentity          `json:"author"`
	Auditor      AuditIdentity          `json:"auditor"`
}

func (f AccidentalCorpusFixture) Bundle() (IssueAuditBundle, error) {
	return BuildIssueAuditBundle(f.Evidence, IssueAuditBundleOptions{})
}

func runAccidentalWitness(f AccidentalCorpusFixture) (bool, error) {
	bundle, err := f.Bundle()
	if err != nil {
		return false, err
	}
	re, err := regexp.Compile(f.Witness.Pattern)
	if err != nil {
		return false, fmt.Errorf("%s witness regexp: %w", f.ID, err)
	}
	return re.MatchString(accidentalWitnessSurface(bundle)), nil
}

func accidentalWitnessSurface(b IssueAuditBundle) string {
	var out strings.Builder
	for _, blob := range b.Blobs {
		out.WriteByte('\n')
		out.WriteString(blob.Content)
	}
	for _, commit := range b.Closure.Commits {
		for _, path := range commit.ChangedPaths {
			out.WriteByte('\n')
			out.WriteString(path)
		}
	}
	for _, refs := range [][]EvidenceRef{b.Evidence.Tests, b.Evidence.CI, b.Evidence.DOS, b.Evidence.Artifacts, b.Evidence.Other} {
		for _, ref := range refs {
			out.WriteByte('\n')
			out.WriteString(ref.Kind)
			out.WriteByte(':')
			out.WriteString(ref.Ref)
		}
	}
	return out.String()
}

// SelfCheckAccidentalCorpus proves pair completeness and deterministic labels.
// A flipped label changes Expected but not the evidence, so this check fails.
func SelfCheckAccidentalCorpus(fixtures []AccidentalCorpusFixture) error {
	if len(fixtures) == 0 {
		return fmt.Errorf("accidental corpus is empty")
	}
	pairs := map[string]map[bool]bool{}
	seenIDs := map[string]bool{}
	for _, f := range fixtures {
		if f.ID == "" || seenIDs[f.ID] {
			return fmt.Errorf("accidental corpus empty or duplicate id %q", f.ID)
		}
		seenIDs[f.ID] = true
		if f.Pair == "" || !f.Class.Valid() {
			return fmt.Errorf("accidental corpus %s has invalid pair/class", f.ID)
		}
		if f.Author.Family == "" || f.Auditor.Family == "" || strings.EqualFold(f.Author.Family, f.Auditor.Family) {
			return fmt.Errorf("accidental corpus %s lacks author/auditor provenance diversity", f.ID)
		}
		got, err := runAccidentalWitness(f)
		if err != nil {
			return err
		}
		if f.Witness.Expected != f.Corrupt {
			return fmt.Errorf("accidental corpus %s label/witness contract diverged", f.ID)
		}
		if got != f.Witness.Expected {
			return fmt.Errorf("accidental corpus %s witness %q = %t, want %t", f.ID, f.Witness.Command, got, f.Witness.Expected)
		}
		if pairs[f.Pair] == nil {
			pairs[f.Pair] = map[bool]bool{}
		}
		pairs[f.Pair][f.Corrupt] = true
	}
	for pair, labels := range pairs {
		if !labels[false] || !labels[true] {
			return fmt.Errorf("accidental corpus pair %s lacks clean/corrupt members", pair)
		}
	}
	return nil
}

func BuildAccidentalCorpusManifest(fixtures []AccidentalCorpusFixture) (AccidentalCorpusManifest, error) {
	if err := SelfCheckAccidentalCorpus(fixtures); err != nil {
		return AccidentalCorpusManifest{}, err
	}
	manifest := AccidentalCorpusManifest{Schema: AccidentalCorpusManifestSchema, ClassSizes: map[AccidentalFailureClass]int{}}
	pairs := map[string]bool{}
	for _, f := range fixtures {
		bundle, err := f.Bundle()
		if err != nil {
			return AccidentalCorpusManifest{}, err
		}
		pairs[f.Pair] = true
		manifest.Fixtures++
		manifest.ClassSizes[f.Class]++
		if f.Corrupt {
			manifest.Corrupt++
		} else {
			manifest.Clean++
		}
		manifest.Rows = append(manifest.Rows, AccidentalCorpusManifestRow{
			ID: f.ID, Pair: f.Pair, Class: f.Class, Corrupt: f.Corrupt,
			BundleDigest: bundle.BundleDigest, Witness: f.Witness, Author: f.Author, Auditor: f.Auditor,
		})
	}
	manifest.Pairs = len(pairs)
	sort.Slice(manifest.Rows, func(i, j int) bool { return manifest.Rows[i].ID < manifest.Rows[j].ID })
	return manifest, nil
}
