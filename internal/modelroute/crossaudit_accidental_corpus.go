package modelroute

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
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
	Command      string `json:"command"`
	Program      string `json:"program"`
	Args         string `json:"args"`
	Input        string `json:"input"`
	OutputSHA256 string `json:"output_sha256"`
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
	program := f.Witness.Program
	args := witnessProgramArgs(program, f.Witness.Args)
	declared := strings.TrimSpace(strings.Join(append([]string{program}, args...), " "))
	if declared != strings.TrimSpace(f.Witness.Command) {
		return false, fmt.Errorf("%s witness command=%q, structured command=%q", f.ID, f.Witness.Command, declared)
	}
	if override := strings.TrimSpace(os.Getenv("FAK_CROSSAUDIT_FIXTURE_BIN")); program == "crossauditfixture" && override != "" {
		program = override
	}
	cmd := exec.Command(program, args...)
	cmd.Stdin = strings.NewReader(f.Witness.Input)
	out, err := cmd.CombinedOutput()
	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) {
			return false, fmt.Errorf("%s witness execution: %w", f.ID, err)
		}
		exitCode = exitErr.ExitCode()
	}
	outcome := strings.TrimSpace(string(out))
	digest := sha256.Sum256([]byte(outcome))
	gotDigest := "sha256:" + hex.EncodeToString(digest[:])
	if gotDigest != f.Witness.OutputSHA256 {
		return false, fmt.Errorf("%s witness output digest=%s, want %s", f.ID, gotDigest, f.Witness.OutputSHA256)
	}
	switch {
	case exitCode == 0 && outcome == "PASS":
		return false, nil
	case exitCode != 0 && outcome == "FAIL":
		return true, nil
	default:
		return false, fmt.Errorf("%s witness produced exit=%d output=%q, want PASS/0 or FAIL/nonzero", f.ID, exitCode, outcome)
	}
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
		if got != f.Corrupt {
			return fmt.Errorf("accidental corpus %s witness %q corrupt=%t, label=%t", f.ID, f.Witness.Command, got, f.Corrupt)
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

func witnessProgramArgs(program, arg string) []string {
	if program == "crossauditfixture" {
		return []string{"--contract-base64", arg}
	}
	return []string{"-c", arg}
}
