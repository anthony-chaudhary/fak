// Package harnesscreationreceipt implements schema parsing, validation,
// and study-row projection for harness creation trial receipts.
package harnesscreationreceipt

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"time"
)

// Schema identifies the canonical v1alpha1 harness creation receipt specification.
const Schema = "fak.harness-creation-receipt/v1alpha1"

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{5,63}$`)
var digestRE = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// Command records an individual CLI invocation and its exit status.
type Command struct {
	Command string `json:"command"`
	Exit    int    `json:"exit"`
}

// Receipt holds trial execution evidence, provenance metadata, and verification metrics.
type Receipt struct {
	Schema                string    `json:"schema"`
	RunID                 string    `json:"run_id"`
	ParticipantID         string    `json:"participant_id"`
	ParticipantClass      string    `json:"participant_class"`
	PriorFamiliarity      string    `json:"prior_fak_internals_familiarity"`
	Track                 string    `json:"track"`
	Arm                   string    `json:"arm"`
	PairID                string    `json:"pair_id"`
	TaskDigest            string    `json:"task_digest"`
	MachineID             string    `json:"machine_id"`
	PairOrder             string    `json:"pair_order"`
	ArmPosition           int       `json:"arm_position"`
	Independent           bool      `json:"independent"`
	Artifact              string    `json:"artifact"`
	ArtifactDigest        string    `json:"artifact_digest"`
	OS                    string    `json:"os"`
	CPU                   string    `json:"cpu"`
	Toolchain             string    `json:"toolchain"`
	NetworkState          string    `json:"network_state"`
	CacheState            string    `json:"cache_state"`
	StartedAt             time.Time `json:"started_at"`
	StoppedAt             time.Time `json:"stopped_at"`
	ElapsedSeconds        float64   `json:"elapsed_seconds"`
	Commands              []Command `json:"commands"`
	FilesChanged          []string  `json:"files_changed"`
	Rebuilds              int       `json:"rebuilds"`
	RebuildSeconds        float64   `json:"rebuild_seconds"`
	Outcome               string    `json:"outcome"`
	HelpRequests          int       `json:"help_requests"`
	Transcript            string    `json:"transcript"`
	Receipt               string    `json:"receipt"`
	IndependentAuthorship string    `json:"independent_authorship,omitempty"`
	Conformance           string    `json:"conformance,omitempty"`
}

// StudyRow represents a projected comparative trial row for cross-arm study analysis.
type StudyRow struct {
	ID               string  `json:"id"`
	ParticipantID    string  `json:"participant_id"`
	Track            string  `json:"track"`
	Arm              string  `json:"arm"`
	PairID           string  `json:"pair_id"`
	TaskDigest       string  `json:"task_digest"`
	MachineID        string  `json:"machine_id"`
	PairOrder        string  `json:"pair_order"`
	ArmPosition      int     `json:"arm_position"`
	ParticipantClass string  `json:"participant_class"`
	Independent      bool    `json:"independent"`
	OS               string  `json:"os"`
	CPU              string  `json:"cpu"`
	NetworkState     string  `json:"network_state"`
	CacheState       string  `json:"cache_state"`
	Outcome          string  `json:"outcome"`
	ElapsedSeconds   float64 `json:"elapsed_seconds"`
	Receipt          string  `json:"receipt"`
	SourceReceipt    string  `json:"source_receipt,omitempty"`
	SourceDigest     string  `json:"source_digest,omitempty"`
}

// Result encapsulates the validation outcome and projected study row from evaluating a receipt.
type Result struct {
	Schema string   `json:"schema"`
	Valid  bool     `json:"valid"`
	Row    StudyRow `json:"study_row"`
}

// Parse unmarshals receipt JSON and validates field formats, timing intervals, and evidence requirements.
func Parse(raw []byte) (Receipt, error) {
	var r Receipt
	if err := json.Unmarshal(raw, &r); err != nil {
		return r, err
	}
	if r.Schema != Schema {
		return r, fmt.Errorf("schema must be %q", Schema)
	}
	for name, value := range map[string]string{"run_id": r.RunID, "participant_id": r.ParticipantID} {
		if !slug.MatchString(value) {
			return r, fmt.Errorf("%s must be a privacy-safe random slug", name)
		}
	}
	for name, value := range map[string]string{"participant_class": r.ParticipantClass, "prior familiarity": r.PriorFamiliarity, "artifact": r.Artifact, "artifact_digest": r.ArtifactDigest, "os": r.OS, "cpu": r.CPU, "toolchain": r.Toolchain, "network_state": r.NetworkState, "cache_state": r.CacheState, "transcript": r.Transcript, "receipt": r.Receipt} {
		if value == "" {
			return r, fmt.Errorf("%s is required", name)
		}
	}
	if r.Track != "ten-minute" && r.Track != "weekend" {
		return r, errors.New("track must be ten-minute or weekend")
	}
	if r.Arm != "fak" && r.Arm != "baseline" {
		return r, errors.New("arm must be fak or baseline")
	}
	if !slug.MatchString(r.PairID) {
		return r, errors.New("pair_id must be a privacy-safe random slug")
	}
	if !slug.MatchString(r.MachineID) {
		return r, errors.New("machine_id must be a privacy-safe random slug")
	}
	if !digestRE.MatchString(r.TaskDigest) {
		return r, errors.New("task_digest must be a lowercase sha256 digest")
	}
	if r.PairOrder != "fak-first" && r.PairOrder != "baseline-first" {
		return r, errors.New("pair_order must be fak-first or baseline-first")
	}
	expectedPosition := 2
	if (r.PairOrder == "fak-first" && r.Arm == "fak") || (r.PairOrder == "baseline-first" && r.Arm == "baseline") {
		expectedPosition = 1
	}
	if r.ArmPosition != expectedPosition {
		return r, fmt.Errorf("arm_position must be %d for %s in %s", expectedPosition, r.Arm, r.PairOrder)
	}
	if r.Track == "weekend" && r.Arm != "fak" {
		return r, errors.New("weekend receipts must use fak arm")
	}
	if r.Outcome != "success" && r.Outcome != "failure" {
		return r, errors.New("outcome must be success or failure")
	}
	if r.StartedAt.IsZero() || r.StoppedAt.IsZero() || !r.StoppedAt.After(r.StartedAt) || r.ElapsedSeconds <= 0 {
		return r, errors.New("valid clock fields are required")
	}
	if interval := r.StoppedAt.Sub(r.StartedAt).Seconds(); r.ElapsedSeconds != interval {
		return r, fmt.Errorf("elapsed_seconds %.9g does not match started_at/stopped_at interval %.9g", r.ElapsedSeconds, interval)
	}
	if len(r.Commands) == 0 || r.Rebuilds < 0 || r.RebuildSeconds < 0 || r.HelpRequests < 0 {
		return r, errors.New("commands and nonnegative rebuild/help evidence are required")
	}
	if (r.Rebuilds == 0) != (r.RebuildSeconds == 0) {
		return r, errors.New("rebuild count and duration must both be zero or both be positive")
	}
	if r.Outcome == "success" && (len(r.FilesChanged) == 0 || r.Rebuilds < 1) {
		return r, errors.New("successful receipt requires changed files and rebuild evidence")
	}
	if r.Track == "weekend" && (r.IndependentAuthorship == "" || r.Conformance == "") {
		return r, errors.New("weekend receipt requires independent_authorship and conformance")
	}
	return r, nil
}

// Evaluate projects an adjudicated Receipt into a validated Result and StudyRow.
func Evaluate(r Receipt) Result {
	return Result{Schema: "fak.harness-creation-receipt-result/v1alpha1", Valid: true, Row: StudyRow{
		ID: r.RunID, ParticipantID: r.ParticipantID, Track: r.Track, Arm: r.Arm, PairID: r.PairID,
		TaskDigest: r.TaskDigest, MachineID: r.MachineID, PairOrder: r.PairOrder, ArmPosition: r.ArmPosition,
		ParticipantClass: r.ParticipantClass, Independent: r.Independent, OS: r.OS, CPU: r.CPU,
		NetworkState: r.NetworkState, CacheState: r.CacheState, Outcome: r.Outcome,
		ElapsedSeconds: r.ElapsedSeconds, Receipt: r.Receipt,
	}}
}

// CheckUnique verifies that a projected StudyRow adheres to study protocol constraints,
// ensuring no duplicate runs, participant collisions, or cross-arm envelope drift occur.
func CheckUnique(studyRaw []byte, row StudyRow) error {
	var study struct {
		Protocol struct {
			TaskDigest string `json:"task_digest"`
		} `json:"protocol"`
		Runs []StudyRow `json:"runs"`
	}
	if err := json.Unmarshal(studyRaw, &study); err != nil {
		return fmt.Errorf("parse study: %w", err)
	}
	if row.PairID != "" {
		if !digestRE.MatchString(study.Protocol.TaskDigest) {
			return errors.New("study protocol task_digest must be lowercase sha256:<64 hex>")
		}
		if row.TaskDigest != study.Protocol.TaskDigest {
			return errors.New("receipt task_digest does not match study protocol task_digest")
		}
	}
	for _, existing := range study.Runs {
		if existing.ID == row.ID {
			return fmt.Errorf("duplicate run_id %q", row.ID)
		}
		if existing.ParticipantID == row.ParticipantID && existing.Track == row.Track && existing.Arm == row.Arm {
			return fmt.Errorf("participant %q already has a %s/%s attempt", row.ParticipantID, row.Track, row.Arm)
		}
		if existing.PairID == row.PairID {
			if existing.TaskDigest != row.TaskDigest || existing.MachineID != row.MachineID || existing.OS != row.OS || existing.CPU != row.CPU || existing.NetworkState != row.NetworkState || existing.CacheState != row.CacheState {
				return fmt.Errorf("pair %q comparison envelope differs between arms", row.PairID)
			}
		}
		if existing.PairID == row.PairID && existing.PairOrder != row.PairOrder {
			return fmt.Errorf("pair %q has conflicting order %q and %q", row.PairID, existing.PairOrder, row.PairOrder)
		}
		if existing.PairID == row.PairID && existing.ParticipantID != row.ParticipantID {
			return fmt.Errorf("pair %q belongs to participant %q, not %q", row.PairID, existing.ParticipantID, row.ParticipantID)
		}
		if existing.PairID == row.PairID && existing.Arm == row.Arm {
			return fmt.Errorf("pair %q already has a %s arm", row.PairID, row.Arm)
		}
	}
	return nil
}
