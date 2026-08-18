package harnesscontrolstudy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-harness-control-study/1"
const ReceiptSchema = "fak-harness-control-receipt/1"

type Study struct {
	Schema     string        `json:"schema"`
	StudyID    string        `json:"study_id"`
	TaskDigest string        `json:"task_digest"`
	MinPairs   int           `json:"min_pairs"`
	Rows       []ReceiptLink `json:"rows"`
}

type ReceiptLink struct {
	Source string `json:"source"`
	Digest string `json:"digest"`
}

type Receipt struct {
	Schema                string   `json:"schema"`
	StudyID               string   `json:"study_id"`
	TaskDigest            string   `json:"task_digest"`
	ParticipantID         string   `json:"participant_id"`
	PairID                string   `json:"pair_id"`
	PairOrder             string   `json:"pair_order"`
	Arm                   string   `json:"arm"`
	ArmPosition           int      `json:"arm_position"`
	StartedAt             string   `json:"started_at"`
	StoppedAt             string   `json:"stopped_at"`
	ElapsedSeconds        float64  `json:"elapsed_seconds"`
	Succeeded             bool     `json:"succeeded"`
	Verified              bool     `json:"verified"`
	Errors                []string `json:"errors"`
	HelpRequests          int      `json:"help_requests"`
	Confidence            int      `json:"confidence"`
	Commands              []string `json:"commands"`
	ArtifactDigest        string   `json:"artifact_digest"`
	BinaryVersion         string   `json:"binary_version"`
	BinaryCommit          string   `json:"binary_commit"`
	BaseLockID            string   `json:"base_lock_id,omitempty"`
	InspectCaptured       bool     `json:"inspect_captured,omitempty"`
	PreviewCaptured       bool     `json:"preview_captured,omitempty"`
	RuntimeVerifyCaptured bool     `json:"runtime_verify_captured,omitempty"`
	Preference            string   `json:"preference,omitempty"`
	PreferenceReason      string   `json:"preference_reason,omitempty"`
}

type ArmSummary struct {
	Runs              int     `json:"runs"`
	Successes         int     `json:"successes"`
	Verified          int     `json:"verified"`
	MedianSeconds     float64 `json:"median_seconds"`
	TotalErrors       int     `json:"total_errors"`
	TotalHelpRequests int     `json:"total_help_requests"`
	MedianConfidence  float64 `json:"median_confidence"`
}

type Report struct {
	Schema               string     `json:"schema"`
	StudyID              string     `json:"study_id"`
	AdmissiblePairs      int        `json:"admissible_pairs"`
	DefaultControl       ArmSummary `json:"default_control"`
	Scratch              ArmSummary `json:"scratch"`
	PreferDefaultControl int        `json:"prefer_default_control"`
	PreferScratch        int        `json:"prefer_scratch"`
	NoPreference         int        `json:"no_preference"`
	Verdict              string     `json:"verdict"`
	Reasons              []string   `json:"reasons,omitempty"`
}

func Evaluate(studyPath string) (Report, error) {
	raw, err := os.ReadFile(studyPath)
	if err != nil {
		return Report{}, err
	}
	var study Study
	if err := decodeStrict(raw, &study); err != nil {
		return Report{}, fmt.Errorf("study: %w", err)
	}
	if study.Schema != Schema || strings.TrimSpace(study.StudyID) == "" || !isSHA256(study.TaskDigest) {
		return Report{}, fmt.Errorf("invalid study identity or task digest")
	}
	if study.MinPairs < 2 {
		return Report{}, fmt.Errorf("min_pairs must be at least 2")
	}
	base := filepath.Dir(studyPath)
	pairs := map[string][]Receipt{}
	for _, link := range study.Rows {
		if filepath.IsAbs(link.Source) || strings.Contains(filepath.Clean(link.Source), "..") {
			return Report{}, fmt.Errorf("receipt source must stay relative to study")
		}
		bytes, err := os.ReadFile(filepath.Join(base, filepath.FromSlash(link.Source)))
		if err != nil {
			return Report{}, err
		}
		if digest(bytes) != link.Digest {
			return Report{}, fmt.Errorf("receipt %s digest mismatch", link.Source)
		}
		var receipt Receipt
		if err := decodeStrict(bytes, &receipt); err != nil {
			return Report{}, fmt.Errorf("receipt %s: %w", link.Source, err)
		}
		if err := validateReceipt(study, receipt); err != nil {
			return Report{}, fmt.Errorf("receipt %s: %w", link.Source, err)
		}
		pairs[receipt.PairID] = append(pairs[receipt.PairID], receipt)
	}
	report := Report{Schema: Schema, StudyID: study.StudyID, Verdict: "not_yet"}
	var defaultRows, scratchRows []Receipt
	participants := map[string]bool{}
	for pairID, rows := range pairs {
		if len(rows) != 2 {
			report.Reasons = append(report.Reasons, "pair "+pairID+" is incomplete")
			continue
		}
		arms := map[string]Receipt{}
		for _, row := range rows {
			arms[row.Arm] = row
		}
		d, dok := arms["default-control"]
		s, sok := arms["scratch"]
		if !dok || !sok || d.ParticipantID != s.ParticipantID || d.PairOrder != s.PairOrder || d.ArmPosition == s.ArmPosition {
			report.Reasons = append(report.Reasons, "pair "+pairID+" is inconsistent")
			continue
		}
		if participants[d.ParticipantID] {
			return Report{}, fmt.Errorf("participant %s appears in multiple pairs", d.ParticipantID)
		}
		participants[d.ParticipantID] = true
		report.AdmissiblePairs++
		defaultRows = append(defaultRows, d)
		scratchRows = append(scratchRows, s)
		preference := d.Preference
		if preference == "" {
			preference = s.Preference
		}
		switch preference {
		case "default-control":
			report.PreferDefaultControl++
		case "scratch":
			report.PreferScratch++
		default:
			report.NoPreference++
		}
	}
	report.DefaultControl = summarize(defaultRows)
	report.Scratch = summarize(scratchRows)
	if report.AdmissiblePairs < study.MinPairs {
		report.Reasons = append(report.Reasons, fmt.Sprintf("need %d admissible pairs; got %d", study.MinPairs, report.AdmissiblePairs))
	} else if report.DefaultControl.Verified != report.DefaultControl.Runs || report.Scratch.Verified != report.Scratch.Runs {
		report.Reasons = append(report.Reasons, "every arm must finish with successful verification")
	} else {
		report.Verdict = "measured"
	}
	return report, nil
}

func validateReceipt(study Study, r Receipt) error {
	if r.Schema != ReceiptSchema || r.StudyID != study.StudyID || r.TaskDigest != study.TaskDigest {
		return fmt.Errorf("identity does not match study")
	}
	if strings.TrimSpace(r.ParticipantID) == "" || strings.TrimSpace(r.PairID) == "" || strings.TrimSpace(r.ArtifactDigest) == "" || !isSHA256(r.ArtifactDigest) {
		return fmt.Errorf("participant, pair, and artifact digest are required")
	}
	if r.Arm != "default-control" && r.Arm != "scratch" {
		return fmt.Errorf("arm must be default-control or scratch")
	}
	if r.PairOrder != "default-first" && r.PairOrder != "scratch-first" {
		return fmt.Errorf("invalid pair_order")
	}
	wantPosition := 2
	if (r.PairOrder == "default-first" && r.Arm == "default-control") || (r.PairOrder == "scratch-first" && r.Arm == "scratch") {
		wantPosition = 1
	}
	if r.ArmPosition != wantPosition {
		return fmt.Errorf("arm_position contradicts pair_order")
	}
	start, err := time.Parse(time.RFC3339, r.StartedAt)
	if err != nil {
		return fmt.Errorf("invalid started_at")
	}
	stop, err := time.Parse(time.RFC3339, r.StoppedAt)
	if err != nil || !stop.After(start) {
		return fmt.Errorf("invalid stopped_at")
	}
	if delta := stop.Sub(start).Seconds(); delta-r.ElapsedSeconds > 0.001 || r.ElapsedSeconds-delta > 0.001 {
		return fmt.Errorf("elapsed_seconds contradicts timestamps")
	}
	if r.Confidence < 1 || r.Confidence > 5 {
		return fmt.Errorf("confidence must be 1..5")
	}
	if len(r.Commands) == 0 {
		return fmt.Errorf("commands are required")
	}
	if strings.TrimSpace(r.BinaryVersion) == "" || strings.TrimSpace(r.BinaryCommit) == "" {
		return fmt.Errorf("binary version and commit are required")
	}
	if r.Arm == "default-control" && (r.BaseLockID == "" || !r.InspectCaptured || !r.PreviewCaptured) {
		return fmt.Errorf("default-control requires base lock, inspect, and preview evidence")
	}
	if r.Succeeded && !r.Verified {
		return fmt.Errorf("success requires verification")
	}
	if r.ArmPosition == 2 && (r.Preference != "default-control" && r.Preference != "scratch" && r.Preference != "none") {
		return fmt.Errorf("second arm requires explicit preference")
	}
	if r.ArmPosition == 2 && strings.TrimSpace(r.PreferenceReason) == "" {
		return fmt.Errorf("second arm requires preference_reason")
	}
	return nil
}

func summarize(rows []Receipt) ArmSummary {
	out := ArmSummary{Runs: len(rows)}
	var times, conf []float64
	for _, r := range rows {
		if r.Succeeded {
			out.Successes++
		}
		if r.Verified {
			out.Verified++
		}
		out.TotalErrors += len(r.Errors)
		out.TotalHelpRequests += r.HelpRequests
		times = append(times, r.ElapsedSeconds)
		conf = append(conf, float64(r.Confidence))
	}
	out.MedianSeconds = median(times)
	out.MedianConfidence = median(conf)
	return out
}
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	m := len(v) / 2
	if len(v)%2 == 1 {
		return v[m]
	}
	return (v[m-1] + v[m]) / 2
}
func decodeStrict(raw []byte, v any) error {
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.DisallowUnknownFields()
	return d.Decode(v)
}
func digest(raw []byte) string { s := sha256.Sum256(raw); return "sha256:" + hex.EncodeToString(s[:]) }
func isSHA256(v string) bool {
	if !strings.HasPrefix(v, "sha256:") || len(v) != 71 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(v, "sha256:"))
	return err == nil
}
