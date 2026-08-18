package harnesscontrolpacket

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnesscontrolstudy"
)

// ReceiptStartOptions declares the assigned identity and order before an arm clock starts.
type ReceiptStartOptions struct {
	Dir           string
	ParticipantID string
	PairID        string
	PairOrder     string
	Now           func() time.Time
}

// ReceiptFinalizeOptions records operator-observed outcomes and binds the produced artifact.
type ReceiptFinalizeOptions struct {
	Dir                   string
	ArtifactPath          string
	CommandsPath          string
	ErrorsPath            string
	Succeeded             bool
	Verified              bool
	HelpRequests          int
	Confidence            int
	InspectCaptured       bool
	PreviewCaptured       bool
	RuntimeVerifyCaptured bool
	Preference            string
	PreferenceReason      string
	Now                   func() time.Time
}

// StartReceipt verifies the unopened packet, then replaces its blank receipt with a clocked receipt.
func StartReceipt(opts ReceiptStartOptions) (harnesscontrolstudy.Receipt, error) {
	if strings.TrimSpace(opts.ParticipantID) == "" || strings.TrimSpace(opts.PairID) == "" {
		return harnesscontrolstudy.Receipt{}, fmt.Errorf("participant-id and pair-id are required")
	}
	if opts.PairOrder != "default-first" && opts.PairOrder != "scratch-first" {
		return harnesscontrolstudy.Receipt{}, fmt.Errorf("pair-order must be default-first or scratch-first")
	}
	manifest, err := readAndVerifyPacket(opts.Dir)
	if err != nil {
		return harnesscontrolstudy.Receipt{}, err
	}
	var blank harnesscontrolstudy.Receipt
	blankRaw, err := os.ReadFile(filepath.Join(opts.Dir, "receipt.json"))
	if err != nil || json.Unmarshal(blankRaw, &blank) != nil || blank.ParticipantID != "person-random" || blank.ArtifactDigest != "sha256:REPLACE" {
		return harnesscontrolstudy.Receipt{}, fmt.Errorf("receipt is not an unopened packet template")
	}
	position := 2
	if (opts.PairOrder == "default-first" && manifest.Arm == "default-control") || (opts.PairOrder == "scratch-first" && manifest.Arm == "scratch") {
		position = 1
	}
	taskDigest, err := digestFile(filepath.Join(opts.Dir, "task-card.md"))
	if err != nil {
		return harnesscontrolstudy.Receipt{}, fmt.Errorf("task card: %w", err)
	}
	baseLockID := ""
	if manifest.Arm == "default-control" {
		var lock struct {
			ID string `json:"id"`
		}
		raw, readErr := os.ReadFile(filepath.Join(opts.Dir, "product.lock.json"))
		if readErr != nil || json.Unmarshal(raw, &lock) != nil || !validDigest(lock.ID) {
			return harnesscontrolstudy.Receipt{}, fmt.Errorf("default-control packet has invalid product.lock.json id")
		}
		baseLockID = lock.ID
	}
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}
	r := harnesscontrolstudy.Receipt{
		Schema: harnesscontrolstudy.ReceiptSchema, StudyID: manifest.StudyID, TaskDigest: taskDigest,
		ParticipantID: strings.TrimSpace(opts.ParticipantID), PairID: strings.TrimSpace(opts.PairID), PairOrder: opts.PairOrder,
		Arm: manifest.Arm, ArmPosition: position, StartedAt: now().UTC().Format(time.RFC3339Nano), Errors: []string{}, Commands: []string{},
		BinaryVersion: manifest.BinaryVersion, BinaryCommit: manifest.SourceCommit, BaseLockID: baseLockID,
	}
	if err := writeReceipt(filepath.Join(opts.Dir, "receipt.json"), r); err != nil {
		return harnesscontrolstudy.Receipt{}, err
	}
	return r, nil
}

// FinalizeReceipt closes a started receipt with exact elapsed time and artifact provenance.
func FinalizeReceipt(opts ReceiptFinalizeOptions) (harnesscontrolstudy.Receipt, error) {
	path := filepath.Join(opts.Dir, "receipt.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		return harnesscontrolstudy.Receipt{}, err
	}
	var r harnesscontrolstudy.Receipt
	if err := json.Unmarshal(raw, &r); err != nil {
		return r, fmt.Errorf("parse started receipt: %w", err)
	}
	if r.Schema != harnesscontrolstudy.ReceiptSchema || r.StartedAt == "" || r.StoppedAt != "" {
		return r, fmt.Errorf("receipt is not in started state")
	}
	if opts.Confidence < 1 || opts.Confidence > 5 || opts.HelpRequests < 0 {
		return r, fmt.Errorf("confidence must be 1..5 and help-requests non-negative")
	}
	commands, err := nonemptyLines(opts.CommandsPath, true)
	if err != nil {
		return r, fmt.Errorf("commands: %w", err)
	}
	errors, err := nonemptyLines(opts.ErrorsPath, false)
	if err != nil {
		return r, fmt.Errorf("errors: %w", err)
	}
	artifactDigest, err := digestFile(opts.ArtifactPath)
	if err != nil {
		return r, fmt.Errorf("artifact: %w", err)
	}
	started, err := time.Parse(time.RFC3339Nano, r.StartedAt)
	if err != nil {
		return r, fmt.Errorf("started_at: %w", err)
	}
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}
	stopped := now().UTC()
	if stopped.Before(started) {
		return r, fmt.Errorf("stop time precedes start time")
	}
	if r.ArmPosition == 2 && (opts.Preference == "" || strings.TrimSpace(opts.PreferenceReason) == "") {
		return r, fmt.Errorf("second arm requires preference and preference-reason")
	}
	if opts.Preference != "" && opts.Preference != "default-control" && opts.Preference != "scratch" && opts.Preference != "none" {
		return r, fmt.Errorf("preference must be default-control, scratch, or none")
	}
	r.StoppedAt = stopped.Format(time.RFC3339Nano)
	r.ElapsedSeconds = stopped.Sub(started).Seconds()
	r.Succeeded, r.Verified = opts.Succeeded, opts.Verified
	r.Errors, r.HelpRequests, r.Confidence, r.Commands = errors, opts.HelpRequests, opts.Confidence, commands
	r.ArtifactDigest = artifactDigest
	r.InspectCaptured, r.PreviewCaptured, r.RuntimeVerifyCaptured = opts.InspectCaptured, opts.PreviewCaptured, opts.RuntimeVerifyCaptured
	r.Preference, r.PreferenceReason = opts.Preference, strings.TrimSpace(opts.PreferenceReason)
	if err := writeReceipt(path, r); err != nil {
		return r, err
	}
	return r, nil
}

func readAndVerifyPacket(dir string) (Manifest, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "packet.json"))
	if err != nil {
		return Manifest{}, err
	}
	m, err := Parse(raw)
	if err == nil {
		err = Verify(dir, m)
	}
	return m, err
}

func digestFile(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("path is required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func validDigest(v string) bool {
	if !strings.HasPrefix(v, "sha256:") || len(v) != 71 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(v, "sha256:"))
	return err == nil
}

func nonemptyLines(path string, required bool) ([]string, error) {
	if strings.TrimSpace(path) == "" {
		if required {
			return nil, fmt.Errorf("file is required")
		}
		return []string{}, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		if line := strings.TrimSpace(s.Text()); line != "" {
			lines = append(lines, line)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if required && len(lines) == 0 {
		return nil, fmt.Errorf("file has no commands")
	}
	return lines, nil
}

func writeReceipt(path string, r harnesscontrolstudy.Receipt) error {
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
