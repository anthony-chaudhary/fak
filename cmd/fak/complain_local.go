package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardcomplaint"
)

const complaintPendingSchema = "fak.complaint.pending.v1"

type complaintPendingReceipt struct {
	Schema    string                   `json:"schema"`
	Key       string                   `json:"key"`
	Repo      string                   `json:"repo,omitempty"`
	Complaint guardcomplaint.Complaint `json:"complaint"`
	Time      time.Time                `json:"time"`
	Path      string                   `json:"path,omitempty"`
}

type complaintPendingResult struct {
	Schema      string                  `json:"schema"`
	Mode        string                  `json:"mode"`
	PendingPath string                  `json:"pending_path"`
	Synced      []dogfoodissues.SyncRow `json:"synced"`
}

func complaintPendingDir(workspace string) string {
	return filepath.Join(workspace, ".fak", "complaints", "pending")
}

func complaintPendingPath(workspace, key string) string {
	digest := sha256.Sum256([]byte(key))
	return filepath.Join(complaintPendingDir(workspace), hex.EncodeToString(digest[:])+".json")
}

func writeComplaintPending(workspace, repo string, complaint guardcomplaint.Complaint) (string, error) {
	complaint = scrubPendingComplaint(complaint)
	key := complaint.Key()
	path := complaintPendingPath(workspace, key)
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return path, err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return path, err
	}
	// A retry for the same stable key reuses the already durable receipt. Besides
	// being idempotent, this avoids relying on rename-over-existing behavior, which
	// differs between Unix and Windows.
	if info, err := os.Stat(path); err == nil {
		if info.IsDir() {
			return path, fmt.Errorf("pending receipt path is a directory")
		}
		if err := validateComplaintPending(path, key); err != nil {
			return path, err
		}
		if err := os.Chmod(path, 0o600); err != nil {
			return path, err
		}
		return path, nil
	} else if !os.IsNotExist(err) {
		return path, err
	}

	receipt := complaintPendingReceipt{
		Schema:    complaintPendingSchema,
		Key:       key,
		Repo:      strings.TrimSpace(repo),
		Complaint: complaint,
		Time:      time.Now().UTC(),
	}
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return path, err
	}
	raw = append(raw, '\n')

	tmp, err := os.CreateTemp(dir, ".pending-*.tmp")
	if err != nil {
		return path, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return path, err
	}
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return path, err
	}
	if err := tmp.Close(); err != nil {
		return path, err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		// Another writer can win between the existence check and rename. Its
		// same-key receipt satisfies this write; any other collision remains an
		// error and the winning file is left untouched.
		if validateErr := validateComplaintPending(path, key); validateErr == nil {
			return path, nil
		}
		return path, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		return path, err
	}
	return path, nil
}

func validateComplaintPending(path, key string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var receipt complaintPendingReceipt
	if err := json.Unmarshal(raw, &receipt); err != nil {
		return err
	}
	if receipt.Schema != complaintPendingSchema || receipt.Key != key {
		return fmt.Errorf("existing pending receipt is invalid for stable key %q", key)
	}
	return nil
}

func scrubPendingComplaint(c guardcomplaint.Complaint) guardcomplaint.Complaint {
	c.Reason = strings.TrimSpace(agentopt.ScrubText(c.Reason))
	c.Tool = strings.TrimSpace(agentopt.ScrubText(c.Tool))
	c.Summary = strings.TrimSpace(agentopt.ScrubText(c.Summary))
	c.Rationale = strings.TrimSpace(agentopt.ScrubText(c.Rationale))
	if c.Evidence != nil {
		e := *c.Evidence
		e.Source = agentopt.ScrubText(e.Source)
		e.JournalPath = agentopt.ScrubText(e.JournalPath)
		e.Verdict = agentopt.ScrubText(e.Verdict)
		e.Tool = agentopt.ScrubText(e.Tool)
		e.Reason = agentopt.ScrubText(e.Reason)
		e.By = agentopt.ScrubText(e.By)
		e.TraceID = agentopt.ScrubText(e.TraceID)
		e.ArgsDigest = agentopt.ScrubText(e.ArgsDigest)
		e.DenyRule = agentopt.ScrubText(e.DenyRule)
		c.Evidence = &e
	}
	return c
}

func removeComplaintPending(path string) error {
	err := os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func readComplaintPending(workspace string) ([]complaintPendingReceipt, error) {
	dir := complaintPendingDir(workspace)
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return []complaintPendingReceipt{}, nil
	}
	if err != nil {
		return nil, err
	}

	receipts := make([]complaintPendingReceipt, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var receipt complaintPendingReceipt
		if err := json.Unmarshal(raw, &receipt); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if receipt.Schema != complaintPendingSchema || receipt.Key == "" {
			return nil, fmt.Errorf("%s: invalid pending complaint receipt", path)
		}
		receipt.Path = path
		receipts = append(receipts, receipt)
	}
	sort.Slice(receipts, func(i, j int) bool {
		if receipts[i].Time.Equal(receipts[j].Time) {
			return receipts[i].Key < receipts[j].Key
		}
		return receipts[i].Time.Before(receipts[j].Time)
	})
	return receipts, nil
}

func runComplainPending(stdout, stderr io.Writer, workspace string, asJSON bool) int {
	receipts, err := readComplaintPending(workspace)
	if err != nil {
		fmt.Fprintf(stderr, "fak complain: list pending complaints: %v\n", err)
		return 2
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(receipts); err != nil {
			fmt.Fprintf(stderr, "fak complain: encode pending complaints: %v\n", err)
			return 2
		}
		return 0
	}
	for _, receipt := range receipts {
		fmt.Fprintf(stdout, "%s\t%s\n", receipt.Key, receipt.Path)
	}
	return 0
}

func emitComplaintPending(stdout, stderr io.Writer, path string, asJSON bool) error {
	result := complaintPendingResult{
		Schema:      complaintPendingSchema,
		Mode:        "pending-local",
		PendingPath: path,
		Synced:      []dogfoodissues.SyncRow{},
	}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}
	_, err := fmt.Fprintf(stdout, "pending-local\t%s\n", path)
	return err
}
