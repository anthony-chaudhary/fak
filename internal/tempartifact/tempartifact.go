// Package tempartifact inventories and conservatively reaps direct fak artifacts
// from the resolved OS temporary directory.
package tempartifact

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-temp-artifacts/1"

const (
	ReasonEligible                   = "eligible"
	ReasonFresh                      = "fresh"
	ReasonActiveReference            = "active_reference"
	ReasonInspectionUnavailable      = "process_inspection_unavailable"
	ReasonReparsePoint               = "reparse_point"
	ReasonNotRegular                 = "not_regular"
	ReasonInaccessible               = "inaccessible"
	ReasonChangedSinceScan           = "changed_since_scan"
	ReasonMoveFailed                 = "move_failed"
	ReasonPostMoveReference          = "post_move_reference"
	ReasonPostMoveInspectUnavailable = "post_move_inspection_unavailable"
	ReasonPostMoveRecheckFailed      = "post_move_recheck_failed"
	ReasonDeleteFailed               = "delete_failed"
	ReasonReaped                     = "reaped"
	ReasonQuarantineCreateFailed     = "quarantine_create_failed"
)

// Inspection is a candidate-scoped snapshot of live process references. The
// map contains canonical exact paths only; command-line contents never leave
// the platform adapter.
type Inspection struct {
	Complete   bool
	References map[string]bool
	Reason     string
}

type InspectFunc func(context.Context, []string) Inspection

type Config struct {
	Root       string
	MinAge     time.Duration
	Now        func() time.Time
	Apply      bool
	Inspect    InspectFunc
	BeforeMove func(string)
	AfterMove  func(string, string)
	Rename     func(string, string) error
	Remove     func(string) error
}

type Item struct {
	Path           string `json:"path"`
	QuarantinePath string `json:"quarantine_path,omitempty"`
	AgeSeconds     int64  `json:"age_seconds"`
	Bytes          int64  `json:"bytes"`
	Eligible       bool   `json:"eligible"`
	Reason         string `json:"reason"`
}

type Summary struct {
	MatchingCount  int   `json:"matching_count"`
	MatchingBytes  int64 `json:"matching_bytes"`
	EligibleCount  int   `json:"eligible_count"`
	EligibleBytes  int64 `json:"eligible_bytes"`
	PreservedCount int   `json:"preserved_count"`
	PreservedBytes int64 `json:"preserved_bytes"`
	ReapedCount    int   `json:"reaped_count"`
	ReapedBytes    int64 `json:"reaped_bytes"`
}

type Report struct {
	Schema        string   `json:"schema"`
	Mode          string   `json:"mode"`
	Root          string   `json:"root"`
	MinAgeSeconds int64    `json:"min_age_seconds"`
	Inspection    string   `json:"inspection"`
	Items         []Item   `json:"items"`
	Summary       Summary  `json:"summary"`
	Warnings      []string `json:"warnings,omitempty"`
}

type scannedFile struct {
	itemIndex int
	info      os.FileInfo
}

func Run(ctx context.Context, cfg Config) (Report, error) {
	report := Report{
		Schema:        Schema,
		Mode:          "preview",
		MinAgeSeconds: int64(cfg.MinAge / time.Second),
		Inspection:    "complete",
		Items:         []Item{},
	}
	if cfg.Apply {
		report.Mode = "apply"
	}
	if cfg.MinAge <= 0 {
		return report, fmt.Errorf("minimum age must be positive")
	}

	root, err := resolveRoot(cfg.Root)
	if err != nil {
		return report, err
	}
	report.Root = root

	now := time.Now
	if cfg.Now != nil {
		now = cfg.Now
	}
	inspect := cfg.Inspect
	if inspect == nil {
		inspect = inspectLiveProcessPaths
	}
	rename := cfg.Rename
	if rename == nil {
		rename = os.Rename
	}
	remove := cfg.Remove
	if remove == nil {
		remove = os.Remove
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return report, fmt.Errorf("read OS temp root %q: %w", root, err)
	}
	currentTime := now()
	var stale []scannedFile
	for _, entry := range entries {
		name := entry.Name()
		if !allowedName(name) {
			continue
		}
		path := filepath.Clean(filepath.Join(root, name))
		if !directChild(root, path) {
			continue
		}
		info, statErr := os.Lstat(path)
		if statErr != nil {
			report.Items = append(report.Items, Item{Path: path, Reason: ReasonInaccessible})
			continue
		}
		age := currentTime.Sub(info.ModTime())
		if age < 0 {
			age = 0
		}
		item := Item{Path: path, AgeSeconds: int64(age / time.Second), Bytes: info.Size()}
		reparse, reparseErr := isReparsePoint(path, info)
		switch {
		case reparseErr != nil:
			item.Reason = ReasonInaccessible
		case reparse:
			item.Reason = ReasonReparsePoint
		case !info.Mode().IsRegular():
			item.Reason = ReasonNotRegular
		case age < cfg.MinAge:
			item.Reason = ReasonFresh
		default:
			item.Reason = ReasonEligible
		}
		report.Items = append(report.Items, item)
		if item.Reason == ReasonEligible {
			stale = append(stale, scannedFile{itemIndex: len(report.Items) - 1, info: info})
		}
	}

	paths := make([]string, 0, len(stale))
	for _, file := range stale {
		paths = append(paths, report.Items[file.itemIndex].Path)
	}
	initialInspection := inspect(ctx, paths)
	if !initialInspection.Complete && len(paths) > 0 {
		report.Inspection = inspectionReason(initialInspection)
	}
	for _, file := range stale {
		item := &report.Items[file.itemIndex]
		switch {
		case !initialInspection.Complete:
			item.Reason = ReasonInspectionUnavailable
		case inspectionReferences(initialInspection, item.Path):
			item.Reason = ReasonActiveReference
		}
		item.Eligible = item.Reason == ReasonEligible
	}

	if cfg.Apply {
		apply(ctx, &report, stale, inspect, rename, remove, cfg.BeforeMove, cfg.AfterMove)
	}
	sort.Slice(report.Items, func(i, j int) bool { return pathKey(report.Items[i].Path) < pathKey(report.Items[j].Path) })
	report.Summary = summarize(report.Items)
	return report, nil
}

func apply(
	ctx context.Context,
	report *Report,
	stale []scannedFile,
	inspect InspectFunc,
	rename func(string, string) error,
	remove func(string) error,
	beforeMove func(string),
	afterMove func(string, string),
) {
	eligible := make([]scannedFile, 0, len(stale))
	for _, file := range stale {
		if report.Items[file.itemIndex].Reason == ReasonEligible {
			eligible = append(eligible, file)
		}
	}
	if len(eligible) == 0 {
		return
	}

	quarantine, err := os.MkdirTemp(report.Root, "fak-maintenance-quarantine-")
	if err != nil {
		for _, file := range eligible {
			report.Items[file.itemIndex].Reason = ReasonQuarantineCreateFailed
		}
		return
	}

	for _, file := range eligible {
		item := &report.Items[file.itemIndex]
		source := item.Path
		if beforeMove != nil {
			beforeMove(source)
		}
		current, err := os.Lstat(source)
		if err != nil || !sameIdentity(file.info, current) {
			item.Reason = ReasonChangedSinceScan
			continue
		}
		preMove := inspect(ctx, []string{source})
		switch {
		case !preMove.Complete:
			item.Reason = ReasonInspectionUnavailable
			continue
		case inspectionReferences(preMove, source):
			item.Reason = ReasonActiveReference
			continue
		}

		destination := filepath.Join(quarantine, filepath.Base(source))
		if err := rename(source, destination); err != nil {
			item.Reason = ReasonMoveFailed
			continue
		}
		item.QuarantinePath = destination
		if afterMove != nil {
			afterMove(source, destination)
		}
		moved, err := os.Lstat(destination)
		if err != nil || !sameIdentity(file.info, moved) || sourceStillExists(source) {
			item.Reason = ReasonPostMoveRecheckFailed
			continue
		}
		postMove := inspect(ctx, []string{source, destination})
		switch {
		case !postMove.Complete:
			item.Reason = ReasonPostMoveInspectUnavailable
			continue
		case inspectionReferences(postMove, source) || inspectionReferences(postMove, destination):
			item.Reason = ReasonPostMoveReference
			continue
		}
		if err := remove(destination); err != nil {
			item.Reason = ReasonDeleteFailed
			continue
		}
		item.QuarantinePath = ""
		item.Reason = ReasonReaped
	}

	if err := os.Remove(quarantine); err != nil && !errors.Is(err, os.ErrNotExist) {
		report.Warnings = append(report.Warnings, "quarantine_not_empty")
	}
}

func resolveRoot(configured string) (string, error) {
	root := strings.TrimSpace(configured)
	if root == "" {
		root = os.TempDir()
	}
	abs, err := filepath.Abs(filepath.Clean(root))
	if err != nil {
		return "", fmt.Errorf("canonicalize OS temp root: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("resolve OS temp root %q: %w", abs, err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("stat OS temp root %q: %w", resolved, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("OS temp root %q is not a directory", resolved)
	}
	return filepath.Clean(resolved), nil
}

func allowedName(name string) bool {
	if !strings.HasPrefix(name, "fak-") {
		return false
	}
	switch strings.ToLower(filepath.Ext(name)) {
	case ".exe", ".tar", ".zip":
		return true
	default:
		return false
	}
}

func directChild(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	return err == nil && relative != "." && relative != ".." && !filepath.IsAbs(relative) && filepath.Dir(relative) == "."
}

func sameIdentity(before, after os.FileInfo) bool {
	return before != nil && after != nil && os.SameFile(before, after) && before.Size() == after.Size() && before.ModTime().Equal(after.ModTime()) && before.Mode() == after.Mode()
}

func sourceStillExists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil || !errors.Is(err, os.ErrNotExist)
}

func inspectionReason(inspection Inspection) string {
	if strings.TrimSpace(inspection.Reason) != "" {
		return inspection.Reason
	}
	return ReasonInspectionUnavailable
}

func inspectionReferences(inspection Inspection, path string) bool {
	return inspection.References[pathKey(path)] || inspection.References[path]
}

func pathKey(path string) string {
	clean := filepath.Clean(path)
	if filepath.Separator == '\\' {
		return strings.ToLower(clean)
	}
	return clean
}

func summarize(items []Item) Summary {
	var summary Summary
	for _, item := range items {
		summary.MatchingCount++
		summary.MatchingBytes += item.Bytes
		if item.Eligible {
			summary.EligibleCount++
			summary.EligibleBytes += item.Bytes
		}
		switch item.Reason {
		case ReasonReaped:
			summary.ReapedCount++
			summary.ReapedBytes += item.Bytes
		case ReasonEligible:
		default:
			summary.PreservedCount++
			summary.PreservedBytes += item.Bytes
		}
	}
	return summary
}
