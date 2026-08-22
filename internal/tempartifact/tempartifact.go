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
	ReasonNestedUnknown              = "nested_unknown"
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
	RemoveAll  func(string) error
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
	directory bool
	tree      []treeEntry
}

type treeEntry struct {
	relative string
	info     os.FileInfo
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
	removeAll := cfg.RemoveAll
	if removeAll == nil {
		removeAll = os.RemoveAll
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return report, fmt.Errorf("read OS temp root %q: %w", root, err)
	}
	currentTime := now()
	var stale []scannedFile
	for _, entry := range entries {
		name := entry.Name()
		if !allowedName(name) && !allowedDirectoryName(name) {
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
		if info.IsDir() && !allowedDirectoryName(name) {
			continue
		}
		age := currentTime.Sub(info.ModTime())
		if age < 0 {
			age = 0
		}
		item := Item{Path: path, AgeSeconds: int64(age / time.Second), Bytes: info.Size()}
		reparse, reparseErr := isReparsePoint(path, info)
		var tree []treeEntry
		switch {
		case reparseErr != nil:
			item.Reason = ReasonInaccessible
		case reparse:
			item.Reason = ReasonReparsePoint
		case info.IsDir():
			var newest time.Time
			tree, item.Bytes, newest, item.Reason = scanDirectory(path)
			if newest.After(info.ModTime()) {
				age = currentTime.Sub(newest)
				if age < 0 {
					age = 0
				}
				item.AgeSeconds = int64(age / time.Second)
			}
		case !info.Mode().IsRegular():
			item.Reason = ReasonNotRegular
		}
		if item.Reason == "" {
			if age < cfg.MinAge {
				item.Reason = ReasonFresh
			} else {
				item.Reason = ReasonEligible
			}
		}
		report.Items = append(report.Items, item)
		if item.Reason == ReasonEligible {
			stale = append(stale, scannedFile{itemIndex: len(report.Items) - 1, info: info, directory: info.IsDir(), tree: tree})
		}
	}

	paths := make([]string, 0, len(stale))
	for _, file := range stale {
		paths = append(paths, inspectionPaths(report.Items[file.itemIndex].Path, file.tree)...)
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
		case inspectionReferencesAny(initialInspection, inspectionPaths(item.Path, file.tree)):
			item.Reason = ReasonActiveReference
		}
		item.Eligible = item.Reason == ReasonEligible
	}

	if cfg.Apply {
		apply(ctx, &report, stale, inspect, rename, remove, removeAll, cfg.BeforeMove, cfg.AfterMove)
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
	removeAll func(string) error,
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
		if err != nil || !sameIdentity(file.info, current) || !sameTree(source, file.tree) {
			item.Reason = ReasonChangedSinceScan
			continue
		}
		preMove := inspect(ctx, inspectionPaths(source, file.tree))
		switch {
		case !preMove.Complete:
			item.Reason = ReasonInspectionUnavailable
			continue
		case inspectionReferencesAny(preMove, inspectionPaths(source, file.tree)):
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
		if err != nil || !sameIdentity(file.info, moved) || !sameTree(destination, file.tree) || sourceStillExists(source) {
			item.Reason = ReasonPostMoveRecheckFailed
			continue
		}
		postMove := inspect(ctx, append(inspectionPaths(source, file.tree), inspectionPaths(destination, file.tree)...))
		switch {
		case !postMove.Complete:
			item.Reason = ReasonPostMoveInspectUnavailable
			continue
		case inspectionReferencesAny(postMove, append(inspectionPaths(source, file.tree), inspectionPaths(destination, file.tree)...)):
			item.Reason = ReasonPostMoveReference
			continue
		}
		deleteArtifact := remove
		if file.directory {
			deleteArtifact = removeAll
		}
		if err := deleteArtifact(destination); err != nil {
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

func allowedDirectoryName(name string) bool {
	return strings.HasPrefix(name, "fak-issue-") && name != "fak-issue-"
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

func scanDirectory(root string) ([]treeEntry, int64, time.Time, string) {
	entries := []treeEntry{}
	var bytes int64
	var newest time.Time
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}
		reparse, err := isReparsePoint(path, info)
		if err != nil {
			return err
		}
		if reparse || (!info.IsDir() && !info.Mode().IsRegular()) {
			return errNestedUnknown
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		entries = append(entries, treeEntry{relative: relative, info: info})
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		if info.Mode().IsRegular() {
			bytes += info.Size()
		}
		return nil
	})
	if errors.Is(err, errNestedUnknown) {
		return nil, bytes, newest, ReasonNestedUnknown
	}
	if err != nil {
		return nil, bytes, newest, ReasonInaccessible
	}
	return entries, bytes, newest, ""
}

var errNestedUnknown = errors.New("nested unknown entry")

func sameTree(root string, before []treeEntry) bool {
	after, _, _, reason := scanDirectory(root)
	if reason != "" || len(before) != len(after) {
		return false
	}
	for index := range before {
		if before[index].relative != after[index].relative || !sameIdentity(before[index].info, after[index].info) {
			return false
		}
	}
	return true
}

func inspectionPaths(root string, tree []treeEntry) []string {
	paths := make([]string, 0, len(tree)+1)
	paths = append(paths, root)
	for _, entry := range tree {
		paths = append(paths, filepath.Join(root, entry.relative))
	}
	return paths
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

func inspectionReferencesAny(inspection Inspection, paths []string) bool {
	for _, path := range paths {
		if inspectionReferences(inspection, path) {
			return true
		}
	}
	return false
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
