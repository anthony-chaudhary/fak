package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

const syncPublicLeakGate = "PUBLIC_LEAK"

type syncCheckReport struct {
	safesync.Assessment
	PublicLeak *syncPublicLeakPreflight `json:"public_leak,omitempty"`
}

type syncPublicLeakPreflight struct {
	Gate            string                  `json:"gate"`
	OK              bool                    `json:"ok"`
	Count           int                     `json:"count"`
	BlockingCount   int                     `json:"blocking_count"`
	IntroducedCount int                     `json:"introduced_count"`
	InheritedCount  int                     `json:"inherited_count"`
	UnknownCount    int                     `json:"unknown_count"`
	Findings        []syncPublicLeakFinding `json:"findings"`
	RepairSlices    []syncPublicLeakRepair  `json:"repair_slices,omitempty"`
	LaneResolution  string                  `json:"lane_resolution,omitempty"`
	TargetedRecheck string                  `json:"targeted_recheck,omitempty"`
	ResumeToken     string                  `json:"resume_token,omitempty"`
	ResumeCommand   string                  `json:"resume_command,omitempty"`
	RecheckedPaths  []string                `json:"rechecked_paths,omitempty"`
	ResumeValidated bool                    `json:"resume_validated,omitempty"`
}

type syncPublicLeakFinding struct {
	ID          string `json:"id"`
	Path        string `json:"path"`
	Gate        string `json:"gate"`
	Line        int    `json:"line,omitempty"`
	Detail      string `json:"detail"`
	Provenance  string `json:"provenance"`
	Blocking    bool   `json:"blocking"`
	Attributive bool   `json:"attributive"`
}

type syncPublicLeakRepair struct {
	ID             string   `json:"id"`
	Lane           string   `json:"lane,omitempty"`
	Paths          []string `json:"paths"`
	LaneResolution string   `json:"lane_resolution"`
}

func normalizeSyncRecheckPaths(paths []string) ([]string, error) {
	seen := map[string]bool{}
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
		clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(path)))
		if path == "" || filepath.IsAbs(filepath.FromSlash(path)) || clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
			return nil, fmt.Errorf("--recheck-path must be a repo-relative path: %q", path)
		}
		if !seen[clean] {
			seen[clean] = true
			out = append(out, clean)
		}
	}
	sort.Strings(out)
	return out, nil
}

func assessSyncPublicLeak(repo, remote string, info safesync.Assessment, only []string) (syncPublicLeakPreflight, error) {
	report := syncPublicLeakPreflight{
		Gate:           syncPublicLeakGate,
		OK:             true,
		RecheckedPaths: append([]string(nil), only...),
	}
	tree, err := hooks.AuditPublicLeakTree(repo)
	if err != nil {
		return report, err
	}
	current := filterSyncPublicLeakFindings(tree.Findings, only)

	introduced := map[string]bool{}
	if info.Target != "" && info.Head != "" && info.Target != info.Head {
		findings, scanErr := hooks.ScanRangePublicLeak(repo, info.Target+".."+info.Head)
		if scanErr == nil {
			for _, finding := range findings {
				introduced[syncPublicLeakFindingKey(finding)] = true
			}
		}
	}

	sort.Slice(current, func(i, j int) bool {
		if current[i].File != current[j].File {
			return current[i].File < current[j].File
		}
		if current[i].Line != current[j].Line {
			return current[i].Line < current[j].Line
		}
		return current[i].Detail < current[j].Detail
	})
	for _, finding := range current {
		provenance := "unknown"
		if introduced[syncPublicLeakFindingKey(finding)] {
			provenance = "introduced"
		} else if syncPublicLeakExistsAtBaseline(repo, info.Target, finding) {
			provenance = "inherited"
		}
		blocking := provenance != "inherited"
		item := syncPublicLeakFinding{
			ID:          syncPublicLeakFindingID(finding),
			Path:        filepath.ToSlash(finding.File),
			Gate:        syncPublicLeakGate,
			Line:        finding.Line,
			Detail:      finding.Detail,
			Provenance:  provenance,
			Blocking:    blocking,
			Attributive: provenance == "introduced",
		}
		report.Findings = append(report.Findings, item)
		switch provenance {
		case "introduced":
			report.IntroducedCount++
		case "inherited":
			report.InheritedCount++
		default:
			report.UnknownCount++
		}
		if blocking {
			report.BlockingCount++
		}
	}
	report.Count = len(report.Findings)
	report.OK = report.BlockingCount == 0
	if report.BlockingCount == 0 {
		return report, nil
	}

	actionable := syncPublicLeakActionablePaths(report.Findings)
	report.RepairSlices = syncPublicLeakRepairSlices(repo, actionable)
	report.LaneResolution = "actionable paths are classified with classifyDirty and hooksLaneResolver; resolved lanes are disjoint worker slices and unresolved paths stay isolated"
	report.TargetedRecheck = shellJoin(syncPublicLeakCommand("check", repo, remote, info.Branch, actionable, ""))
	report.ResumeToken = syncPublicLeakOperationToken(repo, remote, info)
	report.ResumeCommand = shellJoin(syncPublicLeakCommand("check", repo, remote, info.Branch, nil, report.ResumeToken))
	return report, nil
}

func filterSyncPublicLeakFindings(findings []hooks.Finding, only []string) []hooks.Finding {
	if len(only) == 0 {
		return append([]hooks.Finding(nil), findings...)
	}
	wanted := map[string]bool{}
	for _, path := range only {
		wanted[path] = true
	}
	var out []hooks.Finding
	for _, finding := range findings {
		if wanted[filepath.ToSlash(finding.File)] {
			out = append(out, finding)
		}
	}
	return out
}

func syncPublicLeakFindingKey(finding hooks.Finding) string {
	return fmt.Sprintf("%s\x00%s\x00%d\x00%s", finding.Gate, filepath.ToSlash(finding.File), finding.Line, finding.Detail)
}

func syncPublicLeakFindingID(finding hooks.Finding) string {
	sum := sha256.Sum256([]byte(syncPublicLeakFindingKey(finding)))
	return "public-leak:" + hex.EncodeToString(sum[:12])
}

// syncPublicLeakExistsAtBaseline proves inheritance only when the exact current path or line
// exists at the assessed remote target. Line shifts, missing refs, and unreadable blobs remain
// unknown; this intentionally prefers a blocking unknown over guessed attribution.
func syncPublicLeakExistsAtBaseline(repo, target string, finding hooks.Finding) bool {
	if target == "" || finding.File == "" {
		return false
	}
	cmd := exec.Command("git", "show", target+":"+filepath.ToSlash(finding.File))
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	baseline, err := cmd.Output()
	if err != nil {
		return false
	}
	if finding.Line == 0 {
		return true
	}
	current, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(finding.File)))
	if err != nil {
		return false
	}
	baseLines := strings.Split(string(baseline), "\n")
	currentLines := strings.Split(string(current), "\n")
	line := finding.Line - 1
	return line >= 0 && line < len(baseLines) && line < len(currentLines) && strings.TrimRight(baseLines[line], "\r") == strings.TrimRight(currentLines[line], "\r")
}

func syncPublicLeakActionablePaths(findings []syncPublicLeakFinding) []string {
	seen := map[string]bool{}
	var paths []string
	for _, finding := range findings {
		if !finding.Blocking || finding.Path == "" || seen[finding.Path] {
			continue
		}
		seen[finding.Path] = true
		paths = append(paths, finding.Path)
	}
	sort.Strings(paths)
	return paths
}

func syncPublicLeakRepairSlices(repo string, paths []string) []syncPublicLeakRepair {
	entries := make([]dirtyEntry, 0, len(paths))
	for _, path := range paths {
		entries = append(entries, dirtyEntry{Path: path, Status: "M", WorktreeDirty: true})
	}
	plan := classifyDirty(entries, hooksLaneResolver(repo), nil)
	var repairs []syncPublicLeakRepair
	for _, group := range plan.Groups {
		repairs = append(repairs, syncPublicLeakRepair{
			Lane: group.Lane, Paths: append([]string(nil), group.Paths...), LaneResolution: "resolved",
		})
	}
	for _, entry := range plan.NoLane {
		repairs = append(repairs, syncPublicLeakRepair{Paths: []string{entry.Path}, LaneResolution: "required"})
	}
	for _, entry := range plan.Junk {
		repairs = append(repairs, syncPublicLeakRepair{Paths: []string{entry.Path}, LaneResolution: "manual"})
	}
	sort.Slice(repairs, func(i, j int) bool {
		return strings.Join(repairs[i].Paths, "\x00") < strings.Join(repairs[j].Paths, "\x00")
	})
	for i := range repairs {
		repairs[i].ID = fmt.Sprintf("repair-%03d", i+1)
	}
	return repairs
}

func syncPublicLeakOperationToken(repo, remote string, info safesync.Assessment) string {
	absRepo, err := filepath.Abs(repo)
	if err != nil {
		absRepo = filepath.Clean(repo)
	}
	payload := strings.Join([]string{
		"fak-sync-public-leak-v1", filepath.Clean(absRepo), remote, info.Branch,
		info.Head, info.Target, info.TargetRef,
	}, "\x00")
	sum := sha256.Sum256([]byte(payload))
	return "sync-public-leak-v1:" + hex.EncodeToString(sum[:])
}

func syncPublicLeakCommand(command, repo, remote, branch string, paths []string, token string) []string {
	argv := []string{"fak", "sync", command, "--repo", repo}
	if remote != "" {
		argv = append(argv, "--remote", remote)
	}
	if branch != "" {
		argv = append(argv, "--branch", branch)
	}
	for _, path := range paths {
		argv = append(argv, "--recheck-path", path)
	}
	if token != "" {
		argv = append(argv, "--resume-token", token)
	}
	return argv
}

func renderSyncPublicLeak(w io.Writer, report syncPublicLeakPreflight) {
	if report.Count == 0 {
		if len(report.RecheckedPaths) > 0 {
			fmt.Fprintln(w, "PUBLIC_LEAK targeted recheck: clean")
		} else {
			fmt.Fprintln(w, "PUBLIC_LEAK preflight: clean (resume token validated; whole gate rerun)")
		}
		return
	}
	status := "CLEAR"
	if !report.OK {
		status = "BLOCKED"
	}
	fmt.Fprintf(w, "PUBLIC_LEAK preflight: %s (%d finding(s): %d introduced, %d inherited, %d unknown; %d blocking)\n",
		status, report.Count, report.IntroducedCount, report.InheritedCount, report.UnknownCount, report.BlockingCount)
	for _, finding := range report.Findings {
		fmt.Fprintf(w, "  %s  %s:%d  provenance=%s blocking=%t  %s\n",
			finding.ID, finding.Path, finding.Line, finding.Provenance, finding.Blocking, finding.Detail)
	}
	for _, repair := range report.RepairSlices {
		lane := repair.Lane
		if lane == "" {
			lane = "(resolve lane)"
		}
		fmt.Fprintf(w, "  %s  lane=%s  paths=%s\n", repair.ID, lane, strings.Join(repair.Paths, ", "))
	}
	if report.TargetedRecheck != "" {
		fmt.Fprintf(w, "  targeted recheck: %s\n", report.TargetedRecheck)
	}
	if report.ResumeCommand != "" {
		fmt.Fprintf(w, "  resume: %s\n", report.ResumeCommand)
	}
}
