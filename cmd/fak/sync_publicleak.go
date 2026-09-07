package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
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
	ID          string                            `json:"id"`
	Path        string                            `json:"path"`
	Gate        string                            `json:"gate"`
	Line        int                               `json:"line,omitempty"`
	Detail      string                            `json:"detail"`
	Provenance  string                            `json:"provenance"`
	Blocking    bool                              `json:"blocking"`
	Attributive bool                              `json:"attributive"`
	Evidence    *syncPublicLeakOccurrenceEvidence `json:"evidence,omitempty"`
}

type syncPublicLeakOccurrenceEvidence struct {
	TargetCommit string `json:"target_commit,omitempty"`
	HeadCommit   string `json:"head_commit,omitempty"`
	Path         string `json:"path,omitempty"`
	ContentSHA   string `json:"content_sha,omitempty"`
	BaselineLine int    `json:"baseline_line,omitempty"`
	CurrentLine  int    `json:"current_line,omitempty"`
	Witness      string `json:"witness,omitempty"`
}

func (e *syncPublicLeakOccurrenceEvidence) Valid(repo string, finding syncPublicLeakFinding, target, head string) bool {
	if e == nil {
		return false
	}
	normPath := filepath.ToSlash(filepath.Clean(finding.Path))
	if normPath != e.Path || finding.Line != e.CurrentLine {
		return false
	}
	if target != "" {
		targetSHA := resolveGitCommitSHA(repo, target)
		if targetSHA != "" && targetSHA != e.TargetCommit {
			return false
		}
	}
	if head != "" {
		headSHA := resolveGitCommitSHA(repo, head)
		if headSHA != "" && headSHA != e.HeadCommit {
			return false
		}
	}
	currentContent, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(finding.Path)))
	if err != nil {
		return false
	}
	sum := sha256.Sum256(currentContent)
	if hex.EncodeToString(sum[:]) != e.ContentSHA {
		return false
	}
	return true
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
	classifier := newSyncPublicLeakClassifier(repo, info.Target, info.Head)
	for _, finding := range current {
		provenance := "unknown"
		var evidence *syncPublicLeakOccurrenceEvidence
		if introduced[syncPublicLeakFindingKey(finding)] {
			provenance = "introduced"
		} else {
			var ok bool
			ok, evidence = classifier.Adjudicate(finding)
			if ok {
				provenance = "inherited"
			}
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
			Evidence:    evidence,
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

// syncPublicLeakExistsAtBaseline proves inheritance when the finding exists at the assessed
// remote target, either at the exact line or witnessed at a relocated line via unambiguous
// Git diff occurrence mapping. Changed content, changed path, ambiguous duplicate mapping,
// unreadable baseline, or missing refs remain false (unknown/blocking).
func syncPublicLeakExistsAtBaseline(repo, target string, finding hooks.Finding) bool {
	if target == "" || finding.File == "" {
		return false
	}
	classifier := newSyncPublicLeakClassifier(repo, target, "")
	ok, _ := classifier.Adjudicate(finding)
	return ok
}

var syncPublicLeakHunkRE = regexp.MustCompile(`(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@`)

type syncPublicLeakDiffHunk struct {
	oldStart int
	oldCount int
	newStart int
	newCount int
}

func parseSyncPublicLeakDiffHunks(diff string) []syncPublicLeakDiffHunk {
	matches := syncPublicLeakHunkRE.FindAllStringSubmatch(diff, -1)
	if len(matches) == 0 {
		return nil
	}
	hunks := make([]syncPublicLeakDiffHunk, 0, len(matches))
	for _, m := range matches {
		oldStart, _ := strconv.Atoi(m[1])
		oldCount := 1
		if m[2] != "" {
			oldCount, _ = strconv.Atoi(m[2])
		}
		newStart, _ := strconv.Atoi(m[3])
		newCount := 1
		if m[4] != "" {
			newCount, _ = strconv.Atoi(m[4])
		}
		hunks = append(hunks, syncPublicLeakDiffHunk{
			oldStart: oldStart,
			oldCount: oldCount,
			newStart: newStart,
			newCount: newCount,
		})
	}
	return hunks
}

func mapSyncPublicLeakLine(hunks []syncPublicLeakDiffHunk, currentLine int) (int, bool) {
	if currentLine <= 0 {
		return 0, false
	}
	for _, h := range hunks {
		if h.newCount > 0 && currentLine >= h.newStart && currentLine < h.newStart+h.newCount {
			return 0, false
		}
	}
	cumulativeShift := 0
	for _, h := range hunks {
		if (h.newCount > 0 && h.newStart+h.newCount <= currentLine) || (h.newCount == 0 && h.newStart < currentLine) {
			cumulativeShift += (h.newCount - h.oldCount)
		}
	}
	oldLine := currentLine - cumulativeShift
	if oldLine <= 0 {
		return 0, false
	}
	return oldLine, true
}

func resolveGitCommitSHA(repo, rev string) string {
	if rev == "" {
		return ""
	}
	cmd := exec.Command("git", "rev-parse", "--verify", rev+"^{commit}")
	cmd.Dir = repo
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

type syncPublicLeakFileContext struct {
	targetCommit string
	headCommit   string
	baseLines    []string
	currentLines []string
	contentSHA   string
	hunks        []syncPublicLeakDiffHunk
	claimedLines map[int]bool
	valid        bool
}

type syncPublicLeakClassifier struct {
	repo         string
	target       string
	head         string
	targetCommit string
	headCommit   string
	files        map[string]*syncPublicLeakFileContext
}

func newSyncPublicLeakClassifier(repo, target, head string) *syncPublicLeakClassifier {
	c := &syncPublicLeakClassifier{
		repo:   repo,
		target: target,
		head:   head,
		files:  make(map[string]*syncPublicLeakFileContext),
	}
	if target != "" {
		c.targetCommit = resolveGitCommitSHA(repo, target)
	}
	if head != "" {
		c.headCommit = resolveGitCommitSHA(repo, head)
	}
	if c.headCommit == "" {
		c.headCommit = resolveGitCommitSHA(repo, "HEAD")
	}
	return c
}

func (c *syncPublicLeakClassifier) getFileContext(relPath string) *syncPublicLeakFileContext {
	cleanPath := filepath.ToSlash(filepath.Clean(relPath))
	if ctx, exists := c.files[cleanPath]; exists {
		return ctx
	}
	ctx := &syncPublicLeakFileContext{
		targetCommit: c.targetCommit,
		headCommit:   c.headCommit,
		claimedLines: make(map[int]bool),
	}
	c.files[cleanPath] = ctx

	if c.targetCommit == "" || cleanPath == "" {
		return ctx
	}

	cmd := exec.Command("git", "show", c.targetCommit+":"+cleanPath)
	cmd.Dir = c.repo
	configureDispatchHelperCommand(cmd)
	baseline, err := cmd.Output()
	if err != nil {
		return ctx
	}

	current, err := os.ReadFile(filepath.Join(c.repo, filepath.FromSlash(cleanPath)))
	if err != nil {
		return ctx
	}

	sum := sha256.Sum256(current)
	ctx.contentSHA = hex.EncodeToString(sum[:])
	ctx.baseLines = strings.Split(string(baseline), "\n")
	ctx.currentLines = strings.Split(string(current), "\n")

	diffCmd := exec.Command("git", "diff", "--no-color", "--ignore-space-at-eol", "-U0", c.targetCommit, "--", cleanPath)
	diffCmd.Dir = c.repo
	configureDispatchHelperCommand(diffCmd)
	diffOut, err := diffCmd.Output()
	if err != nil {
		return ctx
	}

	ctx.hunks = parseSyncPublicLeakDiffHunks(string(diffOut))
	ctx.valid = true
	return ctx
}

func (c *syncPublicLeakClassifier) Adjudicate(finding hooks.Finding) (bool, *syncPublicLeakOccurrenceEvidence) {
	cleanPath := filepath.ToSlash(filepath.Clean(finding.File))
	ctx := c.getFileContext(cleanPath)
	if !ctx.valid {
		return false, nil
	}

	if finding.Line == 0 {
		return true, &syncPublicLeakOccurrenceEvidence{
			TargetCommit: ctx.targetCommit,
			HeadCommit:   ctx.headCommit,
			Path:         cleanPath,
			ContentSHA:   ctx.contentSHA,
			BaselineLine: 0,
			CurrentLine:  0,
			Witness:      "file-exists",
		}
	}

	oldLine, ok := mapSyncPublicLeakLine(ctx.hunks, finding.Line)
	if !ok {
		return false, nil
	}

	lineIdx := finding.Line - 1
	oldIdx := oldLine - 1
	if lineIdx < 0 || lineIdx >= len(ctx.currentLines) || oldIdx < 0 || oldIdx >= len(ctx.baseLines) {
		return false, nil
	}

	if strings.TrimRight(ctx.baseLines[oldIdx], "\r") != strings.TrimRight(ctx.currentLines[lineIdx], "\r") {
		return false, nil
	}

	if ctx.claimedLines[oldLine] {
		return false, nil
	}
	ctx.claimedLines[oldLine] = true

	witness := "exact"
	if oldLine != finding.Line {
		witness = "relocated"
	}

	return true, &syncPublicLeakOccurrenceEvidence{
		TargetCommit: ctx.targetCommit,
		HeadCommit:   ctx.headCommit,
		Path:         cleanPath,
		ContentSHA:   ctx.contentSHA,
		BaselineLine: oldLine,
		CurrentLine:  finding.Line,
		Witness:      witness,
	}
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
