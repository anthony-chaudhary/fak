package devcmd

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	codexMarketplaceReportSchema  = "fak/codex-marketplace-maintenance/v1"
	codexMarketplaceAttemptSchema = "fak/codex-marketplace-attempt/v1"
	codexMarketplaceReceiptName   = ".fak-marketplace-attempt.json"
	codexMarketplaceStagePrefix   = "marketplace-upgrade-"
	codexMarketplaceReceiptLimit  = 64
	codexMarketplaceWalkLimit     = 100000
	codexMarketplaceDetailLimit   = 2048
	defaultCodexMarketplaceGrace  = 20 * time.Minute
	defaultCodexMarketplace       = "dos"
)

type codexMarketplaceAttemptStatus string

const (
	codexMarketplaceAttemptRunning  codexMarketplaceAttemptStatus = "running"
	codexMarketplaceAttemptFailed   codexMarketplaceAttemptStatus = "failed"
	codexMarketplaceAttemptPromoted codexMarketplaceAttemptStatus = "promoted"
)

const (
	codexMarketplaceReasonEligible                 = "eligible"
	codexMarketplaceReasonReclaimed                = "reclaimed"
	codexMarketplaceReasonFresh                    = "fresh"
	codexMarketplaceReasonMissingReceipt           = "missing-terminal-receipt"
	codexMarketplaceReasonNonTerminalReceipt       = "non-terminal-receipt"
	codexMarketplaceReasonReceiptMismatch          = "receipt-path-mismatch"
	codexMarketplaceReasonOutsideRoot              = "outside-root"
	codexMarketplaceReasonReparse                  = "reparse-point"
	codexMarketplaceReasonNotDirectory             = "not-directory"
	codexMarketplaceReasonScanFailed               = "scan-failed"
	codexMarketplaceReasonWalkLimit                = "walk-limit"
	codexMarketplaceReasonActive                   = "process-referenced"
	codexMarketplaceReasonProcessEnumerationFailed = "process-enumeration-failed"
	codexMarketplaceReasonPostMoveActive           = "post-move-referenced"
	codexMarketplaceReasonPostMoveProcessFailed    = "post-move-process-enumeration-failed"
	codexMarketplaceReasonQuarantined              = "quarantined"
)

type codexMarketplaceAttemptReceipt struct {
	Schema          string                        `json:"schema"`
	AttemptID       string                        `json:"attempt_id"`
	Marketplace     string                        `json:"marketplace"`
	CandidatePath   string                        `json:"candidate_path"`
	PromotionPath   string                        `json:"promotion_path"`
	Source          string                        `json:"source,omitempty"`
	RefName         string                        `json:"ref_name,omitempty"`
	Revision        string                        `json:"revision,omitempty"`
	Status          codexMarketplaceAttemptStatus `json:"status"`
	StartedAt       time.Time                     `json:"started_at"`
	FinishedAt      time.Time                     `json:"finished_at,omitempty"`
	FailureStage    string                        `json:"failure_stage,omitempty"`
	Detail          string                        `json:"detail,omitempty"`
	DiagnosticBytes int                           `json:"diagnostic_bytes,omitempty"`
}

type codexMarketplaceProcess struct {
	PID            int    `json:"pid"`
	CommandLine    string `json:"command_line,omitempty"`
	ExecutablePath string `json:"executable_path,omitempty"`
}

type codexMarketplaceOptions struct {
	Home           string
	Now            time.Time
	Grace          time.Duration
	QuarantineRoot string
	Processes      func() ([]codexMarketplaceProcess, error)
	RunGit         func(string, ...string) (string, error)
	Rename         func(string, string) error
}

type codexMarketplaceEntry struct {
	Name           string  `json:"name"`
	Path           string  `json:"path"`
	Receipt        string  `json:"receipt,omitempty"`
	Marketplace    string  `json:"marketplace,omitempty"`
	AttemptID      string  `json:"attempt_id,omitempty"`
	Status         string  `json:"status,omitempty"`
	Files          int     `json:"files"`
	Directories    int     `json:"directories"`
	Bytes          int64   `json:"bytes"`
	NewestAgeSec   float64 `json:"newest_age_sec"`
	TerminalAgeSec float64 `json:"terminal_age_sec,omitempty"`
	Reason         string  `json:"reason"`
	ReferencedBy   []int   `json:"referenced_by,omitempty"`
	QuarantinePath string  `json:"quarantine_path,omitempty"`
	RemoveErr      string  `json:"remove_err,omitempty"`
	Removed        bool    `json:"removed,omitempty"`
}

type codexMarketplaceReport struct {
	Schema           string                          `json:"schema"`
	Root             string                          `json:"root"`
	DryRun           bool                            `json:"dry_run"`
	GraceSec         float64                         `json:"grace_sec"`
	Entries          []codexMarketplaceEntry         `json:"entries,omitempty"`
	RetainedClones   int                             `json:"retained_clones"`
	RetainedFiles    int                             `json:"retained_files"`
	RetainedBytes    int64                           `json:"retained_bytes"`
	EligibleClones   int                             `json:"eligible_clones"`
	ReclaimedClones  int                             `json:"reclaimed_clones"`
	ReclaimedFiles   int                             `json:"reclaimed_files"`
	ReclaimedBytes   int64                           `json:"reclaimed_bytes"`
	SkipReasons      map[string]int                  `json:"skip_reasons"`
	ProcessSnapshots int                             `json:"process_snapshots"`
	ProcessErr       string                          `json:"process_err,omitempty"`
	Quarantine       string                          `json:"quarantine,omitempty"`
	Attempt          *codexMarketplaceAttemptReceipt `json:"attempt,omitempty"`
	Err              string                          `json:"err,omitempty"`
}

type codexMarketplaceInstallReceipt struct {
	SourceType  string   `json:"source_type"`
	Source      string   `json:"source"`
	RefName     string   `json:"ref_name"`
	SparsePaths []string `json:"sparse_paths"`
	Revision    string   `json:"revision"`
}

func runCodexMarketplaceMaintenance(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-plugin-sync marketplace-maintenance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("codex-home", os.Getenv("CODEX_HOME"), "active Codex home")
	grace := fs.Duration("grace", defaultCodexMarketplaceGrace, "minimum quiet period before receipt-owned recovery")
	apply := fs.Bool("apply", false, "reclaim eligible receipt-owned staging clones")
	jsonOut := fs.Bool("json", false, "emit JSON report")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *grace <= 0 {
		fmt.Fprintln(stderr, "usage: fak-dev codex-plugin-sync marketplace-maintenance [--codex-home DIR] [--grace DURATION] [--apply] [--json]")
		return 2
	}
	resolvedHome, err := resolveCodexHome(*home)
	if err != nil {
		fmt.Fprintf(stderr, "codex marketplace maintenance: %v\n", err)
		return 1
	}
	report := sweepCodexMarketplace(codexMarketplaceOptions{Home: resolvedHome, Grace: *grace}, *apply)
	writeCodexMarketplaceReport(stdout, report, *jsonOut)
	if report.Err != "" || report.ProcessErr != "" || codexMarketplaceReportFailed(report) {
		return 1
	}
	return 0
}

func runCodexMarketplaceUpgrade(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("codex-plugin-sync marketplace-upgrade", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("codex-home", os.Getenv("CODEX_HOME"), "active Codex home")
	marketplace := fs.String("marketplace", defaultCodexMarketplace, "configured Git marketplace name")
	grace := fs.Duration("grace", defaultCodexMarketplaceGrace, "minimum quiet period for pre-upgrade recovery")
	jsonOut := fs.Bool("json", false, "emit JSON report")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *grace <= 0 || !validMarketplaceToken(*marketplace) {
		fmt.Fprintln(stderr, "usage: fak-dev codex-plugin-sync marketplace-upgrade [--codex-home DIR] [--marketplace NAME] [--grace DURATION] [--json]")
		return 2
	}
	resolvedHome, err := resolveCodexHome(*home)
	if err != nil {
		fmt.Fprintf(stderr, "codex marketplace upgrade: %v\n", err)
		return 1
	}
	opts := codexMarketplaceOptions{Home: resolvedHome, Grace: *grace}
	report := sweepCodexMarketplace(opts, true)
	if report.Err == "" && report.ProcessErr == "" && !codexMarketplaceReportFailed(report) {
		attempt, upgradeErr := upgradeCodexMarketplace(opts, *marketplace)
		report.Attempt = &attempt
		if upgradeErr != nil {
			report.Err = upgradeErr.Error()
		}
	}
	writeCodexMarketplaceReport(stdout, report, *jsonOut)
	if report.Err != "" || report.ProcessErr != "" || codexMarketplaceReportFailed(report) {
		return 1
	}
	return 0
}

func resolveCodexHome(home string) (string, error) {
	if strings.TrimSpace(home) == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve user home: %w", err)
		}
		home = filepath.Join(userHome, ".codex")
	}
	abs, err := filepath.Abs(home)
	if err != nil {
		return "", fmt.Errorf("resolve Codex home: %w", err)
	}
	return filepath.Clean(abs), nil
}

func sweepCodexMarketplace(opts codexMarketplaceOptions, apply bool) codexMarketplaceReport {
	opts = normalizeCodexMarketplaceOptions(opts)
	root := filepath.Join(opts.Home, ".tmp", "marketplaces", ".staging")
	report := codexMarketplaceReport{
		Schema: codexMarketplaceReportSchema, Root: root, DryRun: !apply,
		GraceSec: opts.Grace.Seconds(), SkipReasons: map[string]int{},
	}
	entries, err := scanCodexMarketplaceCandidates(root, opts.Now, opts.Grace)
	if errors.Is(err, os.ErrNotExist) {
		return report
	}
	if err != nil {
		report.Err = err.Error()
		return report
	}
	report.Entries = entries
	if len(entries) == 0 {
		return report
	}
	hasEligible := false
	for _, entry := range entries {
		if entry.Reason == codexMarketplaceReasonEligible {
			hasEligible = true
			break
		}
	}
	if !hasEligible {
		foldCodexMarketplaceReport(&report)
		return report
	}
	processes, processErr := opts.Processes()
	report.ProcessSnapshots = 1
	if processErr != nil {
		report.ProcessErr = processErr.Error()
	}
	for i := range report.Entries {
		entry := &report.Entries[i]
		if entry.Reason != codexMarketplaceReasonEligible {
			continue
		}
		if processErr != nil {
			entry.Reason = codexMarketplaceReasonProcessEnumerationFailed
			continue
		}
		if pids := codexMarketplaceReferencingPIDs(processes, entry.Path); len(pids) > 0 {
			entry.Reason = codexMarketplaceReasonActive
			entry.ReferencedBy = pids
		}
	}
	if apply {
		applyCodexMarketplaceRecovery(opts, &report)
	}
	foldCodexMarketplaceReport(&report)
	return report
}

func scanCodexMarketplaceCandidates(root string, now time.Time, grace time.Duration) ([]codexMarketplaceEntry, error) {
	canonicalRoot, err := canonicalCodexMarketplaceStagingRoot(root)
	if err != nil {
		return nil, err
	}
	children, err := os.ReadDir(canonicalRoot)
	if err != nil {
		return nil, err
	}
	if len(children) > 4096 {
		return nil, fmt.Errorf("candidate bound exceeded: %d immediate children > 4096", len(children))
	}
	entries := make([]codexMarketplaceEntry, 0, len(children))
	for _, child := range children {
		path := filepath.Join(canonicalRoot, child.Name())
		entry := inspectCodexMarketplaceCandidate(canonicalRoot, path, path, now, grace)
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return entries, nil
}

func inspectCodexMarketplaceCandidate(root, path, receiptPathExpectation string, now time.Time, grace time.Duration) codexMarketplaceEntry {
	entry := codexMarketplaceEntry{Name: filepath.Base(path), Path: filepath.Clean(path)}
	info, err := os.Lstat(path)
	if err != nil {
		entry.Reason = codexMarketplaceReasonScanFailed
		entry.RemoveErr = err.Error()
		return entry
	}
	if info.Mode()&os.ModeSymlink != 0 {
		entry.Reason = codexMarketplaceReasonReparse
		return entry
	}
	if !info.IsDir() {
		entry.Bytes = info.Size()
		entry.Reason = codexMarketplaceReasonNotDirectory
		return entry
	}
	canonical, err := filepath.EvalSymlinks(path)
	if err != nil {
		entry.Reason = codexMarketplaceReasonScanFailed
		entry.RemoveErr = err.Error()
		return entry
	}
	canonical = filepath.Clean(canonical)
	if !sameCanonicalPath(canonical, filepath.Clean(path)) {
		entry.Reason = codexMarketplaceReasonReparse
		return entry
	}
	if !exactChildOf(root, canonical) {
		entry.Reason = codexMarketplaceReasonOutsideRoot
		return entry
	}
	newest, unsafeReason, walkErr := measureCodexMarketplaceTree(canonical, &entry)
	if walkErr != nil {
		entry.Reason = codexMarketplaceReasonScanFailed
		entry.RemoveErr = walkErr.Error()
		return entry
	}
	if unsafeReason != "" {
		entry.Reason = unsafeReason
		return entry
	}
	if newest.IsZero() {
		newest = info.ModTime()
	}
	entry.NewestAgeSec = now.Sub(newest).Seconds()
	receiptPath := filepath.Join(canonical, codexMarketplaceReceiptName)
	receipt, err := readCodexMarketplaceAttemptReceipt(receiptPath)
	if err != nil {
		entry.Reason = codexMarketplaceReasonMissingReceipt
		return entry
	}
	entry.Receipt = receiptPath
	entry.Marketplace = receipt.Marketplace
	entry.AttemptID = receipt.AttemptID
	entry.Status = string(receipt.Status)
	if !validCodexMarketplaceReceipt(receipt, receiptPathExpectation) {
		entry.Reason = codexMarketplaceReasonReceiptMismatch
		return entry
	}
	if receipt.Status != codexMarketplaceAttemptFailed || receipt.FinishedAt.IsZero() {
		entry.Reason = codexMarketplaceReasonNonTerminalReceipt
		return entry
	}
	entry.TerminalAgeSec = now.Sub(receipt.FinishedAt).Seconds()
	if entry.NewestAgeSec < grace.Seconds() || entry.TerminalAgeSec < grace.Seconds() {
		entry.Reason = codexMarketplaceReasonFresh
		return entry
	}
	entry.Reason = codexMarketplaceReasonEligible
	return entry
}

func measureCodexMarketplaceTree(root string, entry *codexMarketplaceEntry) (time.Time, string, error) {
	var newest time.Time
	seen := 0
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		seen++
		if seen > codexMarketplaceWalkLimit {
			return filepath.SkipAll
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return errCodexMarketplaceReparse
		}
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			return err
		}
		if !sameCanonicalPath(filepath.Clean(resolved), filepath.Clean(path)) {
			return errCodexMarketplaceReparse
		}
		if d.IsDir() {
			entry.Directories++
			return nil
		}
		entry.Files++
		entry.Bytes += info.Size()
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		return nil
	})
	if seen > codexMarketplaceWalkLimit {
		return newest, codexMarketplaceReasonWalkLimit, nil
	}
	if errors.Is(err, errCodexMarketplaceReparse) {
		return newest, codexMarketplaceReasonReparse, nil
	}
	return newest, "", err
}

var errCodexMarketplaceReparse = errors.New("reparse point")

func applyCodexMarketplaceRecovery(opts codexMarketplaceOptions, report *codexMarketplaceReport) {
	eligible := 0
	for _, entry := range report.Entries {
		if entry.Reason == codexMarketplaceReasonEligible {
			eligible++
		}
	}
	if eligible == 0 {
		return
	}
	quarantineParent := opts.QuarantineRoot
	if quarantineParent == "" {
		quarantineParent = filepath.Join(os.TempDir(), "fak-marketplace-quarantine")
	}
	if err := os.MkdirAll(quarantineParent, 0o700); err != nil {
		markCodexMarketplaceEligibleFailed(report, fmt.Errorf("create quarantine parent: %w", err))
		return
	}
	runDir, err := os.MkdirTemp(quarantineParent, "run-")
	if err != nil {
		markCodexMarketplaceEligibleFailed(report, fmt.Errorf("create unique quarantine: %w", err))
		return
	}
	report.Quarantine = runDir
	root := filepath.Clean(report.Root)
	for i := range report.Entries {
		entry := &report.Entries[i]
		if entry.Reason != codexMarketplaceReasonEligible {
			continue
		}
		refreshed := inspectCodexMarketplaceCandidate(root, entry.Path, entry.Path, opts.Now, opts.Grace)
		if refreshed.Reason != codexMarketplaceReasonEligible {
			*entry = refreshed
			continue
		}
		processes, processErr := opts.Processes()
		report.ProcessSnapshots++
		if processErr != nil {
			entry.Reason = codexMarketplaceReasonProcessEnumerationFailed
			entry.RemoveErr = processErr.Error()
			continue
		}
		if pids := codexMarketplaceReferencingPIDs(processes, entry.Path); len(pids) > 0 {
			entry.Reason = codexMarketplaceReasonActive
			entry.ReferencedBy = pids
			continue
		}
		destination := filepath.Join(runDir, entry.Name)
		if err := opts.Rename(entry.Path, destination); err != nil {
			entry.Reason = codexMarketplaceReasonQuarantined
			entry.RemoveErr = fmt.Sprintf("move exact candidate to quarantine: %v", err)
			continue
		}
		entry.QuarantinePath = destination
		postProcesses, postErr := opts.Processes()
		report.ProcessSnapshots++
		if postErr != nil {
			entry.Reason = codexMarketplaceReasonPostMoveProcessFailed
			entry.RemoveErr = postErr.Error()
			rollbackCodexMarketplaceCandidate(entry, destination)
			continue
		}
		pids := append(codexMarketplaceReferencingPIDs(postProcesses, entry.Path), codexMarketplaceReferencingPIDs(postProcesses, destination)...)
		if len(pids) > 0 {
			sort.Ints(pids)
			entry.Reason = codexMarketplaceReasonPostMoveActive
			entry.ReferencedBy = compactCodexMarketplaceInts(pids)
			rollbackCodexMarketplaceCandidate(entry, destination)
			continue
		}
		post := inspectCodexMarketplaceCandidate(runDir, destination, entry.Path, opts.Now, opts.Grace)
		if post.Reason != codexMarketplaceReasonEligible {
			entry.Reason = post.Reason
			entry.RemoveErr = post.RemoveErr
			rollbackCodexMarketplaceCandidate(entry, destination)
			continue
		}
		files, dirs, deleteErr := deleteExactCodexMarketplaceTree(destination)
		if deleteErr != nil {
			entry.Reason = codexMarketplaceReasonQuarantined
			entry.RemoveErr = fmt.Sprintf("delete exact quarantine tree: %v", deleteErr)
			continue
		}
		entry.Files = files
		entry.Directories = dirs
		entry.Reason = codexMarketplaceReasonReclaimed
		entry.Removed = true
		entry.QuarantinePath = ""
	}
	if err := os.Remove(runDir); err == nil {
		report.Quarantine = ""
	}
}

func rollbackCodexMarketplaceCandidate(entry *codexMarketplaceEntry, destination string) {
	if err := os.Rename(destination, entry.Path); err != nil {
		entry.RemoveErr = strings.TrimSpace(entry.RemoveErr + "; restore exact candidate: " + err.Error())
		return
	}
	entry.QuarantinePath = ""
}

func deleteExactCodexMarketplaceTree(root string) (int, int, error) {
	files := make([]string, 0)
	directories := make([]string, 0)
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("reparse point appeared after quarantine: %s", path)
		}
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil || !sameCanonicalPath(filepath.Clean(resolved), filepath.Clean(path)) {
			return fmt.Errorf("reparse point appeared after quarantine: %s", path)
		}
		if d.IsDir() {
			directories = append(directories, path)
		} else {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return 0, 0, err
	}
	for _, path := range files {
		if err := os.Remove(path); err != nil {
			return 0, 0, err
		}
	}
	sort.Slice(directories, func(i, j int) bool { return len(directories[i]) > len(directories[j]) })
	removedDirs := 0
	for _, path := range directories {
		if err := os.Remove(path); err != nil {
			return len(files), removedDirs, err
		}
		removedDirs++
	}
	return len(files), removedDirs, nil
}

func upgradeCodexMarketplace(opts codexMarketplaceOptions, marketplace string) (codexMarketplaceAttemptReceipt, error) {
	opts = normalizeCodexMarketplaceOptions(opts)
	marketRoot := filepath.Join(opts.Home, ".tmp", "marketplaces")
	destination := filepath.Join(marketRoot, marketplace)
	install, err := readCodexMarketplaceInstallReceipt(filepath.Join(destination, ".codex-marketplace-install.json"))
	if err != nil {
		return codexMarketplaceAttemptReceipt{}, fmt.Errorf("read promoted marketplace receipt: %w", err)
	}
	if install.SourceType != "git" || strings.TrimSpace(install.Source) == "" || strings.TrimSpace(install.RefName) == "" {
		return codexMarketplaceAttemptReceipt{}, errors.New("promoted marketplace receipt is not a bounded Git source")
	}
	stagingRoot := filepath.Join(marketRoot, ".staging")
	if err := os.MkdirAll(stagingRoot, 0o700); err != nil {
		return codexMarketplaceAttemptReceipt{}, fmt.Errorf("create marketplace staging root: %w", err)
	}
	stagingRoot, err = canonicalCodexMarketplaceStagingRoot(stagingRoot)
	if err != nil {
		return codexMarketplaceAttemptReceipt{}, fmt.Errorf("validate marketplace staging root: %w", err)
	}
	stage, err := os.MkdirTemp(stagingRoot, codexMarketplaceStagePrefix)
	if err != nil {
		return codexMarketplaceAttemptReceipt{}, fmt.Errorf("allocate exact marketplace stage: %w", err)
	}
	now := opts.Now
	receipt := codexMarketplaceAttemptReceipt{
		Schema: codexMarketplaceAttemptSchema, AttemptID: filepath.Base(stage), Marketplace: marketplace,
		CandidatePath: stage, PromotionPath: destination, Source: install.Source, RefName: install.RefName,
		Status: codexMarketplaceAttemptRunning, StartedAt: now,
	}
	if err := persistCodexMarketplaceAttempt(opts, stage, receipt, true); err != nil {
		_, _, _ = deleteExactCodexMarketplaceTree(stage)
		return receipt, fmt.Errorf("write running ownership receipt: %w", err)
	}
	fail := func(stageName string, cause error) (codexMarketplaceAttemptReceipt, error) {
		receipt.Status = codexMarketplaceAttemptFailed
		receipt.FinishedAt = codexMarketplaceNow(opts)
		receipt.FailureStage = stageName
		receipt.Detail = boundedCodexMarketplaceDetail(cause.Error())
		receipt.DiagnosticBytes = len(receipt.Detail)
		persistErr := persistCodexMarketplaceAttempt(opts, stage, receipt, true)
		cleanupErr := reclaimFreshCodexMarketplaceAttempt(opts, stage)
		if persistErr != nil {
			return receipt, fmt.Errorf("%s: %w; persist terminal receipt: %v", stageName, cause, persistErr)
		}
		if cleanupErr != nil {
			return receipt, fmt.Errorf("%s: %w; cleanup retained for recovery: %v", stageName, cause, cleanupErr)
		}
		return receipt, fmt.Errorf("%s: %w", stageName, cause)
	}
	commands := [][]string{
		{"init"},
		{"remote", "add", "origin", install.Source},
		{"fetch", "--depth=1", "origin", install.RefName},
		{"checkout", "--detach", "FETCH_HEAD"},
	}
	for _, command := range commands {
		if _, err := opts.RunGit(stage, command...); err != nil {
			return fail("clone", err)
		}
	}
	revision, err := opts.RunGit(stage, "rev-parse", "HEAD")
	if err != nil || strings.TrimSpace(revision) == "" {
		if err == nil {
			err = errors.New("empty staged revision")
		}
		return fail("verify_stage", err)
	}
	receipt.Revision = strings.TrimSpace(revision)
	install.Revision = receipt.Revision
	if err := writeJSONAtomic(filepath.Join(stage, ".codex-marketplace-install.json"), install); err != nil {
		return fail("stage_receipt", err)
	}
	stageCheck := inspectCodexMarketplaceCandidate(stagingRoot, stage, stage, codexMarketplaceNow(opts), 0)
	if stageCheck.Reason == codexMarketplaceReasonReparse || stageCheck.Reason == codexMarketplaceReasonOutsideRoot || stageCheck.Reason == codexMarketplaceReasonScanFailed || stageCheck.Reason == codexMarketplaceReasonWalkLimit {
		return fail("verify_stage", fmt.Errorf("unsafe staged clone: %s", stageCheck.Reason))
	}
	processes, err := opts.Processes()
	if err != nil {
		return fail("cutover_process_census", err)
	}
	if pids := codexMarketplaceReferencingPIDs(processes, destination); len(pids) > 0 {
		return fail("cutover_active", fmt.Errorf("promoted marketplace is process-referenced by %v", pids))
	}
	backup := filepath.Join(marketRoot, ".fak-marketplace-backup-"+marketplace+"-"+receipt.AttemptID)
	destinationExists := pathExists(destination)
	if destinationExists {
		if err := opts.Rename(destination, backup); err != nil {
			return fail("cutover_backup", err)
		}
	}
	if err := opts.Rename(stage, destination); err != nil {
		if destinationExists {
			_ = opts.Rename(backup, destination)
		}
		return fail("cutover_promote", err)
	}
	stage = ""
	rollback := func(stageName string, cause error) (codexMarketplaceAttemptReceipt, error) {
		stage = receipt.CandidatePath
		_ = opts.Rename(destination, stage)
		if destinationExists {
			_ = opts.Rename(backup, destination)
		}
		return fail(stageName, cause)
	}
	if err := os.Remove(filepath.Join(destination, codexMarketplaceReceiptName)); err != nil {
		return rollback("remove_stage_marker", err)
	}
	installed, err := readCodexMarketplaceInstallReceipt(filepath.Join(destination, ".codex-marketplace-install.json"))
	if err != nil || installed.Revision != receipt.Revision {
		if err == nil {
			err = fmt.Errorf("installed revision %q differs from staged %q", installed.Revision, receipt.Revision)
		}
		return rollback("verify_promotion", err)
	}
	installedHead, err := opts.RunGit(destination, "rev-parse", "HEAD")
	if err != nil || strings.TrimSpace(installedHead) != receipt.Revision {
		if err == nil {
			err = fmt.Errorf("promoted HEAD %q differs from staged %q", strings.TrimSpace(installedHead), receipt.Revision)
		}
		return rollback("verify_promotion", err)
	}
	if destinationExists {
		if err := removeOwnedCodexMarketplaceBackup(opts, backup); err != nil {
			return receipt, fmt.Errorf("remove verified marketplace backup: %w", err)
		}
	}
	receipt.Status = codexMarketplaceAttemptPromoted
	receipt.FinishedAt = codexMarketplaceNow(opts)
	if err := persistCodexMarketplaceAttempt(opts, "", receipt, false); err != nil {
		return receipt, fmt.Errorf("write promoted terminal receipt: %w", err)
	}
	return receipt, nil
}

func reclaimFreshCodexMarketplaceAttempt(opts codexMarketplaceOptions, stage string) error {
	if stage == "" || !pathExists(stage) {
		return nil
	}
	processes, err := opts.Processes()
	if err != nil {
		return err
	}
	if pids := codexMarketplaceReferencingPIDs(processes, stage); len(pids) > 0 {
		return fmt.Errorf("candidate still process-referenced by %v", pids)
	}
	return quarantineAndDeleteOwnedCodexMarketplace(opts, stage)
}

func removeOwnedCodexMarketplaceBackup(opts codexMarketplaceOptions, backup string) error {
	processes, err := opts.Processes()
	if err != nil {
		return err
	}
	if pids := codexMarketplaceReferencingPIDs(processes, backup); len(pids) > 0 {
		return fmt.Errorf("backup still process-referenced by %v", pids)
	}
	return quarantineAndDeleteOwnedCodexMarketplace(opts, backup)
}

func quarantineAndDeleteOwnedCodexMarketplace(opts codexMarketplaceOptions, source string) error {
	parent := opts.QuarantineRoot
	if parent == "" {
		parent = filepath.Join(os.TempDir(), "fak-marketplace-quarantine")
	}
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return err
	}
	runDir, err := os.MkdirTemp(parent, "run-")
	if err != nil {
		return err
	}
	destination := filepath.Join(runDir, filepath.Base(source))
	if err := opts.Rename(source, destination); err != nil {
		_ = os.Remove(runDir)
		return err
	}
	processes, processErr := opts.Processes()
	if processErr != nil {
		_ = opts.Rename(destination, source)
		_ = os.Remove(runDir)
		return processErr
	}
	if pids := append(codexMarketplaceReferencingPIDs(processes, source), codexMarketplaceReferencingPIDs(processes, destination)...); len(pids) > 0 {
		_ = opts.Rename(destination, source)
		_ = os.Remove(runDir)
		return fmt.Errorf("source or quarantine became process-referenced by %v", pids)
	}
	if _, _, err := deleteExactCodexMarketplaceTree(destination); err != nil {
		return err
	}
	_ = os.Remove(runDir)
	return nil
}

func persistCodexMarketplaceAttempt(opts codexMarketplaceOptions, candidate string, receipt codexMarketplaceAttemptReceipt, writeCandidate bool) error {
	if writeCandidate && candidate != "" && pathExists(candidate) {
		if err := writeCodexMarketplaceAttemptReceipt(candidate, receipt); err != nil {
			return err
		}
	}
	receiptsRoot := filepath.Join(opts.Home, ".tmp", "marketplaces", ".fak-upgrade-receipts")
	if err := os.MkdirAll(receiptsRoot, 0o700); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(receiptsRoot, receipt.AttemptID+".json"), receipt); err != nil {
		return err
	}
	return trimCodexMarketplaceReceipts(receiptsRoot)
}

func writeCodexMarketplaceAttemptReceipt(candidate string, receipt codexMarketplaceAttemptReceipt) error {
	return writeJSONAtomic(filepath.Join(candidate, codexMarketplaceReceiptName), receipt)
}

func readCodexMarketplaceAttemptReceipt(path string) (codexMarketplaceAttemptReceipt, error) {
	return readCodexMarketplaceReceipt[codexMarketplaceAttemptReceipt](path)
}

func readCodexMarketplaceInstallReceipt(path string) (codexMarketplaceInstallReceipt, error) {
	return readCodexMarketplaceReceipt[codexMarketplaceInstallReceipt](path)
}

func readCodexMarketplaceReceipt[T any](path string) (T, error) {
	var receipt T
	b, err := os.ReadFile(path)
	if err != nil {
		return receipt, err
	}
	if err := json.Unmarshal(b, &receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

func validCodexMarketplaceReceipt(receipt codexMarketplaceAttemptReceipt, expectedPath string) bool {
	return receipt.Schema == codexMarketplaceAttemptSchema && validMarketplaceToken(receipt.AttemptID) &&
		validMarketplaceToken(receipt.Marketplace) && sameCanonicalPath(filepath.Clean(receipt.CandidatePath), filepath.Clean(expectedPath)) &&
		!receipt.StartedAt.IsZero() && (receipt.FinishedAt.IsZero() || !receipt.FinishedAt.Before(receipt.StartedAt))
}

func writeJSONAtomic(path string, value any) error {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	parent := filepath.Dir(path)
	if err := os.MkdirAll(parent, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(parent, ".receipt-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err == nil {
		return nil
	} else if !pathExists(path) {
		return err
	}
	// Windows cannot rename over an existing file. A crash in this fallback leaves the
	// receipt absent or invalid, which recovery treats as a keep, never as ownership proof.
	if err := os.Remove(path); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func trimCodexMarketplaceReceipts(root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}
	type receiptFile struct {
		name    string
		modTime time.Time
	}
	files := make([]receiptFile, 0, len(entries))
	for _, entry := range entries {
		if entry.Type().IsRegular() && strings.HasSuffix(entry.Name(), ".json") {
			info, err := entry.Info()
			if err != nil {
				return err
			}
			files = append(files, receiptFile{name: entry.Name(), modTime: info.ModTime()})
		}
	}
	if len(files) <= codexMarketplaceReceiptLimit {
		return nil
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.Before(files[j].modTime)
	})
	for _, entry := range files[:len(files)-codexMarketplaceReceiptLimit] {
		if err := os.Remove(filepath.Join(root, entry.name)); err != nil {
			return err
		}
	}
	return nil
}

func normalizeCodexMarketplaceOptions(opts codexMarketplaceOptions) codexMarketplaceOptions {
	if opts.Now.IsZero() {
		opts.Now = time.Now().UTC()
	}
	if opts.Grace <= 0 {
		opts.Grace = defaultCodexMarketplaceGrace
	}
	if opts.Processes == nil {
		opts.Processes = listCodexMarketplaceProcesses
	}
	if opts.RunGit == nil {
		opts.RunGit = runCodexMarketplaceGit
	}
	if opts.Rename == nil {
		opts.Rename = os.Rename
	}
	return opts
}

func codexMarketplaceNow(opts codexMarketplaceOptions) time.Time {
	if !opts.Now.IsZero() {
		return opts.Now
	}
	return time.Now().UTC()
}

func runCodexMarketplaceGit(dir string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	commandArgs := append([]string{"-C", dir}, args...)
	cmd := exec.CommandContext(ctx, "git", commandArgs...)
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return "", fmt.Errorf("git %s timed out: %w", args[0], ctx.Err())
	}
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", args[0], err, boundedCodexMarketplaceDetail(string(b)))
	}
	return strings.TrimSpace(string(b)), nil
}

func listCodexMarketplaceProcesses() ([]codexMarketplaceProcess, error) {
	if runtime.GOOS == "windows" {
		const script = `$ErrorActionPreference='Stop'; ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine,ExecutablePath)`
		cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
		windowgate.ConfigureBackgroundCommand(cmd)
		b, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("enumerate Win32_Process: %w", err)
		}
		var rows []struct {
			PID        int     `json:"ProcessId"`
			Command    *string `json:"CommandLine"`
			Executable *string `json:"ExecutablePath"`
		}
		if err := json.Unmarshal(b, &rows); err != nil {
			return nil, fmt.Errorf("decode Win32_Process: %w", err)
		}
		out := make([]codexMarketplaceProcess, 0, len(rows))
		for _, row := range rows {
			process := codexMarketplaceProcess{PID: row.PID}
			if row.Command != nil {
				process.CommandLine = *row.Command
			}
			if row.Executable != nil {
				process.ExecutablePath = *row.Executable
			}
			out = append(out, process)
		}
		return out, nil
	}
	procEntries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("enumerate /proc: %w", err)
	}
	out := make([]codexMarketplaceProcess, 0)
	for _, entry := range procEntries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		base := filepath.Join("/proc", entry.Name())
		commandBytes, _ := os.ReadFile(filepath.Join(base, "cmdline"))
		executable, _ := os.Readlink(filepath.Join(base, "exe"))
		out = append(out, codexMarketplaceProcess{PID: pid, CommandLine: strings.ReplaceAll(string(commandBytes), "\x00", " "), ExecutablePath: executable})
	}
	return out, nil
}

func codexMarketplaceReferencingPIDs(processes []codexMarketplaceProcess, candidate string) []int {
	pids := make([]int, 0)
	for _, process := range processes {
		if codexMarketplaceFieldReferencesPath(process.CommandLine, candidate) || codexMarketplaceFieldReferencesPath(process.ExecutablePath, candidate) {
			pids = append(pids, process.PID)
		}
	}
	sort.Ints(pids)
	return compactCodexMarketplaceInts(pids)
}

func codexMarketplaceFieldReferencesPath(field, candidate string) bool {
	if field == "" || candidate == "" {
		return false
	}
	field = filepath.ToSlash(field)
	candidate = strings.TrimSuffix(filepath.ToSlash(filepath.Clean(candidate)), "/")
	if runtime.GOOS == "windows" {
		field = strings.ToLower(field)
		candidate = strings.ToLower(candidate)
	}
	for offset := 0; offset <= len(field)-len(candidate); {
		index := strings.Index(field[offset:], candidate)
		if index < 0 {
			return false
		}
		index += offset
		beforeOK := index == 0 || codexMarketplacePathBoundary(field[index-1])
		after := index + len(candidate)
		afterOK := after == len(field) || field[after] == '/' || codexMarketplacePathBoundary(field[after])
		if beforeOK && afterOK {
			return true
		}
		offset = index + 1
	}
	return false
}

func codexMarketplacePathBoundary(value byte) bool {
	switch value {
	case 0, ' ', '\t', '\r', '\n', '"', '\'', '=', ':', ',', ';', '(', ')', '[', ']':
		return true
	default:
		return false
	}
}

func compactCodexMarketplaceInts(values []int) []int {
	if len(values) < 2 {
		return values
	}
	out := values[:1]
	for _, value := range values[1:] {
		if value != out[len(out)-1] {
			out = append(out, value)
		}
	}
	return out
}

func foldCodexMarketplaceReport(report *codexMarketplaceReport) {
	report.RetainedClones = 0
	report.RetainedFiles = 0
	report.RetainedBytes = 0
	report.EligibleClones = 0
	report.ReclaimedClones = 0
	report.ReclaimedFiles = 0
	report.ReclaimedBytes = 0
	report.SkipReasons = map[string]int{}
	for _, entry := range report.Entries {
		switch entry.Reason {
		case codexMarketplaceReasonReclaimed:
			report.ReclaimedClones++
			report.ReclaimedFiles += entry.Files
			report.ReclaimedBytes += entry.Bytes
		case codexMarketplaceReasonEligible:
			report.EligibleClones++
			report.RetainedClones++
			report.RetainedFiles += entry.Files
			report.RetainedBytes += entry.Bytes
		default:
			report.RetainedClones++
			report.RetainedFiles += entry.Files
			report.RetainedBytes += entry.Bytes
			report.SkipReasons[entry.Reason]++
		}
	}
}

func writeCodexMarketplaceReport(w io.Writer, report codexMarketplaceReport, jsonOut bool) {
	if jsonOut {
		_ = json.NewEncoder(w).Encode(report)
		return
	}
	action := "reclaimed"
	if report.DryRun {
		action = "eligible"
	}
	fmt.Fprintf(w, "Codex marketplace staging: retained %d clone(s), %s %d, reclaimed %d; retained %d file(s)/%d byte(s), reclaimed %d file(s)/%d byte(s)\n",
		report.RetainedClones, action, report.EligibleClones, report.ReclaimedClones,
		report.RetainedFiles, report.RetainedBytes, report.ReclaimedFiles, report.ReclaimedBytes)
	reasons := make([]string, 0, len(report.SkipReasons))
	for reason := range report.SkipReasons {
		reasons = append(reasons, reason)
	}
	sort.Strings(reasons)
	for _, reason := range reasons {
		fmt.Fprintf(w, "  kept %-36s %d\n", reason, report.SkipReasons[reason])
	}
	if report.Attempt != nil {
		fmt.Fprintf(w, "upgrade %s: %s @ %s\n", report.Attempt.Marketplace, report.Attempt.Status, report.Attempt.Revision)
	}
	if report.ProcessErr != "" {
		fmt.Fprintf(w, "process census: %s\n", report.ProcessErr)
	}
	if report.Err != "" {
		fmt.Fprintf(w, "error: %s\n", report.Err)
	}
}

func codexMarketplaceReportFailed(report codexMarketplaceReport) bool {
	for _, entry := range report.Entries {
		if entry.RemoveErr != "" {
			return true
		}
	}
	return false
}

func markCodexMarketplaceEligibleFailed(report *codexMarketplaceReport, err error) {
	for i := range report.Entries {
		if report.Entries[i].Reason == codexMarketplaceReasonEligible {
			report.Entries[i].Reason = codexMarketplaceReasonQuarantined
			report.Entries[i].RemoveErr = err.Error()
		}
	}
}

func canonicalExistingPath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	return filepath.Clean(resolved), nil
}

func canonicalCodexMarketplaceStagingRoot(root string) (string, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(absRoot)
	if err != nil {
		return "", err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return "", fmt.Errorf("marketplace staging root is a reparse point or not a directory: %s", absRoot)
	}
	canonicalParent, err := canonicalExistingPath(filepath.Dir(absRoot))
	if err != nil {
		return "", err
	}
	canonicalRoot, err := canonicalExistingPath(absRoot)
	if err != nil {
		return "", err
	}
	if !sameCanonicalPath(canonicalRoot, filepath.Join(canonicalParent, ".staging")) || !exactChildOf(canonicalParent, canonicalRoot) {
		return "", fmt.Errorf("marketplace staging root escapes its canonical marketplace parent: %s", absRoot)
	}
	return canonicalRoot, nil
}

func exactChildOf(root, candidate string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(candidate))
	return err == nil && rel != "." && rel != ".." && !filepath.IsAbs(rel) && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !strings.Contains(rel, string(filepath.Separator))
}

func sameCanonicalPath(left, right string) bool {
	if runtime.GOOS == "windows" {
		return strings.EqualFold(left, right)
	}
	return left == right
}

func validMarketplaceToken(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" || value == "." || value == ".." || filepath.IsAbs(value) || strings.ContainsAny(value, `/\\`) {
		return false
	}
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			continue
		}
		return false
	}
	return true
}

func boundedCodexMarketplaceDetail(detail string) string {
	detail = strings.TrimSpace(detail)
	if len(detail) <= codexMarketplaceDetailLimit {
		return detail
	}
	return detail[:codexMarketplaceDetailLimit]
}
