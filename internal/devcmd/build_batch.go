package devcmd

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	buildBatchRecordSchema = "fak.build-batch.v1"
	buildBatchViewSchema   = "fak.build-batch-view.v1"
	// buildBatchRecordMax bounds the user-supplied record so plan/status stay
	// cheap and bounded; inputs are streamed, not capped.
	buildBatchRecordMax = 1 << 20
)

// buildBatchRecord is the user-supplied, nonsecret data record managed by
// 'fak-dev build batch'. One authoritative manifest (Inputs) plus declared
// patch ownership, ordered stage observations, and immutable evidence
// references join into one advisory view. Nothing in this surface executes:
// setup only emits a seed to stdout, and plan/status only read explicitly
// named files below an explicit root. Supplied records are observations,
// never execution grants.
type buildBatchRecord struct {
	Schema           string            `json:"schema"`
	Root             string            `json:"root,omitempty"`
	TargetCommit     string            `json:"target_commit,omitempty"`
	Profile          string            `json:"profile,omitempty"`
	Target           *buildBatchTarget `json:"target,omitempty"`
	Inputs           []buildBatchInput `json:"inputs"`
	Patches          []buildBatchPatch `json:"patches"`
	DeclaredCounts   *buildBatchCounts `json:"declared_counts,omitempty"`
	Stages           []buildBatchStage `json:"stages"`
	VerificationDebt []string          `json:"verification_debt"`
	Publisher        string            `json:"publisher,omitempty"`
}

type buildBatchTarget struct {
	Commit  string `json:"commit,omitempty"`
	Profile string `json:"profile,omitempty"`
}

type buildBatchInput struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256,omitempty"`
}

type buildBatchPatch struct {
	Issue string   `json:"issue"`
	Paths []string `json:"paths"`
}

type buildBatchStage struct {
	Name        string `json:"name"`
	Outcome     string `json:"outcome,omitempty"`
	EvidenceRef string `json:"evidence_ref,omitempty"`
}

// buildBatchCounts are independently editable descriptive fields; any value
// present must agree with the derived count or the record is refused.
type buildBatchCounts struct {
	Inputs           int `json:"inputs,omitempty"`
	Patches          int `json:"patches,omitempty"`
	AffectedPackages int `json:"affected_packages,omitempty"`
}

type buildBatchInputView struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Pinned bool   `json:"pinned"`
	State  string `json:"state"` // fresh | stale | unpinned
}

type buildBatchStageView struct {
	Name        string `json:"name"`
	Outcome     string `json:"outcome,omitempty"`
	EvidenceRef string `json:"evidence_ref,omitempty"`
	Resolution  string `json:"resolution"` // settled | unresolved
}

type buildBatchViewCounts struct {
	Inputs           int `json:"inputs"`
	Patches          int `json:"patches"`
	AffectedPackages int `json:"affected_packages"`
}

type buildBatchView struct {
	Schema           string                `json:"schema"`
	Command          string                `json:"command"`
	Record           string                `json:"record"`
	Root             string                `json:"root"`
	TargetCommit     string                `json:"target_commit,omitempty"`
	Profile          string                `json:"profile,omitempty"`
	Inputs           []buildBatchInputView `json:"inputs"`
	AffectedPackages []string              `json:"affected_packages"`
	Counts           buildBatchViewCounts  `json:"counts"`
	Stages           []buildBatchStageView `json:"stages,omitempty"`
	VerificationDebt []string              `json:"verification_debt,omitempty"`
	Publisher        string                `json:"publisher,omitempty"`
	Advisory         []string              `json:"advisory"`
}

// RunBuildBatch implements the optional 'fak-dev build batch' subcommand:
// setup emits a seed record to stdout; plan and status inspect an explicitly
// named record and its explicitly named inputs below an explicit root. They
// never invoke a compiler, native mutation, lease, claim, sync, or push.
// Exit 0 reports the advisory view (stale or unresolved findings included);
// exit 2 refuses invalid JSON, ambiguous file identities, path escapes, and
// conflicting declarations.
func RunBuildBatch(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeBuildBatchUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "setup":
		return runBuildBatchSetup(stdout, stderr, argv[1:])
	case "plan", "status":
		return runBuildBatchInspect(stdout, stderr, argv[0], argv[1:])
	default:
		fmt.Fprintf(stderr, "fak build batch: unknown subcommand %q\n", argv[0])
		writeBuildBatchUsage(stderr)
		return 2
	}
}

func writeBuildBatchUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak-dev build batch <setup|plan|status> [flags]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "  setup [--root R] [--target-commit C] [--profile P] [--publisher N]")
	fmt.Fprintln(w, "        [--input PATH]... [--patch ISSUE=PATH]...")
	fmt.Fprintln(w, "        emit a seed record to stdout; save it wherever you choose")
	fmt.Fprintln(w, "  plan   --record FILE [--root R] [--json]")
	fmt.Fprintln(w, "  status --record FILE [--root R] [--json]")
	fmt.Fprintln(w, "        inspect the named record and its named inputs below R; advisory only")
}

func runBuildBatchSetup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak build batch setup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "explicit root the seed record describes")
	targetCommit := fs.String("target-commit", "", "current target commit the batch is bound to")
	profile := fs.String("profile", "", "build profile binding (release|dev|race)")
	publisher := fs.String("publisher", "", "names one publisher; advisory only, grants no ownership")
	var inputs, patches pathList
	fs.Var(&inputs, "input", "seed input path below --root (repeatable; unpinned until plan/status)")
	fs.Var(&patches, "patch", "ISSUE=PATH patch ownership declaration (repeatable)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak build batch setup: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	if err := validateBuildBatchProfile(*profile); err != nil {
		fmt.Fprintf(stderr, "fak build batch setup: %v\n", err)
		return 2
	}
	record := buildBatchRecord{
		Schema:           buildBatchRecordSchema,
		Root:             strings.TrimSpace(*root),
		Inputs:           []buildBatchInput{},
		Patches:          []buildBatchPatch{},
		Stages:           []buildBatchStage{},
		VerificationDebt: []string{},
		Publisher:        strings.TrimSpace(*publisher),
	}
	if strings.TrimSpace(*targetCommit) != "" || strings.TrimSpace(*profile) != "" {
		record.Target = &buildBatchTarget{Commit: strings.TrimSpace(*targetCommit), Profile: strings.TrimSpace(*profile)}
	}
	for _, raw := range inputs {
		if _, err := batchSafePath(".", raw); err != nil {
			fmt.Fprintf(stderr, "fak build batch setup: %v\n", err)
			return 2
		}
		record.Inputs = append(record.Inputs, buildBatchInput{Path: raw})
	}
	for _, raw := range patches {
		issue, path, ok := strings.Cut(raw, "=")
		if !ok || strings.TrimSpace(issue) == "" || strings.TrimSpace(path) == "" {
			fmt.Fprintf(stderr, "fak build batch setup: --patch must be ISSUE=PATH (got %q)\n", raw)
			return 2
		}
		if _, err := batchSafePath(".", strings.TrimSpace(path)); err != nil {
			fmt.Fprintf(stderr, "fak build batch setup: %v\n", err)
			return 2
		}
		record.Patches = append(record.Patches, buildBatchPatch{Issue: strings.TrimSpace(issue), Paths: []string{strings.TrimSpace(path)}})
	}
	if err := checkBuildBatchConflicts(&record); err != nil {
		fmt.Fprintf(stderr, "fak build batch setup: %v\n", err)
		return 2
	}
	if err := writeIndentedJSONNoEscape(stdout, record); err != nil {
		fmt.Fprintf(stderr, "fak build batch setup: write seed record: %v\n", err)
		return 1
	}
	return 0
}

func runBuildBatchInspect(stdout, stderr io.Writer, command string, argv []string) int {
	fs := flag.NewFlagSet("fak build batch "+command, flag.ContinueOnError)
	fs.SetOutput(stderr)
	recordFlag := fs.String("record", "", "record file below --root to inspect (required)")
	rootFlag := fs.String("root", ".", "explicit root the record and its inputs live below")
	asJSON := fs.Bool("json", false, "write the view as one JSON object to stdout")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak build batch %s: unexpected arguments: %s\n", command, strings.Join(fs.Args(), " "))
		return 2
	}
	if strings.TrimSpace(*recordFlag) == "" {
		fmt.Fprintf(stderr, "fak build batch %s: --record is required\n", command)
		return 2
	}
	rootAbs, err := filepath.Abs(*rootFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak build batch %s: resolve --root: %v\n", command, err)
		return 2
	}
	if info, statErr := os.Stat(rootAbs); statErr != nil || !info.IsDir() {
		fmt.Fprintf(stderr, "fak build batch %s: --root %s must be an existing directory\n", command, *rootFlag)
		return 2
	}
	record, err := loadBuildBatchRecord(rootAbs, *recordFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak build batch %s: %v\n", command, err)
		return 2
	}
	view, err := evaluateBuildBatch(record, rootAbs, *recordFlag, *rootFlag)
	if err != nil {
		fmt.Fprintf(stderr, "fak build batch %s: %v\n", command, err)
		return 2
	}
	view.Command = command
	view.Root = *rootFlag
	if command == "plan" {
		// Plan is forward-looking: it reports inputs, the union, and counts.
		// Status is where recorded stage observations get resolved.
		view.Stages = nil
	}
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, view); err != nil {
			fmt.Fprintf(stderr, "fak build batch %s: write JSON view: %v\n", command, err)
			return 1
		}
		return 0
	}
	renderBuildBatchView(stdout, *view)
	return 0
}

func loadBuildBatchRecord(rootAbs, recordRel string) (*buildBatchRecord, error) {
	recordPath, err := batchSafePath(rootAbs, recordRel)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(recordPath)
	if err != nil {
		return nil, fmt.Errorf("read record %s: %w", recordRel, err)
	}
	if len(data) > buildBatchRecordMax {
		return nil, fmt.Errorf("record %s exceeds the %d-byte bound", recordRel, buildBatchRecordMax)
	}
	var record buildBatchRecord
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&record); err != nil {
		return nil, fmt.Errorf("invalid record JSON in %s: %w", recordRel, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("invalid record JSON in %s: trailing content after the record object", recordRel)
	}
	if record.Schema != buildBatchRecordSchema {
		return nil, fmt.Errorf("unsupported schema %q; want %s", record.Schema, buildBatchRecordSchema)
	}
	return &record, nil
}

// checkBuildBatchConflicts enforces the closed contract: one authoritative
// target, disjoint patch ownership, unique input identities, a closed stage
// outcome vocabulary, and declared counts that agree with derived counts.
func checkBuildBatchConflicts(record *buildBatchRecord) error {
	if len(record.Inputs) > 256 {
		return errors.New("too many inputs; the record is bounded to 256")
	}
	if len(record.Patches) > 64 {
		return errors.New("too many patches; the record is bounded to 64")
	}
	if len(record.Stages) > 128 {
		return errors.New("too many stages; the record is bounded to 128")
	}
	if len(record.VerificationDebt) > 64 {
		return errors.New("too many verification-debt entries; the record is bounded to 64")
	}
	if len(record.Publisher) > 128 || len(record.Root) > 512 {
		return errors.New("overlong publisher or root field")
	}
	commit, profile := "", ""
	if record.Target != nil {
		commit, profile = record.Target.Commit, record.Target.Profile
	}
	if record.TargetCommit != "" && commit != "" && record.TargetCommit != commit {
		return fmt.Errorf("conflicting current-target declaration: target.commit %q vs target_commit %q", commit, record.TargetCommit)
	}
	if record.Profile != "" && profile != "" && record.Profile != profile {
		return fmt.Errorf("conflicting current-target declaration: target.profile %q vs profile %q", profile, record.Profile)
	}
	profile = firstNonempty(profile, record.Profile)
	if err := validateBuildBatchProfile(profile); err != nil {
		return err
	}
	if len(commit) > 64 || len(firstNonempty(record.TargetCommit, commit)) > 64 {
		return errors.New("overlong target commit field")
	}
	seenInputs := map[string]struct{}{}
	for _, input := range record.Inputs {
		if len(input.Path) > 512 {
			return errors.New("overlong input path field")
		}
		if _, err := normalizeBuildBatchSHA(input.SHA256); err != nil {
			return fmt.Errorf("input %s: %w", input.Path, err)
		}
		key := batchPathKey(input.Path)
		if _, dup := seenInputs[key]; dup {
			return fmt.Errorf("ambiguous file identity: duplicate input path %q", input.Path)
		}
		seenInputs[key] = struct{}{}
	}
	owners := map[string]string{}
	for _, patch := range record.Patches {
		issue := strings.TrimSpace(patch.Issue)
		if issue == "" || len(issue) > 128 {
			return errors.New("each patch needs a non-empty issue identifier of at most 128 characters")
		}
		if len(patch.Paths) == 0 || len(patch.Paths) > 256 {
			return fmt.Errorf("patch %s: needs 1..256 owned paths", issue)
		}
		for _, raw := range patch.Paths {
			if len(raw) > 512 {
				return errors.New("overlong patch path field")
			}
			key := batchPathKey(raw)
			prev, taken := owners[key]
			if taken && prev != issue {
				return fmt.Errorf("membership conflict: path %q is owned by both %s and %s", key, prev, issue)
			}
			if taken {
				return fmt.Errorf("ambiguous file identity: path %q is declared more than once for %s", key, issue)
			}
			owners[key] = issue
		}
	}
	seenStages := map[string]struct{}{}
	for _, stage := range record.Stages {
		name := strings.TrimSpace(stage.Name)
		if name == "" || len(name) > 128 {
			return errors.New("each stage needs a non-empty name of at most 128 characters")
		}
		switch stage.Outcome {
		case "", "success", "failed", "unknown":
		default:
			return fmt.Errorf("unsupported stage outcome %q for %s; closed vocabulary is success, failed, unknown", stage.Outcome, name)
		}
		if len(stage.EvidenceRef) > 512 {
			return errors.New("overlong stage evidence reference")
		}
		if _, dup := seenStages[name]; dup {
			return fmt.Errorf("ambiguous file identity: duplicate stage %q", name)
		}
		seenStages[name] = struct{}{}
	}
	for _, item := range record.VerificationDebt {
		if len(item) > 512 {
			return errors.New("overlong verification-debt entry")
		}
	}
	return nil
}

func evaluateBuildBatch(record *buildBatchRecord, rootAbs, recordRel, rootDisplay string) (*buildBatchView, error) {
	if err := checkBuildBatchConflicts(record); err != nil {
		return nil, err
	}
	inputViews := make([]buildBatchInputView, 0, len(record.Inputs))
	for _, input := range record.Inputs {
		view := buildBatchInputView{Path: batchPathKey(input.Path), Pinned: input.SHA256 != ""}
		actual, err := hashBatchInputFile(rootAbs, input.Path)
		if err != nil {
			return nil, err
		}
		view.SHA256 = actual
		pinned, _ := normalizeBuildBatchSHA(input.SHA256)
		switch {
		case !view.Pinned:
			view.State = "unpinned"
		case actual == pinned:
			view.State = "fresh"
		default:
			// Stale source/profile evidence is invalidated in place, not
			// fabricated over: the view keeps naming the exact drifted input.
			view.State = "stale"
		}
		inputViews = append(inputViews, view)
	}
	patchPaths := make([]string, 0, len(record.Patches))
	for _, patch := range record.Patches {
		patchPaths = append(patchPaths, patch.Paths...)
	}
	affected := deriveBuildBatchPackages(patchPaths)
	counts := buildBatchViewCounts{Inputs: len(record.Inputs), Patches: len(record.Patches), AffectedPackages: len(affected)}
	if declared := record.DeclaredCounts; declared != nil {
		if declared.Inputs > 0 && declared.Inputs != counts.Inputs {
			return nil, fmt.Errorf("conflicting count declaration: inputs declared %d, derived %d", declared.Inputs, counts.Inputs)
		}
		if declared.Patches > 0 && declared.Patches != counts.Patches {
			return nil, fmt.Errorf("conflicting count declaration: patches declared %d, derived %d", declared.Patches, counts.Patches)
		}
		if declared.AffectedPackages > 0 && declared.AffectedPackages != counts.AffectedPackages {
			return nil, fmt.Errorf("conflicting count declaration: affected_packages declared %d, derived %d", declared.AffectedPackages, counts.AffectedPackages)
		}
	}
	stageViews := make([]buildBatchStageView, 0, len(record.Stages))
	for _, stage := range record.Stages {
		view := buildBatchStageView{Name: strings.TrimSpace(stage.Name), Outcome: stage.Outcome, EvidenceRef: stage.EvidenceRef, Resolution: "unresolved"}
		// Terminal outcomes settle only with settlement evidence; an unknown
		// outcome is preserved, never treated as a retry instruction.
		if (stage.Outcome == "success" || stage.Outcome == "failed") && stage.EvidenceRef != "" {
			view.Resolution = "settled"
		}
		stageViews = append(stageViews, view)
	}
	commit, profile := "", ""
	if record.Target != nil {
		commit, profile = record.Target.Commit, record.Target.Profile
	}
	commit = firstNonempty(commit, record.TargetCommit)
	profile = firstNonempty(profile, record.Profile)
	return &buildBatchView{
		Schema:           buildBatchViewSchema,
		Record:           batchPathKey(recordRel),
		TargetCommit:     commit,
		Profile:          profile,
		Inputs:           inputViews,
		AffectedPackages: affected,
		Counts:           counts,
		Stages:           stageViews,
		VerificationDebt: record.VerificationDebt,
		Publisher:        record.Publisher,
		Advisory:         buildBatchAdvisories(inputViews, stageViews),
	}, nil
}

func buildBatchAdvisories(inputs []buildBatchInputView, stages []buildBatchStageView) []string {
	advisory := []string{
		"advisory record only: no compiler, lease, claim, sync, or push is invoked",
		"combined patch manifest is not a dependency-closure proof for intermediate issue-scoped commits",
		"existing native primitives remain authoritative: fak validate, fak-flow commit, fak-sync push",
	}
	for _, input := range inputs {
		if input.State == "stale" {
			advisory = append(advisory, "stale source/profile evidence is invalidated; re-pin hashes before relying on shared batch build evidence")
			break
		}
	}
	for _, stage := range stages {
		if stage.Resolution == "unresolved" {
			advisory = append(advisory, "unresolved attempts are preserved as unknown; an unknown outcome is not a retry instruction")
			break
		}
	}
	return advisory
}

func renderBuildBatchView(w io.Writer, view buildBatchView) {
	fmt.Fprintf(w, "fak build batch %s: record=%s root=%s\n", view.Command, view.Record, view.Root)
	target := "unpinned"
	if view.TargetCommit != "" {
		target = view.TargetCommit
	}
	profile := "unpinned"
	if view.Profile != "" {
		profile = view.Profile
	}
	fmt.Fprintf(w, "  target              %s  profile=%s\n", target, profile)
	fresh, stale, unpinned := 0, 0, 0
	for _, input := range view.Inputs {
		switch input.State {
		case "fresh":
			fresh++
		case "stale":
			stale++
		default:
			unpinned++
		}
	}
	fmt.Fprintf(w, "  inputs              %d (fresh=%d stale=%d unpinned=%d)\n", len(view.Inputs), fresh, stale, unpinned)
	for _, input := range view.Inputs {
		fmt.Fprintf(w, "    %-40s sha256:%s  %s\n", input.Path, input.SHA256, input.State)
	}
	fmt.Fprintf(w, "  affected_packages   %d\n", len(view.AffectedPackages))
	for _, pkg := range view.AffectedPackages {
		fmt.Fprintf(w, "    %s\n", pkg)
	}
	fmt.Fprintf(w, "  counts              inputs=%d patches=%d affected_packages=%d\n",
		view.Counts.Inputs, view.Counts.Patches, view.Counts.AffectedPackages)
	if view.Command == "status" {
		settled, unresolved := 0, 0
		for _, stage := range view.Stages {
			if stage.Resolution == "settled" {
				settled++
			} else {
				unresolved++
			}
		}
		fmt.Fprintf(w, "  stages              %d (settled=%d unresolved=%d)\n", len(view.Stages), settled, unresolved)
		for _, stage := range view.Stages {
			outcome := stage.Outcome
			if outcome == "" {
				outcome = "none"
			}
			evidence := ""
			if stage.EvidenceRef != "" {
				evidence = "  evidence=" + stage.EvidenceRef
			}
			fmt.Fprintf(w, "    %-24s %-8s %s%s\n", stage.Name, outcome, stage.Resolution, evidence)
		}
		fmt.Fprintf(w, "  verification_debt   %d\n", len(view.VerificationDebt))
		for _, item := range view.VerificationDebt {
			fmt.Fprintf(w, "    %s\n", item)
		}
		if view.Publisher != "" {
			fmt.Fprintf(w, "  publisher           %s (advisory; grants no ownership)\n", view.Publisher)
		}
	}
	for _, note := range view.Advisory {
		fmt.Fprintf(w, "  advisory            %s\n", note)
	}
}

// batchSafePath resolves a record-relative path strictly below rootAbs and
// refuses anything that could escape it: absolute paths, parent traversal,
// and empty identities.
func batchSafePath(rootAbs, rel string) (string, error) {
	if !filepath.IsAbs(rootAbs) {
		if absRoot, err := filepath.Abs(rootAbs); err == nil {
			rootAbs = absRoot
		}
	}
	rel = strings.TrimSpace(rel)
	if rel == "" {
		return "", errors.New("empty path")
	}
	if strings.ContainsRune(rel, 0) {
		return "", fmt.Errorf("path %q contains a NUL byte", rel)
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("path %q is absolute; only paths below the explicit root are admitted", rel)
	}
	cleaned := filepath.Clean(filepath.FromSlash(rel))
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes the explicit root", rel)
	}
	abs := filepath.Join(rootAbs, cleaned)
	rootPrefix := rootAbs + string(filepath.Separator)
	if abs != rootAbs && !strings.HasPrefix(abs, rootPrefix) {
		return "", fmt.Errorf("path %q escapes the explicit root", rel)
	}
	return abs, nil
}

// batchPathKey normalizes a record-relative path to a slash-separated,
// cleaned key so identity comparisons are platform-stable.
func batchPathKey(rel string) string {
	return filepath.ToSlash(filepath.Clean(filepath.FromSlash(strings.TrimSpace(rel))))
}

func hashBatchInputFile(rootAbs, rel string) (string, error) {
	abs, err := batchSafePath(rootAbs, rel)
	if err != nil {
		return "", err
	}
	file, err := os.Open(abs)
	if err != nil {
		return "", fmt.Errorf("input %s: %w", batchPathKey(rel), err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("input %s: %w", batchPathKey(rel), err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return "", fmt.Errorf("input %s is not a regular file; ambiguous file identity", batchPathKey(rel))
	}
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return "", fmt.Errorf("hash input %s: %w", batchPathKey(rel), err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func normalizeBuildBatchSHA(raw string) (string, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "", nil
	}
	value = strings.TrimPrefix(value, "sha256:")
	if len(value) != 64 {
		return "", fmt.Errorf("pinned hash must be 64 hex characters (got %d)", len(value))
	}
	lower := strings.ToLower(value)
	for _, r := range lower {
		isHex := (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')
		if !isHex {
			return "", fmt.Errorf("pinned hash %q is not lowercase hex", raw)
		}
	}
	return lower, nil
}

func deriveBuildBatchPackages(paths []string) []string {
	set := map[string]struct{}{}
	for _, raw := range paths {
		dir := filepath.ToSlash(filepath.Dir(filepath.FromSlash(strings.TrimSpace(raw))))
		if dir == "." {
			dir = "(root)"
		}
		set[dir] = struct{}{}
	}
	out := make([]string, 0, len(set))
	for dir := range set {
		out = append(out, dir)
	}
	sort.Strings(out)
	return out
}

func validateBuildBatchProfile(profile string) error {
	if profile == "" {
		return nil
	}
	if profile != "release" && profile != "dev" && profile != "race" {
		return fmt.Errorf("--profile must be release, dev, or race (got %q)", profile)
	}
	return nil
}

func firstNonempty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
