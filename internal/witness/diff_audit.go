package witness

import (
	"bufio"
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// AuditBand defines the attention classification band for a commit or patch.
// It reflects the HUMAN_RESIDUAL doctrine: 100% of human review attention is
// concentrated on RESIDUAL, while machine-verified changes are CLEARED.
type AuditBand string

const (
	// BandCleared indicates that the change is corroborated by non-forgeable
	// machine evidence (source modifications, congruent test assertions).
	BandCleared AuditBand = "CLEARED"

	// BandResidual indicates that a checkable claim was made that could NOT be
	// witnessed by the diff (empty diff with done claim, doc-only fix, deleted
	// test assertions, or comment-only no-ops).
	BandResidual AuditBand = "RESIDUAL"

	// BandUnverifiable indicates that the commit makes no checkable claim or
	// cannot be parsed, leaving review priority low/neutral.
	BandUnverifiable AuditBand = "UNVERIFIABLE"
)

// WitnessRung is the forgeability classification rung of the witness evidence.
type WitnessRung string

const (
	// RungDiffWitnessed represents immutable machine-witnessed evidence.
	RungDiffWitnessed WitnessRung = "diff-witnessed"

	// RungSubjectOnly indicates that the claim rests solely on author-authored prose.
	RungSubjectOnly WitnessRung = "subject-only"

	// RungAbstain indicates no conclusive determination could be reached.
	RungAbstain WitnessRung = "abstain"
)

// SuspiciousPattern identifies specific automated falsification or evasion behaviors.
type SuspiciousPattern string

const (
	// PatternEmptyDiffWithDoneClaim: an agent claims completion/done or a code fix over an empty diff.
	PatternEmptyDiffWithDoneClaim SuspiciousPattern = "EMPTY_COMMIT_WITH_DONE_CLAIM"

	// PatternAssertionDeletionNoReplacement: test assertions were deleted without equivalent additions.
	PatternAssertionDeletionNoReplacement SuspiciousPattern = "ASSERTION_DELETION_WITHOUT_REPLACEMENT"

	// PatternDocOnlyCodeFix: an agent claims a code fix or feature while touching only documentation.
	PatternDocOnlyCodeFix SuspiciousPattern = "DOC_ONLY_CODE_FIX_CLAIM"

	// PatternNoOpCodeEdit: source files were touched but contain only comments or whitespace changes.
	PatternNoOpCodeEdit SuspiciousPattern = "NO_OP_CODE_EDIT"
)

// TaskIntent describes the declared or inferred intention of the change.
type TaskIntent struct {
	Kind          string   `json:"kind,omitempty"`           // "fix", "feat", "docs", "test", "refactor", "chore", etc.
	Description   string   `json:"description,omitempty"`    // High-level task description or prompt
	ClaimedDone   bool     `json:"claimed_done,omitempty"`   // Explicit done / resolved claim
	ExpectedPaths []string `json:"expected_paths,omitempty"` // Expected path globs if specified
	RequireTests  bool     `json:"require_tests,omitempty"`  // Whether test modification is required
}

// DiffAuditVerdict is the inspectable typed verdict produced by the diff-audit runner.
type DiffAuditVerdict struct {
	Band               AuditBand           `json:"band"`
	WitnessRung        WitnessRung         `json:"witness_rung"`
	Reasons            []string            `json:"reasons"`
	Confidence         float64             `json:"confidence"`
	SuspiciousPatterns []SuspiciousPattern `json:"suspicious_patterns,omitempty"`
	ClaimKind          string              `json:"claim_kind,omitempty"`
	SourceFiles        []string            `json:"source_files,omitempty"`
	TestFiles          []string            `json:"test_files,omitempty"`
	DocFiles           []string            `json:"doc_files,omitempty"`
	ConfigFiles        []string            `json:"config_files,omitempty"`
	DeletedAssertions  int                 `json:"deleted_assertions"`
	AddedAssertions    int                 `json:"added_assertions"`
	NetAssertionDelta  int                 `json:"net_assertion_delta"`
	TotalAddedLines    int                 `json:"total_added_lines"`
	TotalDeletedLines  int                 `json:"total_deleted_lines"`
}

// IsCleared reports whether the change was cleared by diff evidence.
func (v DiffAuditVerdict) IsCleared() bool {
	return v.Band == BandCleared
}

// IsResidual reports whether the change requires human residual attention.
func (v DiffAuditVerdict) IsResidual() bool {
	return v.Band == BandResidual
}

// IsUnverifiable reports whether the change was uncheckable.
func (v DiffAuditVerdict) IsUnverifiable() bool {
	return v.Band == BandUnverifiable
}

// HasSuspiciousPattern reports whether a specific suspicious pattern was flagged.
func (v DiffAuditVerdict) HasSuspiciousPattern(p SuspiciousPattern) bool {
	for _, sp := range v.SuspiciousPatterns {
		if sp == p {
			return true
		}
	}
	return false
}

// WitnessOutcome converts the verdict to the kernel's ABI witness outcome.
func (v DiffAuditVerdict) WitnessOutcome() abi.WitnessOutcome {
	switch v.Band {
	case BandCleared:
		return abi.WitnessConfirmed
	case BandResidual:
		return abi.WitnessRefuted
	default:
		return abi.WitnessAbstain
	}
}

// LineType classifies a line within a diff hunk.
type LineType int

const (
	LineContext LineType = iota
	LineAdd
	LineDelete
)

// DiffLine represents a single line in a unified diff hunk.
type DiffLine struct {
	Type    LineType `json:"type"`
	Content string   `json:"content"`
}

// DiffHunk represents a hunk in a unified diff.
type DiffHunk struct {
	OldStart int        `json:"old_start"`
	OldLines int        `json:"old_lines"`
	NewStart int        `json:"new_start"`
	NewLines int        `json:"new_lines"`
	Lines    []DiffLine `json:"lines,omitempty"`
}

// DiffFile represents the diff information for a single file.
type DiffFile struct {
	Path         string     `json:"path"`
	OldPath      string     `json:"old_path,omitempty"`
	Status       string     `json:"status"` // "added", "modified", "deleted", "renamed"
	AddedLines   int        `json:"added_lines"`
	DeletedLines int        `json:"deleted_lines"`
	Hunks        []DiffHunk `json:"hunks,omitempty"`
}

// DiffAuditRequest encapsulates the complete input to the diff-audit runner.
type DiffAuditRequest struct {
	SHA     string     `json:"sha,omitempty"`
	Subject string     `json:"subject"`
	Body    string     `json:"body,omitempty"`
	RawDiff string     `json:"raw_diff,omitempty"`
	Files   []DiffFile `json:"files,omitempty"`
	Intent  TaskIntent `json:"intent,omitempty"`
}

var (
	conventionalCommitRE = regexp.MustCompile(`^([a-zA-Z]+)(?:\([^\)]+\))?!?:\s*(.*)$`)
	doneClaimKeywordsRE  = regexp.MustCompile(`(?i)\b(done|shipped|fixed|fixes|resolved|resolves|closed|closes|implemented|implements|completed|completes)\b`)
	codeFixKeywordsRE    = regexp.MustCompile(`(?i)\b(fix|fixes|bug|panic|crash|leak|race|overflow|error|null|nil|repair|repaired|deadlock)\b`)
	codeFeatKeywordsRE   = regexp.MustCompile(`(?i)\b(feat|feature|add|adds|implement|implements|support|supports|create|creates|new)\b`)

	assertionRE = regexp.MustCompile(`(?i)(` +
		`\bt\.(Error|Errorf|Fatal|Fatalf|Fail|FailNow)\b` +
		`|\b(assert|require)\.(Equal|NotEqual|True|False|Nil|NotNil|NoError|Error|Contains|ElementsMatch|Empty|NotEmpty|Len|Same|NotSame|Panics|Zero|NotZero|InDelta|InEpsilon|Implements)\b` +
		`|\bassert\s+` +
		`|\bself\.assert(Equal|NotEqual|True|False|Is|IsNot|IsNone|IsNotNone|In|NotIn|Raises|AlmostEqual|Greater|Less)\b` +
		`|\bexpect\(.+\)\.(to|toBe|toEqual|toStrictEqual|toBeTruthy|toBeFalsy|toThrow|toContain)\b` +
		`|\bqt\.Assert\b` +
		`|\bis\.(Equal|True|False|Nil|NotNil|OK)\b` +
		`|\bassertEquals\b|\bassertTrue\b|\bassertFalse\b|\bassertNotNull\b|\bassertNull\b` +
		`)`)
)

// DiffAuditRunner executes diff-audit inspections against commits, patches, or file structures.
type DiffAuditRunner struct {
	dir string
	run Runner
}

// NewDiffAuditRunner constructs a DiffAuditRunner for the given directory and git runner.
func NewDiffAuditRunner(dir string, run Runner) *DiffAuditRunner {
	if run == nil {
		run = gitRunner
	}
	return &DiffAuditRunner{dir: dir, run: run}
}

// AuditCommit inspects a git commit ref and evaluates it against the task intent.
func (r *DiffAuditRunner) AuditCommit(ctx context.Context, ref string, intent TaskIntent) (DiffAuditVerdict, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return DiffAuditVerdict{
			Band:        BandUnverifiable,
			WitnessRung: RungAbstain,
			Reasons:     []string{"empty commit ref provided"},
			Confidence:  0.0,
		}, nil
	}

	// 1. Read commit message
	msgOut, code, err := r.run(ctx, r.dir, "show", "-s", "--format=%B", ref)
	if err != nil || code != 0 {
		return DiffAuditVerdict{
			Band:        BandUnverifiable,
			WitnessRung: RungAbstain,
			Reasons:     []string{fmt.Sprintf("failed to read commit message for ref %q (exit %d)", ref, code)},
			Confidence:  0.0,
		}, err
	}

	lines := strings.Split(strings.TrimSpace(msgOut), "\n")
	subject := ""
	body := ""
	if len(lines) > 0 {
		subject = strings.TrimSpace(lines[0])
		if len(lines) > 1 {
			body = strings.TrimSpace(strings.Join(lines[1:], "\n"))
		}
	}

	// 2. Read full patch
	patchOut, code, err := r.run(ctx, r.dir, "show", "-p", "--format=", ref)
	if err != nil || code != 0 {
		return DiffAuditVerdict{
			Band:        BandUnverifiable,
			WitnessRung: RungAbstain,
			Reasons:     []string{fmt.Sprintf("failed to read patch for ref %q (exit %d)", ref, code)},
			Confidence:  0.0,
		}, err
	}

	files, _ := ParseUnifiedDiff(patchOut)
	req := DiffAuditRequest{
		SHA:     ref,
		Subject: subject,
		Body:    body,
		RawDiff: patchOut,
		Files:   files,
		Intent:  intent,
	}

	return r.AuditRequest(req), nil
}

// AuditPatch evaluates a raw unified patch string and commit subject.
func (r *DiffAuditRunner) AuditPatch(subject, rawDiff string, intent TaskIntent) DiffAuditVerdict {
	files, _ := ParseUnifiedDiff(rawDiff)
	return r.AuditRequest(DiffAuditRequest{
		Subject: subject,
		RawDiff: rawDiff,
		Files:   files,
		Intent:  intent,
	})
}

// AuditFiles evaluates pre-parsed diff files against a subject and task intent.
func (r *DiffAuditRunner) AuditFiles(subject string, files []DiffFile, intent TaskIntent) DiffAuditVerdict {
	return r.AuditRequest(DiffAuditRequest{
		Subject: subject,
		Files:   files,
		Intent:  intent,
	})
}

// AuditRequest is the core pure evaluation engine of DiffAuditRunner.
func (r *DiffAuditRunner) AuditRequest(req DiffAuditRequest) DiffAuditVerdict {
	files := req.Files
	if len(files) == 0 && strings.TrimSpace(req.RawDiff) != "" {
		files, _ = ParseUnifiedDiff(req.RawDiff)
	}

	claimKind, claimedDone, requiresTests := inferClaim(req.Subject, req.Body, req.Intent)

	var sourceFiles, testFiles, docFiles, configFiles []string
	totalAdded := 0
	totalDeleted := 0
	activeCodeAdded := 0
	activeCodeDeleted := 0
	addedAssertions := 0
	deletedAssertions := 0

	for _, file := range files {
		totalAdded += file.AddedLines
		totalDeleted += file.DeletedLines

		isSource, isTest, isDoc, isConfig := classifyFilePath(file.Path)
		if isTest {
			testFiles = append(testFiles, file.Path)
		} else if isDoc {
			docFiles = append(docFiles, file.Path)
		} else if isConfig {
			configFiles = append(configFiles, file.Path)
		} else if isSource {
			sourceFiles = append(sourceFiles, file.Path)
		}

		for _, hunk := range file.Hunks {
			for _, line := range hunk.Lines {
				content := strings.TrimSpace(line.Content)
				isCommentOrBlank := isCommentOrBlankLine(content)

				switch line.Type {
				case LineAdd:
					if isSource && !isCommentOrBlank {
						activeCodeAdded++
					}
					if isTest && !isCommentOrBlank && isAssertionLine(content) {
						addedAssertions++
					}
				case LineDelete:
					if isSource && !isCommentOrBlank {
						activeCodeDeleted++
					}
					if isTest && !isCommentOrBlank && isAssertionLine(content) {
						deletedAssertions++
					}
				}
			}
		}
	}

	netAssertionDelta := addedAssertions - deletedAssertions

	verdict := DiffAuditVerdict{
		ClaimKind:         claimKind,
		SourceFiles:       sourceFiles,
		TestFiles:         testFiles,
		DocFiles:          docFiles,
		ConfigFiles:       configFiles,
		DeletedAssertions: deletedAssertions,
		AddedAssertions:   addedAssertions,
		NetAssertionDelta: netAssertionDelta,
		TotalAddedLines:   totalAdded,
		TotalDeletedLines: totalDeleted,
	}

	// RULE 1: Empty Diff Check
	if len(files) == 0 || (totalAdded == 0 && totalDeleted == 0) {
		if claimedDone || isCodeEffectClaim(claimKind) || claimKind == "test" || claimKind == "docs" {
			verdict.Band = BandResidual
			verdict.WitnessRung = RungSubjectOnly
			verdict.SuspiciousPatterns = append(verdict.SuspiciousPatterns, PatternEmptyDiffWithDoneClaim)
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("commit claims completion or effect (%q) but the diff is completely empty (0 files/lines modified)", claimKind))
			verdict.Confidence = 1.00
			return verdict
		}

		verdict.Band = BandUnverifiable
		verdict.WitnessRung = RungAbstain
		verdict.Reasons = append(verdict.Reasons, "empty diff with no checkable effect claim")
		verdict.Confidence = 0.50
		return verdict
	}

	// RULE 2: Monotonic Test Assertion Invariant (Anti-Reward Hacking)
	if deletedAssertions > addedAssertions {
		verdict.Band = BandResidual
		verdict.WitnessRung = RungSubjectOnly
		verdict.SuspiciousPatterns = append(verdict.SuspiciousPatterns, PatternAssertionDeletionNoReplacement)
		verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("test assertions deleted without replacement: deleted %d, added %d (net delta: %d)", deletedAssertions, addedAssertions, netAssertionDelta))
		verdict.Confidence = 0.98
		return verdict
	}

	// RULE 3: Documentation-Only Edits Claiming Code Fixes / Features
	if len(sourceFiles) == 0 && len(testFiles) == 0 && len(docFiles) > 0 {
		if isCodeEffectClaim(claimKind) {
			verdict.Band = BandResidual
			verdict.WitnessRung = RungSubjectOnly
			verdict.SuspiciousPatterns = append(verdict.SuspiciousPatterns, PatternDocOnlyCodeFix)
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("code-effect claim (%q) but diff touches only documentation files (%d doc files)", claimKind, len(docFiles)))
			verdict.Confidence = 1.00
			return verdict
		}
		if claimKind == "docs" {
			verdict.Band = BandCleared
			verdict.WitnessRung = RungDiffWitnessed
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("documentation claim (%q) witnessed by %d modified documentation file(s)", claimKind, len(docFiles)))
			verdict.Confidence = 1.00
			return verdict
		}
		verdict.Band = BandUnverifiable
		verdict.WitnessRung = RungAbstain
		verdict.Reasons = append(verdict.Reasons, "documentation changes with no checkable code-effect claim")
		verdict.Confidence = 0.70
		return verdict
	}

	// RULE 4: No-Op / Comment-Only Source Edit Claiming Code Fix / Feature
	if len(sourceFiles) > 0 && isCodeEffectClaim(claimKind) {
		if activeCodeAdded == 0 && activeCodeDeleted == 0 {
			verdict.Band = BandResidual
			verdict.WitnessRung = RungSubjectOnly
			verdict.SuspiciousPatterns = append(verdict.SuspiciousPatterns, PatternNoOpCodeEdit)
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("code-effect claim (%q) but source diff contains only comments or whitespace modifications", claimKind))
			verdict.Confidence = 0.95
			return verdict
		}
	}

	// RULE 5: Test-Only Diff
	if len(sourceFiles) == 0 && len(testFiles) > 0 {
		if claimKind == "test" {
			verdict.Band = BandCleared
			verdict.WitnessRung = RungDiffWitnessed
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("test claim witnessed by %d test file(s) with %d added assertion(s)", len(testFiles), addedAssertions))
			verdict.Confidence = 1.00
			return verdict
		}
		if isCodeEffectClaim(claimKind) {
			verdict.Band = BandResidual
			verdict.WitnessRung = RungSubjectOnly
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("code-effect claim (%q) but diff touches only test files without source code changes", claimKind))
			verdict.Confidence = 0.95
			return verdict
		}
	}

	// RULE 6: Code + Test Diffs (Legitimate)
	if len(sourceFiles) > 0 {
		if len(testFiles) > 0 {
			verdict.Band = BandCleared
			verdict.WitnessRung = RungDiffWitnessed
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("diff witnessed: code changes corroborated by %d source file(s) and %d test file(s) (assertions: +%d/-%d)", len(sourceFiles), len(testFiles), addedAssertions, deletedAssertions))
			verdict.Confidence = 1.00
			return verdict
		}

		// Source touched, but no tests in diff
		if requiresTests {
			verdict.Band = BandResidual
			verdict.WitnessRung = RungSubjectOnly
			verdict.Reasons = append(verdict.Reasons, "task intent explicitly requires test modifications, but diff touches no test files")
			verdict.Confidence = 0.90
			return verdict
		}

		if isCodeEffectClaim(claimKind) || claimKind == "refactor" {
			verdict.Band = BandCleared
			verdict.WitnessRung = RungDiffWitnessed
			verdict.Reasons = append(verdict.Reasons, fmt.Sprintf("diff witnessed: code-effect claim (%q) corroborated by %d source file(s)", claimKind, len(sourceFiles)))
			verdict.Confidence = 0.85
			return verdict
		}
	}

	// RULE 7: Fallthrough / Ambiguous
	verdict.Band = BandUnverifiable
	verdict.WitnessRung = RungAbstain
	verdict.Reasons = append(verdict.Reasons, "diff contains no verifiable claim-to-artifact mapping")
	verdict.Confidence = 0.50
	return verdict
}

// Top-level convenience audit functions.

// AuditPatch evaluates a raw unified patch and subject against an intent.
func AuditPatch(subject, rawDiff string, intent TaskIntent) DiffAuditVerdict {
	return NewDiffAuditRunner("", nil).AuditPatch(subject, rawDiff, intent)
}

// AuditRequest evaluates a structured DiffAuditRequest.
func AuditRequest(req DiffAuditRequest) DiffAuditVerdict {
	return NewDiffAuditRunner("", nil).AuditRequest(req)
}

// ParseUnifiedDiff parses a raw git patch into structured DiffFile slices.
func ParseUnifiedDiff(rawDiff string) ([]DiffFile, error) {
	var files []DiffFile
	scanner := bufio.NewScanner(strings.NewReader(rawDiff))

	var currentFile *DiffFile
	var currentHunk *DiffHunk

	flushHunk := func() {
		if currentFile != nil && currentHunk != nil {
			currentFile.Hunks = append(currentFile.Hunks, *currentHunk)
			currentHunk = nil
		}
	}

	flushFile := func() {
		flushHunk()
		if currentFile != nil {
			files = append(files, *currentFile)
			currentFile = nil
		}
	}

	hunkHeaderRE := regexp.MustCompile(`^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@`)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "diff --git ") {
			flushFile()
			parts := strings.Fields(line)
			filePath := ""
			if len(parts) >= 4 {
				filePath = strings.TrimPrefix(parts[3], "b/")
			}
			currentFile = &DiffFile{
				Path:   filePath,
				Status: "modified",
			}
			continue
		}

		if currentFile == nil {
			continue
		}

		if strings.HasPrefix(line, "new file mode ") {
			currentFile.Status = "added"
			continue
		}
		if strings.HasPrefix(line, "deleted file mode ") {
			currentFile.Status = "deleted"
			continue
		}
		if strings.HasPrefix(line, "rename from ") {
			currentFile.OldPath = strings.TrimPrefix(line, "rename from ")
			currentFile.Status = "renamed"
			continue
		}
		if strings.HasPrefix(line, "rename to ") {
			currentFile.Path = strings.TrimPrefix(line, "rename to ")
			continue
		}
		if strings.HasPrefix(line, "--- ") {
			oldPath := strings.TrimPrefix(line, "--- ")
			if oldPath != "/dev/null" {
				currentFile.OldPath = strings.TrimPrefix(oldPath, "a/")
			}
			continue
		}
		if strings.HasPrefix(line, "+++ ") {
			newPath := strings.TrimPrefix(line, "+++ ")
			if newPath != "/dev/null" {
				currentFile.Path = strings.TrimPrefix(newPath, "b/")
			}
			continue
		}

		if strings.HasPrefix(line, "@@ ") {
			flushHunk()
			matches := hunkHeaderRE.FindStringSubmatch(line)
			if len(matches) >= 4 {
				oldStart, _ := strconv.Atoi(matches[1])
				oldLines := 1
				if matches[2] != "" {
					oldLines, _ = strconv.Atoi(matches[2])
				}
				newStart, _ := strconv.Atoi(matches[3])
				newLines := 1
				if len(matches) >= 5 && matches[4] != "" {
					newLines, _ = strconv.Atoi(matches[4])
				}
				currentHunk = &DiffHunk{
					OldStart: oldStart,
					OldLines: oldLines,
					NewStart: newStart,
					NewLines: newLines,
				}
			}
			continue
		}

		if currentHunk != nil {
			if strings.HasPrefix(line, "+") {
				currentFile.AddedLines++
				currentHunk.Lines = append(currentHunk.Lines, DiffLine{
					Type:    LineAdd,
					Content: line[1:],
				})
			} else if strings.HasPrefix(line, "-") {
				currentFile.DeletedLines++
				currentHunk.Lines = append(currentHunk.Lines, DiffLine{
					Type:    LineDelete,
					Content: line[1:],
				})
			} else if strings.HasPrefix(line, " ") {
				currentHunk.Lines = append(currentHunk.Lines, DiffLine{
					Type:    LineContext,
					Content: line[1:],
				})
			}
		}
	}

	flushFile()
	return files, scanner.Err()
}

func classifyFilePath(p string) (isSource, isTest, isDoc, isConfig bool) {
	norm := strings.ReplaceAll(filepath.ToSlash(strings.TrimSpace(p)), "\\", "/")
	base := filepath.Base(norm)
	ext := strings.ToLower(filepath.Ext(norm))

	// Test files
	if strings.HasSuffix(norm, "_test.go") ||
		strings.HasPrefix(base, "test_") ||
		strings.HasSuffix(base, "_test.py") ||
		strings.Contains(base, ".test.") ||
		strings.Contains(base, ".spec.") ||
		strings.HasSuffix(base, "Test.java") ||
		strings.HasSuffix(base, "Tests.java") ||
		strings.HasPrefix(norm, "test/") ||
		strings.HasPrefix(norm, "tests/") {
		return false, true, false, false
	}

	// Doc files
	switch ext {
	case ".md", ".txt", ".rst", ".adoc", ".markdown":
		return false, false, true, false
	}
	if strings.HasPrefix(norm, "docs/") || strings.HasPrefix(norm, "doc/") {
		return false, false, true, false
	}
	baseUpper := strings.ToUpper(base)
	if baseUpper == "README" || baseUpper == "CONTRIBUTING" || baseUpper == "CHANGELOG" ||
		baseUpper == "LICENSE" || baseUpper == "NOTICE" || baseUpper == "AUTHORS" || baseUpper == "ROADMAP" {
		return false, false, true, false
	}

	// Config / Meta files
	switch ext {
	case ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg":
		return false, false, false, true
	}
	switch base {
	case "go.mod", "go.sum", "Cargo.toml", "Cargo.lock", "package.json", "package-lock.json",
		"Makefile", "Dockerfile", ".gitignore", ".gitattributes", ".dockerignore":
		return false, false, false, true
	}

	// Source files
	switch ext {
	case ".go", ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".rs", ".ts", ".tsx",
		".js", ".jsx", ".java", ".sh", ".bash", ".zsh", ".sql", ".proto", ".graphql",
		".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".lua", ".zig":
		return true, false, false, false
	}

	return false, false, false, false
}

func isCommentOrBlankLine(line string) bool {
	t := strings.TrimSpace(line)
	if t == "" {
		return true
	}
	if strings.HasPrefix(t, "//") || strings.HasPrefix(t, "/*") || strings.HasPrefix(t, "*") ||
		strings.HasPrefix(t, "*/") || strings.HasPrefix(t, "#") || strings.HasPrefix(t, ";") ||
		strings.HasPrefix(t, "--") || strings.HasPrefix(t, "<!--") {
		return true
	}
	return false
}

func isAssertionLine(content string) bool {
	return assertionRE.MatchString(content)
}

func isCodeEffectClaim(kind string) bool {
	switch strings.ToLower(kind) {
	case "fix", "feat", "feature", "perf", "bug":
		return true
	default:
		return false
	}
}

func inferClaim(subject, body string, intent TaskIntent) (claimKind string, claimedDone bool, requiresTests bool) {
	requiresTests = intent.RequireTests
	claimedDone = intent.ClaimedDone

	if intent.Kind != "" {
		claimKind = strings.ToLower(strings.TrimSpace(intent.Kind))
	} else {
		// Try conventional commit
		matches := conventionalCommitRE.FindStringSubmatch(subject)
		if len(matches) >= 2 {
			claimKind = strings.ToLower(matches[1])
		} else {
			// Infer from subject keywords
			subLower := strings.ToLower(subject)
			if codeFixKeywordsRE.MatchString(subLower) {
				claimKind = "fix"
			} else if codeFeatKeywordsRE.MatchString(subLower) {
				claimKind = "feat"
			} else if strings.Contains(subLower, "test") {
				claimKind = "test"
			} else if strings.Contains(subLower, "doc") || strings.Contains(subLower, "readme") {
				claimKind = "docs"
			} else if strings.Contains(subLower, "refactor") {
				claimKind = "refactor"
			} else if strings.Contains(subLower, "chore") {
				claimKind = "chore"
			}
		}
	}

	if !claimedDone {
		combined := subject + " " + body
		if doneClaimKeywordsRE.MatchString(combined) || isCodeEffectClaim(claimKind) {
			claimedDone = true
		}
	}

	return claimKind, claimedDone, requiresTests
}
