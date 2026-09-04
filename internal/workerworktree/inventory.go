package workerworktree

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	inventorySchemaV1 = "fak-worker-worktree-intent/1"
	inventorySchema   = "fak-worker-worktree-intent/2"
)

var (
	intentIssueTokenRE = regexp.MustCompile(`#[A-Za-z0-9_-]+`)
	intentIssueIDRE    = regexp.MustCompile(`^#[1-9][0-9]*$`)
)

type Intent struct {
	Schema      string   `json:"schema"`
	Path        string   `json:"path"`
	BaseSHA     string   `json:"base_sha"`
	Message     string   `json:"message,omitempty"`
	IssueNumber int      `json:"issue_number,omitempty"`
	Paths       []string `json:"paths,omitempty"`
}

type InventoryRow struct {
	Path          string   `json:"path"`
	BaseSHA       string   `json:"base_sha,omitempty"`
	HeadSHA       string   `json:"head_sha,omitempty"`
	DirtyPaths    []string `json:"dirty_paths,omitempty"`
	State         string   `json:"state"`
	NeedsOperator bool     `json:"needs_operator,omitempty"`
	Reason        string   `json:"reason,omitempty"`
	LandArgv      []string `json:"land_argv,omitempty"`
}

func intentDir(wtPath string) string {
	return filepath.Join(filepath.Dir(wtPath), ".fak-worker-intents")
}
func intentPath(wtPath string) string {
	return filepath.Join(intentDir(wtPath), filepath.Base(wtPath)+".json")
}
func messagePath(wtPath string) string {
	return filepath.Join(intentDir(wtPath), filepath.Base(wtPath)+".message")
}

func intendedIssueNumber(message string) (int, error) {
	subject := strings.SplitN(strings.ReplaceAll(message, "\r\n", "\n"), "\n", 2)[0]
	tokens := intentIssueTokenRE.FindAllString(subject, -1)
	if len(tokens) == 0 {
		return 0, nil
	}
	if len(tokens) != 1 {
		return 0, fmt.Errorf("intended commit subject must contain at most one issue reference, got %d", len(tokens))
	}
	if !intentIssueIDRE.MatchString(tokens[0]) {
		return 0, fmt.Errorf("intended commit subject has malformed issue reference %q", tokens[0])
	}
	issue, err := strconv.Atoi(strings.TrimPrefix(tokens[0], "#"))
	if err != nil || issue <= 0 {
		return 0, fmt.Errorf("intended commit subject has malformed issue reference %q", tokens[0])
	}
	return issue, nil
}

func canonicalIntentPaths(paths []string) []string {
	clean := make([]string, 0, len(paths))
	seen := map[string]bool{}
	for _, p := range paths {
		p = strings.Trim(strings.ReplaceAll(p, "\\", "/"), "/")
		if p != "" && !seen[p] {
			seen[p] = true
			clean = append(clean, p)
		}
	}
	sort.Strings(clean)
	return clean
}

// SaveIntent records operator-supplied land metadata outside the managed worktree.
// It cannot become part of the worker diff and is safe from git clean/reset.
func SaveIntent(wtPath, baseSHA, message string, paths []string) error {
	wtPath = filepath.Clean(strings.TrimSpace(wtPath))
	if wtPath == "." || wtPath == "" {
		return fmt.Errorf("worktree path is required")
	}
	message = strings.TrimSpace(message)
	issue, err := intendedIssueNumber(message)
	if err != nil {
		return err
	}
	in := Intent{
		Schema: inventorySchema, Path: wtPath, BaseSHA: strings.TrimSpace(baseSHA),
		Message: message, IssueNumber: issue, Paths: canonicalIntentPaths(paths),
	}
	if err := os.MkdirAll(intentDir(wtPath), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(in, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if existing, err := os.ReadFile(intentPath(wtPath)); err == nil {
		if bytes.Equal(existing, b) {
			_, err = LoadIntent(wtPath)
			return err
		}
		prior, err := LoadIntent(wtPath)
		if err != nil {
			return fmt.Errorf("refuse replacement of invalid existing worker intent: %w", err)
		}
		if prior.Schema != in.Schema || prior.Path != in.Path || prior.BaseSHA != in.BaseSHA || prior.Message != in.Message || prior.IssueNumber != in.IssueNumber {
			return fmt.Errorf("worker intent already exists with different coordinator metadata")
		}
		if !containsAllPaths(in.Paths, prior.Paths) {
			return fmt.Errorf("worker intent path update may expand but not remove coordinator paths")
		}
		// Re-preparing a reused worktree may deliberately expand its land paths. The
		// signed message and derived issue authority remain immutable; only this
		// monotonic path-superset update uses the replacement path.
		return os.WriteFile(intentPath(wtPath), b, 0o600)
	} else if !os.IsNotExist(err) {
		return err
	}
	intentFile, err := os.OpenFile(intentPath(wtPath), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create immutable worker intent: %w", err)
	}
	if _, err := intentFile.Write(b); err != nil {
		_ = intentFile.Close()
		_ = os.Remove(intentPath(wtPath))
		return err
	}
	if err := intentFile.Close(); err != nil {
		_ = os.Remove(intentPath(wtPath))
		return err
	}
	if in.Message == "" {
		return nil
	}
	messageFile, err := os.OpenFile(messagePath(wtPath), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		_ = os.Remove(intentPath(wtPath))
		return fmt.Errorf("create immutable worker intent message: %w", err)
	}
	if _, err := messageFile.Write([]byte(in.Message + "\n")); err != nil {
		_ = messageFile.Close()
		_ = os.Remove(messagePath(wtPath))
		_ = os.Remove(intentPath(wtPath))
		return err
	}
	if err := messageFile.Close(); err != nil {
		_ = os.Remove(messagePath(wtPath))
		_ = os.Remove(intentPath(wtPath))
		return err
	}
	return nil
}

// LoadIntent reads coordinator-owned land metadata stored outside the managed
// worktree. It validates the closed schema, canonical worktree identity, immutable
// message mirror, and issue number derived from the intended commit subject.
func LoadIntent(wtPath string) (Intent, error) {
	wtPath = filepath.Clean(strings.TrimSpace(wtPath))
	if wtPath == "." || wtPath == "" {
		return Intent{}, fmt.Errorf("worktree path is required")
	}
	b, err := os.ReadFile(intentPath(wtPath))
	if err != nil {
		return Intent{}, err
	}
	var in Intent
	decoder := json.NewDecoder(bytes.NewReader(b))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&in); err != nil {
		return Intent{}, fmt.Errorf("decode worker intent: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return Intent{}, fmt.Errorf("decode worker intent: trailing data")
	}
	storedSchema := in.Schema
	if (storedSchema != inventorySchemaV1 && storedSchema != inventorySchema) || !samePath(in.Path, wtPath) {
		return Intent{}, fmt.Errorf("invalid worker intent")
	}
	if storedSchema == inventorySchemaV1 {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(b, &fields); err != nil {
			return Intent{}, fmt.Errorf("decode legacy worker intent fields: %w", err)
		}
		if _, added := fields["issue_number"]; added {
			return Intent{}, fmt.Errorf("legacy worker intent contains non-v1 issue_number field")
		}
	}
	if in.Path != filepath.Clean(strings.TrimSpace(in.Path)) || in.BaseSHA != strings.TrimSpace(in.BaseSHA) || in.Message != strings.TrimSpace(in.Message) {
		return Intent{}, fmt.Errorf("worker intent contains non-canonical path, base SHA, or message")
	}
	if !equalStrings(in.Paths, canonicalIntentPaths(in.Paths)) {
		return Intent{}, fmt.Errorf("worker intent paths are not canonical sorted unique paths")
	}
	issue, err := intendedIssueNumber(in.Message)
	if err != nil {
		return Intent{}, err
	}
	if storedSchema == inventorySchemaV1 {
		in.Schema = inventorySchema
		in.IssueNumber = issue
	} else if in.IssueNumber != issue {
		return Intent{}, fmt.Errorf("worker intent issue_number %d does not match subject issue %d", in.IssueNumber, issue)
	}
	if in.Message == "" {
		if _, err := os.Stat(messagePath(wtPath)); err == nil || !os.IsNotExist(err) {
			return Intent{}, fmt.Errorf("worker intent has an unexpected message mirror")
		}
		return in, nil
	}
	message, err := os.ReadFile(messagePath(wtPath))
	if err != nil {
		return Intent{}, fmt.Errorf("read worker intent message mirror: %w", err)
	}
	if !bytes.Equal(message, []byte(in.Message+"\n")) {
		return Intent{}, fmt.Errorf("worker intent message mirror does not match coordinator metadata")
	}
	return in, nil
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func containsAllPaths(have, required []string) bool {
	set := make(map[string]bool, len(have))
	for _, path := range have {
		set[path] = true
	}
	for _, path := range required {
		if !set[path] {
			return false
		}
	}
	return true
}

// Inventory inspects managed worktrees without writing refs, indexes, or worktree bytes.
func Inventory(root string, git GitRunner) ([]InventoryRow, error) {
	if git == nil {
		git = defaultGit
	}
	_, paths := Count(root, git)
	return InventoryForPaths(root, paths, git)
}

// InventoryForPaths inspects the provided managed worktree paths without writing refs, indexes, or worktree bytes.
func InventoryForPaths(root string, paths []string, git GitRunner) ([]InventoryRow, error) {
	rows, _, err := InventoryForPathsContext(context.Background(), root, paths, git)
	return rows, err
}

// InventoryForPathsContext inspects the provided managed worktree paths bounded by ctx.
func InventoryForPathsContext(ctx context.Context, root string, paths []string, git GitRunner) ([]InventoryRow, bool, error) {
	if git == nil {
		git = defaultGit
	}
	rows := make([]InventoryRow, 0, len(paths))
	for _, wt := range paths {
		if ctx != nil && ctx.Err() != nil {
			return rows, true, nil
		}
		wt = filepath.Clean(wt)
		row := InventoryRow{Path: wt, State: "NEEDS_OPERATOR", NeedsOperator: true}
		in, err := LoadIntent(wt)
		if err != nil {
			row.Reason = "missing or invalid prepare metadata"
			rows = append(rows, row)
			continue
		}
		// Git may enumerate the macOS /var symlink as /private/var. Preserve the
		// coordinator-owned prepare path in receipts and ready-to-run argv after
		// LoadIntent has proved both spellings identify the same worktree.
		wt = in.Path
		row.Path = wt
		row.BaseSHA = in.BaseSHA
		if code, out := run(git, wt, []string{"rev-parse", "HEAD"}); code == 0 {
			row.HeadSHA = strings.TrimSpace(out)
		}
		code, out := run(git, wt, []string{"status", "--porcelain=v1", "--untracked-files=all"})
		if code != 0 {
			row.Reason = "cannot inspect worktree status"
			rows = append(rows, row)
			continue
		}
		for _, line := range strings.Split(strings.TrimRight(out, "\r\n"), "\n") {
			line = strings.TrimSuffix(line, "\r")
			if len(line) < 4 {
				continue
			}
			name := strings.TrimSpace(line[3:])
			if i := strings.LastIndex(name, " -> "); i >= 0 {
				name = name[i+4:]
			}
			name = strings.Trim(name, `"`)
			row.DirtyPaths = append(row.DirtyPaths, strings.ReplaceAll(name, "\\", "/"))
		}
		sort.Strings(row.DirtyPaths)
		if len(row.DirtyPaths) == 0 {
			row.State = "CLEAN"
			row.NeedsOperator = false
			row.Reason = ""
			rows = append(rows, row)
			continue
		}
		if in.BaseSHA == "" || in.Message == "" || len(in.Paths) == 0 {
			row.Reason = "prepare metadata lacks base SHA, message, or intended paths"
			rows = append(rows, row)
			continue
		}
		allowed := map[string]bool{}
		for _, p := range in.Paths {
			allowed[p] = true
		}
		ambiguous := false
		for _, p := range row.DirtyPaths {
			if !allowed[p] {
				ambiguous = true
			}
		}
		if ambiguous {
			row.Reason = "dirty paths exceed explicit intended paths"
			rows = append(rows, row)
			continue
		}
		row.State, row.NeedsOperator, row.Reason = "LAND_READY", false, ""
		row.LandArgv = []string{"fak", "worktree", "worker", "land", "--worktree", wt, "--base-sha", in.BaseSHA, "--msg-file", messagePath(wt)}
		for _, p := range in.Paths {
			row.LandArgv = append(row.LandArgv, "--paths", p)
		}
		rows = append(rows, row)
	}
	if ctx != nil && ctx.Err() != nil {
		return rows, true, nil
	}
	return rows, false, nil
}
