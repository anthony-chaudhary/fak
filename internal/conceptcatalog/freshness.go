package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	GeneratedReadme = "docs/concept-disambiguation-scorecard/README.md"
	// GeneratedIndex is the reverse lookup: every NAME a reader can meet in the tree
	// mapped back to the concept it denotes, plus the twins that name is mistakable
	// for. It is generated from the same catalog as the scorecard and must age with
	// it - a fresh scorecard beside a stale index would answer "what is this concept"
	// correctly while answering "which concept is this name" from a retired catalog.
	GeneratedIndex = "docs/concept-disambiguation-scorecard/INDEX.md"
	// RegenerateCommand is the WORKTREE-mode cure, and it answers CheckFresh only.
	// The generator derives its numbers by walking the whole workspace, so run from
	// the repo root it scores every peer's unsaved edit too - a different tree from
	// the one CheckGitTree scores, and therefore a different answer.
	RegenerateCommand = "fak concept generate"
	// RegenerateStagedCommand is the TREE-mode cure, and it is the only one that can
	// clear a CheckGitTree refusal: it regenerates inside the same clean-room export
	// of the staged tree that CheckGitTree scored, then writes the artifacts back to
	// the worktree for the operator to stage (#5829).
	//
	// Bare, it resolves the CALLER's index. That is not good enough for a refusal to
	// quote: a pre-commit hook runs against git's temporary partial-commit index (HEAD
	// plus the pathspec), while the operator who reads the refusal runs the cure in a
	// plain shell, where the same words resolve the SHARED .git/index - HEAD plus
	// whatever every peer left staged. Measured on this trunk, those two trees give
	// different artifacts and the retry refuses again. Refusals must therefore print
	// RegenerateStagedCommandFor, never this constant alone.
	RegenerateStagedCommand = "fak concept generate --staged"
)

// RegenerateStagedCommandFor pins the tree-mode cure to the exact tree that was scored,
// so the printed command means the same thing in the operator's shell as it did inside
// the hook (#5829).
//
// Safe to hand out because git write-tree does not merely hash the index, it WRITES the
// tree object into the repository - so the SHA still resolves after the hook's temporary
// index is gone. An empty treeish degrades to the bare command rather than emitting a
// dangling `--tree`, which would be worse than imprecise.
func RegenerateStagedCommandFor(treeish string) string {
	if treeish == "" {
		return RegenerateStagedCommand
	}
	return RegenerateStagedCommand + " --tree " + treeish
}

// generatedArtifacts pairs each tracked artifact with its filename under the
// generator's --markdown-dir output.
var generatedArtifacts = []struct{ Tracked, Name string }{
	{GeneratedReadme, "README.md"},
	{GeneratedIndex, "INDEX.md"},
}

// FreshnessResult describes deterministic generated-artifact freshness.
type FreshnessResult struct {
	Fresh      bool     `json:"fresh"`
	StalePaths []string `json:"stale_paths,omitempty"`
	Regenerate string   `json:"regenerate"`
}
type InvariantResult struct {
	Freshness      FreshnessResult    `json:"freshness"`
	SemanticValid  bool               `json:"semantic_valid"`
	CriticalClean  bool               `json:"critical_clean"`
	ClarityDebt    int                `json:"clarity_debt"`
	Coverage       float64            `json:"coverage"`
	CoverageDebt   int                `json:"coverage_debt"`
	FamilyCoverage map[string]float64 `json:"family_coverage,omitempty"`
	Detail         string             `json:"detail,omitempty"`
}

// RelevantPath reports whether a change can affect the disambiguation snapshot.
func RelevantPath(path string) bool {
	p := filepath.ToSlash(path)
	return p == "tools/concept_disambiguation_scorecard.py" ||
		p == "docs/fak/concept-glossary.md" ||
		p == GeneratedReadme ||
		p == GeneratedIndex ||
		strings.HasPrefix(p, "tools/concept_disambiguation_scorecard.data/")
}

// CheckFresh regenerates in scratch space and compares every tracked generated artifact.
func CheckFresh(root string) (FreshnessResult, error) {
	out, err := os.MkdirTemp("", "fak-concept-fresh-*")
	if err != nil {
		return FreshnessResult{}, err
	}
	defer os.RemoveAll(out)
	generated := filepath.Join(out, "generated")
	if err := generate(root, generated); err != nil {
		return FreshnessResult{}, err
	}
	return compareGeneratedFreshness(root, generated)
}

func compareGeneratedFreshness(root, generated string) (FreshnessResult, error) {
	result := FreshnessResult{Fresh: true, Regenerate: RegenerateCommand}
	for _, art := range generatedArtifacts {
		expected, err := os.ReadFile(filepath.Join(generated, art.Name))
		if err != nil {
			return result, fmt.Errorf("read generated %s: %w", art.Name, err)
		}
		actual, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(art.Tracked)))
		if err != nil || !generatedBytesEqual(actual, expected) {
			result.Fresh = false
			result.StalePaths = append(result.StalePaths, art.Tracked)
		}
	}
	sort.Strings(result.StalePaths)
	return result, nil
}

// generatedBytesEqual compares the generated Markdown contract after normalizing
// checkout line endings. Git may materialize a tracked text file as CRLF on Windows,
// while the generator subprocess writes LF (or vice versa); that transport detail must
// not make a semantically identical committed artifact stale. No other bytes are
// normalized, so content drift remains a hard freshness failure.
func generatedBytesEqual(actual, expected []byte) bool {
	return bytes.Equal(normalizeGeneratedNewlines(actual), normalizeGeneratedNewlines(expected))
}

func normalizeGeneratedNewlines(b []byte) []byte {
	return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n"))
}

// CheckGitTree checks a committed or staged git tree, immune to peer working-tree files.
// An empty treeish means the current index (git write-tree).
//
// The result's Regenerate names the tree-scoped cure, never RegenerateCommand: a refusal
// produced by scoring a git tree can only be cleared by regenerating from that same tree,
// and the worktree command answers a tree this check never looked at (#5829).
//
// The treeish is resolved HERE rather than inside materializeGitTree so the resolved SHA
// can be baked into that cure. Naming the tree is what makes the printed command portable
// out of the hook's environment: "the current index" denotes git's temporary partial-commit
// index to the hook and the shared .git/index to the operator reading the refusal, and on a
// multi-session trunk those are different trees with different artifacts.
func CheckGitTree(root, treeish string) (FreshnessResult, error) {
	resolved, err := resolveTreeish(root, treeish)
	if err != nil {
		return FreshnessResult{}, err
	}
	tree, cleanup, err := materializeGitTree(root, resolved)
	if err != nil {
		return FreshnessResult{}, err
	}
	defer cleanup()
	res, err := CheckFresh(tree)
	res.Regenerate = RegenerateStagedCommandFor(resolved)
	return res, err
}

// resolveTreeish turns the empty treeish into a concrete tree SHA by writing the current
// index out as a tree. git write-tree persists that tree object, so the SHA outlives the
// index it came from - including a hook's temporary partial-commit index.
func resolveTreeish(root, treeish string) (string, error) {
	if treeish != "" {
		return treeish, nil
	}
	b, err := git(root, "write-tree")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

// RegenerateFromGitTree runs the generator against the same clean-room export of
// treeish that CheckGitTree scores, and writes every tracked artifact under dest
// (the repo root when empty). It returns the tracked paths written, for the caller
// to name in its own pathspec - staging is the operator's call on a shared trunk.
//
// This is the cure CheckGitTree's refusal prints. The extraction is shared with the
// check by construction, so the bytes written here are the bytes the check compared
// against; regenerating from the worktree instead answers a different tree and leaves
// the refusal standing.
func RegenerateFromGitTree(root, treeish, dest string) ([]string, error) {
	if dest == "" {
		dest = root
	}
	rendered, err := renderFromGitTree(root, treeish, nil)
	if err != nil {
		return nil, err
	}
	written := make([]string, 0, len(generatedArtifacts))
	for _, art := range generatedArtifacts {
		// Copied byte for byte: the generator emits LF and git staging must leave the
		// blob unchanged (#5136). Any rewriting here re-breaks that round trip.
		b := rendered[art.Tracked]
		target := filepath.Join(dest, filepath.FromSlash(art.Tracked))
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return written, err
		}
		if err := os.WriteFile(target, b, 0644); err != nil {
			return written, err
		}
		written = append(written, art.Tracked)
	}
	return written, nil
}

// materializeGitTree extracts treeish into a fresh temp directory and returns it with
// its cleanup. An empty treeish means the current index (git write-tree) - under a
// pathspec commit that index is git's temporary partial-commit index, so the tree is
// HEAD plus exactly the committer's pathspec.
//
// Deliberately NOT merged with CheckFresh: callers hand CheckFresh the extraction root,
// never the repo, and collapsing the two roots is how the peer-dirty bug this export
// exists to prevent gets reintroduced.
func materializeGitTree(root, treeish string) (string, func(), error) {
	treeish, err := resolveTreeish(root, treeish)
	if err != nil {
		return "", func() {}, err
	}
	tmp, err := os.MkdirTemp("", "fak-concept-tree-*")
	if err != nil {
		return "", func() {}, err
	}
	cleanup := func() { os.RemoveAll(tmp) }
	if err := extractGitTree(root, treeish, tmp); err != nil {
		cleanup()
		return "", func() {}, err
	}
	return tmp, cleanup, nil
}

func extractGitTree(root, treeish, tmp string) error {
	cmd := exec.Command("git", "archive", "--format=tar", treeish)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	raw, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("git archive %s: %w", treeish, err)
	}
	tr := tar.NewReader(bytes.NewReader(raw))
	for {
		h, e := tr.Next()
		if errors.Is(e, io.EOF) {
			break
		}
		if e != nil {
			return e
		}
		clean := filepath.Clean(filepath.FromSlash(h.Name))
		if filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
			return fmt.Errorf("unsafe archive path %q", h.Name)
		}
		dst := filepath.Join(tmp, clean)
		if h.FileInfo().IsDir() {
			if e := os.MkdirAll(dst, 0755); e != nil {
				return e
			}
			continue
		}
		if h.Typeflag != tar.TypeReg && h.Typeflag != tar.TypeRegA {
			continue
		}
		if e := os.MkdirAll(filepath.Dir(dst), 0755); e != nil {
			return e
		}
		f, e := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, h.FileInfo().Mode())
		if e != nil {
			return e
		}
		_, copyErr := io.Copy(f, tr)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

// Regenerate runs the canonical concept writer and returns every artifact it
// refreshed. Classifications live in the input catalog, so generation reads and
// preserves the same source that concept classify updates.
func Regenerate(root string) ([]string, error) {
	out := filepath.Join(root, filepath.FromSlash("docs/concept-disambiguation-scorecard"))
	if err := generate(root, out); err != nil {
		return nil, err
	}
	files := make([]string, 0, len(generatedArtifacts))
	for _, artifact := range generatedArtifacts {
		files = append(files, filepath.Join(out, artifact.Name))
	}
	return files, nil
}

// ResolvePython returns the path to an available Python interpreter ("python" or "python3").
// It tries the platform default first (python on Windows, python3 elsewhere), then the fallback.
// If neither is found, it returns an error.
func ResolvePython() (string, error) {
	primary, fallback := "python3", "python"
	if runtime.GOOS == "windows" {
		primary, fallback = "python", "python3"
	}
	if p, err := exec.LookPath(primary); err == nil {
		return p, nil
	}
	if p, err := exec.LookPath(fallback); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("Python interpreter not found (tried %q and %q)", primary, fallback)
}

func generate(root, out string) error {
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	python, err := ResolvePython()
	if err != nil {
		return fmt.Errorf("generate scorecard: %w", err)
	}
	cmd := exec.Command(python, script, "--workspace", root, "--markdown-dir", out)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stdout = io.Discard
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
			return fmt.Errorf("generate scorecard: %w: %s", err, strings.TrimSpace(stderr.String()))
		}
		for _, art := range generatedArtifacts {
			if _, statErr := os.Stat(filepath.Join(out, art.Name)); statErr != nil {
				return fmt.Errorf("generate scorecard: %w: %s", err, strings.TrimSpace(stderr.String()))
			}
		}
	}
	return nil
}

// runInvariantSnapshot is the single scorecard snapshot CheckInvariant consumes. It
// emits the freshness artifacts and JSON verdict together, so both answers describe
// exactly the same workspace walk.
var runInvariantSnapshot = executeInvariantSnapshot

// invariantSnapshotCommand is a test seam for process exit and transport behavior.
// Production always uses the canonical Python scorecard command below.
var invariantSnapshotCommand = newInvariantSnapshotCommand

func newInvariantSnapshotCommand(root, generated string) *exec.Cmd {
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	python, err := ResolvePython()
	if err != nil {
		python = "python3"
		if runtime.GOOS == "windows" {
			python = "python"
		}
	}
	return exec.Command(python, script, "--workspace", root, "--markdown-dir", generated, "--json")
}

func executeInvariantSnapshot(root, generated string) ([]byte, error) {
	cmd := invariantSnapshotCommand(root, generated)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			return nil, err
		}
		// ACTION is a valid scorecard verdict. A non-zero exit is therefore a
		// usable snapshot when stdout and every generated artifact are present.
		if len(bytes.TrimSpace(out)) == 0 {
			return nil, fmt.Errorf("generate invariant snapshot: %w: %s", err, strings.TrimSpace(stderr.String()))
		}
	}
	for _, art := range generatedArtifacts {
		if _, statErr := os.Stat(filepath.Join(generated, art.Name)); statErr != nil {
			return nil, fmt.Errorf("generate invariant snapshot: missing %s: %w", art.Name, statErr)
		}
	}
	return out, nil
}

// CheckInvariant validates freshness, semantic catalog structure and scorecard critical state.
func CheckInvariant(root string) (InvariantResult, error) {
	tmp, err := os.MkdirTemp("", "fak-concept-invariant-*")
	if err != nil {
		return InvariantResult{}, err
	}
	defer os.RemoveAll(tmp)
	generated := filepath.Join(tmp, "generated")
	out, err := runInvariantSnapshot(root, generated)
	if err != nil {
		return InvariantResult{}, err
	}
	fresh, err := compareGeneratedFreshness(root, generated)
	if err != nil {
		return InvariantResult{}, err
	}
	inv := InvariantResult{Freshness: fresh, SemanticValid: true, CriticalClean: true, FamilyCoverage: map[string]float64{}}
	cat, err := Load(root)
	if err != nil {
		inv.SemanticValid = false
		inv.Detail = err.Error()
	} else if ds := Validate(cat); len(ds) > 0 {
		inv.SemanticValid = false
		b, _ := json.Marshal(ds)
		inv.Detail = string(b)
	}
	var payload struct {
		OK     bool   `json:"ok"`
		Reason string `json:"reason"`
		Corpus struct {
			CoverageDebt int `json:"coverage_debt"`
			ClarityDebt  int `json:"clarity_defects"`
			Coverage     struct {
				CoveragePct float64 `json:"coverage_pct"`
				PerFamily   []struct {
					Family     string `json:"family"`
					Discovered int    `json:"discovered"`
					Covered    int    `json:"covered"`
				} `json:"per_family"`
			} `json:"coverage"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal(out, &payload); err != nil {
		return inv, fmt.Errorf("decode scorecard: %w", err)
	}
	inv.ClarityDebt = payload.Corpus.ClarityDebt
	inv.CriticalClean = inv.SemanticValid && inv.ClarityDebt == 0
	inv.Coverage = payload.Corpus.Coverage.CoveragePct
	inv.CoverageDebt = payload.Corpus.CoverageDebt
	for _, f := range payload.Corpus.Coverage.PerFamily {
		if f.Discovered > 0 {
			inv.FamilyCoverage[f.Family] = 100 * float64(f.Covered) / float64(f.Discovered)
		}
	}
	if !payload.OK && inv.Detail == "" {
		inv.Detail = payload.Reason
	}
	return inv, nil
}

func git(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(b)))
	}
	return b, nil
}

func (r FreshnessResult) JSON() []byte { b, _ := json.Marshal(r); return b }
