package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"context"
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

	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/conceptcatalog/scorecard"
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

// RelevantPath reports whether a change can affect the disambiguation snapshot. It must
// cover every input the generator reads, or a commit touching only an omitted input
// stales the committed artifacts while the gate stays silent: the generator's
// identifier map records which production Go file each name appears in, and it grounds
// rows in doc headings, so Go sources and docs are inputs, not just the script.
func RelevantPath(path string) bool {
	p := filepath.ToSlash(path)
	switch p {
	case "tools/concept_disambiguation_scorecard.py", "docs/fak/concept-glossary.md",
		GeneratedReadme, GeneratedIndex:
		return true
	}
	if strings.HasPrefix(p, "tools/concept_disambiguation_scorecard.data/") {
		return true
	}
	if strings.HasPrefix(p, "docs/") && strings.HasSuffix(p, ".md") {
		return true
	}
	if rootDocInputs[p] {
		return true
	}
	return (strings.HasPrefix(p, "internal/") || strings.HasPrefix(p, "cmd/")) &&
		strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go")
}

// rootDocInputs mirrors the root documents the generator reads for structural concepts.
var rootDocInputs = map[string]bool{
	"CLAIMS.md": true, "ARCHITECTURE.md": true, "README.md": true,
	"AGENTS.md": true, "INDEX.md": true, "GLOSSARY.md": true,
}

// CheckFresh regenerates in scratch space and compares every tracked generated artifact.
func CheckFresh(root string) (FreshnessResult, error) {
	return checkFreshProfile(root, nil)
}

func checkFreshProfile(root string, profile *conceptTreeProfile) (FreshnessResult, error) {
	out, err := os.MkdirTemp("", "fak-concept-fresh-*")
	if err != nil {
		return FreshnessResult{}, err
	}
	defer os.RemoveAll(out)
	generated := filepath.Join(out, "generated")
	if err := generateProfile(root, generated, profile); err != nil {
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
	res, err := checkFreshProfile(tree.root, tree.profile)
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
func materializeGitTree(root, treeish string) (conceptGitTree, func(), error) {
	treeish, err := resolveTreeish(root, treeish)
	if err != nil {
		return conceptGitTree{}, func() {}, err
	}
	pinned, err := git(root, "rev-parse", "--verify", treeish)
	if err != nil {
		return conceptGitTree{}, func() {}, err
	}
	treeish = strings.TrimSpace(string(pinned))
	qualifiedTree, err := git(root, "rev-parse", "--verify", treeish+"^{tree}")
	if err != nil {
		return conceptGitTree{}, func() {}, err
	}
	profile, err := qualifyConceptTreeProfile(root, strings.TrimSpace(string(qualifiedTree)))
	if err != nil {
		return conceptGitTree{}, func() {}, err
	}
	tmp, err := os.MkdirTemp("", "fak-concept-tree-*")
	if err != nil {
		return conceptGitTree{}, func() {}, err
	}
	cleanup := func() { os.RemoveAll(tmp) }
	extract := extractGitTar
	if profile != nil {
		extract = extractConceptProfileTar
	}
	if err = extractCommittedTreeUsing(root, treeish, tmp, extract); err != nil {
		cleanup()
		return conceptGitTree{}, func() {}, err
	}
	if err = expandCommitSubstitution(root, treeish, tmp); err != nil {
		cleanup()
		return conceptGitTree{}, func() {}, err
	}
	if profile != nil {
		h, e := conceptFileDigest(filepath.Join(tmp, "tools/concept_disambiguation_scorecard.py"))
		if e != nil || h != conceptScriptSHA {
			cleanup()
			return conceptGitTree{}, func() {}, fmt.Errorf("qualified archived generator identity")
		}
	}
	return conceptGitTree{root: tmp, profile: profile}, cleanup, nil
}

// extractCommittedTreeUsing streams the committed tree of treeish into tmp. The tree is
// read through committedtree.WriteTar rather than git archive: export-ignore rows would
// strip paths such as docs/ and tools/ from the snapshot, and the generator and its
// freshness comparison read exactly those paths.
func extractCommittedTreeUsing(root, treeish, tmp string, extract func(io.Reader, string) error) error {
	return extractTarStreamUsing(treeish, tmp, extract, func(w io.Writer) error {
		return committedtree.WriteTar(context.Background(), root, treeish, w)
	})
}

const (
	exportSubstOpen  = "$Format:"
	exportSubstClose = '$'
)

// commitIdentity is what git's export-subst placeholders can name for one commit.
type commitIdentity struct {
	hash, abbrevHash       string
	tree, abbrevTree       string
	parents, abbrevParents string
}

// expandCommitSubstitution applies git archive's export-subst expansion to the regular
// files of treeish that carry the attribute, so the snapshot holds the bytes git archive
// wrote. A tree object has no commit identity and git archive never expanded one, so it
// is left as extracted.
func expandCommitSubstitution(root, treeish, tmp string) error {
	commit, err := peelToCommit(root, treeish)
	if err != nil || commit == "" {
		return err
	}
	paths, err := exportSubstPaths(root, commit)
	if err != nil || len(paths) == 0 {
		return err
	}
	id, err := identityOf(root, commit)
	if err != nil {
		return err
	}
	for _, p := range paths {
		target := filepath.Join(tmp, filepath.FromSlash(p))
		data, err := os.ReadFile(target)
		if err != nil {
			return fmt.Errorf("export-subst %s: %w", p, err)
		}
		out, err := expandExportSubst(data, id)
		if err != nil {
			return fmt.Errorf("export-subst %s: %w", p, err)
		}
		if !bytes.Equal(out, data) {
			if err := os.WriteFile(target, out, 0o644); err != nil {
				return err
			}
		}
	}
	return nil
}

// peelToCommit returns the commit treeish names, or "" when it names a tree.
func peelToCommit(root, treeish string) (string, error) {
	out, err := gitBytes(root, nil, "rev-parse", "--verify", "--quiet", treeish+"^{commit}")
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// exportSubstPaths lists the regular files of commit whose export-subst attribute is set,
// as read from the commit's own .gitattributes rather than the shared worktree.
func exportSubstPaths(root, commit string) ([]string, error) {
	listing, err := gitBytes(root, nil, "ls-tree", "-r", "-z", commit)
	if err != nil {
		return nil, err
	}
	var regular [][]byte
	for _, rec := range bytes.Split(listing, []byte{0}) {
		meta, path, ok := bytes.Cut(rec, []byte{'\t'})
		if !ok {
			continue
		}
		fields := strings.Fields(string(meta))
		if len(fields) > 0 && (fields[0] == "100644" || fields[0] == "100755") {
			regular = append(regular, path)
		}
	}
	if len(regular) == 0 {
		return nil, nil
	}
	query := append(bytes.Join(regular, []byte{0}), 0)
	attrs, err := gitBytes(root, query, "check-attr", "-z", "--source="+commit, "export-subst", "--stdin")
	if err != nil {
		return nil, err
	}
	fields := bytes.Split(attrs, []byte{0})
	var paths []string
	for i := 0; i+2 < len(fields); i += 3 {
		if string(fields[i+1]) == "export-subst" && string(fields[i+2]) == "set" {
			paths = append(paths, string(fields[i]))
		}
	}
	return paths, nil
}

// identityOf reads the commit fields export-subst can name. One log call supplies them
// all, and its abbreviations match the ones git archive's placeholders produce.
func identityOf(root, commit string) (commitIdentity, error) {
	out, err := gitBytes(root, nil, "log", "-1", "--no-show-signature",
		"--format=%H%x00%h%x00%T%x00%t%x00%P%x00%p", commit)
	if err != nil {
		return commitIdentity{}, err
	}
	f := strings.Split(strings.TrimRight(string(out), "\n"), "\x00")
	if len(f) != 6 {
		return commitIdentity{}, fmt.Errorf("commit identity for %s: unexpected fields %q", commit, f)
	}
	return commitIdentity{hash: f[0], abbrevHash: f[1], tree: f[2], abbrevTree: f[3], parents: f[4], abbrevParents: f[5]}, nil
}

// expandExportSubst replaces each $Format:...$ span with its expansion. An unterminated
// span stays verbatim, as git leaves it. A placeholder the expansion does not support
// is an error, never passed through.
func expandExportSubst(data []byte, id commitIdentity) ([]byte, error) {
	var out []byte
	rest := data
	expanded := false
	for {
		i := bytes.Index(rest, []byte(exportSubstOpen))
		if i < 0 {
			break
		}
		body := rest[i+len(exportSubstOpen):]
		j := bytes.IndexByte(body, exportSubstClose)
		if j < 0 {
			break
		}
		expansion, err := formatCommitPlaceholder(body[:j], id)
		if err != nil {
			return nil, err
		}
		out = append(out, rest[:i]...)
		out = append(out, expansion...)
		rest = body[j+1:]
		expanded = true
	}
	if !expanded {
		return data, nil
	}
	return append(out, rest...), nil
}

// formatCommitPlaceholder expands one placeholder body in git's pretty-format syntax,
// restricted to the commit-identity codes.
func formatCommitPlaceholder(format []byte, id commitIdentity) ([]byte, error) {
	var out []byte
	for k := 0; k < len(format); k++ {
		if format[k] != '%' {
			out = append(out, format[k])
			continue
		}
		k++
		if k == len(format) {
			return nil, fmt.Errorf("placeholder %q ends in %%", format)
		}
		switch format[k] {
		case '%':
			out = append(out, '%')
		case 'n':
			out = append(out, '\n')
		case 'H':
			out = append(out, id.hash...)
		case 'h':
			out = append(out, id.abbrevHash...)
		case 'T':
			out = append(out, id.tree...)
		case 't':
			out = append(out, id.abbrevTree...)
		case 'P':
			out = append(out, id.parents...)
		case 'p':
			out = append(out, id.abbrevParents...)
		default:
			return nil, fmt.Errorf("unsupported placeholder %%%s in %q", string(format[k]), format)
		}
	}
	return out, nil
}

// gitBytes runs git in root and returns stdout alone, so NUL-delimited output is not
// mixed with stderr. stdin is fed when non-nil.
func gitBytes(root string, stdin []byte, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	if stdin != nil {
		cmd.Stdin = bytes.NewReader(stdin)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

// extractTarStreamUsing runs produce concurrently and feeds its tar stream to extract.
// A consumer that fails closes the pipe so a producer blocked on a write returns; a
// consumer that finishes early drains the remaining padding so the producer can
// complete. Consumer errors are reported first.
func extractTarStreamUsing(treeish, tmp string, extract func(io.Reader, string) error, produce func(io.Writer) error) error {
	pr, pw := io.Pipe()
	produced := make(chan error, 1)
	go func() {
		err := produce(pw)
		_ = pw.CloseWithError(err)
		produced <- err
	}()
	extractErr := extract(pr, tmp)
	var drainErr error
	if extractErr == nil {
		_, drainErr = io.Copy(io.Discard, pr)
	}
	_ = pr.Close()
	var produceErr error
	if err := <-produced; err != nil {
		produceErr = fmt.Errorf("committed tree %s: %w", treeish, err)
	}
	return errors.Join(extractErr, drainErr, produceErr)
}

func extractGitTar(archive io.Reader, tmp string) error {
	tr := tar.NewReader(archive)
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
	// Render into an empty directory: an ACTION exit is usable only when this
	// invocation produced the artifacts, never because old worktree files exist.
	out, err := os.MkdirTemp("", "fak-concept-generate-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(out)
	if err := generate(root, out); err != nil {
		return nil, err
	}
	// Read every artifact before publishing any, including on exit zero. A
	// partial render must leave the previous worktree snapshot untouched.
	rendered := make([][]byte, len(generatedArtifacts))
	for i, artifact := range generatedArtifacts {
		rendered[i], err = os.ReadFile(filepath.Join(out, artifact.Name))
		if err != nil {
			return nil, fmt.Errorf("read generated %s: %w", artifact.Name, err)
		}
	}
	files := make([]string, 0, len(generatedArtifacts))
	for i, artifact := range generatedArtifacts {
		target := filepath.Join(root, filepath.FromSlash(artifact.Tracked))
		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return files, err
		}
		if err := os.WriteFile(target, rendered[i], 0644); err != nil {
			return files, err
		}
		files = append(files, target)
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
	return generateProfile(root, out, nil)
}

// generateProfile renders the doc folder in-process. A profile only selects the
// sparse export, which keeps every byte the scorecard reads, so it does not
// change what is generated.
func generateProfile(root, out string, _ *conceptTreeProfile) error {
	if _, err := scorecard.Generate(context.Background(), root, "", out); err != nil {
		return fmt.Errorf("generate scorecard: %w", err)
	}
	return nil
}

// runInvariantSnapshot is the single scorecard snapshot CheckInvariant consumes. It
// writes the freshness artifacts and returns the verdict from the same fold, so both
// answers describe exactly the same workspace walk.
var runInvariantSnapshot = executeInvariantSnapshot

func executeInvariantSnapshot(root, generated string) (scorecard.Summary, error) {
	p, err := scorecard.Generate(context.Background(), root, "", generated)
	if err != nil {
		return scorecard.Summary{}, fmt.Errorf("generate invariant snapshot: %w", err)
	}
	return p.Summary(), nil
}

// CheckInvariant validates freshness, semantic catalog structure and scorecard critical state.
func CheckInvariant(root string) (InvariantResult, error) {
	tmp, err := os.MkdirTemp("", "fak-concept-invariant-*")
	if err != nil {
		return InvariantResult{}, err
	}
	defer os.RemoveAll(tmp)
	generated := filepath.Join(tmp, "generated")
	payload, err := runInvariantSnapshot(root, generated)
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
	inv.ClarityDebt = payload.Corpus.ClarityDefects
	inv.CriticalClean = inv.SemanticValid && inv.ClarityDebt == 0
	inv.Coverage = float64(payload.Corpus.Coverage.CoveragePct)
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
