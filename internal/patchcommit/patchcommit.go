// Package patchcommit commits one explicitly supplied unified patch through a
// temporary Git index. It is the hunk-scoped sibling of safecommit's path commit.
package patchcommit

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	// ReasonPatchInvalid indicates that the patch is malformed, empty, or uses forbidden diff features.
	ReasonPatchInvalid = "PATCH_INVALID"
	// ReasonPatchPaths indicates that the files modified by the patch differ from the explicit allowlist.
	ReasonPatchPaths = "PATCH_PATH_MISMATCH"
	// ReasonPatchStaged indicates that one of the target paths already has changes staged in the index.
	ReasonPatchStaged = "PATCH_PATH_PRESTAGED"
	// ReasonHeadRace indicates that the branch ref moved or concurrent writer collision occurred.
	ReasonHeadRace = "PATCH_HEAD_RACE"
	// ReasonIndexRace indicates that the shared Git index changed during patch transaction staging.
	ReasonIndexRace = "PATCH_INDEX_RACE"
	// ReasonVerify indicates that pre-commit hooks, commit-msg hooks, or read-back validation failed.
	ReasonVerify = "PATCH_VERIFY_FAILED"
)

// Options configures a patch commit transaction, specifying repository path, patch input, and targets.
type Options struct {
	Dir       string
	PatchFile string
	Paths     []string
	Message   string
	Signoff   bool
	BeforeCAS func() // test seam, called after candidate commit creation
}

// Result reports the outcome of a patch commit transaction, including the commit SHA or refusal reason.
type Result struct {
	SHA    string   `json:"sha,omitempty"`
	Paths  []string `json:"paths,omitempty"`
	Reason string   `json:"reason,omitempty"`
	Detail string   `json:"detail,omitempty"`
}

type gitRunner struct{ dir, index string }

func (g gitRunner) run(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = g.dir
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	if g.index != "" {
		cmd.Env = append(cmd.Env, "GIT_INDEX_FILE="+g.index, "FAK_SAFECOMMIT_VETTED=1")
	}
	var out, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &stderr
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out.String(), nil
}

// Commit applies an explicit unified patch through an isolated temporary index and commits it to HEAD.
func Commit(ctx context.Context, opts Options) (Result, error) {
	var res Result
	if strings.TrimSpace(opts.PatchFile) == "" || strings.TrimSpace(opts.Message) == "" {
		res.Reason, res.Detail = ReasonPatchInvalid, "--patch-file and message are required"
		return res, nil
	}
	root, err := repoRoot(ctx, opts.Dir)
	if err != nil {
		return res, err
	}
	paths, ok := normalizePaths(opts.Paths)
	if !ok || len(paths) == 0 {
		res.Reason, res.Detail = ReasonPatchPaths, "at least one explicit repository-relative --path is required"
		return res, nil
	}
	res.Paths = paths
	patch, err := os.ReadFile(opts.PatchFile)
	if err != nil {
		return res, err
	}
	if len(bytes.TrimSpace(patch)) == 0 {
		res.Reason, res.Detail = ReasonPatchInvalid, "patch is empty"
		return res, nil
	}
	if forbiddenPatchForm(patch) {
		res.Reason, res.Detail = ReasonPatchInvalid, "binary patches, renames, copies, and mode changes are not admitted"
		return res, nil
	}

	// The candidate patch must still describe the exact owned postimage in the
	// live worktree. Reverse-checking the patch catches an overlapping peer edit
	// before any commit object or ref is written while permitting disjoint dirty
	// hunks in the same file.
	if _, err = (gitRunner{dir: root}).run(ctx, "apply", "--reverse", "--check", "--unidiff-zero", "--whitespace=error-all", opts.PatchFile); err != nil {
		res.Reason, res.Detail = ReasonPatchInvalid, err.Error()
		return res, nil
	}

	unlock, err := safecommit.AcquireSharedLock(safecommit.LockOptions{Path: filepath.Join(root, ".git", "fak-commit.lock")})
	if err != nil {
		return res, err
	}
	defer unlock()

	real := gitRunner{dir: root}
	parent, err := oneLine(real.run(ctx, "rev-parse", "HEAD"))
	if err != nil {
		return res, err
	}
	branch, err := oneLine(real.run(ctx, "branch", "--show-current"))
	if err != nil {
		return res, err
	}
	if branch == "" {
		res.Reason, res.Detail = ReasonPatchInvalid, "detached HEAD is not an admitted commit target"
		return res, nil
	}
	if staged, err := real.run(ctx, append([]string{"diff", "--cached", "--name-only", "--"}, paths...)...); err != nil {
		return res, err
	} else if strings.TrimSpace(staged) != "" {
		res.Reason, res.Detail = ReasonPatchStaged, "a requested patch path already has staged content: "+strings.Join(lines(staged), ", ")
		return res, nil
	}
	indexBefore, err := real.run(ctx, "diff", "--cached", "--binary", parent)
	if err != nil {
		return res, err
	}
	indexPath, err := tempIndex(root)
	if err != nil {
		return res, err
	}
	defer os.Remove(indexPath)
	tmp := gitRunner{dir: root, index: indexPath}
	if _, err = tmp.run(ctx, "read-tree", parent); err != nil {
		return res, err
	}
	if _, err = tmp.run(ctx, "apply", "--cached", "--check", "--unidiff-zero", "--whitespace=error-all", opts.PatchFile); err != nil {
		res.Reason, res.Detail = ReasonPatchInvalid, err.Error()
		return res, nil
	}
	if _, err = tmp.run(ctx, "apply", "--cached", "--unidiff-zero", "--whitespace=error-all", opts.PatchFile); err != nil {
		return res, err
	}
	candidatePaths, err := changedPaths(ctx, tmp, parent)
	if err != nil {
		return res, err
	}
	if !sameStrings(candidatePaths, paths) {
		res.Reason, res.Detail = ReasonPatchPaths, fmt.Sprintf("patch changes %v; explicit allowlist is %v", candidatePaths, paths)
		return res, nil
	}
	tree, err := oneLine(tmp.run(ctx, "write-tree"))
	if err != nil {
		return res, err
	}
	msgPath, err := writeMessage(root, opts.Message, opts.Signoff)
	if err != nil {
		return res, err
	}
	defer os.Remove(msgPath)
	if _, err = tmp.run(ctx, "hook", "run", "--ignore-missing", "pre-commit"); err != nil {
		res.Reason, res.Detail = ReasonVerify, err.Error()
		return res, nil
	}
	if _, err = tmp.run(ctx, "hook", "run", "--ignore-missing", "commit-msg", "--", msgPath); err != nil {
		res.Reason, res.Detail = ReasonVerify, err.Error()
		return res, nil
	}
	sha, err := oneLine(tmp.run(ctx, "commit-tree", tree, "-p", parent, "-F", msgPath))
	if err != nil {
		return res, err
	}
	if opts.BeforeCAS != nil {
		opts.BeforeCAS()
	}
	current, err := oneLine(real.run(ctx, "rev-parse", "HEAD"))
	if err != nil {
		return res, err
	}
	if current != parent {
		res.Reason, res.Detail = ReasonHeadRace, fmt.Sprintf("HEAD moved from %s to %s before ref update", parent, current)
		return res, nil
	}
	indexNow, err := real.run(ctx, "diff", "--cached", "--binary", parent)
	if err != nil {
		return res, err
	}
	if indexNow != indexBefore {
		res.Reason, res.Detail = ReasonIndexRace, "shared index changed while the patch transaction was in flight"
		return res, nil
	}
	// update-ref's old-value argument is the authoritative CAS; a raw Git writer
	// cannot slip between the check above and this effect.
	if _, err = real.run(ctx, "update-ref", "HEAD", sha, parent); err != nil {
		res.Reason, res.Detail = ReasonHeadRace, err.Error()
		return res, nil
	}
	// Bring only committed patch paths in the real index to the new HEAD. Other
	// staged entries remain byte-for-byte entries in the shared index.
	if _, err = real.run(ctx, append([]string{"reset", "-q", "HEAD", "--"}, paths...)...); err != nil {
		return res, err
	}
	gotParent, err := oneLine(real.run(ctx, "rev-parse", sha+"^"))
	if err != nil {
		return res, err
	}
	gotTree, err := oneLine(real.run(ctx, "rev-parse", sha+"^{tree}"))
	if err != nil {
		return res, err
	}
	gotPaths, err := changedPaths(ctx, real, parent+".."+sha)
	if err != nil {
		return res, err
	}
	if gotParent != parent || gotTree != tree || !sameStrings(gotPaths, paths) {
		res.Reason, res.Detail = ReasonVerify, "committed parent/tree/path read-back differs from the candidate"
		return res, errors.New(res.Detail)
	}
	_, _ = tmp.run(ctx, "hook", "run", "--ignore-missing", "post-commit")
	res.SHA = sha
	return res, nil
}

func repoRoot(ctx context.Context, dir string) (string, error) {
	if dir == "" {
		dir = "."
	}
	return oneLine(gitRunner{dir: dir}.run(ctx, "rev-parse", "--show-toplevel"))
}
func tempIndex(root string) (string, error) {
	f, err := os.CreateTemp(filepath.Join(root, ".git"), "fak-patch-index-")
	if err != nil {
		return "", err
	}
	p := f.Name()
	if err := f.Close(); err != nil {
		return "", err
	}
	if err := os.Remove(p); err != nil {
		return "", err
	}
	return p, nil
}
func writeMessage(root, msg string, signoff bool) (string, error) {
	if signoff && !strings.Contains(msg, "Signed-off-by:") {
		name, _ := oneLine(gitRunner{dir: root}.run(context.Background(), "config", "user.name"))
		email, _ := oneLine(gitRunner{dir: root}.run(context.Background(), "config", "user.email"))
		msg = strings.TrimRight(msg, "\r\n") + "\n\nSigned-off-by: " + name + " <" + email + ">\n"
	}
	f, err := os.CreateTemp(filepath.Join(root, ".git"), "fak-patch-msg-")
	if err != nil {
		return "", err
	}
	if _, err = f.WriteString(msg); err != nil {
		f.Close()
		return "", err
	}
	if err = f.Close(); err != nil {
		return "", err
	}
	return f.Name(), nil
}
func changedPaths(ctx context.Context, g gitRunner, rev string) ([]string, error) {
	args := []string{"diff", "--cached", "--name-only", rev}
	if strings.Contains(rev, "..") {
		args = []string{"diff", "--name-only", rev}
	}
	out, err := g.run(ctx, args...)
	if err != nil {
		return nil, err
	}
	p := lines(out)
	sort.Strings(p)
	return p, nil
}
func normalizePaths(in []string) ([]string, bool) {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, p := range in {
		p = filepath.ToSlash(filepath.Clean(strings.TrimSpace(p)))
		if p == "." || p == "" || filepath.IsAbs(p) || p == ".." || strings.HasPrefix(p, "../") {
			return nil, false
		}
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out, true
}
func oneLine(s string, err error) (string, error) { return strings.TrimSpace(s), err }
func lines(s string) []string {
	var out []string
	for _, x := range strings.Split(strings.ReplaceAll(s, "\r\n", "\n"), "\n") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, filepath.ToSlash(x))
		}
	}
	return out
}
func sameStrings(a, b []string) bool {
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

func forbiddenPatchForm(patch []byte) bool {
	text := "\n" + strings.ReplaceAll(string(patch), "\r\n", "\n")
	for _, marker := range []string{
		"\nGIT binary patch\n", "\nBinary files ", "\nrename from ", "\nrename to ",
		"\ncopy from ", "\ncopy to ", "\nold mode ", "\nnew mode ",
	} {
		if strings.Contains(text, marker) {
			return true
		}
	}
	return false
}
