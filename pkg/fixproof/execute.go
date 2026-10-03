package fixproof

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

type checkoutSnapshot struct {
	head string
	tree string
}

type changedPath struct {
	status string
	path   string
}

// Execute validates immutable Git inputs, delegates the behavioral verdict to
// the one canonical witness resolver, then rejects any checkout movement or
// dirt observed after the proof. ExecutionContext is identity asserted by the
// retained caller; Execute binds but does not independently measure it.
func Execute(ctx context.Context, roots Roots, request Request) (Result, error) {
	if err := request.Validate(); err != nil {
		return Result{}, err
	}
	if len(request.TestOverlays) == 0 {
		return Result{}, fmt.Errorf("%w: canonical resolver requires a changed Go test overlay", ErrUnsupportedProofShape)
	}
	repo, companion, err := validateRoots(ctx, roots)
	if err != nil {
		return Result{}, err
	}

	beforeRepo, err := snapshotCheckout(ctx, repo)
	if err != nil {
		return Result{}, err
	}
	beforeCompanion, err := snapshotCheckout(ctx, companion)
	if err != nil {
		return Result{}, err
	}
	if beforeRepo.head != request.CandidateCommit || beforeRepo.tree != request.CandidateTree {
		return Result{}, fmt.Errorf("%w: repository checkout does not match candidate", ErrIdentityMismatch)
	}
	if beforeCompanion.head != request.CompanionCommit || beforeCompanion.tree != request.CompanionTree {
		return Result{}, fmt.Errorf("%w: companion checkout does not match request", ErrIdentityMismatch)
	}
	if err := validateCandidateIdentity(ctx, repo, request); err != nil {
		return Result{}, err
	}

	proofCtx, cancel := context.WithTimeout(ctx, time.Duration(request.TimeoutMillis)*time.Millisecond)
	resolver := witness.NewWithRunner(nil, repo).
		WithSymptomTags(request.BuildTags).
		WithSymptomTests(request.Selectors)
	outcome, detail := resolver.ResolveSymptomWithDetail(proofCtx, request.CandidateCommit, true)
	cancel()

	// The proof context may have expired. Give each sequential read-only
	// snapshot its own short context so a slow repository check cannot consume
	// the companion's entire budget and suppress its movement and dirt check.
	repoPostCtx, repoPostCancel := context.WithTimeout(context.Background(), 15*time.Second)
	afterRepo, repoErr := snapshotCheckout(repoPostCtx, repo)
	repoPostCancel()
	companionPostCtx, companionPostCancel := context.WithTimeout(context.Background(), 15*time.Second)
	afterCompanion, companionErr := snapshotCheckout(companionPostCtx, companion)
	companionPostCancel()
	var checkoutChanges []error
	if repoErr != nil {
		checkoutChanges = append(checkoutChanges, fmt.Errorf("repository post-check failed: %w", repoErr))
	} else if afterRepo != beforeRepo {
		checkoutChanges = append(checkoutChanges, errors.New("repository identity moved during proof"))
	}
	if companionErr != nil {
		checkoutChanges = append(checkoutChanges, fmt.Errorf("companion post-check failed: %w", companionErr))
	} else if afterCompanion != beforeCompanion {
		checkoutChanges = append(checkoutChanges, errors.New("companion identity moved during proof"))
	}
	if len(checkoutChanges) > 0 {
		detail := strings.ReplaceAll(errors.Join(checkoutChanges...).Error(), "\n", "; ")
		return Result{}, fmt.Errorf("%w: %s", ErrCheckoutChanged, boundedDetail(detail))
	}

	digest, err := request.Digest()
	if err != nil {
		return Result{}, err
	}
	result := Result{Schema: ResultSchema, Contract: Contract, RequestDigest: digest, Detail: boundedDetail(detail)}
	switch outcome {
	case abi.WitnessConfirmed:
		result.Verdict = VerdictConfirmed
	case abi.WitnessRefuted:
		result.Verdict = VerdictRefuted
	default:
		result.Verdict = VerdictAbstained
	}
	if result.Detail == "" {
		result.Detail = "canonical witness returned no diagnostic"
	}
	if err := result.Validate(request); err != nil {
		return Result{}, err
	}
	return result, nil
}

func validateRoots(ctx context.Context, roots Roots) (string, string, error) {
	repo, err := filepath.Abs(roots.RepositoryDir)
	if err != nil || roots.RepositoryDir == "" || filepath.Clean(roots.RepositoryDir) != repo {
		return "", "", fmt.Errorf("%w: repository root must be an absolute clean path", ErrInvalidRequest)
	}
	companion, err := filepath.Abs(roots.CompanionDir)
	if err != nil || roots.CompanionDir == "" || filepath.Clean(roots.CompanionDir) != companion {
		return "", "", fmt.Errorf("%w: companion root must be an absolute clean path", ErrInvalidRequest)
	}
	if samePath(repo, companion) || !samePath(filepath.Dir(repo), filepath.Dir(companion)) {
		return "", "", fmt.Errorf("%w: repository and companion must be distinct sibling checkouts", ErrInvalidRequest)
	}
	repoPhysical, repoCommon, err := independentCloneRoot(ctx, repo)
	if err != nil {
		return "", "", err
	}
	companionPhysical, companionCommon, err := independentCloneRoot(ctx, companion)
	if err != nil {
		return "", "", err
	}
	if !samePath(filepath.Dir(repoPhysical), filepath.Dir(companionPhysical)) {
		return "", "", fmt.Errorf("%w: physical repository roots are not siblings", ErrIdentityMismatch)
	}
	if samePath(repoCommon, companionCommon) {
		return "", "", fmt.Errorf("%w: repository and companion share one Git common namespace", ErrIdentityMismatch)
	}
	return repoPhysical, companionPhysical, nil
}

// independentCloneRoot accepts only a physical, ordinary clone whose .git is
// a real directory owned by that clone. A linked worktree's .git pointer, a
// symlink/junction escaping the clone, or a shared common namespace is not an
// isolated qualification root. The canonical resolver may create scratch
// linked worktrees later, but only inside this sandbox clone's namespace.
func independentCloneRoot(ctx context.Context, root string) (physicalRoot, commonDir string, err error) {
	physicalRoot, err = filepath.EvalSymlinks(root)
	if err != nil || !samePath(physicalRoot, root) {
		return "", "", fmt.Errorf("%w: %s is not a physical clone root", ErrIdentityMismatch, root)
	}
	dotGit := filepath.Join(physicalRoot, ".git")
	info, statErr := os.Lstat(dotGit)
	if statErr != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", "", fmt.Errorf("%w: %s must own a physical .git directory", ErrIdentityMismatch, root)
	}
	physicalGit, evalErr := filepath.EvalSymlinks(dotGit)
	if evalErr != nil || !samePath(physicalGit, dotGit) {
		return "", "", fmt.Errorf("%w: %s .git escapes through a symlink or junction", ErrIdentityMismatch, root)
	}
	gitDir, resolveErr := gitPhysicalDir(ctx, physicalRoot, "--git-dir")
	if resolveErr != nil || !samePath(gitDir, physicalGit) {
		return "", "", fmt.Errorf("%w: %s Git directory is not owned by the clone", ErrIdentityMismatch, root)
	}
	commonDir, resolveErr = gitPhysicalDir(ctx, physicalRoot, "--git-common-dir")
	if resolveErr != nil || !samePath(commonDir, physicalGit) {
		return "", "", fmt.Errorf("%w: %s Git common directory is not owned by the clone", ErrIdentityMismatch, root)
	}
	return physicalRoot, commonDir, nil
}

func gitPhysicalDir(ctx context.Context, root, which string) (string, error) {
	value, err := gitText(ctx, root, "rev-parse", "--path-format=absolute", which)
	if err != nil {
		return "", err
	}
	physical, err := filepath.EvalSymlinks(filepath.Clean(value))
	if err != nil {
		return "", err
	}
	return physical, nil
}

func snapshotCheckout(ctx context.Context, dir string) (checkoutSnapshot, error) {
	inside, err := gitText(ctx, dir, "rev-parse", "--is-inside-work-tree")
	if err != nil || inside != "true" {
		return checkoutSnapshot{}, fmt.Errorf("%w: %s is not a worktree", ErrIdentityMismatch, dir)
	}
	if _, code, runErr := gitRun(ctx, dir, "symbolic-ref", "-q", "HEAD"); runErr != nil || code != 1 {
		return checkoutSnapshot{}, fmt.Errorf("%w: %s must have detached HEAD", ErrIdentityMismatch, dir)
	}
	head, err := gitText(ctx, dir, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return checkoutSnapshot{}, fmt.Errorf("%w: resolve HEAD in %s: %v", ErrIdentityMismatch, dir, err)
	}
	tree, err := gitText(ctx, dir, "rev-parse", "--verify", head+"^{tree}")
	if err != nil {
		return checkoutSnapshot{}, fmt.Errorf("%w: resolve tree in %s: %v", ErrIdentityMismatch, dir, err)
	}
	status, code, runErr := gitRun(ctx, dir, "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if runErr != nil || code != 0 {
		return checkoutSnapshot{}, fmt.Errorf("%w: inspect checkout status in %s", ErrIdentityMismatch, dir)
	}
	if len(status) != 0 {
		return checkoutSnapshot{}, fmt.Errorf("%w: %s is not clean", ErrCheckoutChanged, dir)
	}
	return checkoutSnapshot{head: head, tree: tree}, nil
}

func validateCandidateIdentity(ctx context.Context, repo string, request Request) error {
	parents, err := gitText(ctx, repo, "rev-list", "--parents", "-n", "1", request.CandidateCommit)
	if err != nil {
		return fmt.Errorf("%w: resolve candidate parents: %v", ErrIdentityMismatch, err)
	}
	fields := strings.Fields(parents)
	if len(fields) != 2 || fields[0] != request.CandidateCommit || fields[1] != request.ParentCommit {
		return fmt.Errorf("%w: candidate must have sole parent %s", ErrIdentityMismatch, request.ParentCommit)
	}
	parentTree, err := gitText(ctx, repo, "rev-parse", "--verify", request.ParentCommit+"^{tree}")
	if err != nil || parentTree != request.ParentTree {
		return fmt.Errorf("%w: parent tree does not match request", ErrIdentityMismatch)
	}
	candidateTree, err := gitText(ctx, repo, "rev-parse", "--verify", request.CandidateCommit+"^{tree}")
	if err != nil || candidateTree != request.CandidateTree {
		return fmt.Errorf("%w: candidate tree does not match request", ErrIdentityMismatch)
	}
	changes, err := candidateChanges(ctx, repo, request.ParentCommit, request.CandidateCommit)
	if err != nil {
		return err
	}
	paths := make([]string, 0, len(changes))
	changedTests := make(map[string]string)
	for _, change := range changes {
		paths = append(paths, change.path)
		if !strings.HasSuffix(change.path, "_test.go") {
			continue
		}
		if change.status != "A" && change.status != "M" {
			return fmt.Errorf("%w: changed Go test %q has status %s", ErrUnsupportedProofShape, change.path, change.status)
		}
		mode, kind, object, err := lsTreeEntry(ctx, repo, request.CandidateCommit, change.path)
		if err != nil || kind != "blob" || (mode != "100644" && mode != "100755") {
			return fmt.Errorf("%w: changed Go test %q is not a regular candidate blob", ErrUnsupportedProofShape, change.path)
		}
		changedTests[change.path] = object
	}
	sort.Strings(paths)
	if !equalStrings(paths, request.Paths) {
		return fmt.Errorf("%w: candidate changed paths differ from request", ErrIdentityMismatch)
	}
	if len(changedTests) != len(request.TestOverlays) {
		return fmt.Errorf("%w: overlays do not equal changed Go test set", ErrUnsupportedProofShape)
	}
	for _, overlay := range request.TestOverlays {
		if changedTests[overlay.Path] != overlay.Blob {
			return fmt.Errorf("%w: overlay blob mismatch for %s", ErrIdentityMismatch, overlay.Path)
		}
	}
	return nil
}

func candidateChanges(ctx context.Context, repo, parent, candidate string) ([]changedPath, error) {
	out, code, runErr := gitRun(ctx, repo, "diff-tree", "--no-commit-id", "--name-status", "--no-renames", "-r", "-z", parent, candidate)
	if runErr != nil || code != 0 {
		return nil, fmt.Errorf("%w: inspect candidate diff", ErrIdentityMismatch)
	}
	parts := bytes.Split(out, []byte{0})
	if len(parts) > 0 && len(parts[len(parts)-1]) == 0 {
		parts = parts[:len(parts)-1]
	}
	if len(parts)%2 != 0 {
		return nil, fmt.Errorf("%w: malformed candidate diff", ErrIdentityMismatch)
	}
	changes := make([]changedPath, 0, len(parts)/2)
	for i := 0; i < len(parts); i += 2 {
		status, p := string(parts[i]), string(parts[i+1])
		if status == "" || !validRelativePath(p, false) {
			return nil, fmt.Errorf("%w: malformed candidate diff path", ErrIdentityMismatch)
		}
		changes = append(changes, changedPath{status: status, path: p})
	}
	return changes, nil
}

func lsTreeEntry(ctx context.Context, repo, commit, p string) (mode, kind, object string, err error) {
	out, code, runErr := gitRun(ctx, repo, "ls-tree", "-z", commit, "--", ":(literal)"+p)
	if runErr != nil || code != 0 {
		return "", "", "", fmt.Errorf("inspect candidate blob")
	}
	rows := bytes.Split(bytes.TrimSuffix(out, []byte{0}), []byte{0})
	if len(rows) != 1 {
		return "", "", "", fmt.Errorf("candidate path did not resolve uniquely")
	}
	header, returnedPath, ok := bytes.Cut(rows[0], []byte{'\t'})
	if !ok || string(returnedPath) != p {
		return "", "", "", fmt.Errorf("candidate path mismatch")
	}
	fields := strings.Fields(string(header))
	if len(fields) != 3 {
		return "", "", "", fmt.Errorf("malformed ls-tree result")
	}
	return fields[0], fields[1], fields[2], nil
}

func gitText(ctx context.Context, dir string, args ...string) (string, error) {
	out, code, err := gitRun(ctx, dir, args...)
	if err != nil || code != 0 {
		return "", fmt.Errorf("git %s exited %d: %v", args[0], code, err)
	}
	return strings.TrimSpace(string(out)), nil
}

func gitRun(ctx context.Context, dir string, args ...string) ([]byte, int, error) {
	gitArgs := append([]string{"-C", dir}, args...)
	cmd := sysproc.CommandContext(ctx, "git", gitArgs...)
	cmd.Env = isolatedGitEnv(os.Environ())
	out, err := cmd.Output()
	if err == nil {
		return out, 0, nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return out, -1, ctxErr
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return out, exitErr.ExitCode(), nil
	}
	return out, -1, err
}

func isolatedGitEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, entry := range env {
		name, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		switch strings.ToUpper(name) {
		case "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_COMMON_DIR":
			continue
		}
		out = append(out, entry)
	}
	return out
}

func boundedDetail(detail string) string {
	detail = strings.TrimSpace(detail)
	if len(detail) <= MaxDetailBytes {
		return detail
	}
	b := []byte(detail[:MaxDetailBytes])
	for len(b) > 0 && !utf8.Valid(b) {
		b = b[:len(b)-1]
	}
	return strings.TrimSpace(string(b))
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

func samePath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}
