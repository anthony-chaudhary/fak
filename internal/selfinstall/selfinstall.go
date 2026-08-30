// Package selfinstall rebuilds the fak binary from the current checkout and atomically
// swaps it into a target path — but ONLY after the freshly-built binary passes a gate, so
// a tree that does not compile, fails vet, or produces a binary that cannot even print its
// version is NEVER installed over a running fleet.
//
// This is the "make the latest verified fak available" half of keeping an always-on guard
// fleet converged (binstamp is the "am I stale?" detection half). The hard rule it exists
// to enforce: convergence must mean "converge on the latest GOOD commit," never "converge
// on the latest commit." A broken fak.exe swapped under N guards would break every guard at
// once; the gate is therefore not optional polish — it is the whole point.
//
// The flow (Install):
//  0. cache   restore only an exact-input, digest-verified candidate, then smoke it again;
//     any miss or mismatch falls through to the complete gate below.
//  1. build   `go build [-ldflags -X …BuildVersion=<VERSION>] -o <tmp> ./cmd/fak` — a tree
//     that won't compile stops here; the ldflags bake the tree's VERSION into the
//     binary so its reported version does not depend on a guard's runtime cwd.
//  2. vet     `go vet ./cmd/fak`                — a vet failure stops here.
//  3. smoke   `<tmp> version`                   — the built binary must run + self-report.
//  4. swap    atomic replace of target by <tmp> — only reached when 1–3 all pass.
//
// Every effect goes through an injected Runner and an injected swap, so the gate ladder is
// testable with no toolchain and no filesystem race. A failed gate returns a Result with
// the failing Stage and the captured output, and leaves the target binary untouched.
package selfinstall

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Runner runs a command and returns combined output + whether it succeeded. ok=false means
// the command ran and failed OR could not be executed — either way the gate has not passed.
type Runner func(ctx context.Context, dir, name string, args ...string) (out string, ok bool)

// Swapper atomically replaces dst with the file at src (which is consumed/moved). It must be
// atomic enough that a concurrent reader sees either the old or the new binary, never a
// truncated one. On Windows replacing a mapped .exe requires renaming the old aside first;
// the production swap (osSwap) handles that.
type Swapper func(src, dst string) error

// Stage names the step a run reached (the last one attempted).
type Stage string

const (
	StageBuild   Stage = "build"
	StageCopy    Stage = "copy"
	StageVet     Stage = "vet"
	StageSmoke   Stage = "smoke"
	StageSwap    Stage = "swap"
	StageSkipped Stage = "skipped" // nothing to do (e.g. target already current)
)

// Result reports the outcome of an Install attempt.
type Result struct {
	Installed            bool              // the target was replaced with a freshly-verified binary
	Stage                Stage             // the last stage attempted
	Detail               string            // captured output / error context for the failing or final stage
	SourceCommit         string            // selected source revision for this activation
	ArtifactSourceCommit string            // source revision that originally produced the artifact bytes
	BuildInputDigest     string            // complete executable-input identity
	BuildEnvelope        map[string]string // toolchain/platform/options bound by BuildInputDigest
	ArtifactDigest       string            // SHA-256 of the activated candidate
	ArtifactSize         int64
	AppVersion           string
	Reused               bool
}

const candidateCacheSchema = "fak.selfinstall.candidate-cache.v2"

type candidateCacheInput struct {
	Schema           string            `json:"schema"`
	BuildInputDigest string            `json:"build_input_digest"`
	BuildEnvelope    map[string]string `json:"build_envelope"`
	SourceCommit     string            `json:"source_commit"`
	HostOS           string            `json:"host_os"`
	HostArch         string            `json:"host_arch"`
	BuildArgs        []string          `json:"build_args"`
	VetArgs          []string          `json:"vet_args"`
}

type candidateCacheManifest struct {
	Schema               string `json:"schema"`
	ArtifactSourceCommit string `json:"artifact_source_commit"`
	BuildInputDigest     string `json:"build_input_digest"`
	InputDigest          string `json:"input_digest"`
	ArtifactDigest       string `json:"artifact_digest"`
	ArtifactSize         int64  `json:"artifact_size"`
	BoundDigest          string `json:"bound_digest"`
}

type candidateVersionIdentity struct {
	AppVersion string `json:"app_version"`
	Commit     string `json:"commit"`
	Dirty      bool   `json:"dirty"`
	Stamped    bool   `json:"stamped"`
}

// Options configures Install.
type Options struct {
	// RepoRoot is the checkout to build from (the dir `go build` runs in).
	RepoRoot string
	// Target is the binary path to replace (e.g. the guard's os.Executable()).
	Target string
	// BuildTmp is where the candidate binary is built before the swap. Empty => a sibling
	// of Target with a ".new" suffix (same volume, so the swap is a cheap rename).
	BuildTmp string
	// CacheDir stores a previously build/vet/smoke-verified candidate. Empty disables
	// reuse. Cache acceptance is fail-closed on the clean exact commit, Go
	// toolchain/platform/build inputs, artifact SHA-256, and a fresh smoke/provenance check.
	CacheDir string
	// ExpectedCommit is the exact clean commit selected by the caller. When caching is enabled,
	// empty falls back to the checkout's clean HEAD. A valid explicit value must match both the
	// checkout and the candidate's full `version --json` provenance before reuse or activation.
	ExpectedCommit string
}

// Install runs the gated build→vet→smoke→swap ladder. It installs IFF every gate passes; on
// any gate failure it returns the failing Stage and leaves Target untouched.
func Install(ctx context.Context, run Runner, swap Swapper, opts Options) Result {
	tmp := strings.TrimSpace(opts.BuildTmp)
	if tmp == "" {
		tmp = opts.Target + ".new"
	}
	cleanupBuildArtifact := true
	defer func() {
		if cleanupBuildArtifact {
			_ = os.Remove(tmp)
		}
	}()

	sourceCommit := ""
	if out, ok := run(ctx, opts.RepoRoot, "git", "status", "--porcelain"); ok && strings.TrimSpace(out) == "" {
		if out, ok := run(ctx, opts.RepoRoot, "git", "rev-parse", "HEAD"); ok {
			candidate := strings.TrimSpace(out)
			if validCommit(candidate) {
				sourceCommit = strings.ToLower(candidate)
			}
		}
	}

	explicitExpectedCommit := strings.ToLower(strings.TrimSpace(opts.ExpectedCommit))
	if explicitExpectedCommit != "" && !validCommit(explicitExpectedCommit) {
		return Result{Stage: StageSmoke, Detail: "expected commit is not a full 40-character Git object ID"}
	}
	if explicitExpectedCommit != "" && !strings.EqualFold(sourceCommit, explicitExpectedCommit) {
		return Result{Stage: StageSmoke, Detail: fmt.Sprintf("clean source checkout reports commit %s, want exact commit %s", emptyLabel(sourceCommit), explicitExpectedCommit)}
	}
	expectedCommit := explicitExpectedCommit
	if expectedCommit == "" && strings.TrimSpace(opts.CacheDir) != "" {
		expectedCommit = sourceCommit
	}
	if !validCommit(expectedCommit) {
		expectedCommit = ""
	}

	// The output path changes per activation and cannot affect the artifact, so the cache
	// identity binds the stable build arguments rather than BuildTmp.
	buildInputs := []string{"build", "-buildvcs=true"}
	if ld := versionLDFlags(opts.RepoRoot, sourceCommit); ld != "" {
		buildInputs = append(buildInputs, "-ldflags", ld)
	}
	buildArgs := append(append([]string{}, buildInputs...), "-o", tmp, "./cmd/fak")
	buildInputs = append(buildInputs, "./cmd/fak")
	identityBuildInputs := []string{"-buildvcs=true"}
	if ld := versionLDFlags(opts.RepoRoot, ""); ld != "" {
		identityBuildInputs = append(identityBuildInputs, "-ldflags", ld)
	}
	vetArgs := []string{"vet", "./cmd/fak"}

	cacheDir := strings.TrimSpace(opts.CacheDir)
	var cacheInput candidateCacheInput
	cacheEligible := cacheDir != "" &&
		validCommit(sourceCommit) &&
		validCommit(expectedCommit) &&
		strings.EqualFold(sourceCommit, expectedCommit)
	cacheHit := false
	artifactSourceCommit := sourceCommit
	appVersion := ""
	if cacheEligible {
		identity, err := deriveBuildInputIdentity(ctx, opts.RepoRoot, "./cmd/fak", buildInputOptions{BuildFlags: identityBuildInputs})
		if err == nil {
			cacheInput = candidateCacheInput{
				Schema:           candidateCacheSchema,
				BuildInputDigest: identity.Digest,
				BuildEnvelope:    identity.Envelope,
				SourceCommit:     sourceCommit,
				HostOS:           runtime.GOOS,
				HostArch:         runtime.GOARCH,
				BuildArgs:        append([]string{}, identityBuildInputs...),
				VetArgs:          append([]string{}, vetArgs...),
			}
			if manifest, err := restoreCandidateCache(cacheDir, cacheInput, tmp); err == nil {
				if identity, _, ok := smokeCandidate(ctx, run, opts.RepoRoot, tmp, manifest.ArtifactSourceCommit); ok {
					cacheHit = true
					artifactSourceCommit = manifest.ArtifactSourceCommit
					appVersion = identity.AppVersion
				} else {
					_ = os.Remove(tmp)
				}
			}
		} else {
			cacheEligible = false
		}
	}

	cacheRefreshDetail := ""
	if !cacheHit {
		// 1. build the candidate, baking the tree's VERSION into it as
		//    appversion.BuildVersion (see versionLDFlags for why this is load-bearing, not
		//    cosmetic). -buildvcs=true FORCES the VCS stamp: under Go's default
		//    -buildvcs=auto the detached-worktree build (PrepareOrigin) can silently drop the
		//    stamp when VCS detection cannot resolve the repo, shipping a binary that cannot
		//    attest which commit it is (#3350, epic #2218 gap G2). With =true the build FAILS
		//    instead of emitting an unstamped binary.
		if out, ok := run(ctx, opts.RepoRoot, "go", buildArgs...); !ok {
			return Result{Stage: StageBuild, Detail: trim(out)}
		}
		// 2. vet the package (catches a compiling-but-suspect tree).
		if out, ok := run(ctx, opts.RepoRoot, "go", vetArgs...); !ok {
			return Result{Stage: StageVet, Detail: trim(out)}
		}
		// 3. smoke: the freshly-built binary must run and report its full machine-readable
		//    provenance. When the caller selected an exact commit, the candidate must attest
		//    that full commit and a clean build before it can be cached or swapped.
		if identity, out, ok := smokeCandidate(ctx, run, opts.RepoRoot, tmp, expectedCommit); !ok {
			return Result{Stage: StageSmoke, Detail: trim(out)}
		} else {
			appVersion = identity.AppVersion
		}
		if cacheEligible {
			if err := refreshCandidateCache(cacheDir, cacheInput, tmp); err != nil {
				// Cache persistence is an optimization, not an installation gate. The
				// candidate itself still passed build, vet, and exact-provenance smoke.
				cacheRefreshDetail = "; candidate cache refresh failed: " + trim(err.Error())
			}
		}
	}

	// 4. swap: only now is the candidate trusted over the running fleet.
	cleanupBuildArtifact = false
	artifactDigest, artifactSize, err := fileSHA256(tmp)
	if err != nil && !os.IsNotExist(err) {
		return Result{Stage: StageSmoke, Detail: "hash verified candidate: " + err.Error()}
	}
	if err := swap(tmp, opts.Target); err != nil {
		return Result{Stage: StageSwap, Detail: err.Error()}
	}
	cleanupBuildArtifact = false
	detail := "installed " + filepath.Base(opts.Target)
	if cacheHit {
		detail += " from build-input verified candidate cache"
	}
	return Result{Installed: true, Stage: StageSwap, Detail: detail + cacheRefreshDetail,
		SourceCommit: sourceCommit, ArtifactSourceCommit: artifactSourceCommit,
		BuildInputDigest: cacheInput.BuildInputDigest, BuildEnvelope: cacheInput.BuildEnvelope,
		ArtifactDigest: artifactDigest, ArtifactSize: artifactSize, AppVersion: appVersion, Reused: cacheHit}
}

func emptyLabel(value string) string {
	if strings.TrimSpace(value) == "" {
		return "(unverified or dirty)"
	}
	return value
}

func smokeCandidate(ctx context.Context, run Runner, repoRoot, candidate, expectedCommit string) (candidateVersionIdentity, string, bool) {
	out, ok := run(ctx, repoRoot, candidate, "version", "--json")
	if !ok {
		return candidateVersionIdentity{}, out, false
	}
	var identity candidateVersionIdentity
	if err := json.Unmarshal([]byte(out), &identity); err != nil {
		if expectedCommit == "" && versionOutputStamped(out) {
			return candidateVersionIdentity{}, out, true
		}
		return candidateVersionIdentity{}, "candidate provenance is not valid `version --json`: " + trim(out), false
	}
	if !identity.Stamped || identity.Dirty || !validCommit(identity.Commit) {
		return candidateVersionIdentity{}, "candidate binary is unstamped or dirty; refusing to swap: " + trim(out), false
	}
	if expectedCommit != "" && !strings.EqualFold(identity.Commit, expectedCommit) {
		return candidateVersionIdentity{}, fmt.Sprintf("candidate binary reports commit %s, want exact commit %s", identity.Commit, expectedCommit), false
	}
	return identity, out, true
}

func candidateInputDigest(input candidateCacheInput) (string, error) {
	input.SourceCommit = ""
	data, err := json.Marshal(input)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func candidateBoundDigest(inputDigest, artifactDigest string) string {
	sum := sha256.Sum256([]byte(candidateCacheSchema + "\x00" + inputDigest + "\x00" + artifactDigest))
	return hex.EncodeToString(sum[:])
}

// CandidateCacheDir maps the clone-shared Git common directory to the persistent
// self-update cache. An unresolved common directory disables caching rather than creating
// a relative or bogus .git directory inside a linked/custom worktree.
func CandidateCacheDir(gitCommonDir string) string {
	if strings.TrimSpace(gitCommonDir) == "" {
		return ""
	}
	return filepath.Join(filepath.Clean(gitCommonDir), "fak", "self-update-cache")
}

func candidateCachePaths(dir string) (manifest string) {
	return filepath.Join(dir, "manifest.json")
}

func candidateArtifactPath(dir, artifactDigest string) string {
	return filepath.Join(dir, "candidate-"+artifactDigest)
}

func fileSHA256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", 0, err
	}
	if !info.Mode().IsRegular() {
		return "", 0, fmt.Errorf("%s is not a regular file", path)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), info.Size(), nil
}

func restoreCandidateCache(dir string, input candidateCacheInput, dst string) (candidateCacheManifest, error) {
	inputDigest, err := candidateInputDigest(input)
	if err != nil {
		return candidateCacheManifest{}, err
	}
	data, err := os.ReadFile(candidateCachePaths(dir))
	if err != nil {
		return candidateCacheManifest{}, err
	}
	var manifest candidateCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return candidateCacheManifest{}, err
	}
	if manifest.Schema != candidateCacheSchema ||
		manifest.BuildInputDigest != input.BuildInputDigest ||
		!validCommit(manifest.ArtifactSourceCommit) ||
		manifest.InputDigest != inputDigest ||
		manifest.BoundDigest != candidateBoundDigest(inputDigest, manifest.ArtifactDigest) {
		return candidateCacheManifest{}, fmt.Errorf("candidate cache identity mismatch")
	}
	if !validSHA256(manifest.ArtifactDigest) {
		return candidateCacheManifest{}, fmt.Errorf("candidate cache artifact digest is malformed")
	}
	artifact := candidateArtifactPath(dir, manifest.ArtifactDigest)
	digest, size, err := fileSHA256(artifact)
	if err != nil {
		return candidateCacheManifest{}, err
	}
	if digest != manifest.ArtifactDigest || size != manifest.ArtifactSize {
		return candidateCacheManifest{}, fmt.Errorf("candidate cache artifact digest mismatch")
	}
	if err := atomicCopyFile(artifact, dst); err != nil {
		return candidateCacheManifest{}, err
	}
	copiedDigest, copiedSize, err := fileSHA256(dst)
	if err != nil {
		_ = os.Remove(dst)
		return candidateCacheManifest{}, err
	}
	if copiedDigest != manifest.ArtifactDigest || copiedSize != manifest.ArtifactSize {
		_ = os.Remove(dst)
		return candidateCacheManifest{}, fmt.Errorf("restored candidate digest mismatch")
	}
	return manifest, nil
}

func validSHA256(s string) bool {
	if len(s) != sha256.Size*2 || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func refreshCandidateCache(dir string, input candidateCacheInput, src string) error {
	inputDigest, err := candidateInputDigest(input)
	if err != nil {
		return err
	}
	artifactDigest, artifactSize, err := fileSHA256(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	artifact := candidateArtifactPath(dir, artifactDigest)
	if err := atomicCopyFile(src, artifact); err != nil {
		return err
	}
	storedDigest, storedSize, err := fileSHA256(artifact)
	if err != nil {
		return err
	}
	if storedDigest != artifactDigest || storedSize != artifactSize {
		return fmt.Errorf("refreshed candidate digest mismatch")
	}
	manifest := candidateCacheManifest{
		Schema:               candidateCacheSchema,
		ArtifactSourceCommit: input.SourceCommit,
		BuildInputDigest:     input.BuildInputDigest,
		InputDigest:          inputDigest,
		ArtifactDigest:       artifactDigest,
		ArtifactSize:         artifactSize,
		BoundDigest:          candidateBoundDigest(inputDigest, artifactDigest),
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := atomicWriteFile(candidateCachePaths(dir), 0o600, bytes.NewReader(data)); err != nil {
		return err
	}
	pruneCandidateArtifacts(dir, artifact)
	return nil
}

func atomicCopyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	return atomicWriteFile(dst, info.Mode().Perm(), in)
}

func atomicWriteFile(dst string, mode os.FileMode, source io.Reader) (err error) {
	if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), "."+filepath.Base(dst)+".tmp-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}()
	if err := tmp.Chmod(mode); err != nil {
		return err
	}
	if _, err := io.Copy(tmp, source); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return OSSwap(tmpPath, dst)
}

func pruneCandidateArtifacts(dir, keep string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		path := filepath.Join(dir, entry.Name())
		if path == keep || !strings.HasPrefix(entry.Name(), "candidate-") {
			continue
		}
		_ = os.Remove(path)
	}
}

// InstallVerifiedCopy installs an already-gated fak binary at another hot-copy path.
// Self-update uses this after its first build/vet/smoke ladder succeeds, so converging N
// host copies pays that expensive ladder once rather than rebuilding identical bytes N times.
func InstallVerifiedCopy(swap Swapper, source, target string) Result {
	if strings.TrimSpace(source) == "" || strings.TrimSpace(target) == "" {
		return Result{Stage: StageCopy, Detail: "source and target are required"}
	}

	in, err := os.Open(source)
	if err != nil {
		return Result{Stage: StageCopy, Detail: "open verified source: " + err.Error()}
	}
	defer in.Close()

	info, err := in.Stat()
	if err != nil {
		return Result{Stage: StageCopy, Detail: "stat verified source: " + err.Error()}
	}
	tmp := fmt.Sprintf("%s.new.%d", target, os.Getpid())
	_ = os.Remove(tmp)
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, info.Mode().Perm())
	if err != nil {
		return Result{Stage: StageCopy, Detail: "create candidate copy: " + err.Error()}
	}
	_, copyErr := io.Copy(out, in)
	if copyErr == nil {
		copyErr = out.Sync()
	}
	closeErr := out.Close()
	if copyErr == nil {
		copyErr = closeErr
	}
	if copyErr != nil {
		_ = os.Remove(tmp)
		return Result{Stage: StageCopy, Detail: "copy verified candidate: " + copyErr.Error()}
	}
	defer os.Remove(tmp)

	if err := swap(tmp, target); err != nil {
		return Result{Stage: StageSwap, Detail: err.Error()}
	}
	return Result{Installed: true, Stage: StageSwap, Detail: "installed " + filepath.Base(target) + " from verified candidate"}
}

// PrepareOrigin checks out a PRISTINE detached copy of a ref (e.g. "origin/main") into a
// fresh temp worktree under the repo, and returns its path plus a cleanup func. Building
// from this — instead of the live working tree — is what makes self-update viable on a
// permanently-dirty shared trunk: a build from the live tree always stamps vcs.modified=true
// (because peers are mid-edit), which would make every binary look "dirty" and defeat the
// staleness check; worse, it would bake peer work-in-progress INTO the installed binary.
// A detached origin worktree gives a clean VCS stamp AND installs exactly verified
// origin/main, never a contaminated local build.
//
// It is best-effort and self-cleaning: the cleanup removes the worktree (git worktree
// remove --force), prunes the admin entry, and removes the out-of-tree owner stamp only
// after both steps are proven. The cleanup is idempotent so callers may invoke it
// explicitly before os.Exit (which skips deferred functions) while still deferring it
// for ordinary returns. A partial `worktree add` failure is cleaned immediately, but
// only after proving the target did not exist before the add attempt.
func PrepareOrigin(ctx context.Context, run Runner, repoRoot, ref, dir string) (string, func(), error) {
	noop := func() {}
	if strings.TrimSpace(ref) == "" {
		ref = "origin/main"
	}
	dir = filepath.Clean(strings.TrimSpace(dir))
	if dir == "." || dir == "" {
		return "", noop, fmt.Errorf("prepare-origin: empty worktree path")
	}
	if _, err := os.Lstat(dir); err == nil {
		return "", noop, fmt.Errorf("prepare-origin: refusing to reuse existing path %s", dir)
	} else if !os.IsNotExist(err) {
		return "", noop, fmt.Errorf("prepare-origin: cannot inspect worktree path %s: %v", dir, err)
	}
	// Mutable refs need a refresh immediately before materialization. A caller that already
	// selected a full immutable commit must not fetch again: that preserves one attempt's
	// identity and avoids an unnecessary metadata write in the linearized transaction.
	if !validCommit(ref) {
		_, _ = run(ctx, repoRoot, "git", "fetch", "origin", "--quiet")
	}
	if out, ok := run(ctx, repoRoot, "git", "worktree", "add", "--detach", dir, ref); !ok {
		// Git may have materialized part of the directory/admin entry before returning
		// failure. The path was proven absent above, so removing that partial result
		// cannot touch a pre-existing checkout.
		cleanupOriginWorktree(ctx, run, repoRoot, dir)
		return "", noop, fmt.Errorf("prepare-origin: git worktree add %s @ %s failed: %s", dir, ref, trim(out))
	}
	if err := writeBuildOwnerStamp(dir, defaultBuildOwnerStamp()); err != nil {
		cleanupOriginWorktree(ctx, run, repoRoot, dir)
		return "", noop, fmt.Errorf("prepare-origin: owner stamp %s: %v", BuildOwnerStampPath(dir), err)
	}
	var once sync.Once
	cleanup := func() { once.Do(func() { cleanupOriginWorktree(ctx, run, repoRoot, dir) }) }
	return dir, cleanup, nil
}

// cleanupOriginWorktree is the one source-cleanup path for PrepareOrigin. Git removal
// is preferred because it removes the administrative record and directory together.
// If it fails, direct removal is safe here because PrepareOrigin proved the target was
// absent before creating it. The owner stamp survives any incomplete cleanup so a later
// owner-aware GC can repair a directory-gone administrative record without guessing.
func cleanupOriginWorktree(ctx context.Context, run Runner, repoRoot, dir string) {
	if _, ok := run(ctx, repoRoot, "git", "worktree", "remove", "--force", dir); !ok {
		_ = os.RemoveAll(dir)
	}
	if _, err := os.Lstat(dir); !os.IsNotExist(err) {
		return
	}
	if _, ok := run(ctx, repoRoot, "git", "worktree", "prune"); !ok {
		return
	}
	removeBuildOwnerStamp(dir)
}

// versionLDFlags returns the `-ldflags` value that bakes RepoRoot's VERSION marker into the
// built binary as appversion.BuildVersion, or "" when the tree has no readable VERSION file
// (in which case the build stays exactly as before — no ldflags).
//
// Why this is load-bearing, not polish: a bare `go build ./cmd/fak` produces a binary with
// NO embedded version, so at RUN time appversion.Current() falls through to walking the
// filesystem upward for a VERSION file — which makes a guard's reported version depend on
// its working directory. A guard launched with a cwd under a PARENT checkout inherits that
// parent's marker: on the fleet host, guards launched under C:\work reported the workspace's
// stale C:\work\VERSION ("0.1.1") instead of the fleet binary's actual version, and the same
// binary reported "dev" / the real version / the stale one purely by where it was launched.
// BuildVersion wins over the VERSION walk in appversion.Current() precisely so a
// shipped/installed binary "reports the version it was built with instead of inheriting a
// parent checkout's marker" — self-update simply never set it. Baking it here (the same
// -X the release-artifacts workflow uses) pins the installed binary's version to the commit
// it was built from, independent of wherever a guard is later launched.
func versionLDFlags(repoRoot, commit string) string {
	var flags []string
	if b, err := os.ReadFile(filepath.Join(repoRoot, "VERSION")); err == nil {
		if v := strings.TrimSpace(string(b)); v != "" {
			flags = append(flags, "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="+v)
		}
	}
	if validCommit(commit) {
		flags = append(flags, "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildCommit="+commit)
	}
	return strings.Join(flags, " ")
}

func validCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

// versionOutputStamped reports whether `fak version` output carries a REAL VCS stamp: a
// `build: <rev>` line whose revision is an actual commit, not the "(no VCS stamp …)" or
// "module vX" sentinels an unstamped / `go install …@vX` build prints. It mirrors the parse
// cmd/fak's stampOfBinary uses, so the self-update smoke gate reads the exact same provenance
// signal a fleet version-skew witness does (#3350). An unstamped candidate yields false and
// the gate refuses the swap.
func versionOutputStamped(out string) bool {
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "build:") {
			continue
		}
		fields := strings.Fields(strings.TrimSpace(strings.TrimPrefix(line, "build:")))
		if len(fields) == 0 {
			return false // a bare "build:" line — no revision to attest.
		}
		switch fields[0] {
		case "module", "(no": // "module vX" (go install …@vX) or "(no VCS stamp …)".
			return false
		default:
			return true
		}
	}
	return false // no build line at all — treat as unstamped, not as a pass.
}

func trim(s string) string {
	s = strings.TrimSpace(s)
	const max = 2000
	if len(s) > max {
		return s[:max] + "…(truncated)"
	}
	if s == "" {
		return "(no output)"
	}
	return s
}

// FormatResult renders a Result as a single human line for the CLI / logs.
func FormatResult(r Result) string {
	if r.Installed {
		return "self-install: OK — " + r.Detail
	}
	return fmt.Sprintf("self-install: NOT installed — failed at %s gate: %s", r.Stage, r.Detail)
}
