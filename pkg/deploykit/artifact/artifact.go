// Package artifact is the acquire and verify facade of the deployable lifecycle. It turns an
// artifact source into a verified candidate, and a verified candidate into an immutable,
// content-addressed slot, or it refuses.
//
// There are two sources. BuildPinned builds from a clean-HEAD commit tuple across one or more
// repositories, under a generated go.work that uses exactly those checkouts and never the
// operator's ambient workspace, then runs build → vet → smoke. Fetch downloads an artifact and
// checks its size and SHA-256 before anything executes it, then smokes it. Both end in Verify,
// which is selfinstall.VerifyTarget: byte identity first, provenance second. StoreSlot wraps
// selfinstall.StoreVerifiedSlot, the generation+digest slot `fak self-update` already writes,
// so an existing slot is re-verified and reused, never overwritten.
//
// It is a facade, not a second self-update. The slot layout, the byte-identity check and the
// provenance smoke of Verify are internal/selfinstall's; the `version --json` parser is
// pkg/deploykit/stamp's. What is new here is only what those primitives do not take as input
// yet: the descriptor's package and vet targets, the multi-repo clean-HEAD tuple, the pinned
// workspace environment, and the download.
//
// Two paths fail closed by name until their owners export what they need: a signed
// Fetch.Manifest returns ErrManifestNotWired (the ed25519 verifier is unexported in
// internal/selfupdate/cmd), and a BuildPinned for another platform returns
// ErrCrossBuildNotWired (this host cannot smoke a foreign binary).
package artifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/deploykit"
	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

// Source kinds. They match the deploykit.Source.Kind tag of the descriptor a source came from.
const (
	KindBuild = "build"
	KindFetch = "fetch"
)

var (
	// ErrInvalidSource reports a source spec that is incomplete or malformed. Nothing ran.
	ErrInvalidSource = errors.New("artifact: invalid source")
	// ErrInvalidCandidate reports a candidate whose recorded identity is incomplete or malformed.
	ErrInvalidCandidate = errors.New("artifact: invalid candidate")
	// ErrEnvExecutorRequired reports a BuildPinned acquire through an executor that cannot pin
	// the go children's environment. Building without the pinned GOWORK would fall back to the
	// operator's ambient workspace, so it is refused instead.
	ErrEnvExecutorRequired = errors.New("artifact: pinned build needs an EnvExecutor")
	// ErrCrossBuildNotWired reports a BuildPinned whose GOOS/GOARCH is not this host's. The
	// smoke cannot run a foreign binary here, so the build is refused rather than left unsmoked.
	ErrCrossBuildNotWired = errors.New("artifact: cross-platform pinned build is not wired")
	// ErrManifestNotWired reports a Fetch that carries a signed manifest. The ed25519 verifier
	// lives unexported in internal/selfupdate/cmd; until it is exported the facade refuses
	// rather than pass on the SHA-256 alone.
	ErrManifestNotWired = errors.New("artifact: signed manifest verification is not wired")
	// ErrDirtySource reports a repository of the pinned tuple with tracked or untracked changes.
	ErrDirtySource = errors.New("artifact: pinned source tree is not clean")
	// ErrPinnedCommitMismatch reports a repository whose HEAD is not its pinned commit, before or
	// after the build.
	ErrPinnedCommitMismatch = errors.New("artifact: source HEAD is not the pinned commit")
	// ErrBuildFailed reports a failed `go build` of the pinned package.
	ErrBuildFailed = errors.New("artifact: build failed")
	// ErrVetFailed reports a failed `go vet` of the descriptor's vet targets.
	ErrVetFailed = errors.New("artifact: vet failed")
	// ErrFetchFailed reports a download that did not produce a complete response.
	ErrFetchFailed = errors.New("artifact: fetch failed")
	// ErrSizeMismatch reports artifact bytes whose length is not the pinned size.
	ErrSizeMismatch = errors.New("artifact: size mismatch")
	// ErrDigestMismatch reports artifact bytes whose SHA-256 is not the pinned digest.
	ErrDigestMismatch = errors.New("artifact: SHA-256 mismatch")
	// ErrSmokeCommitMismatch reports a candidate whose `version --json` attests a clean commit
	// other than the pinned one.
	ErrSmokeCommitMismatch = errors.New("artifact: smoked commit is not the pinned commit")
	// ErrProvenance reports a candidate that failed the provenance smoke: it did not run, did not
	// print a valid `version --json`, is dirty or unstamped, or reports the wrong app version.
	ErrProvenance = errors.New("artifact: provenance check failed")
	// ErrSlot reports a slot that could not be stored or re-verified. An existing slot that no
	// longer matches is refused and left untouched.
	ErrSlot = errors.New("artifact: verified slot refused")
)

// Source is an artifact source Acquire understands: BuildPinned or Fetch.
type Source interface {
	// Kind is KindBuild or KindFetch.
	Kind() string
	// preflight refuses a malformed spec before anything is created or executed.
	preflight(x deploykit.Executor) error
	// acquire materializes the candidate in dir and gates it. Acquire owns dir.
	acquire(ctx context.Context, x deploykit.Executor, dir string) (Candidate, error)
}

// Candidate is an artifact that passed its source's gate and Verify. Path lives in a staging
// directory the candidate owns; store it with StoreSlot, then Remove it.
type Candidate struct {
	Kind       string `json:"kind"`
	Path       string `json:"path"`
	Generation uint64 `json:"generation"`
	Commit     string `json:"commit"` // the full source commit the binary attests
	Digest     string `json:"sha256"` // lowercase hex SHA-256 of the bytes at Path
	Size       int64  `json:"size"`
	AppVersion string `json:"app_version"`

	dir string // staging directory created by Acquire; "" for a hand-built Candidate
}

// Remove deletes the staging directory Acquire created for the candidate. It never removes
// anything for a Candidate that Acquire did not produce.
func (c Candidate) Remove() error {
	if c.dir == "" {
		return nil
	}
	return os.RemoveAll(c.dir)
}

func (c Candidate) target() selfinstall.VerifiedTarget {
	return selfinstall.VerifiedTarget{
		MetadataGeneration: c.Generation,
		SourceCommit:       c.Commit,
		ArtifactDigest:     c.Digest,
		ArtifactSize:       c.Size,
		AppVersion:         c.AppVersion,
	}
}

func (c Candidate) validate() error {
	switch {
	case strings.TrimSpace(c.Path) == "":
		return fmt.Errorf("%w: empty path", ErrInvalidCandidate)
	case c.Generation == 0:
		return fmt.Errorf("%w: generation must be positive", ErrInvalidCandidate)
	case !fullCommit(c.Commit):
		return fmt.Errorf("%w: commit is not a full 40-hex object ID", ErrInvalidCandidate)
	case !validDigest(c.Digest) || c.Size < 1:
		return fmt.Errorf("%w: digest or size is invalid", ErrInvalidCandidate)
	case strings.TrimSpace(c.AppVersion) == "":
		return fmt.Errorf("%w: app version is required", ErrInvalidCandidate)
	}
	return nil
}

// Acquire turns src into a verified candidate or refuses. A malformed source is refused before
// anything is created; any later refusal removes the staging directory, so a failed Acquire
// never leaves a candidate behind. The executor must run on this host: it smokes the local
// candidate file.
func Acquire(ctx context.Context, x deploykit.Executor, src Source) (Candidate, error) {
	if x == nil {
		return Candidate{}, fmt.Errorf("%w: nil executor", ErrInvalidSource)
	}
	if v := reflect.ValueOf(src); src == nil || (v.Kind() == reflect.Pointer && v.IsNil()) {
		return Candidate{}, fmt.Errorf("%w: nil source", ErrInvalidSource)
	}
	if err := src.preflight(x); err != nil {
		return Candidate{}, err
	}
	dir, err := os.MkdirTemp("", "fak-deploykit-candidate-*")
	if err != nil {
		return Candidate{}, fmt.Errorf("artifact: create staging directory: %w", err)
	}
	c, err := src.acquire(ctx, x, dir)
	if err == nil {
		c.dir = dir
		err = Verify(ctx, x, c)
	}
	if err != nil {
		_ = os.RemoveAll(dir)
		return Candidate{}, err
	}
	return c, nil
}

// Verify re-proves a candidate against its recorded identity: byte identity first (size, then
// SHA-256), provenance second (selfinstall.VerifyTarget: the smoke must attest the exact clean
// commit and the app version).
func Verify(ctx context.Context, x deploykit.Executor, c Candidate) error {
	if x == nil {
		return fmt.Errorf("%w: nil executor", ErrInvalidCandidate)
	}
	if err := c.validate(); err != nil {
		return err
	}
	if err := VerifyBytes(c.Path, c.Digest, c.Size); err != nil {
		return err
	}
	if err := selfinstall.VerifyTarget(ctx, x.Run, c.Path, filepath.Dir(c.Path), c.target()); err != nil {
		return fmt.Errorf("%w: %v", ErrProvenance, err)
	}
	return nil
}

// VerifyBytes checks that the regular file at path is exactly size bytes with the given
// SHA-256 (hex, any case). It executes nothing.
func VerifyBytes(path, digest string, size int64) error {
	want := strings.ToLower(strings.TrimSpace(digest))
	if !validDigest(want) || size < 1 {
		return fmt.Errorf("%w: digest or size is invalid", ErrInvalidCandidate)
	}
	got, n, err := hashFile(path)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidCandidate, err)
	}
	if n != size {
		return fmt.Errorf("%w: got %d bytes, want %d", ErrSizeMismatch, n, size)
	}
	if got != want {
		return fmt.Errorf("%w: got %s, want %s", ErrDigestMismatch, got, want)
	}
	return nil
}

// StoreSlot stores a verified candidate in the immutable generation+digest slot beside
// targetPath (selfinstall.StoreVerifiedSlot, root <target>.self-update-slots) and returns the
// slot artifact path. The candidate's bytes are re-checked first, so bytes that do not match
// their recorded digest never create a slot. An existing identical slot is re-verified and
// reused; a corrupt or mismatched one is refused and left untouched. A slot this call created
// but could not complete is removed.
func StoreSlot(targetPath string, c Candidate) (string, error) {
	if strings.TrimSpace(targetPath) == "" {
		return "", fmt.Errorf("%w: empty target path", ErrInvalidCandidate)
	}
	if err := c.validate(); err != nil {
		return "", err
	}
	if err := VerifyBytes(c.Path, c.Digest, c.Size); err != nil {
		return "", err
	}
	root := slotRoot(targetPath)
	before, rootExisted := slotNames(root)
	path, err := selfinstall.StoreVerifiedSlot(targetPath, c.Path, c.target())
	if err == nil {
		err = VerifyBytes(path, c.Digest, c.Size)
	}
	if err != nil {
		discardNewSlots(root, before, rootExisted, strings.ToLower(c.Digest))
		return "", fmt.Errorf("%w: %v", ErrSlot, err)
	}
	return path, nil
}

// slotRoot is selfinstall.StoreVerifiedSlot's slot root; TestSlotRootMatchesSelfinstall pins
// the two together.
func slotRoot(targetPath string) string {
	return filepath.Clean(targetPath) + ".self-update-slots"
}

func slotNames(root string) (map[string]bool, bool) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, false
	}
	names := make(map[string]bool, len(entries))
	for _, e := range entries {
		names[e.Name()] = true
	}
	return names, true
}

// discardNewSlots removes the slot directories for digest that appeared during a failed store,
// and the slot root if the store created it and it is now empty. Pre-existing slots stay.
func discardNewSlots(root string, before map[string]bool, rootExisted bool, digest string) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() && !before[e.Name()] && strings.HasSuffix(e.Name(), "-"+digest) {
			_ = os.RemoveAll(filepath.Join(root, e.Name()))
		}
	}
	if !rootExisted {
		_ = os.Remove(root) // only succeeds when empty
	}
}

// smoke runs `<path> version --json` in dir and returns the parsed identity when the binary is
// stamped, clean, attests wantCommit, and reports an app version.
func smoke(ctx context.Context, x deploykit.Executor, path, dir, wantCommit string) (stamp.Identity, error) {
	out, ok := x.Run(ctx, dir, path, "version", "--json")
	if !ok {
		return stamp.Identity{}, fmt.Errorf("%w: `version --json` failed: %s", ErrProvenance, clip(out))
	}
	id, err := stamp.ParseVersionJSON([]byte(out))
	if err != nil {
		return stamp.Identity{}, fmt.Errorf("%w: %v", ErrProvenance, err)
	}
	switch fresh, cause := stamp.Explain(id.Stamp(), wantCommit); {
	case fresh == stamp.Fresh:
	case cause == stamp.CauseDiverged:
		return stamp.Identity{}, fmt.Errorf("%w: candidate reports %s, pinned %s", ErrSmokeCommitMismatch, id.Commit, wantCommit)
	default:
		return stamp.Identity{}, fmt.Errorf("%w: candidate is not a clean stamped build of %s (%s)", ErrProvenance, wantCommit, cause)
	}
	if strings.TrimSpace(id.AppVersion) == "" {
		return stamp.Identity{}, fmt.Errorf("%w: candidate reports no app_version", ErrProvenance)
	}
	return id, nil
}

// EnvExecutor is a deploykit.Executor that can also add environment entries to one command.
// BuildPinned needs it: the go children must see the pinned GOWORK and platform while the
// operator's own environment stays untouched.
type EnvExecutor interface {
	deploykit.Executor
	// RunEnv runs the command with env appended to the executor's environment; a later entry
	// for the same variable wins.
	RunEnv(ctx context.Context, dir string, env []string, name string, args ...string) (out string, ok bool)
}

// Local returns the EnvExecutor for this host. Run is deploykit.Local's; RunEnv runs the child
// with the process environment plus env, windowless on Windows the same way.
func Local() EnvExecutor { return localExecutor{deploykit.Local()} }

type localExecutor struct{ deploykit.Executor }

func (localExecutor) RunEnv(ctx context.Context, dir string, env []string, name string, args ...string) (string, bool) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// candidateName is the staging file name. Windows only executes a path with an executable
// extension, so the Windows name carries .exe.
func candidateName(goos string) string {
	if goos == "windows" {
		return "candidate.exe"
	}
	return "candidate"
}

func hashFile(path string) (string, int64, error) {
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
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

// fullCommit reports whether s is a full 40-hex Git object ID (either case).
func fullCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// validDigest reports whether s is a lowercase hex SHA-256.
func validDigest(s string) bool {
	if len(s) != sha256.Size*2 || strings.ToLower(s) != s {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

// clip bounds command output quoted into an error.
func clip(s string) string {
	s = strings.TrimSpace(s)
	const max = 2000
	if len(s) > max {
		return s[:max] + "...(truncated)"
	}
	if s == "" {
		return "(no output)"
	}
	return s
}
