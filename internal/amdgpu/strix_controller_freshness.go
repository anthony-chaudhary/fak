package amdgpu

import (
	"context"
	"errors"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

const (
	strixControllerStaleToken       = "STALE_VALIDATOR_BINARY"
	strixControllerUnsupportedToken = "UNSUPPORTED_VALIDATOR_BINARY"
	strixControllerWSLRecovery      = "use a current stamped WSL build with /proc/self/exe authority"
)

// StrixControllerAuthority is an opaque proof that the mapped controller
// executable carries semantics at or after the fixed Strix epoch. Its zero
// value and any value missing the private seal expose no evidence.
type StrixControllerAuthority struct {
	evidence strixControllerAuthorityEvidence
	seal     *strixControllerAuthoritySeal
}

type strixControllerAuthoritySeal struct {
	marker byte
}

var strixControllerAuthoritySealValue = strixControllerAuthoritySeal{marker: 0xa5}

// Epoch returns the fixed semantic epoch bound by this authority.
func (a StrixControllerAuthority) Epoch() string {
	if !a.valid() {
		return ""
	}
	return a.evidence.epoch
}

// ObservedRevision returns the full revision observed from the mapped binary.
func (a StrixControllerAuthority) ObservedRevision() string {
	if !a.valid() {
		return ""
	}
	return a.evidence.revision
}

// ExecutableSHA256 returns the digest read from the mapped executable handle.
func (a StrixControllerAuthority) ExecutableSHA256() string {
	if !a.valid() {
		return ""
	}
	return a.evidence.binarySHA256
}

// ExecutableBytes returns the executable byte count covered by the digest.
func (a StrixControllerAuthority) ExecutableBytes() int64 {
	if !a.valid() {
		return 0
	}
	return a.evidence.binaryBytes
}

// AdmittedAt returns when classification and snapshot cleanup both succeeded.
func (a StrixControllerAuthority) AdmittedAt() time.Time {
	if !a.valid() {
		return time.Time{}
	}
	return a.evidence.admittedAt
}

func (a StrixControllerAuthority) valid() bool {
	return a.seal == &strixControllerAuthoritySealValue && validStrixControllerEvidence(a.evidence)
}

// StrixControllerAuthorityRefusal is a typed, path-redacted local refusal.
// When snapshot cleanup itself failed, it retains opaque ownership so the
// caller can retry without learning or supplying the private path.
type StrixControllerAuthorityRefusal struct {
	code     string
	recovery string
	cleanup  *strixGitObjectSnapshot
}

func (r *StrixControllerAuthorityRefusal) Error() string {
	if r == nil {
		return ""
	}
	return r.code + ": Strix controller authority refused"
}

// Code returns the stable refusal token.
func (r *StrixControllerAuthorityRefusal) Code() string {
	if r == nil {
		return ""
	}
	return r.code
}

// Recovery returns a redacted operator action when one is available.
func (r *StrixControllerAuthorityRefusal) Recovery() string {
	if r == nil {
		return ""
	}
	return r.recovery
}

// RetryCleanup retries destruction of a retained private snapshot. It never
// exposes the snapshot path or the raw cleanup error.
func (r *StrixControllerAuthorityRefusal) RetryCleanup() error {
	if r == nil || r.cleanup == nil || r.cleanup.gitDir() == "" {
		return nil
	}
	if err := r.cleanup.close(); err != nil {
		return &strixGitSnapshotCleanupError{}
	}
	return nil
}

func newStrixControllerAuthorityRefusal(code string, _ error, recovery string) *StrixControllerAuthorityRefusal {
	return &StrixControllerAuthorityRefusal{code: code, recovery: recovery}
}

func strixControllerObservationRefusal(goos string, cause error) error {
	if goos == "windows" {
		return newStrixControllerAuthorityRefusal(strixControllerUnsupportedToken, cause, strixControllerWSLRecovery)
	}
	return newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, cause, "")
}

// refuseStrixControllerWithSnapshot makes one cleanup attempt and retains
// opaque ownership in the refusal only when retry is still required.
func refuseStrixControllerWithSnapshot(cause error, snapshot *strixGitObjectSnapshot) error {
	refusal := newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, cause, "")
	if snapshot != nil && snapshot.gitDir() != "" {
		if err := snapshot.close(); err != nil {
			refusal.cleanup = snapshot
		}
	}
	return refusal
}

type strixControllerProvenance struct {
	revision     string
	dirty        bool
	binarySHA256 string
	binaryBytes  int64
}

type strixControllerAuthorityEvidence struct {
	epoch        string
	revision     string
	binarySHA256 string
	binaryBytes  int64
	admittedAt   time.Time
}

type strixControllerAuthorityDependencies struct {
	observe  func() (strixControllerProvenance, error)
	snapshot func(context.Context, string) (*strixGitObjectSnapshot, error)
	classify func(context.Context, *strixGitObjectSnapshot, string) (strixGitAncestry, error)
	clock    func() time.Time
	goos     string
}

func productionStrixControllerAuthorityDependencies() strixControllerAuthorityDependencies {
	return strixControllerAuthorityDependencies{
		observe:  observeStrixControllerProvenance,
		snapshot: newStrixGitObjectSnapshot,
		classify: classifyStrixGitSnapshotAncestry,
		clock:    time.Now,
		goos:     runtime.GOOS,
	}
}

func observeStrixControllerProvenance() (strixControllerProvenance, error) {
	observed, err := binstamp.ObserveExecutableProvenance()
	if err != nil {
		return strixControllerProvenance{}, err
	}
	return strixControllerProvenance{
		revision:     observed.Revision(),
		dirty:        observed.Dirty(),
		binarySHA256: observed.BinarySHA256(),
		binaryBytes:  observed.BinaryBytes(),
	}, nil
}

// NewStrixControllerAuthority is the only exported authority constructor. It
// observes the running mapped executable directly, freezes the repository
// object view, and delegates all ancestry to the sterile-snapshot classifier.
// It contains no transport path and accepts no caller-authored identity fact.
func NewStrixControllerAuthority(ctx context.Context, resolvedRepositoryRoot string) (StrixControllerAuthority, error) {
	return newStrixControllerAuthorityWith(ctx, resolvedRepositoryRoot, productionStrixControllerAuthorityDependencies())
}

func newStrixControllerAuthorityWith(ctx context.Context, resolvedRepositoryRoot string, deps strixControllerAuthorityDependencies) (StrixControllerAuthority, error) {
	if ctx == nil {
		return StrixControllerAuthority{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, errors.New("nil context"), "")
	}
	if deps.observe == nil || deps.snapshot == nil || deps.classify == nil || deps.clock == nil || deps.goos == "" {
		return StrixControllerAuthority{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, errors.New("invalid authority dependencies"), "")
	}

	provenance, err := deps.observe()
	if err != nil {
		return StrixControllerAuthority{}, strixControllerObservationRefusal(deps.goos, err)
	}
	if err := validateStrixControllerProvenance(provenance); err != nil {
		return StrixControllerAuthority{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, err, "")
	}

	snapshot, err := deps.snapshot(ctx, resolvedRepositoryRoot)
	if err != nil || snapshot == nil {
		return StrixControllerAuthority{}, refuseStrixControllerWithSnapshot(err, snapshot)
	}
	ancestry, ancestryErr := deps.classify(ctx, snapshot, provenance.revision)
	if ancestryErr != nil || snapshot.gitDir() != "" {
		return StrixControllerAuthority{}, refuseStrixControllerWithSnapshot(ancestryErr, snapshot)
	}

	// Admission time is sampled only after authenticated classification and
	// successful destruction of the consumed private snapshot.
	evidence, err := evaluateStrixControllerAuthority(provenance, ancestry, nil, deps.clock().UTC())
	if err != nil {
		return StrixControllerAuthority{}, err
	}
	authority := StrixControllerAuthority{evidence: evidence, seal: &strixControllerAuthoritySealValue}
	if !authority.valid() {
		return StrixControllerAuthority{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, errors.New("invalid authority evidence"), "")
	}
	return authority, nil
}

// evaluateStrixControllerAuthority is a pure, non-minting join used by the focused
// witness. It returns private evidence, never production authority.
func evaluateStrixControllerAuthority(provenance strixControllerProvenance, ancestry strixGitAncestry, ancestryErr error, admittedAt time.Time) (strixControllerAuthorityEvidence, error) {
	if err := validateStrixControllerProvenance(provenance); err != nil {
		return strixControllerAuthorityEvidence{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, err, "")
	}
	if ancestryErr != nil {
		return strixControllerAuthorityEvidence{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, ancestryErr, "")
	}
	switch ancestry {
	case strixGitAncestryEpochEqual, strixGitAncestryDescendant:
		evidence := strixControllerAuthorityEvidence{
			epoch:        strixGitSemanticEpoch,
			revision:     provenance.revision,
			binarySHA256: provenance.binarySHA256,
			binaryBytes:  provenance.binaryBytes,
			admittedAt:   admittedAt.UTC(),
		}
		if !validStrixControllerEvidence(evidence) {
			return strixControllerAuthorityEvidence{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, errors.New("invalid authority evidence"), "")
		}
		return evidence, nil
	case strixGitAncestryPreEpoch:
		return strixControllerAuthorityEvidence{}, newStrixControllerAuthorityRefusal(strixControllerStaleToken, nil, "")
	default:
		return strixControllerAuthorityEvidence{}, newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, nil, "")
	}
}

func validateStrixControllerProvenance(provenance strixControllerProvenance) error {
	if provenance.dirty || !isLowerHex(provenance.revision, 40) {
		return errors.New("invalid executable revision")
	}
	if !isLowerHex(provenance.binarySHA256, 64) || provenance.binaryBytes <= 0 {
		return errors.New("invalid executable digest")
	}
	return nil
}

func validStrixControllerEvidence(evidence strixControllerAuthorityEvidence) bool {
	return evidence.epoch == strixGitSemanticEpoch &&
		isLowerHex(evidence.revision, 40) &&
		isLowerHex(evidence.binarySHA256, 64) &&
		evidence.binaryBytes > 0 &&
		!evidence.admittedAt.IsZero() &&
		evidence.admittedAt.Location() == time.UTC
}
