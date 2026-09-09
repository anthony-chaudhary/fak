package llamacppinterop

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrInvalidStrixComparatorAuthority is returned when a zero, copied, or
	// already-closed value is used as a capability.
	ErrInvalidStrixComparatorAuthority = errors.New("invalid Strix comparator authority")
	// ErrStrixComparatorAuthorityCleanupPending means the owned child remains
	// live after a bounded pidfd teardown attempt. The same capability retains
	// its handles and may be closed again safely.
	ErrStrixComparatorAuthorityCleanupPending = errors.New("Strix comparator authority cleanup pending")
	// ErrStrixComparatorAuthorityUnsupported reports that the trusted process
	// boundary is unavailable on the current operating system.
	ErrStrixComparatorAuthorityUnsupported = errors.New("Strix comparator authority requires Linux")
)

// StrixComparatorPinnedFile declares a loader, ICD, or runtime dependency
// whose opened bytes must be mapped by the child process.
type StrixComparatorPinnedFile struct {
	Path   string
	SHA256 string
}

// StrixComparatorAuthorityOptions names the already-validated manifest and
// every file whose bytes may contribute to the comparator process. The child
// receives a fixed empty-PATH environment and runs from the literal root cwd.
type StrixComparatorAuthorityOptions struct {
	Manifest          StrixComparatorManifest
	SourceArchivePath string
	BuildManifestPath string
	ServerBinaryPath  string
	ModelPath         string
	LoaderICD         []StrixComparatorPinnedFile
	Dependencies      []StrixComparatorPinnedFile
	Arguments         []string

	// testModelSHA256 permits a tiny deterministic model fixture while the
	// production manifest remains pinned to the real model digest.
	testModelSHA256 string
	// testAfterOpen deterministically exercises replacement between the pinned
	// open and the final path continuity check. It is unavailable to external
	// callers and is never part of the authority state.
	testAfterOpen func() error
	// testDuringValidation injects a state change between the leading process
	// check and retained-handle hashes.
	testDuringValidation func(pid int)
	// testDuringStartupValidation injects a state change immediately before the
	// final startup artifact and process checks.
	testDuringStartupValidation func(pid int)
	// testPidfdSignal injects a bounded pidfd-send failure after minting.
	testPidfdSignal func(pidfd, signal int) error
}

type strixComparatorAuthorityPlatform interface {
	pid() int
	valid() bool
	close() (bool, error)
}

// StrixComparatorAuthorityCleanupError reports a rejected startup whose child
// could not yet be torn down. It owns the only pidfd and pinned handles needed
// to retry cleanup, but it is not and cannot become a valid authority.
type StrixComparatorAuthorityCleanupError struct {
	mu      sync.Mutex
	cause   error
	cleanup strixComparatorAuthorityPlatform
}

func (e *StrixComparatorAuthorityCleanupError) Error() string {
	return e.cause.Error()
}

func (e *StrixComparatorAuthorityCleanupError) Unwrap() error {
	return e.cause
}

// RetryCleanup makes one bounded attempt to terminate the rejected direct
// child and close all retained handles. It is safe to call again while the
// returned error matches ErrStrixComparatorAuthorityCleanupPending.
func (e *StrixComparatorAuthorityCleanupError) RetryCleanup() error {
	if e == nil {
		return ErrInvalidStrixComparatorAuthority
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.cleanup == nil {
		return nil
	}
	terminal, err := e.cleanup.close()
	if terminal {
		e.cleanup = nil
	}
	return err
}

// CleanupPending reports whether this error still owns live cleanup state.
func (e *StrixComparatorAuthorityCleanupError) CleanupPending() bool {
	if e == nil {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cleanup != nil
}

// StrixComparatorAuthority is an opaque, process-bound capability. Its
// self-reference deliberately makes shallow copies invalid; it cannot be
// serialized and reconstructed from claims.
type StrixComparatorAuthority struct {
	mu       *sync.Mutex
	self     *StrixComparatorAuthority
	platform strixComparatorAuthorityPlatform
}

var openStrixComparatorAuthorityPlatform = func(StrixComparatorAuthorityOptions) (*StrixComparatorAuthority, error) {
	return nil, ErrStrixComparatorAuthorityUnsupported
}

// OpenStrixComparatorAuthority opens, hashes, executes, and verifies the
// comparator files through the Linux trusted boundary. Failure returns no
// partial capability and tears down only resources created by this call.
func OpenStrixComparatorAuthority(options StrixComparatorAuthorityOptions) (*StrixComparatorAuthority, error) {
	return openStrixComparatorAuthorityPlatform(options)
}

// PID returns the verified child PID, or zero for an invalid capability.
func (a *StrixComparatorAuthority) PID() int {
	if a == nil || a.mu == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.self != a || a.platform == nil || !a.platform.valid() {
		return 0
	}
	return a.platform.pid()
}

// Valid reports whether this exact minted value still owns a live, verified
// comparator child.
func (a *StrixComparatorAuthority) Valid() bool {
	return a.PID() > 0
}

// Close terminates and reaps only the child created for this capability and
// closes its held artifact handles.
func (a *StrixComparatorAuthority) Close() error {
	if a == nil || a.mu == nil {
		return ErrInvalidStrixComparatorAuthority
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.self != a || a.platform == nil {
		return ErrInvalidStrixComparatorAuthority
	}
	terminal, err := a.platform.close()
	if terminal {
		a.platform = nil
	}
	return err
}

// MarshalJSON rejects claim-shaped copies of this live capability.
func (*StrixComparatorAuthority) MarshalJSON() ([]byte, error) {
	return nil, fmt.Errorf("%w: capability is not serializable", ErrInvalidStrixComparatorAuthority)
}

// UnmarshalJSON rejects reconstruction of this capability from plain fields.
func (*StrixComparatorAuthority) UnmarshalJSON([]byte) error {
	return fmt.Errorf("%w: capability is not deserializable", ErrInvalidStrixComparatorAuthority)
}

var _ json.Marshaler = (*StrixComparatorAuthority)(nil)
var _ json.Unmarshaler = (*StrixComparatorAuthority)(nil)
