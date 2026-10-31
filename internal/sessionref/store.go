package sessionref

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
)

// Runner preserves leaseref's stdout/exit-code/execution-error contract.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// Store is the local descriptor writer. It does not acquire or arbitrate leases.
type Store struct {
	run        Runner
	dir        string
	probeOnce  sync.Once
	publishOff bool
}

// NewInDir uses real Git, including the cached object-DB probe and window gate.
func NewInDir(dir string) *Store { return &Store{run: GitRunner, dir: dir} }

func (s *Store) PublishSession(ctx context.Context, d SessionDescriptor) (string, error) {
	return PublishSession(ctx, d, func(ctx context.Context, ref string, v any) (string, error) {
		return PutBlobRef(ctx, s.run, s.dir, ref, v, PublishingDisabled(ctx, s.run, s.dir, true, &s.probeOnce, &s.publishOff))
	})
}

// PutBlobRef is the existing side-ref blob/update algorithm. Callers retain probe state.
func PutBlobRef(ctx context.Context, run Runner, dir, ref string, v any, disabled bool) (string, error) {
	if disabled {
		// No git object DB in s.dir (a serve running from a non-repo cwd): a blob write
		// would fail with `hash-object -w` exit 128 on EVERY call. Disable the side-ref
		// publisher CLEANLY — a best-effort caller (session_durable.publishBestEffort)
		// sees success and stops logging a per-refresh error, and the cross-machine
		// feature is simply off on a non-repo box, the honest post-state. Returns the ref
		// the caller asked for (nothing was written) so the fail-open contract holds.
		return ref, nil
	}
	blob, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("leaseref: marshal record: %w", err)
	}
	sha, err := WriteBlob(ctx, run, dir, blob)
	if err != nil {
		return "", err
	}
	if _, code, err := run(ctx, dir, "update-ref", ref, sha); err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	} else if code != 0 {
		return "", fmt.Errorf("leaseref: update-ref %s exited %d", ref, code)
	}
	return ref, nil
}

// PublishingDisabled shares the exact one-probe policy while callers retain state identity.
func PublishingDisabled(ctx context.Context, run Runner, dir string, realGit bool, probeOnce *sync.Once, publishOff *bool) bool {
	if !realGit {
		return false
	}
	probeOnce.Do(func() {
		if HasObjectDB(ctx, run, dir) {
			return
		}
		*publishOff = true
		noticeDir := dir
		if noticeDir == "" {
			if wd, err := os.Getwd(); err == nil {
				noticeDir = wd
			}
		}
		log.Printf("leaseref: no git object DB in %s; session side-ref publishing disabled", noticeDir)
	})
	return *publishOff
}

// HasObjectDB accepts bare repositories as well as worktrees.
func HasObjectDB(ctx context.Context, run Runner, dir string) bool {
	_, code, err := run(ctx, dir, "rev-parse", "--git-dir")
	return err == nil && code == 0
}

// WriteBlob preserves temporary-file cleanup and object-ID validation.
func WriteBlob(ctx context.Context, run Runner, dir string, blob []byte) (string, error) {
	f, err := os.CreateTemp("", "fak-lease-*.json")
	if err != nil {
		return "", fmt.Errorf("leaseref: temp blob: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(blob); err != nil {
		f.Close()
		return "", fmt.Errorf("leaseref: write blob: %w", err)
	}
	if err := f.Close(); err != nil {
		return "", fmt.Errorf("leaseref: close blob: %w", err)
	}
	out, code, err := run(ctx, dir, "hash-object", "-w", tmp)
	if err != nil {
		return "", fmt.Errorf("leaseref: git not executable: %w", err)
	}
	if code != 0 {
		return "", fmt.Errorf("leaseref: hash-object exited %d", code)
	}
	sha := strings.TrimSpace(out)
	if !ValidObjectID(sha) {
		return "", fmt.Errorf("leaseref: hash-object produced invalid object id %q", sha)
	}
	return sha, nil
}

// ValidObjectID accepts the same nonzero SHA-1 and SHA-256 hexadecimal object IDs.
func ValidObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	nonzero := false
	for i := range len(s) {
		c := s[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
		if c != '0' {
			nonzero = true
		}
	}
	return nonzero
}
