package storedrv

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/blobfs"
)

// SpillSharder is the validated multi-root spill policy the disaggregated store was
// missing: ONE object that owns a fixed, ordered LIST of storage roots and hands each
// worker a single, deterministic root by worker_id % len(roots). blobfs already shards
// a store across dirs — but by CONTENT HASH within ONE root (store.go pathFor), so KV
// spill from every worker piles onto the same mount. This spreads spill ACROSS mounts:
// worker 0 -> roots[0], worker 1 -> roots[1], ... worker N -> roots[N%len], so a fleet
// with M disks fans its durable writes over all M instead of hammering one.
//
// Two properties make it a policy object and not just a helper:
//   - VALIDATED at construction: an empty list, a blank root, or a duplicate root is a
//     hard error, so a misconfigured spill can never silently collapse two workers onto
//     one mount (or none).
//   - EAGER creation: every root is os.MkdirAll'd once at construction, not lazily on
//     first write. blobfs creates its shard dirs lazily per-write (store.go writeBlob);
//     that is fine WITHIN a root, but a spill ROOT that does not exist until the first
//     large payload would surface a mkdir failure deep inside a hot Put. Failing fast at
//     construction turns "this mount is missing" into a boot error, not a runtime spill loss.
type SpillSharder struct {
	roots []string
}

// NewSpillSharder validates roots and EAGERLY creates every one (os.MkdirAll) so a
// worker's assigned mount is guaranteed to exist before the first spill write. Blank
// entries are dropped (a trailing comma in "disk:/a,/b," is not an error); a duplicate
// root — which would fold two workers onto the same mount, defeating the fan — is a hard
// error, as is an empty resulting list. Roots are cleaned (filepath.Clean) so "/a" and
// "/a/" collapse to one entry and the pick is stable regardless of trailing slashes.
func NewSpillSharder(roots []string) (*SpillSharder, error) {
	cleaned := make([]string, 0, len(roots))
	seen := make(map[string]struct{}, len(roots))
	for _, r := range roots {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		abs := filepath.Clean(r)
		if _, dup := seen[abs]; dup {
			return nil, fmt.Errorf("storedrv: spill sharder: duplicate root %q", abs)
		}
		seen[abs] = struct{}{}
		cleaned = append(cleaned, abs)
	}
	if len(cleaned) == 0 {
		return nil, fmt.Errorf("storedrv: spill sharder needs at least one root")
	}
	for _, r := range cleaned {
		if err := os.MkdirAll(r, 0o755); err != nil {
			return nil, fmt.Errorf("storedrv: spill sharder: mkdir root %q: %w", r, err)
		}
	}
	return &SpillSharder{roots: cleaned}, nil
}

// Len reports the number of validated roots (the modulus of the worker->root pick).
func (s *SpillSharder) Len() int { return len(s.roots) }

// Roots returns a copy of the validated, ordered root list (defensive: the caller
// cannot mutate the sharder's slice through the return value).
func (s *SpillSharder) Roots() []string { return append([]string(nil), s.roots...) }

// Root returns the storage root assigned to workerID: roots[worker_id % len(roots)].
// A negative workerID is folded back into range so the pick is always a valid index
// (Go's % keeps the sign of the dividend, which would panic as a slice index).
func (s *SpillSharder) Root(workerID int) string {
	n := len(s.roots)
	i := workerID % n
	if i < 0 {
		i += n
	}
	return s.roots[i]
}

// DiskDriver builds a durable blobfs disk Driver rooted at the shard assigned to
// workerID — the optional fan that wires the sharder onto storedrv's disk tier so KV
// spill for this worker lands on its own mount. The returned Driver is the same
// content-addressed blobfs store the single-path disk tier uses; only the root differs.
func (s *SpillSharder) DiskDriver(workerID int) (Driver, error) {
	root := s.Root(workerID)
	st, err := blobfs.New(root)
	if err != nil {
		return nil, fmt.Errorf("storedrv: spill sharder: disk driver at %q: %w", root, err)
	}
	return diskDriver{st}, nil
}
