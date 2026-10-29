package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Local-SSD KV overflow tier defaults (perf-parity T-06 / umem TICKET-04): with no operator
// byte count, evicted host-L2 prefix snapshots demote to local disk up to the smaller of
// inKernelRadixLocalSSDCap and this fraction of the free bytes on the store's volume, so a
// high-RAM node keeps its cold prefixes restorable instead of paying a full cold prefill.
// The reserve is a ceiling on REFERENCED bytes, not an allocation: the tier holds nothing
// until L2 pressure actually evicts a staged image into it.
const (
	inKernelRadixLocalSSDBytesEnv          = "FAK_INKERNEL_RADIX_SSD_BYTES"
	inKernelRadixLocalSSDFractionDef       = 0.25
	inKernelRadixLocalSSDCap         int64 = 64 << 30 // 64 GiB
	// inKernelRadixLocalSSDMaxEntries bounds the record COUNT the store owner prunes to after
	// each spill, independent of the byte cap, so a pathological run of tiny snapshots cannot
	// fragment the volume past a bounded directory scan.
	inKernelRadixLocalSSDMaxEntries = 1024
)

// Sources of a resolved local SSD overflow budget, printed at planner construction.
const (
	InKernelRadixLocalSSDSourceEnv     = "env"
	InKernelRadixLocalSSDSourceDefault = "default min(64 GiB, 25% free on volume)"
	InKernelRadixLocalSSDSourceUnknown = "unknown disk capacity"
	InKernelRadixLocalSSDSourceRefused = "refused"
)

// ErrInKernelRadixLocalSSDConfig marks a local SSD env value the resolver refused; the tier
// then fails closed to off (0 bytes), today's behaviour.
var ErrInKernelRadixLocalSSDConfig = errors.New("fak inkernel: invalid radix local SSD KV overflow")

// InKernelRadixLocalSSD is the resolved local-SSD KV overflow budget, its store directory,
// and where the budget came from.
type InKernelRadixLocalSSD struct {
	Bytes  int64
	Dir    string
	Source string
	Err    error
}

// localSSDDir picks the overflow store's directory: the FAK_L3_KVBACKEND root when set to a
// usable directory (sibling of the warm-disk native-prefix slot, never inside it, so the two
// tiers' prune policies cannot delete each other's records), otherwise the per-user cache
// default. It mirrors resolveWarmDiskConfig's root handling.
func localSSDDir() string {
	root := strings.TrimSpace(os.Getenv(l3kv.EnvSpec))
	if root != "" && !strings.EqualFold(root, "off") {
		return filepath.Join(root, "ssd-l3")
	}
	cacheRoot, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cacheRoot, "fak", "ssd-l3")
}

// resolveInKernelRadixLocalSSD sizes the radix tree's local-SSD overflow tier:
//
//  1. FAK_INKERNEL_RADIX_SSD_BYTES set (any value, including 0 = the OFF arm) is honoured
//     exactly; an unparseable or negative value refuses and fails closed to off.
//  2. Otherwise, with the store volume's free bytes known, min(cap, fraction x free) rounded
//     down to MiB.
//  3. Unknown disk capacity: off, today's default.
//
// lookupEnv and diskInfo are injected so the derivation is unit-testable with simulated
// volumes; production passes os.LookupEnv and compute.DiskInfo.
func resolveInKernelRadixLocalSSD(lookupEnv func(string) (string, bool), diskInfo func(string) (int64, int64, bool)) InKernelRadixLocalSSD {
	dir := localSSDDir()
	if lookupEnv != nil {
		if v, ok := lookupEnv(inKernelRadixLocalSSDBytesEnv); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil || n < 0 {
				return InKernelRadixLocalSSD{Source: InKernelRadixLocalSSDSourceRefused, Err: fmt.Errorf("%w: %s=%q: want a non-negative byte count", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDBytesEnv, v)}
			}
			return InKernelRadixLocalSSD{Bytes: n, Dir: dir, Source: InKernelRadixLocalSSDSourceEnv}
		}
	}
	if diskInfo == nil {
		return InKernelRadixLocalSSD{Dir: dir, Source: InKernelRadixLocalSSDSourceUnknown}
	}
	_, free, known := diskInfo(dir)
	if !known || free <= 0 {
		return InKernelRadixLocalSSD{Dir: dir, Source: InKernelRadixLocalSSDSourceUnknown}
	}
	budget := int64(float64(free) * inKernelRadixLocalSSDFractionDef)
	if budget > inKernelRadixLocalSSDCap {
		budget = inKernelRadixLocalSSDCap
	}
	budget &^= 1<<20 - 1
	return InKernelRadixLocalSSD{Bytes: budget, Dir: dir, Source: InKernelRadixLocalSSDSourceDefault}
}

// armRadixLocalSSD installs the local-SSD KV overflow tier on the planner's radix tree at
// construction, so both `fak serve` and turnkey `fak up` demote evicted host-L2 prefixes to
// local disk instead of dropping them. CPU-only planners (backend == nil) stay off: their
// snapshots have no device install target to restore into, exactly like the remote L3 tier.
// A resolution refusal or an unopenable store prints and leaves the tier off; the host L2
// tier is unaffected either way.
func (p *InKernelPlanner) armRadixLocalSSD(m *model.Model, backend compute.Backend, modelID string) {
	if p == nil || p.tree == nil || m == nil || backend == nil {
		return
	}
	ssd := resolveInKernelRadixLocalSSD(os.LookupEnv, compute.DiskInfo)
	if ssd.Err != nil {
		fmt.Fprintln(os.Stderr, ssd.Err)
	}
	p.radixLocalSSD = ssd
	if ssd.Bytes <= 0 || ssd.Dir == "" {
		return
	}
	store, err := l3kv.NewDiskStore(ssd.Dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak inkernel: local SSD L3 KV overflow unavailable: %v\n", err)
		p.radixLocalSSD.Bytes = 0
		return
	}
	if err := p.tree.ConfigureLocalSSDSnapshotStore(newLocalSSDSnapshotStore(store, ssd.Bytes), modelID, backend, m.Cfg, ssd.Bytes); err != nil {
		fmt.Fprintln(os.Stderr, err)
		p.radixLocalSSD.Bytes = 0
		return
	}
	fmt.Fprintf(os.Stderr, "fak inkernel: radix local-SSD L3 KV overflow=%d bytes dir=%s source=%s\n", ssd.Bytes, ssd.Dir, ssd.Source)
}

// RadixLocalSSD reports the local-SSD KV overflow budget this planner resolved at
// construction; the zero value when the tier is off.
func (p *InKernelPlanner) RadixLocalSSD() InKernelRadixLocalSSD {
	if p == nil {
		return InKernelRadixLocalSSD{}
	}
	return p.radixLocalSSD
}

// localSSDSnapshotStore is the SnapshotStore adapter that owns the overflow tier's disk
// hygiene: every confirmed spill re-prunes the store to the byte cap and the record-count
// cap, so records left behind by earlier incarnations (the tree's own accounting restarts
// at zero) cannot grow the volume past the configured budget.
type localSSDSnapshotStore struct {
	disk   *l3kv.DiskStore
	maxB   int64
	maxEnt int
}

func newLocalSSDSnapshotStore(disk *l3kv.DiskStore, budget int64) *localSSDSnapshotStore {
	return &localSSDSnapshotStore{disk: disk, maxB: budget, maxEnt: inKernelRadixLocalSSDMaxEntries}
}

func (s *localSSDSnapshotStore) Put(ctx context.Context, key string, value []byte) error {
	if err := s.disk.Put(ctx, key, value); err != nil {
		return err
	}
	// Best-effort hygiene: a prune failure leaves the record in place (the tree-side
	// budget already refused the spill if the referenced total was over the cap).
	_ = s.disk.Prune(s.maxEnt, s.maxB, key)
	return nil
}

func (s *localSSDSnapshotStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.disk.Get(ctx, key)
}

var _ radixkv.SnapshotStore = (*localSSDSnapshotStore)(nil)
