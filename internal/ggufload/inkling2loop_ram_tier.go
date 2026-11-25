package ggufload

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2RAMBudgetEnv caps the bytes of cold expert stacks prefaulted into
// host RAM, overriding the available-memory probe. 0 disables the RAM tier.
const inkling2RAMBudgetEnv = "FAK_INKLING2_RAM_BUDGET_BYTES"

// The RAM tier leaves max(inkling2RAMHeadroomMin, inkling2RAMHeadroomPct of
// total) unclaimed so the OS, KV cache and peers keep room; the floor matters
// on small hosts, the percentage on large ones.
const (
	inkling2RAMHeadroomMin = int64(4) << 30
	inkling2RAMHeadroomPct = 10
)

// inkling2HostMemoryProbe reports host total and AVAILABLE (not total free
// of reclaim) bytes; a var so tests can fake the host.
var inkling2HostMemoryProbe = compute.HostSystemMemoryInfo

// inkling2RAMMlockEnv=1 pins RAM-tier pages with mlock so reclaim cannot push
// them back to NVMe; best effort, it needs RLIMIT_MEMLOCK or CAP_IPC_LOCK.
const inkling2RAMMlockEnv = "FAK_INKLING2_RAM_MLOCK"

// inkling2RAMLog receives RAM-tier warnings; a var so tests can silence it.
var inkling2RAMLog io.Writer = os.Stderr

// inkling2RAMBudgetInfo is the RAM tier's budget and the probe it came from.
// Budget is 0 when the host could not be probed: without a reading the tier
// stays empty and every cold stack streams from the mapping, as before.
// CgroupClamped means Available is a cgroup's headroom, not the host's.
type inkling2RAMBudgetInfo struct {
	Budget, Total, Available       int64
	Known, Override, CgroupClamped bool
}

func inkling2RAMHeadroom(total int64) int64 {
	return max(inkling2RAMHeadroomMin, total*inkling2RAMHeadroomPct/100)
}

// inkling2RAMBudget reads the live available host memory each call, so adding
// RAM is picked up at the next load with no configuration.
func inkling2RAMBudget(probe func() (int64, int64, bool), lookup func(string) (string, bool)) (inkling2RAMBudgetInfo, error) {
	var info inkling2RAMBudgetInfo
	if probe != nil {
		total, avail, known := probe()
		if known && avail >= 0 && avail != compute.FreeUnknown {
			info.Total, info.Available, info.Known = total, avail, true
			info.Budget = max(0, avail-inkling2RAMHeadroom(total))
		}
	}
	if v, ok := lookup(inkling2RAMBudgetEnv); ok && strings.TrimSpace(v) != "" {
		n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil || n < 0 {
			return info, fmt.Errorf("inkling2: %s=%q is not a non-negative byte count", inkling2RAMBudgetEnv, v)
		}
		info.Budget, info.Override = n, true
	}
	return info, nil
}

// inkling2ApplyCgroupClamp handles a cgroup-clamped probe. Host total would
// size the 10% headroom off memory the cgroup can never claim (a 2 TB host
// under a 64 GiB limit would get no tier at all), so Total is clamped to
// Available, the only figure known to be inside the limit. The need is then
// reported as the plain shortfall: the cure is raising memory.max, not RAM.
func inkling2ApplyCgroupClamp(info inkling2RAMBudgetInfo, clamped bool) inkling2RAMBudgetInfo {
	if !clamped || !info.Known {
		return info
	}
	info.CgroupClamped = true
	info.Total = min(info.Total, info.Available)
	if !info.Override {
		info.Budget = max(0, info.Available-inkling2RAMHeadroom(info.Total))
	}
	return info
}

// inkling2RAMNeededForAll is the extra installed RAM that would let the tier
// hold need bytes. Headroom grows with total, so the 10% branch needs
// need/0.9 more; an override, unknown host or cgroup limit can only report
// the shortfall.
func inkling2RAMNeededForAll(info inkling2RAMBudgetInfo, need int64) int64 {
	if need <= info.Budget {
		return 0
	}
	if info.Override || !info.Known || info.CgroupClamped {
		return need - info.Budget
	}
	floor := need + inkling2RAMHeadroomMin - info.Available
	pct := (need*100 + info.Total*inkling2RAMHeadroomPct - info.Available*100 + 89) / 90
	return max(0, floor, pct)
}

// inkling2PlanRAMTier places cold (non-resident) layers' expert stacks in the
// RAM tier, in residency order, while they fit the budget; a layer that does
// not fit is skipped so later, smaller layers may still fit. The rest stay on
// the mapping (NVMe tier). It fills the RAM fields of p.
func inkling2PlanRAMTier(order []int, resident []bool, hostLayerBytes []int64, info inkling2RAMBudgetInfo, p *Inkling2DevicePlacement) []bool {
	inRAM := make([]bool, len(hostLayerBytes))
	p.RAMBudgetBytes, p.RAMAvailableBytes = info.Budget, -1
	if info.Known {
		p.RAMAvailableBytes = info.Available
	}
	var cold int64
	for _, i := range order {
		b := hostLayerBytes[i]
		if b == 0 || (resident != nil && resident[i]) {
			continue
		}
		cold += b
		if p.RAMBytes+b <= info.Budget {
			inRAM[i] = true
			p.RAMBytes += b
			p.RAMLayers++
			continue
		}
		p.NVMeBytes += b
		p.NVMeLayers++
	}
	p.RAMNeededForAllExpertsBytes = inkling2RAMNeededForAll(info, cold)
	return inRAM
}

func inkling2HostExpertsBytes(e model.Inkling2LoopExperts) int64 {
	if q, ok := e.(*inkling2QuantExperts); ok {
		return int64(len(q.raw))
	}
	return 0
}

// applyInkling2RAMTier plans the RAM tier for m's cold expert stacks and
// returns their mapped regions. It starts nothing: the caller prefetches only
// after the resident uploads, so cold reads do not compete with them on NVMe.
func applyInkling2RAMTier(m *Inkling2LoopModel, resident []bool, p *Inkling2DevicePlacement) ([][]byte, error) {
	info, err := inkling2RAMBudget(inkling2HostMemoryProbe, os.LookupEnv)
	if err != nil {
		return nil, err
	}
	info = inkling2ApplyCgroupClamp(info, info.Known && inkling2CgroupClamped(info.Available))
	w := m.Weights
	hostBytes := make([]int64, len(w.Layers))
	for i := range w.Layers {
		hostBytes[i] = inkling2HostExpertsBytes(w.Layers[i].GateUp) + inkling2HostExpertsBytes(w.Layers[i].Down)
	}
	order := inkling2ExpertResidencyOrder(m.HP, len(w.Layers))
	inRAM := inkling2PlanRAMTier(order, resident, hostBytes, info, p)
	var regions [][]byte
	for _, i := range order {
		if !inRAM[i] {
			continue
		}
		for _, e := range []model.Inkling2LoopExperts{w.Layers[i].GateUp, w.Layers[i].Down} {
			if q, ok := e.(*inkling2QuantExperts); ok && len(q.raw) > 0 {
				regions = append(regions, q.raw)
			}
		}
	}
	return regions, nil
}

// inkling2StartRAMPrefetch is a var so tests can observe when bind starts it.
var inkling2StartRAMPrefetch = startInkling2RAMPrefetch

// inkling2RAMPrefetch faults RAM-tier pages in from a background goroutine, so
// the first decode step does not stall on NVMe: a readahead hint for the whole
// set first (deep NVMe queue), then mlock or a one-byte-per-page touch. It
// works in place and never copies the weights onto the Go heap.
type inkling2RAMPrefetch struct {
	stop    atomic.Bool
	done    chan struct{}
	touched atomic.Int64
	next    io.Closer
	once    sync.Once
}

// inkling2PrefetchSink keeps the page reads observable to the compiler.
var inkling2PrefetchSink atomic.Uint64

func startInkling2RAMPrefetch(regions [][]byte, next io.Closer) *inkling2RAMPrefetch {
	p := &inkling2RAMPrefetch{done: make(chan struct{}), next: next}
	go func() {
		defer close(p.done)
		inkling2RAMAdvise(regions)
		if v, _ := os.LookupEnv(inkling2RAMMlockEnv); strings.TrimSpace(v) == "1" && p.lock(regions) {
			return
		}
		p.run(regions)
	}()
	return p
}

func (p *inkling2RAMPrefetch) run(regions [][]byte) {
	page := os.Getpagesize()
	const chunk = 1 << 20
	var sum uint64
	for _, r := range regions {
		for off := 0; off < len(r); off += chunk {
			if p.stop.Load() {
				inkling2PrefetchSink.Add(sum)
				return
			}
			end := min(off+chunk, len(r))
			for i := off; i < end; i += page {
				sum += uint64(r[i])
			}
			p.touched.Add(int64(end - off))
		}
	}
	inkling2PrefetchSink.Add(sum)
}

// lock mlocks the regions chunk by chunk, so Close need not wait out one huge
// call. On any failure it logs, keeps what it locked and reports false so run
// still faults the rest in.
func (p *inkling2RAMPrefetch) lock(regions [][]byte) bool {
	const chunk = 64 << 20
	for _, r := range regions {
		for off := 0; off < len(r); off += chunk {
			if p.stop.Load() {
				return true
			}
			end := min(off+chunk, len(r))
			if err := inkling2RAMMlock(r[off:end]); err != nil {
				fmt.Fprintf(inkling2RAMLog, "inkling2: %s=1: mlock after %d bytes: %v; continuing with page-touch prefetch\n",
					inkling2RAMMlockEnv, p.touched.Load(), err)
				p.touched.Store(0)
				return false
			}
			p.touched.Add(int64(end - off))
		}
	}
	return true
}

// Close stops the prefetch and waits for it before closing the mapping.
func (p *inkling2RAMPrefetch) Close() error {
	p.once.Do(func() {
		p.stop.Store(true)
		<-p.done
	})
	if p.next != nil {
		return p.next.Close()
	}
	return nil
}
