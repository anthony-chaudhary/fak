package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// expert_warmpins.go — the online-learning resident-pin cache: a persistent per-(layer,expert) usage
// histogram that survives across runs (crash-safe dump per turn, summed at boot) to WARM-START which
// experts pagedRing pins, plus a between-turns quiescent ACTUATOR (RepinPass) that swaps the coldest
// pinned experts for the hottest recent ones under a bounded swaps-per-pass cap and a decaying heat
// model. It personalizes the pinned hot-set to the observed workload with NO offline profiling run
// (issue #4358, epic #3174 expert residency; colibri @1bdaeee c/glm.c:2152 usage-dump/summed-load +
// :1835 repin_pass, `inspire` verdict — clean-room re-implementation, no bytes vendored).
//
// Two gaps in the existing seam this closes. ExpertTraceRecorder (expert_replay.go) observes one
// routed-expert window but its Snapshot is in-memory ONLY — never persisted or reloaded, so nothing
// carries yesterday's routing into today's warm-start. And pagedRing's `pinned` is a STATIC boolean
// passed per matMul call — the resident hot-set is never re-selected from what the workload actually
// touched. This file adds the DURABLE histogram (Persist/Load/Sum) and the pin SELECTION + ACTUATOR
// (WarmStartExpertPins + RepinPass) those two gaps needed.
//
// Scope (honest, matching pagedRing's own): a STANDALONE selection/actuation layer, OFF the live serve
// path. It moves no bytes and links nothing new into the default binary. Wiring the selected pin-set
// into the live matMulStaged `pinned` decision on a real session weight HAL is the follow-on rung of
// #2726; this lands the persistence + warm-start + actuator + the two-phase drift witness that rung
// builds on. #3902 consumes the drift signal RepinPass emits.

// ExpertUsageHistogramSchema is the host-free schema tag for the persisted cross-session histogram.
const ExpertUsageHistogramSchema = "fak.expert-usage-histogram/v1"

// expertUsageKey identifies one resident expert weight. Identity is the (layer, expert) pair — expert 7
// in adjacent layers names different resident weights — the same identity ExpertAccessTraceEvent and
// pagedRing key on.
type expertUsageKey struct {
	layer  int
	expert int
}

// ExpertUsageCount is one persisted per-(layer,expert) counter row. Count is a float so the actuator's
// decaying heat (RepinPass) and the integer per-turn touch counts share one accumulator — colibri's
// histogram both SUMS across turns and DECAYS inside the actuator.
type ExpertUsageCount struct {
	Layer  int     `json:"layer"`
	Expert int     `json:"expert"`
	Count  float64 `json:"count"`
}

// expertUsageHistogramFile is the on-disk envelope: a schema tag plus the counts in deterministic
// (count desc, layer asc, expert asc) order, so a dump is byte-stable and diffable across turns.
type expertUsageHistogramFile struct {
	Schema string             `json:"schema"`
	Counts []ExpertUsageCount `json:"counts"`
}

// ExpertUsageHistogram is a per-(layer,expert) usage accumulator that persists across runs — the
// durable twin of ExpertTraceRecorder's in-memory Snapshot. The recorder observes one window; this
// counts weighted touches, dumps them crash-safely per turn, and sums the dumps at boot to warm-start
// the pinned hot-set.
type ExpertUsageHistogram struct {
	counts map[expertUsageKey]float64
}

// NewExpertUsageHistogram returns an empty histogram.
func NewExpertUsageHistogram() *ExpertUsageHistogram {
	return &ExpertUsageHistogram{counts: map[expertUsageKey]float64{}}
}

// Observe adds weight to the (layer,expert) counter. A non-positive weight or a negative layer/expert
// is ignored — the same identity guard replayEvents applies, so a malformed touch cannot poison the
// prior.
func (h *ExpertUsageHistogram) Observe(layer, expert int, weight float64) {
	if h == nil || layer < 0 || expert < 0 || weight <= 0 {
		return
	}
	h.counts[expertUsageKey{layer, expert}] += weight
}

// ObserveTrace folds one recorded trace into the histogram (+1 per event) — the bridge from
// ExpertTraceRecorder.Snapshot (a window of routed touches) to the durable counter.
func (h *ExpertUsageHistogram) ObserveTrace(trace ExpertAccessTrace) {
	if h == nil {
		return
	}
	for _, e := range trace.Events {
		h.Observe(e.Layer, e.Expert, 1)
	}
}

// Count returns the accumulated weight for (layer,expert), or 0 if never observed.
func (h *ExpertUsageHistogram) Count(layer, expert int) float64 {
	if h == nil {
		return 0
	}
	return h.counts[expertUsageKey{layer, expert}]
}

// Len is the number of distinct (layer,expert) pairs carrying a counter.
func (h *ExpertUsageHistogram) Len() int {
	if h == nil {
		return 0
	}
	return len(h.counts)
}

// Add sums other into h — the summed-at-boot fold that combines several per-turn dumps into one
// warm-start prior. A nil other is a no-op.
func (h *ExpertUsageHistogram) Add(other *ExpertUsageHistogram) {
	if h == nil || other == nil {
		return
	}
	for k, v := range other.counts {
		h.counts[k] += v
	}
}

// Decay multiplies every counter by factor (0<factor<=1) — the actuator's forgetting term, so a burst
// of yesterday's experts fades instead of pinning the cache forever. A factor outside (0,1] is ignored
// (no decay); a caller wanting a hard reset builds a fresh histogram.
func (h *ExpertUsageHistogram) Decay(factor float64) {
	if h == nil || factor <= 0 || factor > 1 {
		return
	}
	for k := range h.counts {
		h.counts[k] *= factor
	}
}

// sortedCounts returns the counters in deterministic (count desc, layer asc, expert asc) order — the
// order HotSet selects in and the order Persist serializes in, so both are reproducible.
func (h *ExpertUsageHistogram) sortedCounts() []ExpertUsageCount {
	out := make([]ExpertUsageCount, 0, len(h.counts))
	for k, v := range h.counts {
		out = append(out, ExpertUsageCount{Layer: k.layer, Expert: k.expert, Count: v})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		if out[i].Layer != out[j].Layer {
			return out[i].Layer < out[j].Layer
		}
		return out[i].Expert < out[j].Expert
	})
	return out
}

// HotSet returns the budget hottest (layer,expert) pairs, deterministically tie-broken by identity —
// the warm-start selection: the experts the ring should pin at startup given the summed prior. A
// budget <= 0 returns nil; a budget past the population returns every pair.
func (h *ExpertUsageHistogram) HotSet(budget int) []ExpertUsageCount {
	if h == nil || budget <= 0 {
		return nil
	}
	sorted := h.sortedCounts()
	if budget > len(sorted) {
		budget = len(sorted)
	}
	return append([]ExpertUsageCount(nil), sorted[:budget]...)
}

// Persist writes the histogram to path crash-safely: it marshals to a UNIQUE temp sibling in the same
// directory, fsyncs it, then atomically renames it over path — so a crash mid-write leaves either the
// intact prior file or the intact new one, never a torn dump. This is the per-turn dump colibri does at
// c/glm.c:2152. (os.Rename replaces an existing destination atomically on both POSIX and Windows.)
func (h *ExpertUsageHistogram) Persist(path string) error {
	if h == nil {
		return fmt.Errorf("model: persist nil histogram")
	}
	file := expertUsageHistogramFile{Schema: ExpertUsageHistogramSchema, Counts: h.sortedCounts()}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return fmt.Errorf("model: marshal histogram: %w", err)
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".expert-usage-*.tmp")
	if err != nil {
		return fmt.Errorf("model: histogram temp in %s: %w", dir, err)
	}
	tmpName := tmp.Name()
	renamed := false
	defer func() {
		if !renamed {
			_ = os.Remove(tmpName) // best-effort cleanup if we failed before the rename landed
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("model: write histogram: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("model: sync histogram: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("model: close histogram: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("model: rename histogram over %s: %w", path, err)
	}
	renamed = true
	return nil
}

// LoadExpertUsageHistogram reads a persisted histogram from path. A missing file is NOT an error — it
// returns an empty histogram, so a boot fold need not special-case the cold first run. A present but
// malformed or wrong-schema file IS an error: a torn or foreign file must not silently zero the prior.
func LoadExpertUsageHistogram(path string) (*ExpertUsageHistogram, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return NewExpertUsageHistogram(), nil
		}
		return nil, fmt.Errorf("model: read histogram %s: %w", path, err)
	}
	var file expertUsageHistogramFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("model: parse histogram %s: %w", path, err)
	}
	if file.Schema != ExpertUsageHistogramSchema {
		return nil, fmt.Errorf("model: histogram %s schema %q, want %q", path, file.Schema, ExpertUsageHistogramSchema)
	}
	h := NewExpertUsageHistogram()
	for _, c := range file.Counts {
		h.Observe(c.Layer, c.Expert, c.Count)
	}
	return h, nil
}

// SumExpertUsageHistograms folds every persisted dump at paths into one warm-start prior — the "summed
// at boot" step: a run may dump one file per turn (or per prior session), and boot combines them. A
// missing path contributes nothing; a malformed one errors (never silently drops history).
func SumExpertUsageHistograms(paths ...string) (*ExpertUsageHistogram, error) {
	sum := NewExpertUsageHistogram()
	for _, p := range paths {
		h, err := LoadExpertUsageHistogram(p)
		if err != nil {
			return nil, err
		}
		sum.Add(h)
	}
	return sum, nil
}

// ExpertPinSet is the resident hot-set pagedRing pins: the experts warm-started from the summed
// histogram and drifted between turns by RepinPass. It carries its own decaying heat model so the
// actuator can weigh a pinned expert's staleness against the hottest recent unpinned one.
type ExpertPinSet struct {
	budget int
	pinned map[expertUsageKey]bool
	heat   *ExpertUsageHistogram // the live, decaying heat the actuator repins against
}

// WarmStartExpertPins selects the budget hottest experts from the summed histogram as the initial
// pinned set — the startup warm-start. The histogram also seeds the heat model, so the first RepinPass
// already knows which pins are cold relative to a fresh burst. A budget <= 0 yields an empty but usable
// pin-set.
func WarmStartExpertPins(hist *ExpertUsageHistogram, budget int) *ExpertPinSet {
	p := &ExpertPinSet{budget: budget, pinned: map[expertUsageKey]bool{}, heat: NewExpertUsageHistogram()}
	p.heat.Add(hist)
	for _, c := range hist.HotSet(budget) {
		p.pinned[expertUsageKey{c.Layer, c.Expert}] = true
	}
	return p
}

// IsPinned reports whether (layer,expert) is currently pinned.
func (p *ExpertPinSet) IsPinned(layer, expert int) bool {
	return p != nil && p.pinned[expertUsageKey{layer, expert}]
}

// Len is the number of currently pinned experts.
func (p *ExpertPinSet) Len() int {
	if p == nil {
		return 0
	}
	return len(p.pinned)
}

// Pins returns the pinned experts in deterministic (layer asc, expert asc) order, each carrying its
// current heat — the snapshot a caller (and the drift witness) reads to see the personalized set.
func (p *ExpertPinSet) Pins() []ExpertUsageCount {
	if p == nil {
		return nil
	}
	out := make([]ExpertUsageCount, 0, len(p.pinned))
	for k := range p.pinned {
		out = append(out, ExpertUsageCount{Layer: k.layer, Expert: k.expert, Count: p.heat.Count(k.layer, k.expert)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Layer != out[j].Layer {
			return out[i].Layer < out[j].Layer
		}
		return out[i].Expert < out[j].Expert
	})
	return out
}

// ExpertPinSwap is one actuator swap: the cold pinned expert evicted and the hot recent expert pinned
// in its place. RepinPass returns the swaps it performed so a caller (and the two-phase witness) can
// watch the pin-set personalize to the workload.
type ExpertPinSwap struct {
	OutLayer, OutExpert int     // the coldest-pinned expert unpinned
	InLayer, InExpert   int     // the hottest-recent unpinned expert pinned in its place
	OutHeat, InHeat     float64 // their heats at the swap (InHeat strictly exceeds OutHeat)
}

// RepinPass is the between-turns quiescent actuator. It (1) DECAYS the standing heat, (2) folds the
// recent turn's usage into it, then (3) swaps up to maxSwaps of the COLDEST pinned experts for the
// HOTTEST unpinned experts whose post-fold heat STRICTLY exceeds the pinned one they would replace. The
// decay plus the bounded swaps make it an online-learning cache: it drifts toward the live workload a
// little each turn instead of reshuffling wholesale. Returns the swaps performed, coldest eviction
// first. A nil/empty recent or maxSwaps<=0 still applies the decay (heat ages) but swaps nothing.
func (p *ExpertPinSet) RepinPass(recent *ExpertUsageHistogram, decay float64, maxSwaps int) []ExpertPinSwap {
	if p == nil {
		return nil
	}
	p.heat.Decay(decay)
	p.heat.Add(recent)
	if maxSwaps <= 0 {
		return nil
	}
	// Rank pinned experts coldest-first and unpinned experts hottest-first, both by the post-fold heat
	// and tie-broken by identity, so a pass is deterministic. Both are snapshots taken before any swap,
	// so mutating p.pinned inside the loop cannot disturb the iteration or double-select a key.
	pinned := p.pinsByHeatAsc()
	candidates := p.unpinnedByHeatDesc()
	swaps := make([]ExpertPinSwap, 0, maxSwaps)
	ci := 0
	for _, cold := range pinned {
		if len(swaps) >= maxSwaps || ci >= len(candidates) {
			break
		}
		hot := candidates[ci]
		coldHeat := p.heat.Count(cold.layer, cold.expert)
		hotHeat := p.heat.Count(hot.layer, hot.expert)
		if hotHeat <= coldHeat {
			break // the hottest remaining candidate no longer beats the coldest pin — stable, stop early
		}
		delete(p.pinned, cold)
		p.pinned[hot] = true
		swaps = append(swaps, ExpertPinSwap{
			OutLayer: cold.layer, OutExpert: cold.expert, OutHeat: coldHeat,
			InLayer: hot.layer, InExpert: hot.expert, InHeat: hotHeat,
		})
		ci++
	}
	return swaps
}

// pinsByHeatAsc returns the pinned experts coldest-heat-first (ties by identity) — the actuator's
// eviction order.
func (p *ExpertPinSet) pinsByHeatAsc() []expertUsageKey {
	keys := make([]expertUsageKey, 0, len(p.pinned))
	for k := range p.pinned {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		hi, hj := p.heat.Count(keys[i].layer, keys[i].expert), p.heat.Count(keys[j].layer, keys[j].expert)
		if hi != hj {
			return hi < hj
		}
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return keys[i].expert < keys[j].expert
	})
	return keys
}

// unpinnedByHeatDesc returns every expert carrying standing heat that is NOT currently pinned, hottest
// first (ties by identity) — the actuator's admission order.
func (p *ExpertPinSet) unpinnedByHeatDesc() []expertUsageKey {
	keys := make([]expertUsageKey, 0, len(p.heat.counts))
	for k := range p.heat.counts {
		if p.pinned[k] {
			continue
		}
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		hi, hj := p.heat.Count(keys[i].layer, keys[i].expert), p.heat.Count(keys[j].layer, keys[j].expert)
		if hi != hj {
			return hi > hj
		}
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return keys[i].expert < keys[j].expert
	})
	return keys
}

// WarmStartPins gives the ring an online-learning pin-set warm-started from the summed histogram: the
// top-budget experts of the workload prior start pinned. It is the ring-side entry to the axis #4358
// targets — the ring's `pinned` boolean was static per call; now a durable, workload-personalized set
// backs it. ON the live serve path as of #5613: weightHALStagedBounded (expert_ring_hal.go:113)
// computes matMulStaged's `pinned` from this set per routed expert, so the pool's
// pinned-never-evicted invariant protects the warm hot-set. See paging_ring.go's `pins` field doc,
// which has carried the corrected wiring status since #5613 while this line still denied it.
func (r *pagedRing) WarmStartPins(hist *ExpertUsageHistogram, budget int) {
	r.pins = WarmStartExpertPins(hist, budget)
}

// RepinPass runs the between-turns actuator over the ring's pin-set (a no-op if the ring was never
// warm-started) — the pagedRing.RepinPass(maxSwaps) the issue names, carrying the recent-heat and decay
// the online cache needs.
func (r *pagedRing) RepinPass(recent *ExpertUsageHistogram, decay float64, maxSwaps int) []ExpertPinSwap {
	if r.pins == nil {
		return nil
	}
	return r.pins.RepinPass(recent, decay, maxSwaps)
}

// isExpertPinned reports whether (layer,expert) is currently in the ring's warm-started/actuated
// pin-set — the seam a future live-path caller consults to compute matMulStaged's `pinned` boolean from
// the personalized set instead of a static per-call flag.
func (r *pagedRing) isExpertPinned(layer, expert int) bool {
	return r.pins != nil && r.pins.IsPinned(layer, expert)
}
