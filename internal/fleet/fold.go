package fleet

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// SnapshotSchema tags the folded fleet snapshot — the machine output of
// `fleetctl status --json` / `fak lab status --json`.
const SnapshotSchema = "fak.fleet.snapshot/v1"

// DefaultStaleSec is how long a box may be silent before it is worth an operator
// glance. A box whose last report is older than this is "stale" even if its last
// state word was healthy.
const DefaultStaleSec = 900 // 15 minutes

// DefaultWasteFloor is how many of a box's GPUs may sit idle before the fold raises a
// utilization crit. A box with 8 GPUs running on 1 wastes 7 — well past this — and is
// the founding case; a box that idles 1-3 GPUs is below the floor and not flagged, so
// the signal stays a real waste alarm, not noise on a lightly-loaded box.
const DefaultWasteFloor = 4

// FoldOpts tunes the fold without reaching for a clock or env — the fold stays PURE
// so the renderer, the JSON, and the tests all share one deterministic shape.
type FoldOpts struct {
	StaleSec   float64 // silence threshold; <= 0 means DefaultStaleSec
	WasteFloor int     // idle-GPU count that trips the utilization crit; <= 0 means DefaultWasteFloor
}

// Item is one ranked "what needs me now" entry — crit before warn before ok.
type Item struct {
	Level  string `json:"level"` // "crit" | "warn" | "ok"
	Title  string `json:"title"`
	Detail string `json:"detail,omitempty"`
}

// BoxRow is the per-box render record: roster identity folded with its report.
type BoxRow struct {
	ID      string    `json:"id"`
	Class   string    `json:"class,omitempty"`
	Group   string    `json:"group,omitempty"`
	State   State     `json:"state"`
	Version string    `json:"version,omitempty"`
	AgeSec  float64   `json:"age_sec,omitempty"`
	Note    string    `json:"note,omitempty"`
	GPU     *GPUStats `json:"gpu,omitempty"`
	Err     string    `json:"err,omitempty"`
}

type countRow struct {
	Key   string `json:"key"`
	Count int    `json:"count"`
}

// Snapshot is the render-ready fold of a roster and its reports. It is PURE data:
// Fold takes no clock, disk, or subprocess, so the same inputs always produce the
// same snapshot and the JSON, the render, and the tests never disagree.
type Snapshot struct {
	Schema       string        `json:"schema"`
	Total        int           `json:"total"`
	Reachable    int           `json:"reachable"`
	ByState      map[State]int `json:"by_state"`
	ByClass      []countRow    `json:"by_class"`
	Versions     []countRow    `json:"versions"`
	ModalVersion string        `json:"modal_version,omitempty"`
	Score        int           `json:"score"` // 0-100 readiness, see scoreOf
	// GPUUtil is the fleet COMPUTE-utilization aggregate over reachable boxes that
	// reported a GPU stat — Busy/Total GPUs and a token-weighted busy percent. It is
	// nil when no reachable box reported one, so a fleet with no util producers shows
	// no utilization line rather than a false 0%. Distinct from Score (readiness) and
	// from cache-tier capacity: this is "is the silicon working", not "is it up" or
	// "does the cache have room".
	GPUUtil   *GPUStats `json:"gpu_util,omitempty"`
	Attention []Item    `json:"attention"`
	Rows      []BoxRow  `json:"rows"`
}

// Fold folds a roster and its (roster-aligned) reports into a Snapshot. reports must
// be in roster order; a slice shorter than the roster is padded with unknowns so the
// fold never panics on a partial probe.
func Fold(ro Roster, reports []Report, opts FoldOpts) Snapshot {
	if opts.StaleSec <= 0 {
		opts.StaleSec = DefaultStaleSec
	}
	if opts.WasteFloor <= 0 {
		opts.WasteFloor = DefaultWasteFloor
	}
	byState := map[State]int{}
	verCount := map[string]int{}
	classCount := map[string]int{}
	rows := make([]BoxRow, len(ro.Boxes))
	reachable, healthy := 0, 0
	// GPU-util aggregate over reachable boxes that reported one. utilWeighted sums
	// per-box UtilPct*Total so the fleet percent is GPU-weighted (a busy 8-GPU box
	// counts more than a busy 1-GPU box); gpuTotal is the denominator for that mean.
	var gpuTotal, gpuBusy, utilWeighted, utilWeight int
	sawGPU := false

	for i, b := range ro.Boxes {
		r := Report{ID: b.ID, State: StateUnknown, Err: "no report"}
		if i < len(reports) {
			r = reports[i]
		}
		st := r.State
		if !st.Known() {
			st = StateUnknown
		}
		byState[st]++
		if st.Healthy() {
			healthy++
		}
		cls := b.Class
		if cls == "" {
			cls = "(unset)"
		}
		classCount[cls]++
		if r.Reachable() {
			reachable++
			if r.Version != "" {
				verCount[r.Version]++
			}
		}
		// Only a HEALTHY box's GPU reading counts: a down box is reachable (we know it
		// is down) but does no work, so its stale "8 busy" must not mask its outage as
		// utilization. Gate on Healthy(), not Reachable().
		if st.Healthy() {
			if g := r.GPU; g != nil && g.Total > 0 {
				sawGPU = true
				gpuTotal += g.Total
				gpuBusy += g.Busy
				utilWeighted += g.UtilPct * g.Total
				utilWeight += g.Total
			}
		}
		rows[i] = BoxRow{
			ID: b.ID, Class: b.Class, Group: b.Group,
			State: st, Version: r.Version, AgeSec: r.AgeSec, Note: r.Note, GPU: r.GPU, Err: r.Err,
		}
	}

	modal, modalN := modeOf(verCount)
	snap := Snapshot{
		Schema:       SnapshotSchema,
		Total:        len(ro.Boxes),
		Reachable:    reachable,
		ByState:      byState,
		ByClass:      sortedCounts(classCount),
		Versions:     sortedCounts(verCount),
		ModalVersion: modal,
		Rows:         rows,
	}
	if sawGPU {
		util := 0
		if utilWeight > 0 {
			util = int(math.Round(float64(utilWeighted) / float64(utilWeight)))
		}
		snap.GPUUtil = &GPUStats{Total: gpuTotal, Busy: gpuBusy, UtilPct: util}
	}
	snap.Score = scoreOf(snap.Total, reachable, healthy, modalN)
	snap.Attention = attentionOf(rows, modal, opts.StaleSec, opts.WasteFloor)
	return snap
}

// scoreOf is the fleet READINESS score, 0-100 — a deliberately simple, documented
// blend an operator can predict:
//
//		score = 100 * ( 0.6*usable_frac + 0.2*reach_frac + 0.2*version_coverage_frac )
//
//	  - usable_frac           — healthy boxes (live|idle|draining) / total
//	  - reach_frac            — boxes that returned a trustworthy report (incl. down) / total
//	  - version_coverage_frac — boxes on the single most common version / total
//
// USABILITY dominates (0.6): an unreachable or down box is the operator's real
// problem. REACH (0.2) gives credit for OBSERVABILITY — knowing a box is down beats
// not knowing. VERSION COVERAGE (0.2) couples consistency with reporting: a box that
// does not report a version is not "covered". So an all-healthy single-version fleet
// scores 100, an all-DOWN-but-visible fleet scores 20, and an all-SILENT fleet
// scores 0. An empty roster scores 0 (there is nothing ready). The score is a fence,
// not a benchmark — the per-state counts and the attention list carry the detail.
func scoreOf(total, reachable, healthy, modalN int) int {
	if total == 0 {
		return 0
	}
	t := float64(total)
	frac := 0.6*(float64(healthy)/t) + 0.2*(float64(reachable)/t) + 0.2*(float64(modalN)/t)
	return int(math.Round(clamp(100*frac, 0, 100)))
}

// attentionOf builds the ranked "what needs me now" list: crit (down/unreachable)
// first, then warn (version skew, stale), then a single ok line if the fleet is
// clean. Each list is capped in the rendered detail so 100 down boxes do not print
// 100 ids.
func attentionOf(rows []BoxRow, modal string, staleSec float64, wasteFloor int) []Item {
	var down, skew, stale, wasting []string
	for _, r := range rows {
		if r.Err != "" || r.State == StateDown || r.State == StateUnknown {
			down = append(down, r.ID)
			continue // a down/unreachable box is not also "skewed" or "stale" — one signal each
		}
		if r.Version != "" && modal != "" && r.Version != modal {
			skew = append(skew, r.ID+"@"+r.Version)
		}
		// staleSec is always positive here — Fold coerces a <= 0 opt to the default.
		if r.AgeSec > staleSec {
			stale = append(stale, r.ID)
		}
		// Utilization waste: a reachable box leaving wasteFloor+ of its GPUs idle is
		// leased-but-unused silicon — the founding 1/8 case. Only fires when the box
		// actually reported GPUs (nil/Total==0 is unknown-util, not idle).
		if g := r.GPU; g != nil && g.Total > 0 && (g.Total-g.Busy) >= wasteFloor {
			wasting = append(wasting, fmt.Sprintf("%s(%d/%d)", r.ID, g.Busy, g.Total))
		}
	}
	var items []Item
	if len(down) > 0 {
		items = append(items, Item{
			Level:  "crit",
			Title:  fmt.Sprintf("%d box(es) down or unreachable", len(down)),
			Detail: previewList(down, 6),
		})
	}
	if len(wasting) > 0 {
		items = append(items, Item{
			Level:  "crit",
			Title:  fmt.Sprintf("%d box(es) wasting >=%d GPUs", len(wasting), wasteFloor),
			Detail: previewList(wasting, 6),
		})
	}
	if len(skew) > 0 {
		items = append(items, Item{
			Level:  "warn",
			Title:  fmt.Sprintf("%d box(es) off the fleet version %s", len(skew), modal),
			Detail: previewList(skew, 6),
		})
	}
	if len(stale) > 0 {
		items = append(items, Item{
			Level:  "warn",
			Title:  fmt.Sprintf("%d box(es) silent > %sm", len(stale), strconv.FormatFloat(staleSec/60, 'g', -1, 64)),
			Detail: previewList(stale, 6),
		})
	}
	if len(items) == 0 {
		items = append(items, Item{Level: "ok", Title: "fleet is healthy — every box reachable, one version, none stale"})
	}
	return items
}

// modeOf returns the most common key and its count, deterministically: highest
// count wins, ties break to the lexically smallest key. An empty map returns ("", 0).
// Callers pass only non-empty keys with counts >= 1, so the first key is always taken
// by the n > bestN arm and `k < best` is the sole tie-break.
func modeOf(m map[string]int) (string, int) {
	best, bestN := "", 0
	for k, n := range m {
		if n > bestN || (n == bestN && k < best) {
			best, bestN = k, n
		}
	}
	return best, bestN
}

// sortedCounts renders a count map as rows ordered by descending count then key, so
// the output is stable across runs.
func sortedCounts(m map[string]int) []countRow {
	rows := make([]countRow, 0, len(m))
	for k, n := range m {
		rows = append(rows, countRow{Key: k, Count: n})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Count != rows[j].Count {
			return rows[i].Count > rows[j].Count
		}
		return rows[i].Key < rows[j].Key
	})
	return rows
}

// previewList joins up to max ids, then summarizes the remainder as "(+k more)" so a
// large fleet's attention detail stays one bounded line.
func previewList(xs []string, max int) string {
	if len(xs) <= max {
		return strings.Join(xs, ", ")
	}
	return strings.Join(xs[:max], ", ") + fmt.Sprintf(" (+%d more)", len(xs)-max)
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}
