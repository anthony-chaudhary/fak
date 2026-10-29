package compute

import (
	"sort"
	"strconv"
	"strings"
)

// MemoryPlanDiffTopN is how many positive (actual > predicted) deltas a
// MemoryPlanLedgerDiff names in TopPositive.
const MemoryPlanDiffTopN = 3

// MemoryPlanDiffRow is one ledger class: the fit planner's predicted device
// bytes against the allocation ledger's live device bytes.
type MemoryPlanDiffRow struct {
	Class          string `json:"class"`
	PredictedBytes int64  `json:"predicted_bytes"`
	ActualBytes    int64  `json:"actual_bytes"`
	DeltaBytes     int64  `json:"delta_bytes"`
}

// MemoryPlanLedgerDiff compares a device MemoryPlan to a DeviceAllocSnapshot at
// one serve phase ("load", "first_prefill"). It is observability only: no
// admission or fit decision reads it.
//
// Actual counts device-local and managed ledger cells; pinned-host cells are
// host memory and stay out. DeltaBytes = ActualBytes - PredictedBytes, so a
// positive delta is memory the planner did not predict. UnaccountedBytes is the
// snapshot's driver-used bytes the ledger cannot attribute to any class.
type MemoryPlanLedgerDiff struct {
	Phase            string              `json:"phase"`
	Rows             []MemoryPlanDiffRow `json:"rows"`
	PredictedBytes   int64               `json:"predicted_bytes"`
	ActualBytes      int64               `json:"actual_bytes"`
	DeltaBytes       int64               `json:"delta_bytes"`
	MemInfoKnown     bool                `json:"mem_info_known"`
	UsedBytes        int64               `json:"used_bytes"`
	UnaccountedBytes int64               `json:"unaccounted_bytes"`
	TopPositive      []MemoryPlanDiffRow `json:"top_positive,omitempty"`
}

// DiffMemoryPlanAgainstLedger folds plan's device-scope demands into ledger
// classes (DeviceAllocClassFor over Class + Detail) and compares them with the
// snapshot's live bytes per class. Rows are in ledger-class order and include
// every class with a nonzero prediction or actual.
func DiffMemoryPlanAgainstLedger(phase string, plan MemoryPlan, snap DeviceAllocSnapshot) MemoryPlanLedgerDiff {
	var predicted, actual [deviceAllocClassCount]int64
	for _, d := range plan {
		if d.Bytes <= 0 || d.ScopeOrDefault() != MemoryScopeDevice {
			continue
		}
		c := DeviceAllocClassFor(d.Class, d.Detail, false)
		predicted[c] = saturatingAddInt64(predicted[c], d.Bytes)
	}
	for _, cell := range snap.Cells {
		if cell.Kind == AllocPinnedHost || cell.LiveBytes <= 0 {
			continue
		}
		c := cell.Class
		if c < 0 || c >= deviceAllocClassCount {
			c = AllocOther
		}
		actual[c] = saturatingAddInt64(actual[c], cell.LiveBytes)
	}
	out := MemoryPlanLedgerDiff{
		Phase:            phase,
		MemInfoKnown:     snap.MemInfoKnown,
		UsedBytes:        snap.UsedBytes,
		UnaccountedBytes: snap.UnaccountedBytes,
	}
	for c := DeviceAllocClass(0); c < deviceAllocClassCount; c++ {
		if predicted[c] == 0 && actual[c] == 0 {
			continue
		}
		row := MemoryPlanDiffRow{
			Class:          c.String(),
			PredictedBytes: predicted[c],
			ActualBytes:    actual[c],
			DeltaBytes:     actual[c] - predicted[c],
		}
		out.Rows = append(out.Rows, row)
		out.PredictedBytes = saturatingAddInt64(out.PredictedBytes, row.PredictedBytes)
		out.ActualBytes = saturatingAddInt64(out.ActualBytes, row.ActualBytes)
	}
	out.DeltaBytes = out.ActualBytes - out.PredictedBytes
	var pos []MemoryPlanDiffRow
	for _, r := range out.Rows {
		if r.DeltaBytes > 0 {
			pos = append(pos, r)
		}
	}
	sort.SliceStable(pos, func(i, j int) bool { return pos[i].DeltaBytes > pos[j].DeltaBytes })
	if len(pos) > MemoryPlanDiffTopN {
		pos = pos[:MemoryPlanDiffTopN]
	}
	out.TopPositive = pos
	return out
}

// LogLine renders the diff as one stable-key line:
//
//	fak_mem_plan_diff phase=<p> predicted=<B> actual=<B> delta=<B> unaccounted=<B>
//	  mem_info_known=<bool> top=<class>:<delta>,... <class>=<pred>/<actual>/<delta> ...
//
// Keys and their order are fixed; byte values are decimal int64.
func (d MemoryPlanLedgerDiff) LogLine() string {
	var b strings.Builder
	b.WriteString("fak_mem_plan_diff phase=")
	phase := d.Phase
	if phase == "" {
		phase = "unknown"
	}
	b.WriteString(phase)
	kv := func(k string, v int64) {
		b.WriteByte(' ')
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(strconv.FormatInt(v, 10))
	}
	kv("predicted", d.PredictedBytes)
	kv("actual", d.ActualBytes)
	kv("delta", d.DeltaBytes)
	kv("unaccounted", d.UnaccountedBytes)
	b.WriteString(" mem_info_known=")
	b.WriteString(strconv.FormatBool(d.MemInfoKnown))
	b.WriteString(" top=")
	if len(d.TopPositive) == 0 {
		b.WriteString("none")
	}
	for i, r := range d.TopPositive {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(r.Class)
		b.WriteByte(':')
		b.WriteString(strconv.FormatInt(r.DeltaBytes, 10))
	}
	for _, r := range d.Rows {
		b.WriteByte(' ')
		b.WriteString(r.Class)
		b.WriteByte('=')
		b.WriteString(strconv.FormatInt(r.PredictedBytes, 10))
		b.WriteByte('/')
		b.WriteString(strconv.FormatInt(r.ActualBytes, 10))
		b.WriteByte('/')
		b.WriteString(strconv.FormatInt(r.DeltaBytes, 10))
	}
	return b.String()
}

// Clone returns a deep copy (nil stays nil) so a published diff never aliases
// the producer's row slices.
func (d *MemoryPlanLedgerDiff) Clone() *MemoryPlanLedgerDiff {
	if d == nil {
		return nil
	}
	out := *d
	out.Rows = append([]MemoryPlanDiffRow(nil), d.Rows...)
	out.TopPositive = append([]MemoryPlanDiffRow(nil), d.TopPositive...)
	return &out
}

// DeviceAllocLedgerOf returns b's ledger snapshot when b carries a tagged
// device-allocation ledger.
func DeviceAllocLedgerOf(b Backend) (DeviceAllocSnapshot, bool) {
	r, ok := b.(DeviceAllocLedgerReader)
	if !ok || r == nil {
		return DeviceAllocSnapshot{}, false
	}
	return r.DeviceAllocLedger(), true
}
