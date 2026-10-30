package model

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ResidentBytesRatioAlarm is the resident/file weight-byte ratio above which a load is
// DEGRADED: the weights were widened (requantized) on the host instead of running on a
// native kernel at their file size.
const ResidentBytesRatioAlarm = 1.5

// RequantizedWeightsDegradedToken leads the perf-line clause of a degraded load.
const RequantizedWeightsDegradedToken = "DEGRADED requantized-weights"

// ErrRequantizedWeights is the sentinel every requantized-weights refusal satisfies.
var ErrRequantizedWeights = errors.New("model: resident weight bytes exceed the file size (requantized weights)")

// WeightResidencyReceipt compares the weight bytes a load keeps resident with the bytes
// its file carries, and names every requantization ("IQ3_XXS->Q8_0") behind the gap.
type WeightResidencyReceipt struct {
	FileWeightBytes     int64    `json:"file_weight_bytes"`
	ResidentWeightBytes int64    `json:"resident_weight_bytes"`
	Ratio               float64  `json:"resident_to_file_ratio"`
	RequantizedTypes    []string `json:"requantized_types"`
	Degraded            bool     `json:"degraded"`
}

// NewWeightResidencyReceipt derives the receipt from the file weight bytes and the
// requantized uploads: each one adds its resident-minus-file growth to the file size.
func NewWeightResidencyReceipt(fileBytes int64, requants []compute.WeightRequant) WeightResidencyReceipt {
	r := WeightResidencyReceipt{FileWeightBytes: max(fileBytes, 0), RequantizedTypes: []string{}}
	r.ResidentWeightBytes = r.FileWeightBytes
	for _, q := range requants {
		if q.Tensors <= 0 {
			continue
		}
		r.ResidentWeightBytes += q.ResidentBytes - q.FileBytes
		r.RequantizedTypes = append(r.RequantizedTypes, q.Label())
	}
	sort.Strings(r.RequantizedTypes)
	if r.FileWeightBytes > 0 {
		r.Ratio = float64(r.ResidentWeightBytes) / float64(r.FileWeightBytes)
	}
	r.Degraded = r.Ratio > ResidentBytesRatioAlarm
	return r
}

// PerfClause is the trailing perf-line clause of a degraded load; empty otherwise.
func (r WeightResidencyReceipt) PerfClause() string {
	if !r.Degraded {
		return ""
	}
	return fmt.Sprintf(" %s=%.2fx(%s)", RequantizedWeightsDegradedToken, r.Ratio, strings.Join(r.RequantizedTypes, ","))
}

// Admit refuses a degraded load on the Halo appliance unless the operator admitted it.
// Off the appliance a degraded load is reported, never refused.
func (r WeightResidencyReceipt) Admit(appliance, admitted bool) error {
	if !r.Degraded || !appliance || admitted {
		return nil
	}
	return fmt.Errorf("%w: resident %d B is %.2fx the %d B file (limit %.1fx); requantized %s",
		ErrRequantizedWeights, r.ResidentWeightBytes, r.Ratio, r.FileWeightBytes, ResidentBytesRatioAlarm, strings.Join(r.RequantizedTypes, ","))
}

// SetFileWeightBytes records the weight payload size of the file this model loaded.
func (m *Model) SetFileWeightBytes(n int64) {
	if m != nil {
		m.fileWeightBytes = n
	}
}

// PlannedWeightRequants predicts, before any staging, the requantizations be will apply
// to this model's packed dense weights: a raw i-quant on a backend without its native
// kernel, or on one that cannot take the raw dtype at all, is resident as Q8_0. Routed experts keep their host path and are not staged.
func (m *Model) PlannedWeightRequants(be compute.Backend) []compute.WeightRequant {
	if m == nil || be == nil || !be.Caps().DeviceMemory {
		return nil
	}
	rows := map[string]*compute.WeightRequant{}
	for name, qt := range m.kqw {
		dt, raw := rawIQDenseHALDtypeFor(qt)
		if !raw || isRoutedExpertWeight(name) {
			continue
		}
		to, widened := compute.BackendRequantizesWeightDtype(be, dt)
		if !compute.BackendSupportsDeviceWeightDtype(be, dt) {
			to, widened = compute.Q8_0, true // weightHALRawIQExpanded stages it expanded
		}
		if !widened {
			continue
		}
		row := &compute.WeightRequant{From: compute.WeightDtypeName(dt), To: compute.WeightDtypeName(to)}
		if cur, ok := rows[row.Label()]; ok {
			row = cur
		} else {
			rows[row.Label()] = row
		}
		row.Tensors++
		row.FileBytes += int64(qt.out) * int64(qt.rowBytes())
		row.ResidentBytes += compute.Q8ResidentBytes(qt.out, qt.in, qBlk)
	}
	out := make([]compute.WeightRequant, 0, len(rows))
	for _, row := range rows {
		out = append(out, *row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Label() < out[j].Label() })
	return out
}

// WeightResidency is the live receipt for this model on be: per requantization label the
// larger of the load-time plan and the uploads compute actually recorded, so the receipt
// is right before staging runs and stays right if staging widens more than planned.
func (m *Model) WeightResidency(be compute.Backend) WeightResidencyReceipt {
	if m == nil {
		return NewWeightResidencyReceipt(0, nil)
	}
	return NewWeightResidencyReceipt(m.fileWeightBytes, mergeWeightRequants(m.PlannedWeightRequants(be), compute.WeightRequantSnapshot()))
}

func mergeWeightRequants(planned, observed []compute.WeightRequant) []compute.WeightRequant {
	byLabel := map[string]compute.WeightRequant{}
	for _, set := range [][]compute.WeightRequant{planned, observed} {
		for _, q := range set {
			if cur, ok := byLabel[q.Label()]; !ok || q.ResidentBytes-q.FileBytes > cur.ResidentBytes-cur.FileBytes {
				byLabel[q.Label()] = q
			}
		}
	}
	out := make([]compute.WeightRequant, 0, len(byLabel))
	for _, q := range byLabel {
		out = append(out, q)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Label() < out[j].Label() })
	return out
}
