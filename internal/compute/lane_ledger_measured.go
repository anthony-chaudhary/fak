package compute

// Joins the lane-slot-wave ledger to a measured window of cycles. Lane-slot-waves are dimensionless;
// lane-slot-cycles exist only once the caller supplies the window, and every one lands in a model cell or in
// Unattributed so the two sum exactly to the capacity.

import (
	"errors"
	"math"
	"math/big"
	"sort"
)

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrMeasuredCycles                 = errors.New("measured window: CyclesPerSM must be positive")
	ErrMeasuredWeightsLength          = errors.New("measured window: WaveWeights needs one entry per modelled wave")
	ErrMeasuredWeightNotPositive      = errors.New("measured window: every WaveWeights entry must be positive")
	ErrMeasuredWeightsExceedExact     = errors.New("measured window: exact WaveWeights sum exceeds CyclesPerSM")
	ErrMeasuredWeightsExceedSampled   = errors.New("measured window: sampled WaveWeights sum exceeds CyclesPerSM; refused, not clamped")
	ErrMeasuredWindowShorterThanWaves = errors.New("measured window: CyclesPerSM is shorter than the wave count, so the final wave would get no cycles")
	ErrMeasuredActiveNegative         = errors.New("measured window: ActiveLaneCycles must not be negative")
	ErrMeasuredActiveOverCapacity     = errors.New("measured window: ActiveLaneCycles exceeds the window capacity")
	ErrMeasuredTolerance              = errors.New("measured window: ActiveTolerance must be a fraction in [0, 1]")
	ErrMeasuredLedgerNotConserved     = errors.New("measured window: ledger does not conserve capacity or split into waves")
	ErrMeasuredOverflow               = errors.New("measured window: lane-slot-cycle capacity overflows int64")
)

// Verdict values of MeasuredLedger.Verdict. The verdict is empty when the caller supplied no active lanes.
const (
	MeasuredVerdictConsistent  = "consistent"
	MeasuredVerdictOverstates  = "model_overstates_active"
	MeasuredVerdictUnderstates = "model_understates_active"
)

// MeasuredWindow is the caller's measurement of one window. Every number is a caller input read from a
// profiler; the join never estimates one.
type MeasuredWindow struct {
	// CyclesPerSM is the elapsed cycles of the window on one SM.
	CyclesPerSM int64
	// WaveWeights optionally gives each modelled wave's cycle duration in wave order. Nil assumes equal waves
	// that fill the whole window, and the result is flagged AssumedEqualWaves.
	WaveWeights []int64
	// ActiveLaneCycles optionally gives the measured integral of occupied lanes over the window.
	ActiveLaneCycles int64
	// ActiveSupplied marks ActiveLaneCycles as measured, so a measured zero gets a verdict instead of reading
	// as absent. A non-zero ActiveLaneCycles is always treated as supplied.
	ActiveSupplied bool
	// ActiveTolerance is the largest |ModelActive-MeasuredActive| accepted as consistent, as a fraction of
	// CapacityLaneCycles. Zero demands exact agreement.
	ActiveTolerance float64
	// Exact is true when the inputs are exact counters rather than sampled. It only selects which refusal a
	// WaveWeights sum above CyclesPerSM reports.
	Exact bool
}

// MeasuredLedger is the lane-slot-cycle accounting of one measured window: each cell is the model's
// lane-slot-waves times the cycles of the waves that carry them. The flags and Verdict are not lane-slot-cycles.
type MeasuredLedger struct {
	CyclesPerSM        int64
	CapacityLaneCycles int64
	// Cells holds the lane-slot-cycles the model assigns to each LaneCell over the window.
	Cells [numLaneCells]int64
	// Unattributed is the window capacity that no modelled wave covers. It is zero under AssumedEqualWaves.
	Unattributed      int64
	AssumedEqualWaves bool
	// ActiveSupplied is true when the caller supplied ActiveLaneCycles. The fields below are zero otherwise.
	ActiveSupplied  bool
	ModelActive     int64
	MeasuredActive  int64
	ActiveShortfall int64 // ModelActive-MeasuredActive; positive means the model calls more lanes active than were seen
	Verdict         string
}

// MeasuredFinding is one lane-slot-cycle loss class of a measured window.
type MeasuredFinding struct {
	Name            string
	Cell            string
	LaneSlotCycles  int64
	ShareOfCapacity float64
	// Additive is true for the findings that partition the lost capacity. It is false for unexplained_inactive,
	// which is a slice of the active cell reported beside them.
	Additive bool
}

// JoinMeasured accounts the window m against the ledger. A wave's cells are its lane-slot-waves times its
// cycles, and every full wave shares one cell shape, so the window's cells are the full-wave shape times the
// cycles of the full waves plus the final-wave shape times the final wave's cycles. Unattributed is the capacity
// that no modelled wave covers.
func (x LaneLedger) JoinMeasured(m MeasuredWindow) (MeasuredLedger, error) {
	if !x.Conserved() || x.DeviceSMs <= 0 || x.LanesPerSM <= 0 || x.Waves <= 0 {
		return MeasuredLedger{}, ErrMeasuredLedgerNotConserved
	}
	if m.CyclesPerSM <= 0 {
		return MeasuredLedger{}, ErrMeasuredCycles
	}
	if !(m.ActiveTolerance >= 0 && m.ActiveTolerance <= 1) {
		return MeasuredLedger{}, ErrMeasuredTolerance
	}
	if m.ActiveLaneCycles < 0 {
		return MeasuredLedger{}, ErrMeasuredActiveNegative
	}
	slotsPerWave := int64(x.DeviceSMs) * int64(x.LanesPerSM)
	capacity, ok := checkedPositiveInt64Product(m.CyclesPerSM, int64(x.DeviceSMs), int64(x.LanesPerSM))
	if !ok {
		return MeasuredLedger{}, ErrMeasuredOverflow
	}
	if m.ActiveLaneCycles > capacity {
		return MeasuredLedger{}, ErrMeasuredActiveOverCapacity
	}
	full, final, err := x.laneWaveShapes(slotsPerWave)
	if err != nil {
		return MeasuredLedger{}, err
	}
	fullCycles, finalCycles, assumed, err := m.waveCycles(x.Waves)
	if err != nil {
		return MeasuredLedger{}, err
	}

	out := MeasuredLedger{
		CyclesPerSM:        m.CyclesPerSM,
		CapacityLaneCycles: capacity,
		AssumedEqualWaves:  assumed,
		ActiveSupplied:     m.ActiveSupplied || m.ActiveLaneCycles != 0,
	}
	// Each product is bounded by slotsPerWave*CyclesPerSM, which is the capacity checked above, so the
	// plain arithmetic cannot overflow.
	var covered int64
	for c := range out.Cells {
		out.Cells[c] = full[c]*fullCycles + final[c]*finalCycles
		covered += out.Cells[c]
	}
	out.Unattributed = capacity - covered
	if out.ActiveSupplied {
		out.ModelActive = out.Cells[CellActive]
		out.MeasuredActive = m.ActiveLaneCycles
		out.ActiveShortfall, out.Verdict = m.activeVerdict(out.ModelActive, out.MeasuredActive, capacity)
	}
	return out, nil
}

// laneWaveShapes recovers the cells of one full wave and of the final wave from the ledger totals. Every full
// wave has the same shape, so the full shape is the total minus the final wave, divided by the number of full
// waves. A ledger whose totals do not divide that way is refused.
func (x LaneLedger) laneWaveShapes(slotsPerWave int64) (full, final [numLaneCells]int64, err error) {
	final = x.FinalWave
	if total, ok := cellsSum(final); !ok || total != slotsPerWave {
		return full, final, ErrMeasuredLedgerNotConserved
	}
	if x.Waves == 1 {
		if x.Cells != final {
			return full, final, ErrMeasuredLedgerNotConserved
		}
		return full, final, nil
	}
	fullWaves := int64(x.Waves - 1)
	for c := range full {
		rest := x.Cells[c] - final[c]
		if rest < 0 || rest%fullWaves != 0 {
			return full, final, ErrMeasuredLedgerNotConserved
		}
		full[c] = rest / fullWaves
	}
	if total, ok := cellsSum(full); !ok || total != slotsPerWave {
		return full, final, ErrMeasuredLedgerNotConserved
	}
	return full, final, nil
}

// waveCycles returns the cycles of the full waves together and of the final wave. With equal waves the final
// wave gets floor(CyclesPerSM/waves) and the remainder goes to the full waves. Those waves share one cell
// shape, so only their total is needed.
func (m MeasuredWindow) waveCycles(waves int) (fullCycles, finalCycles int64, assumed bool, err error) {
	if m.WaveWeights == nil {
		finalCycles = m.CyclesPerSM / int64(waves)
		if finalCycles == 0 {
			return 0, 0, false, ErrMeasuredWindowShorterThanWaves
		}
		return m.CyclesPerSM - finalCycles, finalCycles, true, nil
	}
	if len(m.WaveWeights) != waves {
		return 0, 0, false, ErrMeasuredWeightsLength
	}
	var total int64
	for _, w := range m.WaveWeights {
		if w <= 0 {
			return 0, 0, false, ErrMeasuredWeightNotPositive
		}
		// Checked as a difference so the running total never overflows.
		if w > m.CyclesPerSM-total {
			return 0, 0, false, m.weightsExceedError()
		}
		total += w
	}
	finalCycles = m.WaveWeights[waves-1]
	return total - finalCycles, finalCycles, false, nil
}

func (m MeasuredWindow) weightsExceedError() error {
	if m.Exact {
		return ErrMeasuredWeightsExceedExact
	}
	return ErrMeasuredWeightsExceedSampled
}

// activeVerdict compares the model's active lanes with the measured ones. The shortfall is a plain difference
// and is never assigned a cause.
func (m MeasuredWindow) activeVerdict(model, measured, capacity int64) (int64, string) {
	shortfall := model - measured
	switch {
	case withinTolerance(shortfall, m.ActiveTolerance, capacity):
		return shortfall, MeasuredVerdictConsistent
	case shortfall > 0:
		return shortfall, MeasuredVerdictOverstates
	default:
		return shortfall, MeasuredVerdictUnderstates
	}
}

// withinTolerance reports whether |shortfall| <= tolerance*capacity in exact arithmetic. float64 cannot hold
// capacities above 2^53, so the bound is computed as a rational and the integer shortfall is compared with its
// floor.
func withinTolerance(shortfall int64, tolerance float64, capacity int64) bool {
	bound := new(big.Rat).SetFloat64(tolerance)
	bound.Mul(bound, new(big.Rat).SetInt64(capacity))
	allowed := new(big.Int).Quo(bound.Num(), bound.Denom())
	return new(big.Int).Abs(big.NewInt(shortfall)).Cmp(allowed) <= 0
}

// cellsSum adds non-negative cell counts and reports overflow instead of wrapping.
func cellsSum(cells [numLaneCells]int64) (int64, bool) {
	var total int64
	for _, n := range cells {
		if n < 0 || n > math.MaxInt64-total {
			return 0, false
		}
		total += n
	}
	return total, true
}

// Findings lists the lane-slot-cycle losses of the window, largest first, omitting zero losses. The non-active
// cells and Unattributed are Additive and partition Capacity-Cells[CellActive]. The unexplained_* rows are the
// active-cell residual outside tolerance, signed by the verdict: unexplained_inactive when the model calls more
// lanes active than were seen, unexplained_active when it calls fewer. They are not Additive and are emitted
// only when the verdict is not consistent, so a row never contradicts the verdict beside it.
func (m MeasuredLedger) Findings() []MeasuredFinding {
	var out []MeasuredFinding
	add := func(name, cell string, cycles int64, additive bool) {
		if cycles > 0 {
			out = append(out, MeasuredFinding{
				Name: name, Cell: cell, LaneSlotCycles: cycles,
				ShareOfCapacity: ratio(cycles, m.CapacityLaneCycles), Additive: additive,
			})
		}
	}
	for _, c := range AllLaneCells() {
		if c != CellActive {
			add(c.String(), c.String(), m.Cells[c], true)
		}
	}
	add("unattributed", "unattributed", m.Unattributed, true)
	switch m.Verdict {
	case MeasuredVerdictOverstates:
		add("unexplained_inactive", CellActive.String(), m.ActiveShortfall, false)
	case MeasuredVerdictUnderstates:
		add("unexplained_active", CellActive.String(), -m.ActiveShortfall, false)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].LaneSlotCycles != out[j].LaneSlotCycles {
			return out[i].LaneSlotCycles > out[j].LaneSlotCycles
		}
		return out[i].Name < out[j].Name
	})
	return out
}
