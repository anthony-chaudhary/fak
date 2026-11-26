package compute

// Joins the observed per-launch kernel time of one profile to the modelled lane loss of each launch shape,
// so the lane losses can be ranked by where the device spends its time. The ranking key is a prioritisation
// heuristic and says nothing measured: see ObservedCaveat.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"strings"
)

// ObservedCaveat travels with every ObservedReport. The ranking key multiplies observed kernel time by a
// modelled fraction, so it orders the losses only.
const ObservedCaveat = "time_weighted_idle_ns is a prioritisation heuristic: observed kernel time multiplied by the " +
	"modelled fraction of lane slots that carry no thread. It is not measured occupancy and not a predicted speedup. " +
	"Kernel time includes the launch gap the timestamp profile records, so short kernels are overstated."

// Unmodelled reasons. A row without a reason is modelled; a row with one keeps its time share in the report.
const (
	ObservedReasonRegsUnmeasured  = "regs_unmeasured"
	ObservedReasonThreadsUnjudged = "threads_unjudged"
	ObservedReasonSmemUnjudged    = "smem_unjudged"
	ObservedReasonUnlaunchable    = "unlaunchable"
	ObservedReasonOverflow        = "overflow"
	ObservedReasonInvariant       = "invariant_failed"
)

var observedReasons = []string{
	ObservedReasonRegsUnmeasured, ObservedReasonThreadsUnjudged, ObservedReasonSmemUnjudged,
	ObservedReasonUnlaunchable, ObservedReasonOverflow, ObservedReasonInvariant,
}

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrObservedEmpty         = errors.New("observed profile: no launch rows")
	ErrObservedRow           = errors.New("observed profile: a row needs a kernel name, a positive grid, and non-negative calls and total_ns")
	ErrObservedDuplicate     = errors.New("observed profile: two rows share a kernel name and grid shape")
	ErrObservedArch          = errors.New("observed join: Arch must be a registered NVArch row")
	ErrObservedDeviceSMs     = errors.New("observed join: DeviceSMs must be positive; the SM count is a caller input")
	ErrObservedNegativeInput = errors.New("observed join: Regs, Threads and SmemBytes must not be negative")
	ErrObservedKeyNotBase    = errors.New("observed join: input keys must be kernel base names, without template arguments or a namespace")
	ErrObservedOverflow      = errors.New("observed join: summed observed time overflows int64")
	ErrObservedTopN          = errors.New("observed report: Top needs a positive n")
	ErrObservedInvariant     = errors.New("observed join: internal accounting identity failed")
)

// ObservedLaunch is one row of an observed profile: a kernel name, its grid (blocks are GridX*GridY), the
// number of launches that shared that grid, and their summed device time. The JSON names are the documented
// array form: {"name","gx","gy","calls","total_ns"}.
type ObservedLaunch struct {
	Kernel  string `json:"name"`
	GridX   int    `json:"gx"`
	GridY   int    `json:"gy"`
	Calls   int64  `json:"calls"`
	TotalNS int64  `json:"total_ns"`
}

// ObservedInputs is the caller's model of the device and of each kernel. Regs, Threads and SmemBytes are keyed
// by kernel base name as the timestamp profile reports it. An absent or zero Regs or Threads entry leaves the
// row unmodelled; SmemBytes must be present, because a missing dynamic shared-memory count would silently
// lift the shared-memory limit.
type ObservedInputs struct {
	Arch      NVArch
	DeviceSMs int
	Regs      map[string]int // registers per thread
	Threads   map[string]int // threads per block
	SmemBytes map[string]int // dynamic shared memory per block, in bytes
}

// ObservedModel is the lane-slot-wave model of one row's grid on the device. Lane-slot-wave counts stay in
// the Cells fields; the time fields are nanoseconds. TimeWeightedIdleNS is TotalNS minus the active cell's time.
type ObservedModel struct {
	RegsPerThread      int                `json:"regs_per_thread"`
	ThreadsPerBlock    int                `json:"threads_per_block"`
	SmemBytesPerBlock  int                `json:"smem_bytes_per_block"`
	GridBlocks         int64              `json:"grid_blocks"`
	BlocksPerSM        int                `json:"blocks_per_sm"`
	Limiter            string             `json:"limiter"`
	Waves              int                `json:"waves"`
	LaneWaveCapacity   int64              `json:"capacity_lane_slot_waves"`
	ActiveShare        float64            `json:"active_share"`
	CellShare          map[string]float64 `json:"cell_share"`
	CellTimeNS         map[string]int64   `json:"cell_time_ns"`
	TimeWeightedIdleNS int64              `json:"time_weighted_idle_ns"`
}

// ObservedRow is one observed launch shape. TimeShare is its fraction of the profile's total observed time.
// Model is nil exactly when Reason names why the row could not be modelled, and Error carries the refusal text.
type ObservedRow struct {
	Kernel    string         `json:"kernel"`
	GridX     int            `json:"gx"`
	GridY     int            `json:"gy"`
	Calls     int64          `json:"calls"`
	TotalNS   int64          `json:"total_ns"`
	TimeShare float64        `json:"time_share"`
	Reason    string         `json:"unmodelled_reason,omitempty"`
	Error     string         `json:"error,omitempty"`
	Model     *ObservedModel `json:"model,omitempty"`
}

// ObservedTotals accounts for all observed time: ModelledNS plus UnmodelledNS equals TotalNS exactly.
// UnmodelledNSByReason lists every reason, so zero means none of that reason, not an unreported one.
// CellTimeNS sums the modelled rows' cell times; it sums to ModelledNS exactly.
type ObservedTotals struct {
	TotalNS              int64            `json:"total_ns"`
	ModelledNS           int64            `json:"modelled_ns"`
	UnmodelledNS         int64            `json:"unmodelled_ns"`
	ModelledShare        float64          `json:"modelled_share"`
	ModelledRows         int              `json:"modelled_rows"`
	UnmodelledRows       int              `json:"unmodelled_rows"`
	UnmodelledNSByReason map[string]int64 `json:"unmodelled_ns_by_reason"`
	TimeWeightedIdleNS   int64            `json:"time_weighted_idle_ns"`
	CellTimeNS           map[string]int64 `json:"cell_time_ns"`
}

// ObservedReport ranks the rows: modelled rows first by TimeWeightedIdleNS, then unmodelled rows by TotalNS.
// Ties break on kernel name, then grid x and y, so the order is deterministic.
type ObservedReport struct {
	Arch      string         `json:"arch"`
	DeviceSMs int            `json:"device_sms"`
	Caveat    string         `json:"caveat"`
	Totals    ObservedTotals `json:"totals"`
	Rows      []ObservedRow  `json:"rows"`
}

// observedRecord holds the documented fields as pointers, so a missing field is detected instead of read as zero.
type observedRecord struct {
	Name    *string `json:"name"`
	GX      *int    `json:"gx"`
	GY      *int    `json:"gy"`
	Calls   *int64  `json:"calls"`
	TotalNS *int64  `json:"total_ns"`
}

// ParseObservedProfile reads the documented JSON array form. Fields it does not name are ignored, so the reader
// tolerates extra metadata; every named field is required, and an empty array is refused.
func ParseObservedProfile(r io.Reader) ([]ObservedLaunch, error) {
	dec := json.NewDecoder(r)
	var recs []observedRecord
	if err := dec.Decode(&recs); err != nil {
		return nil, fmt.Errorf("observed profile: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("observed profile: unexpected data after the JSON array")
	}
	rows := make([]ObservedLaunch, len(recs))
	for i, rec := range recs {
		if rec.Name == nil || rec.GX == nil || rec.GY == nil || rec.Calls == nil || rec.TotalNS == nil {
			return nil, fmt.Errorf("%w: row %d is missing a required field", ErrObservedRow, i)
		}
		rows[i] = ObservedLaunch{Kernel: *rec.Name, GridX: *rec.GX, GridY: *rec.GY, Calls: *rec.Calls, TotalNS: *rec.TotalNS}
	}
	if err := observedValidateRows(rows); err != nil {
		return nil, err
	}
	return rows, nil
}

// observedValidateRows refuses an empty profile, a malformed row, and two rows that name the same launch shape.
func observedValidateRows(rows []ObservedLaunch) error {
	if len(rows) == 0 {
		return ErrObservedEmpty
	}
	type shape struct {
		kernel string
		gx, gy int
	}
	seen := make(map[shape]int, len(rows))
	for i, r := range rows {
		if r.Kernel == "" || r.GridX < 1 || r.GridY < 1 || r.Calls < 0 || r.TotalNS < 0 {
			return fmt.Errorf("%w: row %d", ErrObservedRow, i)
		}
		k := shape{r.Kernel, r.GridX, r.GridY}
		if first, dup := seen[k]; dup {
			return fmt.Errorf("%w: rows %d and %d", ErrObservedDuplicate, first, i)
		}
		seen[k] = i
	}
	return nil
}

// JoinObserved models every observed row on the device in in and accounts for all of its time. A row that
// cannot be modelled stays in the report with its time share and a reason. Only inputs that make the whole
// join meaningless are refused: an empty or malformed profile, an unregistered arch, a non-positive SM count,
// negative or non-base input keys, and a total that overflows.
func JoinObserved(rows []ObservedLaunch, in ObservedInputs) (ObservedReport, error) {
	if err := observedValidateRows(rows); err != nil {
		return ObservedReport{}, err
	}
	if row, ok := LookupNVArch(in.Arch.Cap); !ok || row != in.Arch {
		return ObservedReport{}, ErrObservedArch
	}
	if in.DeviceSMs <= 0 {
		return ObservedReport{}, ErrObservedDeviceSMs
	}
	if err := observedCheckInputs(in); err != nil {
		return ObservedReport{}, err
	}
	var total int64
	for _, r := range rows {
		if r.TotalNS > math.MaxInt64-total {
			return ObservedReport{}, ErrObservedOverflow
		}
		total += r.TotalNS
	}
	out := ObservedReport{Arch: in.Arch.Label, DeviceSMs: in.DeviceSMs, Caveat: ObservedCaveat}
	out.Rows = make([]ObservedRow, len(rows))
	for i, r := range rows {
		out.Rows[i] = observedJudgeRow(r, in, total)
	}
	observedSortRows(out.Rows)
	totals, err := observedTotals(out.Rows, total)
	if err != nil {
		return ObservedReport{}, err
	}
	out.Totals = totals
	return out, nil
}

// observedCheckInputs refuses negative counts and keys that are not kernel base names. A key with template
// arguments would never match the profile, so it would read as unjudged rather than as the caller's error.
func observedCheckInputs(in ObservedInputs) error {
	if err := observedCheckMap(in.Regs); err != nil {
		return err
	}
	if err := observedCheckMap(in.Threads); err != nil {
		return err
	}
	return observedCheckMap(in.SmemBytes)
}

func observedCheckMap(m map[string]int) error {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if k == "" || strings.ContainsAny(k, "<>:") {
			return fmt.Errorf("%w: %q", ErrObservedKeyNotBase, k)
		}
		if m[k] < 0 {
			return fmt.Errorf("%w: %q", ErrObservedNegativeInput, k)
		}
	}
	return nil
}

// observedJudgeRow models one row, or records why it cannot be modelled.
func observedJudgeRow(r ObservedLaunch, in ObservedInputs, total int64) ObservedRow {
	row := ObservedRow{
		Kernel: r.Kernel, GridX: r.GridX, GridY: r.GridY, Calls: r.Calls, TotalNS: r.TotalNS,
		TimeShare: ratio(r.TotalNS, total),
	}
	launch, reason := observedLaunch(r.Kernel, in)
	if reason != "" {
		row.Reason = reason
		return row
	}
	m, err := observedModel(r, in, launch)
	if err != nil {
		row.Reason, row.Error = observedLedgerReason(err), err.Error()
		return row
	}
	row.Model = &m
	return row
}

// observedLaunch assembles the launch inputs of one kernel, or names the first input that is missing.
func observedLaunch(kernel string, in ObservedInputs) (DecodeLaunch, string) {
	l := DecodeLaunch{Kernel: kernel, RegsPerThread: in.Regs[kernel], ThreadsPerBlock: in.Threads[kernel]}
	if l.RegsPerThread == 0 {
		return l, ObservedReasonRegsUnmeasured
	}
	if l.ThreadsPerBlock == 0 {
		return l, ObservedReasonThreadsUnjudged
	}
	smem, ok := in.SmemBytes[kernel]
	if !ok {
		return l, ObservedReasonSmemUnjudged
	}
	l.SmemBytesPerBlock = smem
	return l, ""
}

// observedModel builds the lane ledger of the row's exact grid and joins the row's time to its cells.
func observedModel(r ObservedLaunch, in ObservedInputs, launch DecodeLaunch) (ObservedModel, error) {
	blocks, ok := checkedPositiveInt64Product(int64(r.GridX), int64(r.GridY))
	if !ok || blocks > int64(^uint(0)>>1) {
		return ObservedModel{}, fmt.Errorf("%w: grid %dx%d", ErrLaneOverflow, r.GridX, r.GridY)
	}
	launch.GridBlocks = int(blocks)
	led, err := in.Arch.BuildLaneLedger(launch, in.DeviceSMs)
	if err != nil {
		return ObservedModel{}, err
	}
	if !led.Conserved() {
		return ObservedModel{}, fmt.Errorf("%w: %s", ErrObservedInvariant, r.Kernel)
	}
	cellNS := observedCellTimes(r.TotalNS, led.Cells, led.Capacity)
	m := ObservedModel{
		RegsPerThread:      launch.RegsPerThread,
		ThreadsPerBlock:    launch.ThreadsPerBlock,
		SmemBytesPerBlock:  launch.SmemBytesPerBlock,
		GridBlocks:         blocks,
		BlocksPerSM:        led.BlocksPerSM,
		Limiter:            led.Limiter,
		Waves:              led.Waves,
		LaneWaveCapacity:   led.Capacity,
		ActiveShare:        ratio(led.Cells[CellActive], led.Capacity),
		CellShare:          make(map[string]float64, numLaneCells),
		CellTimeNS:         make(map[string]int64, numLaneCells),
		TimeWeightedIdleNS: r.TotalNS - cellNS[CellActive],
	}
	for _, c := range AllLaneCells() {
		m.CellShare[c.String()] = ratio(led.Cells[c], led.Capacity)
		m.CellTimeNS[c.String()] = cellNS[c]
	}
	return m, nil
}

// observedLedgerReason maps a model refusal to its unmodelled reason. A failed accounting identity is a
// model fault, not a launch that cannot run, so it keeps its own reason.
func observedLedgerReason(err error) string {
	switch {
	case errors.Is(err, ErrLaneOverflow):
		return ObservedReasonOverflow
	case errors.Is(err, ErrObservedInvariant):
		return ObservedReasonInvariant
	}
	return ObservedReasonUnlaunchable
}

// observedCellTimes splits totalNS across the cells in proportion to their lane-slot-waves. Each cell takes the
// floor of its exact share; the leftover nanoseconds go to the cells with the largest remainders, ties to the
// lower-numbered cell. The cells therefore sum to totalNS exactly, and no cell moves by more than one nanosecond.
// The products run in big.Int because totalNS times a cell count can exceed int64.
func observedCellTimes(totalNS int64, cells [numLaneCells]int64, capacity int64) [numLaneCells]int64 {
	var out, rem [numLaneCells]int64
	den := big.NewInt(capacity)
	var sum int64
	for c := range cells {
		num := new(big.Int).Mul(big.NewInt(totalNS), big.NewInt(cells[c]))
		q, r := new(big.Int).QuoRem(num, den, new(big.Int))
		out[c], rem[c] = q.Int64(), r.Int64()
		sum += out[c]
	}
	order := AllLaneCells()
	sort.SliceStable(order, func(i, j int) bool { return rem[order[i]] > rem[order[j]] })
	// The remainders sum to leftover*capacity with each below capacity, so leftover is below numLaneCells.
	leftover := totalNS - sum
	for i := int64(0); i < leftover; i++ {
		out[order[i]]++
	}
	return out
}

// observedSortRows orders modelled rows by TimeWeightedIdleNS, then unmodelled rows by TotalNS.
func observedSortRows(rows []ObservedRow) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if (a.Model != nil) != (b.Model != nil) {
			return a.Model != nil
		}
		if a.Model != nil && a.Model.TimeWeightedIdleNS != b.Model.TimeWeightedIdleNS {
			return a.Model.TimeWeightedIdleNS > b.Model.TimeWeightedIdleNS
		}
		if a.TotalNS != b.TotalNS {
			return a.TotalNS > b.TotalNS
		}
		if a.Kernel != b.Kernel {
			return a.Kernel < b.Kernel
		}
		if a.GridX != b.GridX {
			return a.GridX < b.GridX
		}
		return a.GridY < b.GridY
	})
}

// observedTotals sums the rows and checks that modelled plus unmodelled time equals the profile total.
func observedTotals(rows []ObservedRow, total int64) (ObservedTotals, error) {
	t := ObservedTotals{
		TotalNS:              total,
		UnmodelledNSByReason: make(map[string]int64, len(observedReasons)),
		CellTimeNS:           make(map[string]int64, numLaneCells),
	}
	for _, reason := range observedReasons {
		t.UnmodelledNSByReason[reason] = 0
	}
	for _, c := range AllLaneCells() {
		t.CellTimeNS[c.String()] = 0
	}
	for _, r := range rows {
		if r.Model == nil {
			t.UnmodelledRows++
			t.UnmodelledNS += r.TotalNS
			t.UnmodelledNSByReason[r.Reason] += r.TotalNS
			continue
		}
		t.ModelledRows++
		t.ModelledNS += r.TotalNS
		t.TimeWeightedIdleNS += r.Model.TimeWeightedIdleNS
		for name, ns := range r.Model.CellTimeNS {
			t.CellTimeNS[name] += ns
		}
	}
	if t.ModelledNS+t.UnmodelledNS != total {
		return ObservedTotals{}, ErrObservedInvariant
	}
	t.ModelledShare = ratio(t.ModelledNS, total)
	return t, nil
}

// Top returns the first n modelled rows in rank order. Unmodelled rows are never returned as ranked, and n
// larger than the modelled count returns every modelled row.
func (r ObservedReport) Top(n int) ([]ObservedRow, error) {
	if n <= 0 {
		return nil, ErrObservedTopN
	}
	var out []ObservedRow
	for _, row := range r.Rows {
		if row.Model == nil {
			continue
		}
		out = append(out, row)
		if len(out) == n {
			break
		}
	}
	return out, nil
}
