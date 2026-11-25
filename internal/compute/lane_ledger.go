package compute

// lane_ledger.go — the lane-slot-wave ledger for one decode launch. Every warp lane on every SM for
// every wave is classified into exactly one LaneCell, so the capacity lost to underfill, wave tails,
// unfilled block slots, per-SM resource caps and partial warps is an exact partition rather than a set
// of overlapping percentages. The unit is dimensionless: one lane on one SM for one wave, with no clock.
//
// The per-SM arithmetic is the occupancy calculator in decode_occupancy.go (NVArch.blocksPerSM), and the
// SM count is a caller input because it is a SKU property, not an ISA constant. Two choices are model
// assumptions rather than measured scheduler behaviour: blocks go one-per-SM first, and each wave is
// static, so a slot freed inside a wave is not reused by a later block until the whole wave finishes.
// The wave_tail finding is therefore an upper bound on what a dynamic scheduler would lose.

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// LaneCell classifies one lane-slot-wave of a launch.
type LaneCell uint8

const (
	// CellActive is a lane holding a thread of a block resident on its SM.
	CellActive LaneCell = iota
	// CellPartialWarp is a lane of a warp that ThreadsPerBlock does not fill.
	CellPartialWarp
	// CellResidentUnfilled is a lane in a block slot the SM could hold but this wave does not fill.
	CellResidentUnfilled
	// CellResourceCapped is a lane the binding per-SM limit (registers, shared memory, warps or blocks)
	// keeps out of every resident block.
	CellResourceCapped
	// CellUnitIdle is a lane on an SM that receives no block in this wave.
	CellUnitIdle
	numLaneCells
)

var laneCellNames = [numLaneCells]string{
	CellActive:           "active",
	CellPartialWarp:      "partial_warp",
	CellResidentUnfilled: "resident_unfilled",
	CellResourceCapped:   "resource_capped",
	CellUnitIdle:         "unit_idle",
}

// String returns the snake_case cell name used in JSON and findings.
func (c LaneCell) String() string {
	if c >= numLaneCells {
		return "unknown"
	}
	return laneCellNames[c]
}

// AllLaneCells lists every LaneCell in enum order.
func AllLaneCells() []LaneCell {
	cells := make([]LaneCell, 0, numLaneCells)
	for c := LaneCell(0); c < numLaneCells; c++ {
		cells = append(cells, c)
	}
	return cells
}

// Vendor ceilings shared by every modeled compute capability (CUDA Programming Guide, Compute
// Capabilities table). They bound the launch shape before any per-SM arithmetic runs.
const (
	maxRegsPerThread   = 255
	maxThreadsPerBlock = 1024
)

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrLaneNoDeviceSMs    = errors.New("lane ledger: deviceSMs must be positive; the SM count is a caller input")
	ErrLaneRegsUnmeasured = errors.New("lane ledger: RegsPerThread must be measured and positive")
	ErrLaneRegsOverLimit  = errors.New("lane ledger: RegsPerThread exceeds the per-thread register ceiling")
	ErrLaneThreads        = errors.New("lane ledger: ThreadsPerBlock must be in [1, 1024]")
	ErrLaneGrid           = errors.New("lane ledger: GridBlocks must be positive")
	ErrLaneSmem           = errors.New("lane ledger: SmemBytesPerBlock must not be negative")
	ErrLaneUnknownArch    = errors.New("lane ledger: arch is not a registered NVArch row")
	ErrLaneUnlaunchable   = errors.New("lane ledger: launch fits zero blocks per SM")
	ErrLaneOverflow       = errors.New("lane ledger: capacity overflows int64")
)

// LaneLedger is the exact lane-slot-wave accounting of one launch on one device.
type LaneLedger struct {
	Arch               string              `json:"arch"`
	Kernel             string              `json:"kernel"`
	Limiter            string              `json:"limiter"`
	Provenance         string              `json:"provenance"`
	DeviceSMs          int                 `json:"device_sms"`
	LanesPerSM         int                 `json:"lanes_per_sm"`
	BlocksPerSM        int                 `json:"blocks_per_sm"`
	WarpsPerBlock      int                 `json:"warps_per_block"`
	ThreadsPerBlock    int                 `json:"threads_per_block"`
	GridBlocks         int                 `json:"grid_blocks"`
	Waves              int                 `json:"waves"`
	Capacity           int64               `json:"capacity"`
	Cells              [numLaneCells]int64 `json:"cells"`
	BlockSlotsUnfilled int64               `json:"block_slots_unfilled"`

	// FinalWave is the cell breakdown of the last wave alone. Every full wave fills every SM under the
	// placement model, so the final wave is the only source of wave_tail losses. It travels as the
	// final_wave object so Findings gives the same answer on a decoded ledger.
	FinalWave [numLaneCells]int64 `json:"-"`
}

// CellCounts returns Cells as a name->count map keyed by LaneCell.String.
func (x LaneLedger) CellCounts() map[string]int64 {
	return cellCounts(x.Cells)
}

func cellCounts(cells [numLaneCells]int64) map[string]int64 {
	counts := make(map[string]int64, numLaneCells)
	for _, c := range AllLaneCells() {
		counts[c.String()] = cells[c]
	}
	return counts
}

// cellsFromCounts is the inverse of cellCounts. It requires every cell name exactly once, so a renamed or
// truncated document is refused instead of decoding to zeros.
func cellsFromCounts(counts map[string]int64) ([numLaneCells]int64, error) {
	var cells [numLaneCells]int64
	if len(counts) != int(numLaneCells) {
		return cells, fmt.Errorf("want %d named cells, got %d", numLaneCells, len(counts))
	}
	for _, c := range AllLaneCells() {
		n, ok := counts[c.String()]
		if !ok {
			return cells, fmt.Errorf("missing cell %q", c.String())
		}
		cells[c] = n
	}
	return cells, nil
}

// MarshalJSON writes Cells and FinalWave as name->count objects keyed by LaneCell.String, not as
// positional arrays.
func (x LaneLedger) MarshalJSON() ([]byte, error) {
	type plain LaneLedger // same fields and tags, without these methods
	return json.Marshal(struct {
		plain
		Cells     map[string]int64 `json:"cells"`
		FinalWave map[string]int64 `json:"final_wave"`
	}{plain: plain(x), Cells: x.CellCounts(), FinalWave: cellCounts(x.FinalWave)})
}

// UnmarshalJSON reads the document MarshalJSON writes.
func (x *LaneLedger) UnmarshalJSON(data []byte) error {
	type plain LaneLedger
	var raw struct {
		plain
		Cells     map[string]int64 `json:"cells"`
		FinalWave map[string]int64 `json:"final_wave"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	cells, err := cellsFromCounts(raw.Cells)
	if err != nil {
		return fmt.Errorf("lane ledger cells: %w", err)
	}
	finalWave, err := cellsFromCounts(raw.FinalWave)
	if err != nil {
		return fmt.Errorf("lane ledger final_wave: %w", err)
	}
	*x = LaneLedger(raw.plain)
	x.Cells = cells
	x.FinalWave = finalWave
	return nil
}

// GapFinding is one non-zero loss class of a ledger, measured in lane-slot-waves. Cell names the LaneCell
// that carries the loss; a wave_tail finding names the larger of its unit_idle and resident_unfilled parts.
type GapFinding struct {
	Name            string  `json:"name"`
	Cell            string  `json:"cell"`
	LaneSlotWaves   int64   `json:"lane_slot_waves"`
	ShareOfCapacity float64 `json:"share_of_capacity"`
	Trigger         string  `json:"trigger"`
}

// laneGeometry is the launch-constant part of the per-SM and per-wave arithmetic. All fields are
// int64 so every product stays in one type.
type laneGeometry struct {
	lanesPerSM      int64 // MaxWarpsPerSM * WarpSize: every lane an SM has
	warpsPerBlock   int64 // w = ceil(ThreadsPerBlock / WarpSize)
	warpLanes       int64 // w * WarpSize: lanes the warps of one block occupy
	threadsPerBlock int64 // ThreadsPerBlock
	blocksPerSM     int64 // B: resident blocks one SM can hold
}

func (a NVArch) geometryFor(l DecodeLaunch, bpsm int) laneGeometry {
	w := ceilDivInt(l.ThreadsPerBlock, a.WarpSize)
	return laneGeometry{
		lanesPerSM:      int64(a.MaxWarpsPerSM * a.WarpSize),
		warpsPerBlock:   int64(w),
		warpLanes:       int64(w * a.WarpSize),
		threadsPerBlock: int64(l.ThreadsPerBlock),
		blocksPerSM:     int64(bpsm),
	}
}

// smCells returns the cells one SM contributes in a wave when it holds c blocks.
func (g laneGeometry) smCells(c int64) (cells [numLaneCells]int64) {
	if c == 0 {
		cells[CellUnitIdle] = g.lanesPerSM
		return cells
	}
	cells[CellActive] = c * g.threadsPerBlock
	cells[CellPartialWarp] = c * (g.warpLanes - g.threadsPerBlock)
	cells[CellResidentUnfilled] = (g.blocksPerSM - c) * g.warpLanes
	cells[CellResourceCapped] = g.lanesPerSM - g.blocksPerSM*g.warpLanes
	return cells
}

// waveCells places n blocks one-per-SM first, a model placement rather than a measured policy: every SM
// takes n/D blocks and the first n%D take one more. Only the two per-SM shapes are computed, so the cost
// does not grow with the SM count.
func (g laneGeometry) waveCells(n, deviceSMs int64) (cells [numLaneCells]int64) {
	perSM, extra := n/deviceSMs, n%deviceSMs
	if extra > 0 {
		hi := g.smCells(perSM + 1)
		for c := range cells {
			cells[c] += extra * hi[c]
		}
	}
	lo := g.smCells(perSM)
	for c := range cells {
		cells[c] += (deviceSMs - extra) * lo[c]
	}
	return cells
}

// checkLaunch refuses an arch that is not a registered row, and a launch whose inputs the
// occupancy arithmetic cannot price. An unmeasured register count would silently hide a register
// limiter, so it is refused rather than treated as "no register limit".
func (a NVArch) checkLaunch(l DecodeLaunch) error {
	if row, ok := LookupNVArch(a.Cap); !ok || row != a {
		return ErrLaneUnknownArch
	}
	switch {
	case l.RegsPerThread <= 0:
		return ErrLaneRegsUnmeasured
	case l.RegsPerThread > maxRegsPerThread:
		return ErrLaneRegsOverLimit
	case l.ThreadsPerBlock <= 0 || l.ThreadsPerBlock > maxThreadsPerBlock:
		return ErrLaneThreads
	case l.GridBlocks <= 0:
		return ErrLaneGrid
	case l.SmemBytesPerBlock < 0:
		return ErrLaneSmem
	}
	return nil
}

// BuildLaneLedger classifies every lane-slot-wave of launch l on arch a across deviceSMs SMs. deviceSMs
// is a caller input (the SKU's enabled SM count) and has no default, because a wrong count silently
// rescales every share.
func (a NVArch) BuildLaneLedger(l DecodeLaunch, deviceSMs int) (LaneLedger, error) {
	if deviceSMs <= 0 {
		return LaneLedger{}, ErrLaneNoDeviceSMs
	}
	if err := a.checkLaunch(l); err != nil {
		return LaneLedger{}, err
	}
	bpsm, limiter := a.blocksPerSM(l)
	if bpsm == 0 {
		return LaneLedger{}, ErrLaneUnlaunchable
	}
	g := a.geometryFor(l, bpsm)
	slots, ok := checkedPositiveInt64Product(int64(deviceSMs), int64(bpsm))
	if !ok {
		return LaneLedger{}, ErrLaneOverflow
	}
	grid := int64(l.GridBlocks)
	waves := (grid-1)/slots + 1
	capacity, ok := checkedPositiveInt64Product(waves, int64(deviceSMs), g.lanesPerSM)
	if !ok {
		return LaneLedger{}, ErrLaneOverflow
	}

	full := waves - 1
	fullWave := g.waveCells(slots, int64(deviceSMs))
	finalWave := g.waveCells(grid-full*slots, int64(deviceSMs))
	var cells [numLaneCells]int64
	for c := range cells {
		cells[c] = fullWave[c]*full + finalWave[c]
	}
	return LaneLedger{
		Arch:               a.Label,
		Kernel:             l.Kernel,
		Limiter:            limiter,
		Provenance:         a.Source,
		DeviceSMs:          deviceSMs,
		LanesPerSM:         int(g.lanesPerSM),
		BlocksPerSM:        bpsm,
		WarpsPerBlock:      int(g.warpsPerBlock),
		ThreadsPerBlock:    l.ThreadsPerBlock,
		GridBlocks:         l.GridBlocks,
		Waves:              int(waves),
		Capacity:           capacity,
		Cells:              cells,
		BlockSlotsUnfilled: waves*slots - grid,
		FinalWave:          finalWave,
	}, nil
}

// Conserved reports whether the ledger prices a launch (Capacity > 0), every cell is non-negative, and the
// cells partition the whole capacity. The zero value is not conserved, so a caller that ignores a refusal
// cannot read it as a valid partition.
func (x LaneLedger) Conserved() bool {
	if x.Capacity <= 0 {
		return false
	}
	var sum int64
	for _, n := range x.Cells {
		if n < 0 {
			return false
		}
		sum += n
	}
	return sum == x.Capacity
}

// tailCell names the cell that carries a wave_tail loss: unit_idle when the final wave leaves SMs empty,
// otherwise resident_unfilled.
func tailCell(idle, resident int64) LaneCell {
	if idle >= resident {
		return CellUnitIdle
	}
	return CellResidentUnfilled
}

// Findings lists the non-zero losses, largest first. Each lost lane-slot-wave belongs to exactly one
// finding, so the LaneSlotWaves sum to Capacity minus the active cell.
func (x LaneLedger) Findings() []GapFinding {
	var out []GapFinding
	add := func(name, cell string, slots int64, trigger string) {
		if slots > 0 {
			out = append(out, GapFinding{
				Name: name, Cell: cell, LaneSlotWaves: slots,
				ShareOfCapacity: ratio(slots, x.Capacity), Trigger: trigger,
			})
		}
	}
	resident := x.Cells[CellResidentUnfilled]
	if x.Waves == 1 {
		add("underfill", CellUnitIdle.String(), x.Cells[CellUnitIdle],
			"grid is smaller than the device: SMs receive no block in the single wave")
	} else {
		tailIdle, tailResident := x.FinalWave[CellUnitIdle], x.FinalWave[CellResidentUnfilled]
		add("wave_tail", tailCell(tailIdle, tailResident).String(), tailIdle+tailResident,
			"final wave leaves SMs or block slots empty")
		resident -= tailResident
	}
	add("resident_unfilled", CellResidentUnfilled.String(), resident,
		"free block slots on populated SMs")
	add("resource_capped", CellResourceCapped.String(), x.Cells[CellResourceCapped],
		"binding limiter="+x.Limiter)
	add("partial_warp", CellPartialWarp.String(), x.Cells[CellPartialWarp],
		"ThreadsPerBlock is not a multiple of WarpSize")
	sort.Slice(out, func(i, j int) bool {
		if out[i].LaneSlotWaves != out[j].LaneSlotWaves {
			return out[i].LaneSlotWaves > out[j].LaneSlotWaves
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// RegisterFootprint partitions one SM's 32-bit register file for a launch. Its unit is registers per
// SM, never lane-slots, so it is a distinct type and its findings are a distinct type.
type RegisterFootprint struct {
	RegsPerSM        int64 `json:"regs_per_sm"`
	Useful           int64 `json:"useful"`
	RoundingWaste    int64 `json:"rounding_waste"`
	GranularityWaste int64 `json:"granularity_waste"`
	HeadroomUnused   int64 `json:"headroom_unused"`
	Remainder        int64 `json:"remainder"`
}

// RegisterFootprintOf charges launch l's registers against the register file exactly as blocksPerSM
// does (both read registerWarps): a warp's registers round up to RegAllocUnit, the warp count rounds down
// to WarpAllocGranularity, and the resident blocks take whole blocks' worth of those warps.
func (a NVArch) RegisterFootprintOf(l DecodeLaunch) (RegisterFootprint, error) {
	if err := a.checkLaunch(l); err != nil {
		return RegisterFootprint{}, err
	}
	bpsm, _ := a.blocksPerSM(l)
	if bpsm == 0 {
		return RegisterFootprint{}, ErrLaneUnlaunchable
	}
	w := ceilDivInt(l.ThreadsPerBlock, a.WarpSize)
	demand := l.RegsPerThread * a.WarpSize
	perWarp, fitWarps, grantedWarps := a.registerWarps(l)
	residentWarps := bpsm * w
	return RegisterFootprint{
		RegsPerSM:        int64(a.RegsPerSM),
		Useful:           int64(residentWarps * demand),
		RoundingWaste:    int64(residentWarps * (perWarp - demand)),
		GranularityWaste: int64((fitWarps - grantedWarps) * perWarp),
		HeadroomUnused:   int64((grantedWarps - residentWarps) * perWarp),
		Remainder:        int64(a.RegsPerSM - fitWarps*perWarp),
	}, nil
}

// Conserved reports whether the parts are non-negative and partition the whole register file. A zero
// footprint (a refused launch) is not conserved.
func (x RegisterFootprint) Conserved() bool {
	if x.RegsPerSM <= 0 {
		return false
	}
	parts := [...]int64{x.Useful, x.RoundingWaste, x.GranularityWaste, x.HeadroomUnused, x.Remainder}
	var sum int64
	for _, p := range parts {
		if p < 0 {
			return false
		}
		sum += p
	}
	return sum == x.RegsPerSM
}

// RegisterGapFinding is one register-file loss class in 32-bit registers per SM. It is a separate type
// from GapFinding so a register loss cannot be summed with a lane-slot loss by mistake.
type RegisterGapFinding struct {
	Name             string  `json:"name"`
	RegistersPerSM   int64   `json:"registers_per_sm"`
	ShareOfRegsPerSM float64 `json:"share_of_regs_per_sm"`
	Trigger          string  `json:"trigger"`
}

// Findings lists the register-file losses, largest first. Headroom counts only while a non-register
// limiter binds: under a register limit it is the warps left over after the register ceiling has
// already taken whole blocks, so no other resource could use them.
func (x RegisterFootprint) Findings(limiter string) []RegisterGapFinding {
	var out []RegisterGapFinding
	add := func(name string, regs int64, trigger string) {
		if regs > 0 {
			out = append(out, RegisterGapFinding{
				Name: name, RegistersPerSM: regs,
				ShareOfRegsPerSM: ratio(regs, x.RegsPerSM), Trigger: trigger,
			})
		}
	}
	add("register_rounding", x.RoundingWaste, "per-warp registers rounded up to RegAllocUnit")
	add("register_granularity", x.GranularityWaste, "warp count rounded down to WarpAllocGranularity")
	if limiter != "register" {
		add("register_headroom", x.HeadroomUnused, "limiter="+limiter+" binds; registers left for whole warps")
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].RegistersPerSM != out[j].RegistersPerSM {
			return out[i].RegistersPerSM > out[j].RegistersPerSM
		}
		return out[i].Name < out[j].Name
	})
	return out
}
