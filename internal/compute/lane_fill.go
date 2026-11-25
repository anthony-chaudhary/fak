package compute

// Vendor-neutral lane-fill model for one execution unit (wavefront, subgroup, simd-group or SIMD vector).
// Every lane slot of a pass lands in exactly one bucket, so capacity is an exact integer partition. There is
// no clock and no measurement here, and no stall cause is named.

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// TailMode selects how the items left over after the last full group are executed.
type TailMode string

const (
	// TailMasked runs the tail group as a full group: lanes past the last item stay in the unit, masked.
	TailMasked TailMode = "masked"
	// TailScalar runs each remainder item alone, one lane per unit, so Width-1 lanes of its unit are empty.
	TailScalar TailMode = "scalar"
)

// LaneSlots counts lane slots of one pass. It is a distinct type so a lane-slot count cannot be summed
// with an item or group count by accident.
type LaneSlots int64

// LaneFillUnit names the unit every lane-fill finding is measured in.
const LaneFillUnit = "lane_slot"

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrLaneFillFamily      = errors.New("lane fill: unit Family must be non-empty")
	ErrLaneFillWidth       = errors.New("lane fill: unit Width must be positive")
	ErrLaneFillProvenance  = errors.New("lane fill: unit Provenance must be non-empty")
	ErrLaneFillGroupSize   = errors.New("lane fill: GroupSize must be positive")
	ErrLaneFillItems       = errors.New("lane fill: Items must be positive")
	ErrLaneFillTailMode    = errors.New("lane fill: TailMode must be TailMasked or TailScalar")
	ErrLaneFillScalarGroup = errors.New("lane fill: TailScalar needs GroupSize to be a multiple of Width")
	ErrLaneFillOverflow    = errors.New("lane fill: lane-slot count overflows int64")
	ErrLaneUnitNote        = errors.New("lane unit: caller note must be non-empty")
)

// LaneUnit is one execution unit of Width lanes that issue together. Provenance says where Width came
// from and is required before the unit can be filled.
type LaneUnit struct {
	Family     string `json:"family"`
	Width      int    `json:"width"`
	Provenance string `json:"provenance"`
}

// LaneFillLaunch is one launch over Items work items in groups of GroupSize invocations.
type LaneFillLaunch struct {
	Kernel    string   `json:"kernel"`
	Items     int64    `json:"items"`
	GroupSize int      `json:"group_size"`
	TailMode  TailMode `json:"tail_mode"`
}

// LaneFill is the exact lane-slot partition of one launch on one unit. Groups is ceil(Items/GroupSize)
// under both modes; under TailScalar the last Items%GroupSize items run as scalar remainder rather than
// as a group.
type LaneFill struct {
	Family        string    `json:"family"`
	Width         int       `json:"width"`
	Provenance    string    `json:"provenance"`
	Kernel        string    `json:"kernel"`
	TailMode      TailMode  `json:"tail_mode"`
	GroupSize     int       `json:"group_size"`
	Items         int64     `json:"items"`
	Groups        int64     `json:"groups"`
	UnitsPerGroup int64     `json:"units_per_group"`
	Capacity      LaneSlots `json:"capacity"`
	Active        LaneSlots `json:"active"`
	PartialUnit   LaneSlots `json:"partial_unit"`
	TailIdle      LaneSlots `json:"tail_idle"`
	ScalarTail    LaneSlots `json:"scalar_tail"`
}

// LaneFillFinding is one non-zero loss class of a LaneFill, measured in lane slots.
type LaneFillFinding struct {
	Name            string    `json:"name"`
	Unit            string    `json:"unit"`
	LaneSlots       LaneSlots `json:"lane_slots"`
	ShareOfCapacity float64   `json:"share_of_capacity"`
	Trigger         string    `json:"trigger"`
}

// CallerLaneUnit builds a unit whose width is a caller input. note records where the number came from
// and becomes part of Provenance.
func CallerLaneUnit(family string, width int, note string) (LaneUnit, error) {
	u := LaneUnit{Family: family, Width: width, Provenance: "caller input: " + note}
	if err := u.validate(); err != nil {
		return LaneUnit{}, err
	}
	if strings.TrimSpace(note) == "" {
		return LaneUnit{}, ErrLaneUnitNote
	}
	return u, nil
}

// StrixHaloWavefrontLaneUnit is the wavefront unit read from StrixHaloWavefrontSize, which mtp_rdna35.go
// documents as RDNA 3.5 Wave32.
func StrixHaloWavefrontLaneUnit() LaneUnit {
	return LaneUnit{
		Family:     "wavefront",
		Width:      StrixHaloWavefrontSize,
		Provenance: "repo constant StrixHaloWavefrontSize (code assumption, not a vendor citation)",
	}
}

// Wave32KDALaneUnit is the Wave32 wavefront unit of the KDA kernels, read from Wave32WavefrontSize.
func Wave32KDALaneUnit() LaneUnit {
	return LaneUnit{
		Family:     "wavefront",
		Width:      Wave32WavefrontSize,
		Provenance: "repo constant Wave32WavefrontSize (code assumption, not a vendor citation)",
	}
}

// VulkanWave32SubgroupLaneUnit is the Vulkan Wave32 subgroup unit, read from vulkanWave32SubgroupSz.
func VulkanWave32SubgroupLaneUnit() LaneUnit {
	return LaneUnit{
		Family:     "vulkan_subgroup",
		Width:      vulkanWave32SubgroupSz,
		Provenance: "repo constant vulkanWave32SubgroupSz (code assumption, not a vendor citation)",
	}
}

func (u LaneUnit) validate() error {
	switch {
	case u.Family == "":
		return ErrLaneFillFamily
	case u.Width <= 0:
		return ErrLaneFillWidth
	case u.Provenance == "":
		return ErrLaneFillProvenance
	}
	return nil
}

// Fill partitions the lane slots of launch l on unit u. It refuses a unit or launch it cannot price
// rather than returning a partial partition.
func (u LaneUnit) Fill(l LaneFillLaunch) (LaneFill, error) {
	if err := u.validate(); err != nil {
		return LaneFill{}, err
	}
	if l.GroupSize <= 0 {
		return LaneFill{}, ErrLaneFillGroupSize
	}
	if l.Items <= 0 {
		return LaneFill{}, ErrLaneFillItems
	}
	switch l.TailMode {
	case TailMasked:
	case TailScalar:
		if l.GroupSize%u.Width != 0 {
			return LaneFill{}, ErrLaneFillScalarGroup
		}
	default:
		return LaneFill{}, ErrLaneFillTailMode
	}
	f, ok := laneFillPartition(l.Items, int64(u.Width), int64(l.GroupSize), l.TailMode)
	if !ok {
		return LaneFill{}, ErrLaneFillOverflow
	}
	f.Family = u.Family
	f.Width = u.Width
	f.Provenance = u.Provenance
	f.Kernel = l.Kernel
	f.TailMode = l.TailMode
	f.GroupSize = l.GroupSize
	f.Items = l.Items
	return f, nil
}

// laneFillPartition computes the parts of a launch from non-negative counts with width and group positive.
// Under TailMasked every group uses ceil(group/width) units. Under TailScalar the Items/group full groups
// use the same units and the Items%group remainder items take one unit each.
func laneFillPartition(items, width, group int64, mode TailMode) (LaneFill, bool) {
	unitsPerGroup := laneFillCeilDiv(group, width)
	slotsPerGroup, ok := laneFillMul(unitsPerGroup, width)
	if !ok {
		return LaneFill{}, false
	}
	partialPerGroup := slotsPerGroup - group
	groups := laneFillCeilDiv(items, group)
	f := LaneFill{Groups: groups, UnitsPerGroup: unitsPerGroup, Active: LaneSlots(items)}

	if mode == TailMasked {
		capacity, ok1 := laneFillMul(groups, slotsPerGroup)
		partial, ok2 := laneFillMul(groups, partialPerGroup)
		groupItems, ok3 := laneFillMul(groups, group)
		if !(ok1 && ok2 && ok3) {
			return LaneFill{}, false
		}
		f.Capacity = LaneSlots(capacity)
		f.PartialUnit = LaneSlots(partial)
		f.TailIdle = LaneSlots(groupItems - items)
		return f, true
	}

	full, rem := items/group, items%group
	fullSlots, ok1 := laneFillMul(full, slotsPerGroup)
	remSlots, ok2 := laneFillMul(rem, width)
	capacity, ok3 := laneFillAdd(fullSlots, remSlots)
	partial, ok4 := laneFillMul(full, partialPerGroup)
	scalar, ok5 := laneFillMul(rem, width-1)
	if !(ok1 && ok2 && ok3 && ok4 && ok5) {
		return LaneFill{}, false
	}
	f.Capacity = LaneSlots(capacity)
	f.PartialUnit = LaneSlots(partial)
	f.ScalarTail = LaneSlots(scalar)
	return f, true
}

func laneFillCeilDiv(a, b int64) int64 {
	q := a / b
	if a%b != 0 {
		q++
	}
	return q
}

// laneFillMul multiplies two non-negative counts and reports whether the product fits in int64.
func laneFillMul(a, b int64) (int64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > math.MaxInt64/b {
		return 0, false
	}
	return a * b, true
}

// laneFillAdd adds two non-negative counts and reports whether the sum fits in int64.
func laneFillAdd(a, b int64) (int64, bool) {
	if a > math.MaxInt64-b {
		return 0, false
	}
	return a + b, true
}

// Conserved reports whether the four lane-slot parts are non-negative and sum exactly to Capacity. A
// zero Capacity is not conserved, so a refused or unfilled value cannot read as a valid partition.
func (f LaneFill) Conserved() bool {
	if f.Capacity <= 0 {
		return false
	}
	remaining := f.Capacity
	for _, part := range [...]LaneSlots{f.Active, f.PartialUnit, f.TailIdle, f.ScalarTail} {
		if part < 0 || part > remaining {
			return false
		}
		remaining -= part
	}
	return remaining == 0
}

// Findings lists the non-zero losses, largest first, ties broken by name. Each loss names the unit,
// width, group size and item count it was computed from.
func (f LaneFill) Findings() []LaneFillFinding {
	var out []LaneFillFinding
	add := func(name string, slots LaneSlots, text string) {
		if slots <= 0 {
			return
		}
		out = append(out, LaneFillFinding{
			Name:            name,
			Unit:            LaneFillUnit,
			LaneSlots:       slots,
			ShareOfCapacity: ratio(int64(slots), int64(f.Capacity)),
			Trigger:         f.trigger(text),
		})
	}
	add("partial_unit", f.PartialUnit,
		"group size is not a multiple of the width, so lanes beyond the group size stay empty in every group")
	add("tail_idle", f.TailIdle,
		"the last group holds lanes with no item because the item count is not a multiple of the group size")
	add("scalar_tail", f.ScalarTail,
		"remainder items run one per unit, leaving the other lanes of each unit empty")
	sort.Slice(out, func(i, j int) bool {
		if out[i].LaneSlots != out[j].LaneSlots {
			return out[i].LaneSlots > out[j].LaneSlots
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func (f LaneFill) trigger(text string) string {
	return fmt.Sprintf("family=%s width=%d group_size=%d items=%d: %s",
		f.Family, f.Width, f.GroupSize, f.Items, text)
}
