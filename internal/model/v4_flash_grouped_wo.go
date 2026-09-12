package model

import (
	"errors"
	"fmt"
)

// v4_flash_grouped_wo.go — grouped-geometry derivation for DeepSeek V4 Flash's
// published `layers.N.attn.wo_a.weight`. Flash publishes that tensor as a flat
// dense FP8 E4M3 matrix of shape [8192, 4096], but its semantics are a GROUPED
// output projection whose logical shape is [o_groups, o_lora_rank, hidden] =
// [8, 1024, 4096] at the pinned revision. The flat row axis is therefore the
// concatenation of OGroups contiguous group blocks, each GroupRows tall:
//
//	flat row r   -> group g = r / GroupRows, within-group row r % GroupRows
//
// The derivation is a pure shape check: it never allocates or touches the weight
// bytes, so the loader can refuse an inconsistent geometry before any payload
// read. The grouped view is exposed as v4GroupedWoA so a future consumer can
// slice one group without expanding every group at once.

// ErrV4GroupedWoAGeometry classifies every refusal from v4GroupedWoAFromShape
// and v4ValidateGroupedWoAWeightShape. Callers check it with errors.Is.
var ErrV4GroupedWoAGeometry = errors.New("model: DeepSeek V4 Flash wo_a grouped geometry mismatch")

// v4GroupedWoA is the logical geometry of the published wo_a output projection.
// The flat [Groups*GroupRows, Hidden] weight is Groups contiguous blocks, each
// GroupRows rows tall, projected onto a Hidden-wide input.
type v4GroupedWoA struct {
	Groups    int
	GroupRows int
	Hidden    int
}

// flatRows is the row count the flat published [rows, cols] weight must carry.
func (g v4GroupedWoA) flatRows() int { return g.Groups * g.GroupRows }

// v4GroupedWoAFromShape derives the grouped geometry of `wo_a` from the
// admitted config and the flat published weight shape [rows, cols]. It refuses,
// via ErrV4GroupedWoAGeometry, when the config does not define a positive
// grouped geometry, when o_lora_rank is incompatible with the Q8 block size qBlk,
// when the flat row axis does not equal OGroups*OLoraRank, or when either
// dimension is non-positive. No weight bytes are read or allocated.
func v4GroupedWoAFromShape(cfg Config, rows, cols int) (v4GroupedWoA, error) {
	if cfg.OGroups <= 0 || cfg.OLoraRank <= 0 {
		return v4GroupedWoA{}, fmt.Errorf("%w: shape requires positive o_groups and o_lora_rank, got o_groups=%d o_lora_rank=%d", ErrV4GroupedWoAGeometry, cfg.OGroups, cfg.OLoraRank)
	}
	if cfg.OLoraRank%qBlk != 0 {
		return v4GroupedWoA{}, fmt.Errorf("%w: shape requires o_lora_rank divisible by qBlk=%d, got o_lora_rank=%d", ErrV4GroupedWoAGeometry, qBlk, cfg.OLoraRank)
	}
	wantRows := cfg.OGroups * cfg.OLoraRank
	if rows != wantRows {
		return v4GroupedWoA{}, fmt.Errorf("%w: flat weight shape rows=%d, want o_groups*o_lora_rank=%d", ErrV4GroupedWoAGeometry, rows, wantRows)
	}
	if cols <= 0 {
		return v4GroupedWoA{}, fmt.Errorf("%w: flat weight shape cols=%d, want positive hidden", ErrV4GroupedWoAGeometry, cols)
	}
	if cols%qBlk != 0 {
		return v4GroupedWoA{}, fmt.Errorf("%w: flat weight shape cols=%d is not divisible by qBlk=%d", ErrV4GroupedWoAGeometry, cols, qBlk)
	}
	return v4GroupedWoA{Groups: cfg.OGroups, GroupRows: cfg.OLoraRank, Hidden: cols}, nil
}

// v4ValidateGroupedWoAWeightShape validates a loaded/decoded wo_a weight shape
// against the admitted config, refusing before the weight bytes are used. It is
// the loader-facing wrapper over v4GroupedWoAFromShape.
func v4ValidateGroupedWoAWeightShape(cfg Config, rows, cols int) (v4GroupedWoA, error) {
	return v4GroupedWoAFromShape(cfg, rows, cols)
}

// extractGroup returns the GroupRows*Hidden sub-slice for group g from a flat
// row-major slice of Groups*GroupRows*Hidden values. Deterministic, non-copying.
func (g v4GroupedWoA) extractGroup(flat []float32, group int) []float32 {
	if group < 0 || group >= g.Groups {
		panic(fmt.Sprintf("model: v4GroupedWoA group %d out of range [0,%d)", group, g.Groups))
	}
	if g.Groups <= 0 || g.GroupRows <= 0 || g.Hidden <= 0 {
		panic("model: v4GroupedWoA has non-positive geometry")
	}
	want := g.flatRows() * g.Hidden
	if len(flat) != want {
		panic(fmt.Sprintf("model: v4GroupedWoA flat length %d, want %d", len(flat), want))
	}
	start := group * g.GroupRows * g.Hidden
	return flat[start : start+g.GroupRows*g.Hidden]
}
