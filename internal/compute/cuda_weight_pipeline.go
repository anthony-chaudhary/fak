// Clean-room implementation inspired by IST-DASLab/marlin@1f25790bdd49fba53106164a24666dade68d7c90 (Apache-2.0):
// multi-stage cp.async global->shared weight pipeline descriptor and L2 evict_first cache-hint policy.
package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// L2CacheHint encodes the PTX/CUDA L2 residency hint attached to global memory loads.
type L2CacheHint uint8

const (
	L2CacheNormal     L2CacheHint = iota
	L2CacheEvictFirst             // Streaming hint (evict_first) for large weight tiles.
	L2CachePersist                // Residency hint (evict_last / persistent) for reused activations.
)

func (h L2CacheHint) String() string {
	switch h {
	case L2CacheNormal:
		return "evict_normal"
	case L2CacheEvictFirst:
		return "evict_first"
	case L2CachePersist:
		return "evict_last"
	default:
		return fmt.Sprintf("unknown(%d)", h)
	}
}

// PipelineLoadRole distinguishes operand streams sharing the L2 cache.
type PipelineLoadRole uint8

const (
	PipelineRoleWeight PipelineLoadRole = iota
	PipelineRoleActivation
	PipelineRoleScale
)

// PipelineStep models one async producer/consumer step in the circular cp.async schedule.
type PipelineStep struct {
	KTileIndex     int
	StageSlot      int
	IssueWeightDMA bool
	WeightL2Hint   L2CacheHint
	ActL2Hint      L2CacheHint
	WaitGroupDepth int // Argument to cp.async.wait_group(N) before reading StageSlot.
	ComputeSlot    int // -1 during pure prologue fill steps.
}

// WeightPipelineDescriptor captures the validated double/multi-buffered global->shared
// weight staging contract and per-stream L2 eviction policy.
type WeightPipelineDescriptor struct {
	Format           Dtype
	OutDim           int
	InDim            int
	TileK            int
	Stages           int
	SharedBytesStage int
	TotalSharedBytes int
	WeightL2Hint     L2CacheHint
	ActivationL2Hint L2CacheHint
	ScaleL2Hint      L2CacheHint
	Schedule         []PipelineStep
}

// DefaultWeightPipelinePolicy returns the Marlin-derived L2 hint split:
// weights stream with evict_first so they never thrash activations in L2.
func DefaultWeightPipelinePolicy() (weightHint, actHint, scaleHint L2CacheHint) {
	return L2CacheEvictFirst, L2CachePersist, L2CachePersist
}

// BuildWeightPipelineDescriptor constructs and validates a double/multi-buffered weight
// pipeline schedule for AWQ or Q4_K weight matrices without altering packed weight layouts.
func BuildWeightPipelineDescriptor(format Dtype, outDim, inDim, tileK, stages int) (*WeightPipelineDescriptor, error) {
	if outDim <= 0 || inDim <= 0 {
		return nil, fmt.Errorf("weight pipeline: invalid dimensions out=%d in=%d", outDim, inDim)
	}
	if stages < 2 || stages > 4 {
		return nil, fmt.Errorf("weight pipeline: stages must be in [2, 4], got %d", stages)
	}
	if tileK <= 0 {
		return nil, fmt.Errorf("weight pipeline: tileK must be positive, got %d", tileK)
	}

	var bytesPerStage int
	switch format {
	case Q4_K:
		if inDim%q4kSuper != 0 {
			return nil, fmt.Errorf("weight pipeline Q4_K: inDim=%d must be multiple of %d", inDim, q4kSuper)
		}
		if tileK%q4kSuper != 0 {
			return nil, fmt.Errorf("weight pipeline Q4_K: tileK=%d must be multiple of %d", tileK, q4kSuper)
		}
		bytesPerStage = outDim * (tileK / q4kSuper) * q4kSuperBlock
	case I4: // AWQ 4-bit packed nibbles + fp32/fp16 scales.
		if inDim%2 != 0 || tileK%2 != 0 {
			return nil, fmt.Errorf("weight pipeline AWQ: inDim=%d and tileK=%d must be even", inDim, tileK)
		}
		bytesPerStage = outDim*(tileK/2) + outDim*4
	default:
		return nil, fmt.Errorf("weight pipeline: unsupported weight dtype %s (want Q4_K or I4/AWQ)", format)
	}

	wHint, aHint, sHint := DefaultWeightPipelinePolicy()
	nTiles := (inDim + tileK - 1) / tileK
	schedule := make([]PipelineStep, 0, nTiles+stages)

	// Prologue: pre-fill up to min(nTiles, stages-1) stages with cp.async global->shared.
	prefill := stages - 1
	if nTiles < prefill {
		prefill = nTiles
	}
	for t := 0; t < prefill; t++ {
		schedule = append(schedule, PipelineStep{
			KTileIndex:     t,
			StageSlot:      t % stages,
			IssueWeightDMA: true,
			WeightL2Hint:   wHint,
			ActL2Hint:      aHint,
			WaitGroupDepth: t,
			ComputeSlot:    -1,
		})
	}

	// Steady state: issue tile t's cp.async while draining the oldest completed stage.
	computeIdx := 0
	for t := prefill; t < nTiles; t++ {
		schedule = append(schedule, PipelineStep{
			KTileIndex:     t,
			StageSlot:      t % stages,
			IssueWeightDMA: true,
			WeightL2Hint:   wHint,
			ActL2Hint:      aHint,
			WaitGroupDepth: stages - 2,
			ComputeSlot:    computeIdx % stages,
		})
		computeIdx++
	}

	// Epilogue: drain remaining prefetched stages with wait_group(remaining - 1).
	for computeIdx < nTiles {
		remaining := nTiles - computeIdx - 1
		schedule = append(schedule, PipelineStep{
			KTileIndex:     -1,
			StageSlot:      -1,
			IssueWeightDMA: false,
			WeightL2Hint:   wHint,
			ActL2Hint:      aHint,
			WaitGroupDepth: remaining,
			ComputeSlot:    computeIdx % stages,
		})
		computeIdx++
	}

	desc := &WeightPipelineDescriptor{
		Format:           format,
		OutDim:           outDim,
		InDim:            inDim,
		TileK:            tileK,
		Stages:           stages,
		SharedBytesStage: bytesPerStage,
		TotalSharedBytes: bytesPerStage * stages,
		WeightL2Hint:     wHint,
		ActivationL2Hint: aHint,
		ScaleL2Hint:      sHint,
		Schedule:         schedule,
	}
	if err := desc.Validate(); err != nil {
		return nil, err
	}
	return desc, nil
}

// Validate checks double-buffer invariants and ensures L2 evict_first is applied exclusively to weights.
func (d *WeightPipelineDescriptor) Validate() error {
	if d == nil {
		return errors.New("weight pipeline: nil descriptor")
	}
	if d.Stages < 2 {
		return fmt.Errorf("weight pipeline: requires >= 2 stages for overlap, got %d", d.Stages)
	}
	if d.WeightL2Hint != L2CacheEvictFirst {
		return fmt.Errorf("weight pipeline: weight load must use evict_first, got %s", d.WeightL2Hint)
	}
	if d.ActivationL2Hint == L2CacheEvictFirst {
		return errors.New("weight pipeline: activations must not use evict_first L2 policy")
	}
	if d.ScaleL2Hint == L2CacheEvictFirst {
		return errors.New("weight pipeline: scales must not use evict_first L2 policy")
	}
	for i, step := range d.Schedule {
		if step.IssueWeightDMA && step.WeightL2Hint != L2CacheEvictFirst {
			return fmt.Errorf("weight pipeline step %d: weight DMA missing evict_first hint", i)
		}
		if step.ActL2Hint == L2CacheEvictFirst {
			return fmt.Errorf("weight pipeline step %d: activation load wrongly tagged evict_first", i)
		}
		if step.ComputeSlot >= d.Stages {
			return fmt.Errorf("weight pipeline step %d: compute slot %d >= stages %d", i, step.ComputeSlot, d.Stages)
		}
	}
	return nil
}

// PipelineHintForRole resolves the exact L2 cache hint enforced for a memory load stream.
func (d *WeightPipelineDescriptor) PipelineHintForRole(role PipelineLoadRole) L2CacheHint {
	switch role {
	case PipelineRoleWeight:
		return d.WeightL2Hint
	case PipelineRoleActivation:
		return d.ActivationL2Hint
	case PipelineRoleScale:
		return d.ScaleL2Hint
	default:
		return L2CacheNormal
	}
}

// RunWeightPipelineAWQ executes host-side staged AWQ matmul matching the pipeline schedule
// while preserving packed [outDim, inDim/2] byte layout verbatim.
func RunWeightPipelineAWQ(desc *WeightPipelineDescriptor, packed []byte, scales []float32, x []float32, rows int) ([]float32, error) {
	if err := desc.Validate(); err != nil {
		return nil, err
	}
	if desc.Format != I4 {
		return nil, fmt.Errorf("weight pipeline AWQ: descriptor format %s != I4", desc.Format)
	}
	if rows <= 0 {
		return nil, fmt.Errorf("weight pipeline AWQ: rows must be > 0, got %d", rows)
	}
	rowBytes := desc.InDim / 2
	if len(packed) != desc.OutDim*rowBytes {
		return nil, fmt.Errorf("weight pipeline AWQ: packed len %d != %d", len(packed), desc.OutDim*rowBytes)
	}
	if len(scales) != desc.OutDim {
		return nil, fmt.Errorf("weight pipeline AWQ: scales len %d != %d", len(scales), desc.OutDim)
	}
	if len(x) != rows*desc.InDim {
		return nil, fmt.Errorf("weight pipeline AWQ: activation len %d != %d", len(x), rows*desc.InDim)
	}

	stageTiles := make([][]byte, desc.Stages)
	stageTileIdx := make([]int, desc.Stages)
	for s := range stageTiles {
		stageTiles[s] = make([]byte, desc.OutDim*(desc.TileK/2))
		stageTileIdx[s] = -1
	}

	out := make([]float32, rows*desc.OutDim)
	for _, step := range desc.Schedule {
		if step.IssueWeightDMA {
			// Stage weight bytes for kTile from global packed tensor (read-only).
			t := step.KTileIndex
			colStart := t * desc.TileK
			colEnd := colStart + desc.TileK
			if colEnd > desc.InDim {
				colEnd = desc.InDim
			}
			spanBytes := (colEnd - colStart) / 2
			buf := stageTiles[step.StageSlot]
			for r := 0; r < desc.OutDim; r++ {
				srcOff := r*rowBytes + colStart/2
				dstOff := r * (desc.TileK / 2)
				copy(buf[dstOff:dstOff+spanBytes], packed[srcOff:srcOff+spanBytes])
			}
			stageTileIdx[step.StageSlot] = t
		}
		if step.ComputeSlot >= 0 {
			t := stageTileIdx[step.ComputeSlot]
			if t < 0 {
				return nil, fmt.Errorf("weight pipeline AWQ: consume uninitialized slot %d", step.ComputeSlot)
			}
			colStart := t * desc.TileK
			colEnd := colStart + desc.TileK
			if colEnd > desc.InDim {
				colEnd = desc.InDim
			}
			spanBytes := (colEnd - colStart) / 2
			buf := stageTiles[step.ComputeSlot]
			for row := 0; row < rows; row++ {
				xTile := x[row*desc.InDim+colStart : row*desc.InDim+colEnd]
				outRow := out[row*desc.OutDim : (row+1)*desc.OutDim]
				for r := 0; r < desc.OutDim; r++ {
					wBytes := buf[r*(desc.TileK/2) : r*(desc.TileK/2)+spanBytes]
					sc := scales[r]
					var acc float32
					for k := 0; k < spanBytes; k++ {
						b := wBytes[k]
						lo := float32(int(b&0x0f)-8) * sc
						hi := float32(int(b>>4)-8) * sc
						acc += lo*xTile[2*k] + hi*xTile[2*k+1]
					}
					outRow[r] += acc
				}
			}
		}
	}
	return out, nil
}

// RunWeightPipelineQ4K executes host-side staged Q4_K matmul matching the pipeline schedule
// while preserving verbatim 144-byte GGUF Q4_K super-block layout and four-accumulator reduction.
func RunWeightPipelineQ4K(desc *WeightPipelineDescriptor, raw []byte, x []float32, rows int) ([]float32, error) {
	if err := desc.Validate(); err != nil {
		return nil, err
	}
	if desc.Format != Q4_K {
		return nil, fmt.Errorf("weight pipeline Q4_K: descriptor format %s != Q4_K", desc.Format)
	}
	if rows <= 0 {
		return nil, fmt.Errorf("weight pipeline Q4_K: rows must be > 0, got %d", rows)
	}
	blocksPerRow := desc.InDim / q4kSuper
	rowBytes := blocksPerRow * q4kSuperBlock
	if len(raw) != desc.OutDim*rowBytes {
		return nil, fmt.Errorf("weight pipeline Q4_K: raw len %d != %d", len(raw), desc.OutDim*rowBytes)
	}
	if len(x) != rows*desc.InDim {
		return nil, fmt.Errorf("weight pipeline Q4_K: activation len %d != %d", len(x), rows*desc.InDim)
	}

	blocksPerTile := desc.TileK / q4kSuper
	tileRowBytes := blocksPerTile * q4kSuperBlock
	stageTiles := make([][]byte, desc.Stages)
	stageTileIdx := make([]int, desc.Stages)
	for s := range stageTiles {
		stageTiles[s] = make([]byte, desc.OutDim*tileRowBytes)
		stageTileIdx[s] = -1
	}

	dequantBuf := make([]float32, q4kSuper)
	out := make([]float32, rows*desc.OutDim)

	for _, step := range desc.Schedule {
		if step.IssueWeightDMA {
			t := step.KTileIndex
			blkStart := t * blocksPerTile
			buf := stageTiles[step.StageSlot]
			for r := 0; r < desc.OutDim; r++ {
				srcOff := r*rowBytes + blkStart*q4kSuperBlock
				dstOff := r * tileRowBytes
				copy(buf[dstOff:dstOff+tileRowBytes], raw[srcOff:srcOff+tileRowBytes])
			}
			stageTileIdx[step.StageSlot] = t
		}
		if step.ComputeSlot >= 0 {
			t := stageTileIdx[step.ComputeSlot]
			if t < 0 {
				return nil, fmt.Errorf("weight pipeline Q4_K: consume uninitialized slot %d", step.ComputeSlot)
			}
			colStart := t * desc.TileK
			buf := stageTiles[step.ComputeSlot]
			for row := 0; row < rows; row++ {
				xTile := x[row*desc.InDim+colStart : row*desc.InDim+colStart+desc.TileK]
				outRow := out[row*desc.OutDim : (row+1)*desc.OutDim]
				for r := 0; r < desc.OutDim; r++ {
					wRowTile := buf[r*tileRowBytes : (r+1)*tileRowBytes]
					outRow[r] += q4kTileDot(wRowTile, xTile, dequantBuf)
				}
			}
		}
	}
	return out, nil
}

func q4kTileDot(tileBytes []byte, x []float32, dequantBuf []float32) float32 {
	nblk := len(x) / q4kSuper
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := tileBytes[b*q4kSuperBlock : (b+1)*q4kSuperBlock]
		q4kDequantBlockExact(dequantBuf, blk)
		xs := x[b*q4kSuper:]
		var s0, s1, s2, s3 float32
		for i := 0; i < q4kSuper; i += 4 {
			s0 += dequantBuf[i] * xs[i]
			s1 += dequantBuf[i+1] * xs[i+1]
			s2 += dequantBuf[i+2] * xs[i+2]
			s3 += dequantBuf[i+3] * xs[i+3]
		}
		acc += (s0 + s1) + (s2 + s3)
	}
	return acc
}

func q4kDequantBlockExact(dst []float32, blk []byte) {
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
	scales := blk[4 : 4+12]
	q := blk[4+12 : q4kSuperBlock]
	qi, is := 0, 0
	for j := 0; j < q4kSuper; j += 64 {
		sc, m := kquantbits.ScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = kquantbits.ScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			dst[j+l] = d1*float32(q[qi+l]&0x0f) - m1
		}
		for l := 0; l < 32; l++ {
			dst[j+32+l] = d2*float32(q[qi+l]>>4) - m2
		}
		qi += 32
		is += 2
	}
}
