// semaphore-based producer/consumer warp handoff and load-time act-order column_remap permutation.
package compute

import (
	"errors"
	"fmt"
	"math"
)

// AWQWarpRole identifies the producer (global->shared loader) or consumer (MMA/dequant) warp group.
type AWQWarpRole uint8

const (
	AWQRoleProducer AWQWarpRole = iota
	AWQRoleConsumer
)

// AWQTileSemaphore models a bounded multi-stage producer/consumer circular handoff
// queue mirroring warp-group empty/full barrier synchronization in quantized GEMM tiles.
type AWQTileSemaphore struct {
	stages        int
	tileK         int
	emptyCount    int
	filledCount   int
	producerSlot  int
	consumerSlot  int
	inFlightProd  []bool
	inFlightCons  []bool
	totalProduced int
	totalConsumed int
}

// NewAWQTileSemaphore constructs a validated producer/consumer stage semaphore.
// At initialization all stages are empty and ready for producer acquisition.
func NewAWQTileSemaphore(stages, tileK int) (*AWQTileSemaphore, error) {
	if stages < 2 {
		return nil, fmt.Errorf("awq warp sync: stages must be >= 2, got %d", stages)
	}
	if tileK <= 0 || tileK%2 != 0 {
		return nil, fmt.Errorf("awq warp sync: tileK must be positive and even, got %d", tileK)
	}
	return &AWQTileSemaphore{
		stages:       stages,
		tileK:        tileK,
		emptyCount:   stages,
		inFlightProd: make([]bool, stages),
		inFlightCons: make([]bool, stages),
	}, nil
}

// Stages returns the bounded circular pipeline depth.
func (s *AWQTileSemaphore) Stages() int { return s.stages }

// TileK returns the K-tile width handled per stage handoff.
func (s *AWQTileSemaphore) TileK() int { return s.tileK }

// EmptyCount returns stages available for producer writes.
func (s *AWQTileSemaphore) EmptyCount() int { return s.emptyCount }

// FilledCount returns stages committed by producers and ready for consumers.
func (s *AWQTileSemaphore) FilledCount() int { return s.filledCount }

// TotalProduced returns completed producer releases.
func (s *AWQTileSemaphore) TotalProduced() int { return s.totalProduced }

// TotalConsumed returns completed consumer releases.
func (s *AWQTileSemaphore) TotalConsumed() int { return s.totalConsumed }

// AcquireEmpty claims the next empty stage slot for a producer warp.
// Returns the claimed slot index or an error if all stages are occupied.
func (s *AWQTileSemaphore) AcquireEmpty() (int, error) {
	if s.emptyCount <= 0 {
		return -1, errors.New("awq warp sync: producer stall - no empty stage available")
	}
	slot := s.producerSlot
	if s.inFlightProd[slot] || s.inFlightCons[slot] {
		return -1, fmt.Errorf("awq warp sync: stage %d already in flight", slot)
	}
	s.emptyCount--
	s.inFlightProd[slot] = true
	s.producerSlot = (s.producerSlot + 1) % s.stages
	return slot, nil
}

// ReleaseFilled signals that producer writes into slot are visible to consumer warps.
func (s *AWQTileSemaphore) ReleaseFilled(slot int) error {
	if slot < 0 || slot >= s.stages {
		return fmt.Errorf("awq warp sync: invalid producer slot %d", slot)
	}
	if !s.inFlightProd[slot] {
		return fmt.Errorf("awq warp sync: producer slot %d not acquired", slot)
	}
	s.inFlightProd[slot] = false
	s.filledCount++
	s.totalProduced++
	return nil
}

// AcquireFilled claims the next filled stage slot for a consumer warp.
// Returns the claimed slot index or an error if no filled tile is ready.
func (s *AWQTileSemaphore) AcquireFilled() (int, error) {
	if s.filledCount <= 0 {
		return -1, errors.New("awq warp sync: consumer stall - no filled stage available")
	}
	slot := s.consumerSlot
	if s.inFlightCons[slot] || s.inFlightProd[slot] {
		return -1, fmt.Errorf("awq warp sync: stage %d already in flight", slot)
	}
	s.filledCount--
	s.inFlightCons[slot] = true
	s.consumerSlot = (s.consumerSlot + 1) % s.stages
	return slot, nil
}

// ReleaseEmpty retires a consumed slot back to the producer pool.
func (s *AWQTileSemaphore) ReleaseEmpty(slot int) error {
	if slot < 0 || slot >= s.stages {
		return fmt.Errorf("awq warp sync: invalid consumer slot %d", slot)
	}
	if !s.inFlightCons[slot] {
		return fmt.Errorf("awq warp sync: consumer slot %d not acquired", slot)
	}
	s.inFlightCons[slot] = false
	s.emptyCount++
	s.totalConsumed++
	return nil
}

// CheckInvariants verifies deadlock freedom and stage conservation:
// emptyCount + filledCount + activeProducerSlots + activeConsumerSlots == stages.
func (s *AWQTileSemaphore) CheckInvariants() error {
	activeProd := 0
	activeCons := 0
	for i := 0; i < s.stages; i++ {
		if s.inFlightProd[i] && s.inFlightCons[i] {
			return fmt.Errorf("awq warp sync: stage %d simultaneously active for producer and consumer", i)
		}
		if s.inFlightProd[i] {
			activeProd++
		}
		if s.inFlightCons[i] {
			activeCons++
		}
	}
	sum := s.emptyCount + s.filledCount + activeProd + activeCons
	if sum != s.stages {
		return fmt.Errorf("awq warp sync: stage conservation broken (%d != %d)", sum, s.stages)
	}
	if s.totalConsumed > s.totalProduced {
		return fmt.Errorf("awq warp sync: consumed %d exceeds produced %d", s.totalConsumed, s.totalProduced)
	}
	return nil
}

// AWQColumnRemap encodes the load-time activation-order (act_order) column
// permutation derived from g_idx so quantized weight columns within each group
// become contiguous for coalesced warp-sync GEMM execution.
type AWQColumnRemap struct {
	InDim          int
	NumGroups      int
	Perm           []int   // Perm[newCol] = origCol (stable argsort of g_idx)
	InvPerm        []int   // InvPerm[origCol] = newCol
	OrigGIdx       []int32 // Original g_idx[origCol] -> group index
	PermutedGIdx   []int32 // PermutedGIdx[newCol] = OrigGIdx[Perm[newCol]] (monotone non-decreasing)
	IsIdentityPerm bool
}

// BuildAWQColumnRemap computes a deterministic, bijective column_remap from g_idx.
// When g_idx is nil or empty, an identity permutation with one group is returned.
func BuildAWQColumnRemap(inDim int, gIdx []int32) (*AWQColumnRemap, error) {
	if inDim <= 0 || inDim%2 != 0 {
		return nil, fmt.Errorf("awq column_remap: inDim must be positive and even, got %d", inDim)
	}
	origG := make([]int32, inDim)
	if len(gIdx) > 0 {
		if len(gIdx) != inDim {
			return nil, fmt.Errorf("awq column_remap: len(gIdx)=%d does not match inDim=%d", len(gIdx), inDim)
		}
		copy(origG, gIdx)
	}

	maxGroup := int32(0)
	for col, g := range origG {
		if g < 0 {
			return nil, fmt.Errorf("awq column_remap: negative group index gIdx[%d]=%d", col, g)
		}
		if g > maxGroup {
			maxGroup = g
		}
	}
	numGroups := int(maxGroup) + 1

	// Counting sort makes argsort O(inDim + numGroups) and preserves intra-group order.
	counts := make([]int, numGroups)
	for _, g := range origG {
		counts[g]++
	}
	offsets := make([]int, numGroups)
	running := 0
	for g := 0; g < numGroups; g++ {
		offsets[g] = running
		running += counts[g]
	}

	perm := make([]int, inDim)
	invPerm := make([]int, inDim)
	permG := make([]int32, inDim)
	cursor := make([]int, numGroups)
	copy(cursor, offsets)

	isIdentity := true
	for origCol, g := range origG {
		newCol := cursor[g]
		cursor[g]++
		perm[newCol] = origCol
		invPerm[origCol] = newCol
		permG[newCol] = g
		if newCol != origCol {
			isIdentity = false
		}
	}

	return &AWQColumnRemap{
		InDim:          inDim,
		NumGroups:      numGroups,
		Perm:           perm,
		InvPerm:        invPerm,
		OrigGIdx:       origG,
		PermutedGIdx:   permG,
		IsIdentityPerm: isIdentity,
	}, nil
}

func readPacked4BitNibble(rowBytes []byte, col int) byte {
	b := rowBytes[col>>1]
	if col&1 == 0 {
		return b & 0x0f
	}
	return (b >> 4) & 0x0f
}

func writePacked4BitNibble(rowBytes []byte, col int, val byte) {
	idx := col >> 1
	val &= 0x0f
	if col&1 == 0 {
		rowBytes[idx] = (rowBytes[idx] & 0xf0) | val
	} else {
		rowBytes[idx] = (rowBytes[idx] & 0x0f) | (val << 4)
	}
}

// RemapAWQPackedWeights permutes [out, in/2] packed uint4 (zero-point=8) weights
// into act-order column order according to remap.Perm without modifying src.
func RemapAWQPackedWeights(src []byte, outDim int, remap *AWQColumnRemap) ([]byte, error) {
	if remap == nil {
		return nil, errors.New("awq column_remap: nil remap")
	}
	if outDim <= 0 {
		return nil, fmt.Errorf("awq column_remap: outDim must be > 0, got %d", outDim)
	}
	rowBytes := remap.InDim / 2
	want := outDim * rowBytes
	if len(src) != want {
		return nil, fmt.Errorf("awq column_remap: packed length %d != expected %d", len(src), want)
	}

	dst := make([]byte, want)
	for r := 0; r < outDim; r++ {
		srcRow := src[r*rowBytes : (r+1)*rowBytes]
		dstRow := dst[r*rowBytes : (r+1)*rowBytes]
		for newCol := 0; newCol < remap.InDim; newCol++ {
			origCol := remap.Perm[newCol]
			q := readPacked4BitNibble(srcRow, origCol)
			writePacked4BitNibble(dstRow, newCol, q)
		}
	}
	return dst, nil
}

// UnremapAWQPackedWeights recovers exact original [out, in/2] packed bytes from
// remapped packed weights using remap.InvPerm.
func UnremapAWQPackedWeights(remapped []byte, outDim int, remap *AWQColumnRemap) ([]byte, error) {
	if remap == nil {
		return nil, errors.New("awq column_remap: nil remap")
	}
	if outDim <= 0 {
		return nil, fmt.Errorf("awq column_remap: outDim must be > 0, got %d", outDim)
	}
	rowBytes := remap.InDim / 2
	want := outDim * rowBytes
	if len(remapped) != want {
		return nil, fmt.Errorf("awq column_remap: remapped length %d != expected %d", len(remapped), want)
	}

	orig := make([]byte, want)
	for r := 0; r < outDim; r++ {
		remRow := remapped[r*rowBytes : (r+1)*rowBytes]
		origRow := orig[r*rowBytes : (r+1)*rowBytes]
		for origCol := 0; origCol < remap.InDim; origCol++ {
			newCol := remap.InvPerm[origCol]
			q := readPacked4BitNibble(remRow, newCol)
			writePacked4BitNibble(origRow, origCol, q)
		}
	}
	return orig, nil
}

// PermuteAWQActivations reorders input activations [rows, inDim] by remap.Perm
// so that activation columns align 1:1 with remapped weight columns during GEMM.
func PermuteAWQActivations(x []float32, rows int, remap *AWQColumnRemap) ([]float32, error) {
	if remap == nil {
		return nil, errors.New("awq column_remap: nil remap")
	}
	if rows <= 0 {
		return nil, fmt.Errorf("awq column_remap: rows must be > 0, got %d", rows)
	}
	want := rows * remap.InDim
	if len(x) != want {
		return nil, fmt.Errorf("awq column_remap: activation length %d != expected %d", len(x), want)
	}
	out := make([]float32, want)
	for r := 0; r < rows; r++ {
		srcRow := x[r*remap.InDim : (r+1)*remap.InDim]
		dstRow := out[r*remap.InDim : (r+1)*remap.InDim]
		for newCol := 0; newCol < remap.InDim; newCol++ {
			dstRow[newCol] = srcRow[remap.Perm[newCol]]
		}
	}
	return out, nil
}

// AWQWarpSyncGemm executes host-side semaphore-coordinated AWQ matmul with act-order column_remap.
// scales may be either per-channel [outDim] or per-group [outDim * remap.NumGroups].
// x has shape [rows, inDim] and output has shape [rows, outDim].
func AWQWarpSyncGemm(packed []byte, scales []float32, x []float32, rows, outDim int, remap *AWQColumnRemap, stages, tileK int) ([]float32, *AWQTileSemaphore, error) {
	if remap == nil {
		return nil, nil, errors.New("awq warp sync gemm: nil remap")
	}
	if rows <= 0 || outDim <= 0 {
		return nil, nil, fmt.Errorf("awq warp sync gemm: invalid shape rows=%d outDim=%d", rows, outDim)
	}
	perChannel := len(scales) == outDim
	perGroup := len(scales) == outDim*remap.NumGroups
	if !perChannel && !perGroup {
		return nil, nil, fmt.Errorf("awq warp sync gemm: scales length %d matches neither outDim=%d nor outDim*groups=%d",
			len(scales), outDim, outDim*remap.NumGroups)
	}

	sem, err := NewAWQTileSemaphore(stages, tileK)
	if err != nil {
		return nil, nil, err
	}

	remappedW, err := RemapAWQPackedWeights(packed, outDim, remap)
	if err != nil {
		return nil, nil, err
	}
	permX, err := PermuteAWQActivations(x, rows, remap)
	if err != nil {
		return nil, nil, err
	}

	rowBytes := remap.InDim / 2
	nTiles := (remap.InDim + tileK - 1) / tileK
	tileBuf := make([][]float32, stages)
	for s := 0; s < stages; s++ {
		tileBuf[s] = make([]float32, outDim*tileK)
	}

	out := make([]float32, rows*outDim)
	// Prologue/steady-state/epilogue producer-consumer warp handoff across K-tiles.
	inFlightSlots := make([]int, 0, stages)
	produceTile := func(tileIdx int) error {
		slot, err := sem.AcquireEmpty()
		if err != nil {
			return err
		}
		kStart := tileIdx * tileK
		kEnd := kStart + tileK
		if kEnd > remap.InDim {
			kEnd = remap.InDim
		}
		buf := tileBuf[slot]
		for i := range buf {
			buf[i] = 0
		}
		for r := 0; r < outDim; r++ {
			wRow := remappedW[r*rowBytes : (r+1)*rowBytes]
			dst := buf[r*tileK : (r+1)*tileK]
			for c := kStart; c < kEnd; c++ {
				q := float32(int(readPacked4BitNibble(wRow, c)) - 8)
				var sc float32
				if perChannel {
					sc = scales[r]
				} else {
					g := int(remap.PermutedGIdx[c])
					sc = scales[r*remap.NumGroups+g]
				}
				dst[c-kStart] = q * sc
			}
		}
		if err := sem.ReleaseFilled(slot); err != nil {
			return err
		}
		inFlightSlots = append(inFlightSlots, tileIdx)
		return nil
	}

	consumeTile := func() error {
		if len(inFlightSlots) == 0 {
			return errors.New("awq warp sync gemm: missing in-flight tile index")
		}
		tileIdx := inFlightSlots[0]
		inFlightSlots = inFlightSlots[1:]
		slot, err := sem.AcquireFilled()
		if err != nil {
			return err
		}
		kStart := tileIdx * tileK
		kEnd := kStart + tileK
		if kEnd > remap.InDim {
			kEnd = remap.InDim
		}
		span := kEnd - kStart
		buf := tileBuf[slot]
		for row := 0; row < rows; row++ {
			xTile := permX[row*remap.InDim+kStart : row*remap.InDim+kEnd]
			outRow := out[row*outDim : (row+1)*outDim]
			for r := 0; r < outDim; r++ {
				wTile := buf[r*tileK : r*tileK+span]
				var acc float32
				for c := 0; c < span; c++ {
					acc += wTile[c] * xTile[c]
				}
				outRow[r] += acc
			}
		}
		return sem.ReleaseEmpty(slot)
	}

	nextProduce := 0
	for nextProduce < nTiles && nextProduce < stages-1 {
		if err := produceTile(nextProduce); err != nil {
			return nil, nil, err
		}
		nextProduce++
	}
	for nextProduce < nTiles {
		if err := produceTile(nextProduce); err != nil {
			return nil, nil, err
		}
		nextProduce++
		if err := consumeTile(); err != nil {
			return nil, nil, err
		}
		if err := sem.CheckInvariants(); err != nil {
			return nil, nil, err
		}
	}
	for len(inFlightSlots) > 0 {
		if err := consumeTile(); err != nil {
			return nil, nil, err
		}
		if err := sem.CheckInvariants(); err != nil {
			return nil, nil, err
		}
	}

	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, errors.New("awq warp sync gemm: non-finite output")
		}
	}
	return out, sem, nil
}
