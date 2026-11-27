// Adapted from DeepGEMM (https://github.com/deepseek-ai/DeepGEMM),
// pinned at revision 5081f8513af57e61696de4f4f04b0e707ffb503e.
// Copyright (c) 2025 DeepSeek
// SPDX-License-Identifier: MIT
// See THIRD_PARTY_NOTICES.md ("DeepGEMM").

package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

const (
	// ReplayBarrierLineBytes is the cache-line stride of each split-barrier slot.
	// Barrier state uses bytes [0..8); line 0's unused tail holds the execution
	// epoch at byte 64 and the retiring CTA exit count at byte 72 so host
	// buffer sizing and ABI remain unchanged across graph replays.
	ReplayBarrierLineBytes       = 128
	ReplayBarrierStateBytes      = 8
	ReplayBarrierEpochOffset     = ReplayBarrierLineBytes / 2
	ReplayBarrierExitCountOffset = ReplayBarrierEpochOffset + 8

	ReplayDefaultNumBarriers = 2
	ReplayDefaultMaxSplits   = 64
)

var (
	// ErrReplayBarrierInvalidConfig reports invalid barrier geometry or CTA counts.
	ErrReplayBarrierInvalidConfig = errors.New("compute: invalid replay epoch barrier config")
	// ErrReplayBarrierOutOfRange reports an out-of-bounds barrier or block index.
	ErrReplayBarrierOutOfRange = errors.New("compute: replay barrier index out of range")
	// ErrEpochReadBeforePDLSync reports a CTA trying to observe or use the
	// replay epoch before completing its grid-dependency sync.
	ErrEpochReadBeforePDLSync = errors.New("compute: replay epoch accessed before grid-dependency sync")
	// ErrReplayBarrierRetireOrder reports duplicate or unbalanced CTA retirement.
	ErrReplayBarrierRetireOrder = errors.New("compute: invalid replay epoch barrier CTA retire")
)

// ReplayBarrierConfig sizes the flat split-barrier line buffer for graph replay.
type ReplayBarrierConfig struct {
	NumBarriers int
	NumMBlocks  int
	MaxSplits   uint32
}

// normalized fills defaults and validates bounds.
func (c ReplayBarrierConfig) normalized() (ReplayBarrierConfig, error) {
	if c.NumBarriers == 0 {
		c.NumBarriers = ReplayDefaultNumBarriers
	}
	if c.MaxSplits == 0 {
		c.MaxSplits = ReplayDefaultMaxSplits
	}
	if c.NumBarriers <= 0 || c.NumMBlocks <= 0 || c.MaxSplits == 0 {
		return ReplayBarrierConfig{}, fmt.Errorf("%w: barriers=%d mblocks=%d max_splits=%d",
			ErrReplayBarrierInvalidConfig, c.NumBarriers, c.NumMBlocks, c.MaxSplits)
	}
	return c, nil
}

// BufferSizeBytes returns the unchanged host barrier buffer footprint in bytes.
// Epoch and exit-count metadata sit in the unused tail of line 0, so no extra
// trailing bytes are allocated.
func (c ReplayBarrierConfig) BufferSizeBytes() (int, error) {
	norm, err := c.normalized()
	if err != nil {
		return 0, err
	}
	return norm.NumBarriers * norm.NumMBlocks * ReplayBarrierLineBytes, nil
}

// ReplayEpochBarrierBuffer models a flat device split-barrier buffer reused
// across repeated graph replays over the same graph node ID without host-side
// zeroing between executions.
type ReplayEpochBarrierBuffer struct {
	mu  sync.Mutex
	cfg ReplayBarrierConfig
	buf []byte
}

// NewReplayEpochBarrierBuffer allocates a zero-initialized barrier buffer whose
// size matches the unmodified barrier-line layout.
func NewReplayEpochBarrierBuffer(cfg ReplayBarrierConfig) (*ReplayEpochBarrierBuffer, error) {
	norm, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	size := norm.NumBarriers * norm.NumMBlocks * ReplayBarrierLineBytes
	if ReplayBarrierExitCountOffset+4 > ReplayBarrierLineBytes {
		return nil, fmt.Errorf("%w: tail metadata overflows barrier line", ErrReplayBarrierInvalidConfig)
	}
	return &ReplayEpochBarrierBuffer{
		cfg: norm,
		buf: make([]byte, size),
	}, nil
}

// BufferBytes returns the byte length of the backing barrier-line buffer.
func (b *ReplayEpochBarrierBuffer) BufferBytes() int {
	return len(b.buf)
}

// Config returns the normalized barrier configuration.
func (b *ReplayEpochBarrierBuffer) Config() ReplayBarrierConfig {
	return b.cfg
}

func (b *ReplayEpochBarrierBuffer) lineOffset(barrierIdx, mBlockIdx int) (int, error) {
	if barrierIdx < 0 || barrierIdx >= b.cfg.NumBarriers {
		return 0, fmt.Errorf("%w: barrier=%d not in [0,%d)", ErrReplayBarrierOutOfRange, barrierIdx, b.cfg.NumBarriers)
	}
	if mBlockIdx < 0 || mBlockIdx >= b.cfg.NumMBlocks {
		return 0, fmt.Errorf("%w: mblock=%d not in [0,%d)", ErrReplayBarrierOutOfRange, mBlockIdx, b.cfg.NumMBlocks)
	}
	return (barrierIdx*b.cfg.NumMBlocks + mBlockIdx) * ReplayBarrierLineBytes, nil
}

// InspectTail reads the current epoch and exit count from the unused tail of
// barrier line 0. Intended for host-side verification and diagnostics.
func (b *ReplayEpochBarrierBuffer) InspectTail() (epoch uint64, exitCount uint32) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return binary.LittleEndian.Uint64(b.buf[ReplayBarrierEpochOffset : ReplayBarrierEpochOffset+8]),
		binary.LittleEndian.Uint32(b.buf[ReplayBarrierExitCountOffset : ReplayBarrierExitCountOffset+4])
}

// ReadBarrierLine reads the 64-bit state word of barrier line (barrierIdx, mBlockIdx).
func (b *ReplayEpochBarrierBuffer) ReadBarrierLine(barrierIdx, mBlockIdx int) (uint64, error) {
	off, err := b.lineOffset(barrierIdx, mBlockIdx)
	if err != nil {
		return 0, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return binary.LittleEndian.Uint64(b.buf[off : off+ReplayBarrierStateBytes]), nil
}

// EncodeState encodes (epoch, arrivedSplits) monotonically above zero-init.
func (b *ReplayEpochBarrierBuffer) EncodeState(epoch uint64, arrivedSplits uint32) uint64 {
	return (epoch+1)*uint64(b.cfg.NumBarriers)*uint64(b.cfg.MaxSplits) + uint64(arrivedSplits)
}

// NewCTAContext opens a per-CTA barrier execution handle bound to a fixed
// graph node ID and grid ID. The CTA must call PDLSync before reading the
// epoch or interacting with any split barrier line.
func (b *ReplayEpochBarrierBuffer) NewCTAContext(nodeID, gridID uint64, ctaID, numCTAs uint32) (*ReplayBarrierCTAContext, error) {
	if b == nil {
		return nil, fmt.Errorf("%w: nil buffer", ErrReplayBarrierInvalidConfig)
	}
	if numCTAs == 0 || ctaID >= numCTAs {
		return nil, fmt.Errorf("%w: ctaID=%d numCTAs=%d", ErrReplayBarrierInvalidConfig, ctaID, numCTAs)
	}
	return &ReplayBarrierCTAContext{
		buf:     b,
		nodeID:  nodeID,
		gridID:  gridID,
		ctaID:   ctaID,
		numCTAs: numCTAs,
	}, nil
}

// ReplayBarrierCTAContext tracks one CTA's post-PDL epoch read, split-barrier
// operations, and retirement within a single graph node replay.
type ReplayBarrierCTAContext struct {
	buf       *ReplayEpochBarrierBuffer
	nodeID    uint64
	gridID    uint64
	ctaID     uint32
	numCTAs   uint32
	pdlSynced bool
	hasEpoch  bool
	epoch     uint64
	retired   bool
}

// NodeID returns the graph node identifier for this replay execution.
func (c *ReplayBarrierCTAContext) NodeID() uint64 { return c.nodeID }

// GridID returns the CTA grid identifier (constant across graph replays).
func (c *ReplayBarrierCTAContext) GridID() uint64 { return c.gridID }

// PDLSync marks that the CTA completed its grid-dependency synchronization,
// guaranteeing that the prior execution's final retiring CTA has finished
// resetting the exit count and advancing the epoch.
func (c *ReplayBarrierCTAContext) PDLSync() {
	c.pdlSynced = true
}

func (c *ReplayBarrierCTAContext) ensureEpochPostPDL() (uint64, error) {
	if c == nil || c.buf == nil {
		return 0, fmt.Errorf("%w: nil CTA context", ErrReplayBarrierInvalidConfig)
	}
	if !c.pdlSynced {
		return 0, ErrEpochReadBeforePDLSync
	}
	if !c.hasEpoch {
		c.buf.mu.Lock()
		c.epoch = binary.LittleEndian.Uint64(c.buf.buf[ReplayBarrierEpochOffset : ReplayBarrierEpochOffset+8])
		c.buf.mu.Unlock()
		c.hasEpoch = true
	}
	return c.epoch, nil
}

// ReadEpoch reads the execution epoch strictly after PDLSync.
func (c *ReplayBarrierCTAContext) ReadEpoch() (uint64, error) {
	return c.ensureEpochPostPDL()
}

// Init writes the base state for the current epoch onto barrier line (barrierIdx, mBlockIdx).
func (c *ReplayBarrierCTAContext) Init(barrierIdx, mBlockIdx int) error {
	epoch, err := c.ensureEpochPostPDL()
	if err != nil {
		return err
	}
	off, err := c.buf.lineOffset(barrierIdx, mBlockIdx)
	if err != nil {
		return err
	}
	state := c.buf.EncodeState(epoch, 0)
	c.buf.mu.Lock()
	binary.LittleEndian.PutUint64(c.buf.buf[off:off+ReplayBarrierStateBytes], state)
	c.buf.mu.Unlock()
	return nil
}

// PollInit reports whether barrier line (barrierIdx, mBlockIdx) has been
// initialized for the current epoch. Unsigned distance rejects both zeroed lines
// and stale completed states from prior replays, while accepting any in-progress
// arrival count in [0, numSplits).
func (c *ReplayBarrierCTAContext) PollInit(barrierIdx, mBlockIdx int, numSplits uint32) (bool, error) {
	epoch, err := c.ensureEpochPostPDL()
	if err != nil {
		return false, err
	}
	if numSplits == 0 || numSplits > c.buf.cfg.MaxSplits {
		return false, fmt.Errorf("%w: numSplits=%d max=%d", ErrReplayBarrierInvalidConfig, numSplits, c.buf.cfg.MaxSplits)
	}
	off, err := c.buf.lineOffset(barrierIdx, mBlockIdx)
	if err != nil {
		return false, err
	}
	base := c.buf.EncodeState(epoch, 0)
	c.buf.mu.Lock()
	observed := binary.LittleEndian.Uint64(c.buf.buf[off : off+ReplayBarrierStateBytes])
	c.buf.mu.Unlock()
	return (observed - base) < uint64(numSplits), nil
}

// Arrive atomically advances the split-arrival count on barrier line (barrierIdx, mBlockIdx).
func (c *ReplayBarrierCTAContext) Arrive(barrierIdx, mBlockIdx int) error {
	if _, err := c.ensureEpochPostPDL(); err != nil {
		return err
	}
	off, err := c.buf.lineOffset(barrierIdx, mBlockIdx)
	if err != nil {
		return err
	}
	c.buf.mu.Lock()
	cur := binary.LittleEndian.Uint64(c.buf.buf[off : off+ReplayBarrierStateBytes])
	binary.LittleEndian.PutUint64(c.buf.buf[off:off+ReplayBarrierStateBytes], cur+1)
	c.buf.mu.Unlock()
	return nil
}

// PollWait reports whether all numSplits arrivals for the current epoch have
// landed on barrier line (barrierIdx, mBlockIdx).
func (c *ReplayBarrierCTAContext) PollWait(barrierIdx, mBlockIdx int, numSplits uint32) (bool, error) {
	epoch, err := c.ensureEpochPostPDL()
	if err != nil {
		return false, err
	}
	if numSplits == 0 || numSplits > c.buf.cfg.MaxSplits {
		return false, fmt.Errorf("%w: numSplits=%d max=%d", ErrReplayBarrierInvalidConfig, numSplits, c.buf.cfg.MaxSplits)
	}
	off, err := c.buf.lineOffset(barrierIdx, mBlockIdx)
	if err != nil {
		return false, err
	}
	want := c.buf.EncodeState(epoch, numSplits)
	c.buf.mu.Lock()
	observed := binary.LittleEndian.Uint64(c.buf.buf[off : off+ReplayBarrierStateBytes])
	c.buf.mu.Unlock()
	return observed == want, nil
}

// Retire marks this CTA as finished with all barrier uses for this execution.
// The last retiring CTA resets the exit count to 0 and increments the epoch
// stored in line 0's tail.
func (c *ReplayBarrierCTAContext) Retire() (advanced bool, newEpoch uint64, err error) {
	if _, err := c.ensureEpochPostPDL(); err != nil {
		return false, 0, err
	}
	if c.retired {
		return false, 0, fmt.Errorf("%w: cta %d already retired in epoch %d",
			ErrReplayBarrierRetireOrder, c.ctaID, c.epoch)
	}
	c.retired = true

	c.buf.mu.Lock()
	defer c.buf.mu.Unlock()
	exitCount := binary.LittleEndian.Uint32(c.buf.buf[ReplayBarrierExitCountOffset : ReplayBarrierExitCountOffset+4])
	epoch := binary.LittleEndian.Uint64(c.buf.buf[ReplayBarrierEpochOffset : ReplayBarrierEpochOffset+8])
	if exitCount == c.numCTAs-1 {
		binary.LittleEndian.PutUint32(c.buf.buf[ReplayBarrierExitCountOffset:ReplayBarrierExitCountOffset+4], 0)
		next := epoch + 1
		binary.LittleEndian.PutUint64(c.buf.buf[ReplayBarrierEpochOffset:ReplayBarrierEpochOffset+8], next)
		return true, next, nil
	}
	binary.LittleEndian.PutUint32(c.buf.buf[ReplayBarrierExitCountOffset:ReplayBarrierExitCountOffset+4], exitCount+1)
	return false, epoch, nil
}
