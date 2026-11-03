package model

// paged_swap.go — host-DRAM swap/restore for the paged KV allocator (#31).
//
// The scheduler preemption policy operates on serialized KV bytes. This file is the
// concrete serializer for PagedKV: it snapshots the sequence's logical page table into a
// self-describing byte blob, releases the original pages with Free(), and restores the blob
// into owned pages in a compatible PagedKVPool. Float32 values are stored as raw IEEE bits,
// so swap/restore is a byte-exact move, not a numeric re-encoding.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const pagedKVSwapMagic = "FAKPKV1\n"

// SwapToHost serializes this paged KV sequence into host bytes. The returned blob owns a
// copy of every logical block in the page table, including reserved tail blocks, so a later
// RestoreFromHost recreates the same Len, Blocks, and gathered K/V/Kraw content.
func (s *PagedKV) SwapToHost() ([]byte, error) {
	if s == nil || s.pool == nil {
		return nil, errors.New("model: cannot swap nil PagedKV")
	}
	p := s.pool
	header := [...]int{p.blockTokens, p.stride, p.nLayers, p.planes, s.nTokens, len(s.table)}
	for _, n := range header {
		if n < 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with negative geometry %d", n)
		}
	}
	size := len(pagedKVSwapMagic) + len(header)*4
	for _, id := range s.table {
		if id < 0 || id >= len(p.blocks) || p.ref[id] == 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with invalid block id %d", id)
		}
		var err error
		size, err = pagedKVSwapGrowSize(size, len(p.blocks[id]))
		if err != nil {
			return nil, err
		}
	}
	// One owned output preserves the original FAKPKV1 framing and every
	// physical float, including unused slots and reserved tail blocks.
	blob := make([]byte, size)
	copy(blob, pagedKVSwapMagic)
	off := len(pagedKVSwapMagic)
	for _, n := range header {
		binary.LittleEndian.PutUint32(blob[off:off+4], uint32(n))
		off += 4
	}
	for _, id := range s.table {
		for _, f := range p.blocks[id] {
			binary.LittleEndian.PutUint32(blob[off:off+4], math.Float32bits(f))
			off += 4
		}
	}
	return blob, nil
}

// pagedKVSwapGrowSize checks the addressable output extent before allocation.
func pagedKVSwapGrowSize(size, floats int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if size < 0 || floats < 0 || floats > (maxInt-size)/4 {
		return 0, errors.New("model: paged KV swap blob exceeds addressable size")
	}
	return size + floats*4, nil
}

// RestoreFromHost restores a SwapToHost blob into fresh owned pages in this pool. The pool
// must have the same page geometry and plane count as the source; that is the structural
// gate that keeps swap tied to the paged/block allocator rather than the contiguous cache.
func (p *PagedKVPool) RestoreFromHost(data []byte) (*PagedKV, error) {
	if p == nil {
		return nil, errors.New("model: cannot restore PagedKV into nil pool")
	}
	r := bytes.NewReader(data)
	magic := make([]byte, len(pagedKVSwapMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("model: restore paged KV header: %w", err)
	}
	if string(magic) != pagedKVSwapMagic {
		return nil, errors.New("model: invalid paged KV swap blob")
	}
	readU32 := func(name string) (int, error) {
		var v uint32
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return 0, fmt.Errorf("model: restore paged KV %s: %w", name, err)
		}
		return int(v), nil
	}
	blockTokens, err := readU32("blockTokens")
	if err != nil {
		return nil, err
	}
	stride, err := readU32("stride")
	if err != nil {
		return nil, err
	}
	nLayers, err := readU32("layers")
	if err != nil {
		return nil, err
	}
	planes, err := readU32("planes")
	if err != nil {
		return nil, err
	}
	nTokens, err := readU32("tokens")
	if err != nil {
		return nil, err
	}
	nBlocks, err := readU32("blocks")
	if err != nil {
		return nil, err
	}
	if blockTokens != p.blockTokens || stride != p.stride || nLayers != p.nLayers || planes != p.planes {
		return nil, fmt.Errorf("model: paged KV swap geometry mismatch: blob block=%d stride=%d layers=%d planes=%d, pool block=%d stride=%d layers=%d planes=%d",
			blockTokens, stride, nLayers, planes, p.blockTokens, p.stride, p.nLayers, p.planes)
	}
	if nBlocks == 0 && nTokens != 0 {
		return nil, errors.New("model: paged KV swap blob has tokens but no blocks")
	}
	if nBlocks > 0 && nTokens > nBlocks*p.blockTokens {
		return nil, errors.New("model: paged KV swap blob token count exceeds block table")
	}

	// Keep only owned block IDs in the table so failure cannot release another sequence's pages.
	seq := &PagedKV{pool: p, table: make([]int, 0, nBlocks), nTokens: nTokens}
	fail := func(err error) (*PagedKV, error) {
		seq.Free()
		return nil, err
	}
	payload := data[len(data)-r.Len():]
	for i := 0; i < nBlocks; i++ {
		id := p.alloc()
		seq.table = append(seq.table, id)
		blk := p.blocks[id]
		if len(blk) != p.blockFloats() {
			return fail(errors.New("model: paged KV pool block geometry changed under restore"))
		}
		// Check before multiplying or slicing. Preserve scalar binary.Read's
		// EOF versus partial-word error, and release only this restore's pages.
		if len(blk) > len(payload)/4 {
			err := error(io.EOF)
			if len(payload)%4 != 0 {
				err = io.ErrUnexpectedEOF
			}
			return fail(fmt.Errorf("model: restore paged KV block %d: %w", i, err))
		}
		for j := range blk {
			blk[j] = math.Float32frombits(binary.LittleEndian.Uint32(payload[j*4:]))
		}
		payload = payload[len(blk)*4:]
	}
	if len(payload) != 0 {
		return fail(errors.New("model: paged KV swap blob has trailing bytes"))
	}
	return seq, nil
}
