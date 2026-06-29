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
	var b bytes.Buffer
	b.WriteString(pagedKVSwapMagic)
	for _, n := range []int{p.blockTokens, p.stride, p.nLayers, p.planes, s.nTokens, len(s.table)} {
		if n < 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with negative geometry %d", n)
		}
		if err := binary.Write(&b, binary.LittleEndian, uint32(n)); err != nil {
			return nil, err
		}
	}
	for _, id := range s.table {
		if id < 0 || id >= len(p.blocks) || p.ref[id] == 0 {
			return nil, fmt.Errorf("model: cannot swap PagedKV with invalid block id %d", id)
		}
		for _, f := range p.blocks[id] {
			if err := binary.Write(&b, binary.LittleEndian, math.Float32bits(f)); err != nil {
				return nil, err
			}
		}
	}
	return b.Bytes(), nil
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

	seq := &PagedKV{pool: p, table: make([]int, nBlocks), nTokens: nTokens}
	fail := func(err error) (*PagedKV, error) {
		seq.Free()
		return nil, err
	}
	for i := 0; i < nBlocks; i++ {
		id := p.alloc()
		seq.table[i] = id
		blk := p.blocks[id]
		if len(blk) != p.blockFloats() {
			return fail(errors.New("model: paged KV pool block geometry changed under restore"))
		}
		for j := range blk {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				return fail(fmt.Errorf("model: restore paged KV block %d: %w", i, err))
			}
			blk[j] = math.Float32frombits(bits)
		}
	}
	if r.Len() != 0 {
		return fail(errors.New("model: paged KV swap blob has trailing bytes"))
	}
	return seq, nil
}
