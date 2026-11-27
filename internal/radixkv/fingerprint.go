package radixkv

// Any-offset rolling-hash chunk matcher: the first producer of ReuseCandidate
// values for non-prefix reuse. Chunk-hash and any-offset matching shape follows
// LMCache/LMCache (Apache-2.0), lmcache/v1/multiprocess/modules/blend/matcher.py
// and token_hasher.py; this is a clean-room Go implementation.

import (
	"errors"
	"fmt"
	"sort"
)

// fingerprintBase is the odd 64-bit multiplier of the polynomial hash.
const fingerprintBase uint64 = 0x100000001b3

// DefaultFingerprintChunkSize is the chunk width (tokens) used when a caller passes <= 0.
const DefaultFingerprintChunkSize = 256

type fingerprintEntry struct {
	hash   uint64
	srcOff int // absolute source offset of the chunk's first token
	reg    int // registration key (srcOffset passed to Register)
}

// FingerprintIndex maps chunk polynomial hashes to source offsets.
// The table is power-of-two direct-addressed; each slot holds the entries
// whose low hash bits select it, and lookups compare the full 64-bit hash,
// so a bucket collision is rejected rather than reported as a match.
// It is not safe for concurrent use.
type FingerprintIndex struct {
	chunkSize int
	lead      uint64 // base^(chunkSize-1)
	mask      uint64
	table     [][]fingerprintEntry
	count     int
}

// NewFingerprintIndex returns an index for chunkSize-token chunks.
func NewFingerprintIndex(chunkSize int) *FingerprintIndex {
	if chunkSize <= 0 {
		chunkSize = DefaultFingerprintChunkSize
	}
	lead := uint64(1)
	for i := 1; i < chunkSize; i++ {
		lead *= fingerprintBase
	}
	const initial = 64
	return &FingerprintIndex{
		chunkSize: chunkSize,
		lead:      lead,
		mask:      initial - 1,
		table:     make([][]fingerprintEntry, initial),
	}
}

// ChunkSize reports the chunk width in tokens.
func (f *FingerprintIndex) ChunkSize() int { return f.chunkSize }

// Len reports the number of registered chunks.
func (f *FingerprintIndex) Len() int { return f.count }

func fpTok(t int32) uint64 { return uint64(uint32(t)) + 1 }

func (f *FingerprintIndex) hashAt(tokens []int32) uint64 {
	var h uint64
	for _, t := range tokens[:f.chunkSize] {
		h = h*fingerprintBase + fpTok(t)
	}
	return h
}

func (f *FingerprintIndex) roll(h uint64, out, in int32) uint64 {
	return (h-fpTok(out)*f.lead)*fingerprintBase + fpTok(in)
}

// Register hashes each full chunkSize-aligned chunk of tokens (from tokens[0])
// and records it at source offset srcOffset+i*chunkSize. A trailing partial
// chunk is not registered. It returns the number of chunks registered.
func (f *FingerprintIndex) Register(tokens []int32, srcOffset int) (int, error) {
	if srcOffset < 0 {
		return 0, fmt.Errorf("radixkv: negative srcOffset %d", srcOffset)
	}
	if f.hasReg(srcOffset) {
		return 0, errors.New("radixkv: srcOffset already registered; Evict first")
	}
	n := 0
	for i := 0; i+f.chunkSize <= len(tokens); i += f.chunkSize {
		f.insert(fingerprintEntry{hash: f.hashAt(tokens[i:]), srcOff: srcOffset + i, reg: srcOffset})
		n++
	}
	return n, nil
}

func (f *FingerprintIndex) hasReg(reg int) bool {
	for _, b := range f.table {
		for _, e := range b {
			if e.reg == reg {
				return true
			}
		}
	}
	return false
}

func (f *FingerprintIndex) insert(e fingerprintEntry) {
	if f.count >= len(f.table) {
		f.grow()
	}
	s := e.hash & f.mask
	f.table[s] = append(f.table[s], e)
	f.count++
}

func (f *FingerprintIndex) grow() {
	old := f.table
	f.table = make([][]fingerprintEntry, len(old)*2)
	f.mask = uint64(len(f.table) - 1)
	for _, b := range old {
		for _, e := range b {
			s := e.hash & f.mask
			f.table[s] = append(f.table[s], e)
		}
	}
}

// Evict removes every chunk registered under srcOffset and returns how many.
func (f *FingerprintIndex) Evict(srcOffset int) int {
	removed := 0
	for s, b := range f.table {
		kept := b[:0]
		for _, e := range b {
			if e.reg == srcOffset {
				removed++
				continue
			}
			kept = append(kept, e)
		}
		f.table[s] = kept
	}
	f.count -= removed
	return removed
}

// lookup returns the entry whose full hash equals h (lowest srcOff wins, for determinism).
func (f *FingerprintIndex) lookup(h uint64) (fingerprintEntry, bool) {
	var best fingerprintEntry
	found := false
	for _, e := range f.table[h&f.mask] {
		if e.hash == h && (!found || e.srcOff < best.srcOff) {
			best, found = e, true
		}
	}
	return best, found
}

// Match slides a rolling hash over tokens at every offset and returns one
// ReuseCandidate per maximal run of consecutive matched chunks whose source
// offsets are contiguous. Matches do not overlap in the destination.
// Candidates are sorted by DstStart; Compatible is true, cost fields are left
// for the caller's CostModel.
func (f *FingerprintIndex) Match(tokens []int32) []ReuseCandidate {
	w := f.chunkSize
	var out []ReuseCandidate
	if len(tokens) < w || f.count == 0 {
		return nil
	}
	i := 0
	h := f.hashAt(tokens)
	for i+w <= len(tokens) {
		if e, ok := f.lookup(h); ok {
			if n := len(out); n > 0 && out[n-1].DstEnd == i && out[n-1].SrcEnd == e.srcOff {
				out[n-1].DstEnd += w
				out[n-1].SrcEnd += w
			} else {
				out = append(out, ReuseCandidate{DstStart: i, DstEnd: i + w, SrcStart: e.srcOff, SrcEnd: e.srcOff + w, Compatible: true})
			}
			i += w
			if i+w <= len(tokens) {
				h = f.hashAt(tokens[i:])
			}
			continue
		}
		if i+w < len(tokens) {
			h = f.roll(h, tokens[i], tokens[i+w])
		}
		i++
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].DstStart < out[b].DstStart })
	for k := range out {
		out[k].ID = fmt.Sprintf("fp:%d", out[k].SrcStart)
	}
	return out
}
