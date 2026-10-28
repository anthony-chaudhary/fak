package ggufload

// Tensor-carried Engram dialect adapted from Strata tools/dsv41/pack_gguf.py
// (int_tensors_to_engram), revision 5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d.
// Unlike the packer, this adapter bounds reads, checks redundant declarations,
// and retains FAK's published tokenizer and hash-layout admission.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// This is a bound on auxiliary metadata, not a new weight dtype capability.
// The published token map has 129280 entries; each other array has at most 48.
// Keeping this cap independent of untrusted dimensions prevents a header from
// turning configuration admission into an unbounded tensor allocation.
const deepSeek41MaxAuxElements = 1 << 20

func deepSeek41EngramAuxTensor(name string) bool {
	switch name {
	case "engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight":
		return true
	}
	return false
}

func deepSeek41AuxValueEqual(a, b Value) bool {
	if a.Type == TypeArray && b.Type == TypeArray {
		x, xok := a.Value.([]Value)
		y, yok := b.Value.([]Value)
		if !xok || !yok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !deepSeek41AuxValueEqual(x[i], y[i]) {
				return false
			}
		}
		return true
	}
	x, xok := valueUint64(a)
	y, yok := valueUint64(b)
	return xok && yok && x == y
}

// readDeepSeek41EngramAux is bound at source construction but called only by
// Config, after split/authenticated sources have finished wiring shard readers.
// It builds a temporary metadata view; neither the source directory nor the
// caller's metadata is rewritten, and no floating-point conversion is involved.
func (s *WeightSource) readDeepSeek41EngramAux(f *File, p string, cfg *model.Config) (*File, error) {
	found := false
	for _, info := range f.Tensors {
		found = found || deepSeek41EngramAuxTensor(info.Name)
	}
	if !found {
		return f, nil
	}
	refuse := func(reason string) (*File, error) {
		return nil, fmt.Errorf("gguf: deepseek41 Engram auxiliaries: %s: %w", reason, model.ErrV41ConfigAdmission)
	}
	release, _, ok := s.holdSourceReads()
	if !ok {
		return refuse("checkpoint source is closed")
	}
	defer release()
	view := *f
	view.Metadata = maps.Clone(f.Metadata)
	for _, alias := range [][2]string{
		{ds41KeyEngramLayerIDs, ds41LegacyEngramLayerIDs},
		{ds41KeyEngramNumEmbedding, "engram.num_embeddings"},
		{ds41KeyEngramNumEmbedding, ds41LegacyEngramNumEmbedding},
		{ds41KeyEngramMaxNgramSize, ds41LegacyEngramMaxNgramSize},
		{ds41KeyEngramHeadCount, ds41LegacyEngramNHeads},
		{ds41KeyEngramKeyLength, ds41LegacyEngramHeadDim},
		{ds41KeyEngramPadTokenID, ds41LegacyEngramPadTokenID},
		{ds41KeyEngramCompVocab, ds41LegacyEngramCompVocab},
	} {
		if value, present := f.lookup(p + alias[1]); present {
			if existing, present := view.lookup(p + alias[0]); present && !deepSeek41AuxValueEqual(existing, value) {
				return refuse("conflicting " + p + alias[0] + " and " + p + alias[1])
			}
			view.Metadata[p+alias[0]] = value
		}
	}
	layers, validLayers := view.IntArray(p + ds41KeyEngramLayerIDs)
	ngram, heads := intValueOrZero(&view, p+ds41KeyEngramMaxNgramSize), intValueOrZero(&view, p+ds41KeyEngramHeadCount)
	if !validLayers || len(layers) == 0 || cfg.VocabSize <= 0 || cfg.VocabSize > deepSeek41MaxAuxElements ||
		ngram < 2 || ngram > deepSeek41MaxAuxElements/len(layers) || heads <= 0 ||
		heads > deepSeek41MaxAuxElements/len(layers)/(ngram-1) {
		return refuse("missing, invalid, or oversized vocabulary/hash geometry")
	}
	cols := (ngram - 1) * heads
	for _, item := range []struct {
		name string
		key  string
		typ  TensorType
		dims []uint64
	}{
		{"engram_token_map.weight", ds41KeyEngramTokenMap, TensorType(26), []uint64{uint64(cfg.VocabSize)}},
		{"engram_primes.weight", ds41KeyEngramPrimes, TensorType(27), []uint64{uint64(cols), uint64(len(layers))}},
		{"engram_offsets.weight", ds41KeyEngramOffsets, TensorType(27), []uint64{uint64(cols), uint64(len(layers))}},
		{"engram_multipliers.weight", ds41KeyEngramMultipliers, TensorType(27), []uint64{uint64(ngram), uint64(len(layers))}},
	} {
		info, present := s.Tensor(item.name)
		if !present || info.Type != item.typ || !slices.Equal(info.Dims, item.dims) {
			return refuse(fmt.Sprintf("%s must be type %d with dimensions %v", item.name, item.typ, item.dims))
		}
		count := int(item.dims[0])
		if len(item.dims) == 2 {
			count *= int(item.dims[1])
		}
		width := 8
		if item.typ == TensorType(26) {
			width = 4
		}
		r, size, err := s.tensorReader(info)
		if err != nil {
			return refuse(err.Error())
		}
		bytes := int64(count * width)
		if info.FileOffset < 0 || info.FileOffset > size || bytes > size-info.FileOffset {
			return refuse(item.name + " overruns its shard")
		}
		raw := make([]byte, int(bytes))
		n, err := r.ReadAt(raw, info.FileOffset)
		if err != nil || n != len(raw) {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return refuse(fmt.Sprintf("read %s: %v", item.name, err))
		}
		values := make([]Value, count)
		for i := range values {
			var v int64
			if width == 4 {
				v = int64(int32(binary.LittleEndian.Uint32(raw[i*width:])))
			} else {
				v = int64(binary.LittleEndian.Uint64(raw[i*width:]))
			}
			if v < 0 {
				return refuse(fmt.Sprintf("%s[%d] is negative", item.name, i))
			}
			values[i] = Value{Type: TypeUint64, Value: uint64(v)}
		}
		value := Value{Type: TypeArray, Value: values}
		if existing, present := view.lookup(p + item.key); present && !deepSeek41AuxValueEqual(existing, value) {
			return refuse("tensor disagrees with " + p + item.key)
		}
		view.Metadata[p+item.key] = value
	}
	// Offsets must describe the same per-layer cumulative-prime layout that the
	// existing hash stage derives; rows, when supplied, must agree as well.
	primes := intArrayOrNil(&view, p+ds41KeyEngramPrimes)
	offsets := intArrayOrNil(&view, p+ds41KeyEngramOffsets)
	rows := intArrayOrNil(&view, p+ds41KeyEngramNumEmbedding)
	if len(primes) != cols*len(layers) || len(offsets) != len(primes) ||
		(view.hasKey(p+ds41KeyEngramNumEmbedding) && len(rows) != len(layers)) {
		return refuse("hash arrays exceed host integer range or row counts disagree")
	}
	for l := range layers {
		var total uint64
		for j := range cols {
			i := l*cols + j
			if primes[i] < 2 || uint64(offsets[i]) != total || uint64(primes[i]) > uint64(^uint32(0))-total {
				return refuse("offsets/primes do not describe bounded cumulative table rows")
			}
			total += uint64(primes[i])
		}
		if rows != nil && uint64(rows[l]) != total {
			return refuse("declared table rows disagree with primes")
		}
	}
	return &view, nil
}
