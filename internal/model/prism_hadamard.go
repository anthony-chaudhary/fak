package model

// Bonsai-2 activation transform, adapted from PrismML's llama.cpp fork
// (src/llama-graph.cpp and src/models/qwen35.cpp, adfffbe41b2c) and its MLX
// runtime (runtime/runtime.py, fcba37d). The stored projection weight has
// already been rotated on its input axis. At execution, multiply the incoming
// activation by its explicit signs, then apply normalized H_block to each block.
// The embedding is stored in that basis and uses the inverse order: H first,
// then signs. H is self-inverse; HadamardTransform supplies its normalization.
//
// PrismML-Eng/llama.cpp original notice:
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 PrismML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import "fmt"

// PrismHadamardSpec is the canonicalized prism.hadamard GGUF contract. The
// concatenated SignValues contain one +/-1 vector for each SignWidths entry.
// WeightNames and InverseNames are canonical model tensor names.
type PrismHadamardSpec struct {
	BlockSize    int
	SignWidths   []int
	SignValues   []int
	WeightNames  []string
	InverseNames []string
	GDNVGrouped  bool
}

type prismHadamardState struct {
	blockSize    int
	signs        map[int][]int8
	weightWidth  map[string]int
	inverseWidth map[string]int
	// The loader uses gdnVGrouped to keep the grouped ssm_out input-column
	// order. Fak's GDN core writes core[h*vHd:] with kh=h/(nV/nK)
	// (qwen35.go:495-561), i.e. h=k*repeat+r. The loader reorders GGUF's
	// tiled V/gate rows from h=r*nK+k to that grouped h before recurrence,
	// so upstream llama.cpp's perm_rep is already satisfied here.
	gdnVGrouped bool
}

// HasPrismHadamard reports whether validated Prism rotation metadata was
// attached to this loaded model. The selection receipt uses it to distinguish
// Bonsai-2 PQ2_0 from the older unrotated Q2_0 resident format.
func (m *Model) HasPrismHadamard() bool { return m != nil && m.prism != nil }

// SetPrismHadamard validates and attaches the signed Hadamard contract. It
// retains inverse embedding declarations for transform after row gathering.
// A malformed or unresolved declared weight is rejected before attachment.
func (m *Model) SetPrismHadamard(spec PrismHadamardSpec) error {
	if m == nil || m.prism != nil {
		return fmt.Errorf("model: Prism Hadamard requires an unconfigured model")
	}
	if !isPowerOfTwo(spec.BlockSize) || len(spec.WeightNames) == 0 ||
		len(spec.SignWidths) == 0 || len(spec.SignValues) == 0 {
		return fmt.Errorf("model: malformed Prism Hadamard geometry")
	}
	state := &prismHadamardState{
		blockSize: spec.BlockSize, signs: make(map[int][]int8),
		weightWidth: make(map[string]int), inverseWidth: make(map[string]int),
		gdnVGrouped: spec.GDNVGrouped,
	}
	offset := 0
	for _, width := range spec.SignWidths {
		if width <= 0 || width%spec.BlockSize != 0 || width > len(spec.SignValues)-offset || state.signs[width] != nil {
			return fmt.Errorf("model: malformed Prism Hadamard sign width %d", width)
		}
		signs := make([]int8, width)
		for i, v := range spec.SignValues[offset : offset+width] {
			if v != -1 && v != 1 {
				return fmt.Errorf("model: Prism Hadamard sign %d at width %d index %d", v, width, i)
			}
			signs[i] = int8(v)
		}
		state.signs[width] = signs
		offset += width
	}
	if offset != len(spec.SignValues) {
		return fmt.Errorf("model: Prism Hadamard sign payload has %d trailing entries", len(spec.SignValues)-offset)
	}
	for _, name := range spec.WeightNames {
		if name == "" {
			return fmt.Errorf("model: empty Prism Hadamard weight name")
		}
		if _, duplicate := state.weightWidth[name]; duplicate {
			return fmt.Errorf("model: duplicate Prism Hadamard weight %s", name)
		}
		_, in, ok := m.residentShape(name)
		if !ok {
			return fmt.Errorf("model: Prism Hadamard weight %s is missing", name)
		}
		if state.signs[in] == nil {
			return fmt.Errorf("model: Prism Hadamard weight %s has undeclared input width %d", name, in)
		}
		state.weightWidth[name] = in
	}
	for _, name := range spec.InverseNames {
		if name != "model.embed_tokens.weight" {
			return fmt.Errorf("model: Prism Hadamard inverse %s has no row-gather execution seam", name)
		}
		if _, duplicate := state.inverseWidth[name]; duplicate {
			return fmt.Errorf("model: duplicate Prism Hadamard inverse %s", name)
		}
		width := 0
		if name == "model.embed_tokens.weight" && m.Q2KEmbedding != nil {
			width = m.Q2KEmbedding.Hidden()
		} else if meta, ok := m.manifest[name]; ok && len(meta.Shape) == 2 && meta.Dtype == "f32" {
			width = meta.Shape[1]
		}
		if width == 0 {
			return fmt.Errorf("model: Prism Hadamard inverse %s needs resident packed or f32 rows", name)
		}
		if state.signs[width] == nil || width%spec.BlockSize != 0 {
			return fmt.Errorf("model: Prism Hadamard inverse %s has undeclared width %d", name, width)
		}
		if _, forward := state.weightWidth[name]; forward {
			return fmt.Errorf("model: Prism Hadamard weight %s declared forward and inverse", name)
		}
		state.inverseWidth[name] = width
	}
	m.prism = state
	if state.inverseWidth["model.embed_tokens.weight"] != 0 && m.Q2KEmbedding != nil {
		m.Q2KEmbedding.prism = state
	}
	return nil
}

// prismInverseEmbeddingRow converts one gathered row from the stored rotated
// basis into model activation space: H first, then signs. It never changes the
// packed backing or the f32 embedding table.
func (m *Model) prismInverseEmbeddingRow(name string, row []float32) {
	if m == nil || m.prism == nil {
		return
	}
	width := m.prism.inverseWidth[name]
	if width == 0 {
		return
	}
	m.prism.inverseRow(row, width)
}

func (p *prismHadamardState) inverseEmbeddingRow(row []float32) {
	p.inverseRow(row, p.inverseWidth["model.embed_tokens.weight"])
}

func (p *prismHadamardState) inverseRow(row []float32, width int) {
	if len(row) != width {
		panic(fmt.Sprintf("model: Prism Hadamard inverse embedding row width %d, want %d", len(row), width))
	}
	for start := 0; start < width; start += p.blockSize {
		_ = HadamardTransform(row[start : start+p.blockSize]) // prevalidated power of two
	}
	signs := p.signs[width]
	for i := range row {
		row[i] *= float32(signs[i])
	}
}

// prismProjectInput returns x unchanged for an undeclared weight. For a
// rotated weight, it leaves the caller's activation untouched; group members
// may share x yet require different sign vectors.
func (m *Model) prismProjectInput(name string, x []float32) []float32 {
	if m == nil || m.prism == nil {
		return x
	}
	width := m.prism.weightWidth[name]
	if width == 0 {
		return x
	}
	if len(x) != width {
		panic(fmt.Sprintf("model: Prism Hadamard %s activation width %d, want %d", name, len(x), width))
	}
	y := append([]float32(nil), x...)
	signs := m.prism.signs[width]
	for i := range y {
		y[i] *= float32(signs[i])
	}
	for start := 0; start < width; start += m.prism.blockSize {
		_ = HadamardTransform(y[start : start+m.prism.blockSize]) // prevalidated power of two
	}
	return y
}

// prismProjectPanel is the batched [rows,width] twin used by resident prefill.
func (m *Model) prismProjectPanel(name string, x []float32, rows int) []float32 {
	if m == nil || m.prism == nil || m.prism.weightWidth[name] == 0 {
		return x
	}
	width := m.prism.weightWidth[name]
	if len(x) != rows*width {
		panic(fmt.Sprintf("model: Prism Hadamard %s panel shape [%d,%d] needs %d elements, got %d", name, rows, width, rows*width, len(x)))
	}
	y := make([]float32, len(x))
	signs := m.prism.signs[width]
	for row := 0; row < rows; row++ {
		base := row * width
		for i := 0; i < width; i++ {
			y[base+i] = x[base+i] * float32(signs[i])
		}
		for start := 0; start < width; start += m.prism.blockSize {
			_ = HadamardTransform(y[base+start : base+start+m.prism.blockSize])
		}
	}
	return y
}
