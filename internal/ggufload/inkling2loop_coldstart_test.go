package ggufload

import (
	"encoding/binary"
	"math"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type coldStartTraceDev struct {
	*fakeInk2Budget
	allocSizes []int
	writes     [][]float32
	onWrite    func()
}

func (d *coldStartTraceDev) Ink2Alloc(n int) (compute.Tensor, error) {
	d.allocSizes = append(d.allocSizes, n)
	return d.fakeInk2Budget.Ink2Alloc(n)
}

func (d *coldStartTraceDev) Ink2Write(r compute.Ink2Ref, v []float32) error {
	if d.onWrite != nil {
		d.onWrite()
	}
	cp := make([]float32, len(v))
	copy(cp, v)
	d.writes = append(d.writes, cp)
	return d.fakeInk2Budget.Ink2Write(r, v)
}

func makeColdStartF32QuantMatrix(name string, rows, cols int, seed float32) *inkling2QuantMatrix {
	raw := make([]byte, rows*cols*4)
	for i := 0; i < rows*cols; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(seed+float32(i)*0.25))
	}
	return &inkling2QuantMatrix{name: name, typ: TensorF32, raw: raw, rows: rows, cols: cols, rowBytes: cols * 4}
}

func makeColdStartDenseMatrix(rows, cols int, seed float32) *model.Inkling2LoopDenseMatrix {
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = seed + float32(i)*0.5
	}
	return &model.Inkling2LoopDenseMatrix{R: rows, C: cols, Data: data}
}

func makeColdStartTestModel(layers, ctrls int, dev compute.Backend) *Inkling2LoopModel {
	const hidden = 4
	hp := model.Inkling2LoopHParams{
		Hidden:        hidden,
		Layers:        layers,
		Heads:         1,
		KVHeads:       1,
		HeadDim:       hidden,
		SlidingWindow: 16,
		ConvKernel:    2,
		CoreStart:     0,
		CoreEnd:       layers,
		Passes:        2,
		LocalLayer:    make([]bool, layers),
		VocabUnpadded: 8,
	}
	vec := func(seed float32) []float32 {
		return []float32{seed, seed + 1, seed + 2, seed + 3}
	}
	layerWeights := make([]model.Inkling2LoopLayerWeights, layers)
	for i := range layerWeights {
		s := float32(i + 1)
		stack := compute.Tensor{Dtype: compute.IQ4_NL}
		layerWeights[i] = model.Inkling2LoopLayerWeights{
			Wq:           makeColdStartF32QuantMatrix("wq", hidden, hidden, s),
			Wk:           makeColdStartF32QuantMatrix("wk", hidden, hidden, s+10),
			Wv:           makeColdStartF32QuantMatrix("wv", hidden, hidden, s+20),
			Wo:           makeColdStartF32QuantMatrix("wo", hidden, hidden, s+30),
			Gate:         makeColdStartDenseMatrix(hidden, hidden, s+40),
			Rel:          makeColdStartDenseMatrix(hidden, hidden, s+50),
			SinkProj:     makeColdStartF32QuantMatrix("sink", 1, hidden, s+60),
			KConvProj:    makeColdStartF32QuantMatrix("kconv", hidden, hidden, s+70),
			VConvProj:    makeColdStartF32QuantMatrix("vconv", hidden, hidden, s+80),
			Router:       makeColdStartF32QuantMatrix("router", 2, hidden, s+90),
			AttnNorm:     vec(s),
			QInpNorm:     vec(s + 1),
			KInpNorm:     vec(s + 2),
			VInpNorm:     vec(s + 3),
			QClip:        vec(s + 4),
			RelBias:      vec(s + 5),
			RelClip:      vec(s + 6),
			KConv:        vec(s + 7),
			VConv:        vec(s + 8),
			AttnPostConv: vec(s + 9),
			FFNPostConv:  vec(s + 10),
			AttnResQuery: [][]float32{vec(s + 11), vec(s + 12)},
			FFNResQuery:  vec(s + 13),
			FFNNorm:      vec(s + 14),
			ExpProbsBias: vec(s + 15),
			GateUp:       &inkling2DeviceExperts{be: dev, stack: stack, rows: hidden, padK: hidden},
			Down:         &inkling2DeviceExperts{be: dev, stack: stack, rows: hidden, padK: hidden},
			GateUpScale:  vec(s + 16),
			DownScale:    vec(s + 17),
		}
	}
	ctrlWeights := make([]model.Inkling2LoopControllerWeights, ctrls)
	for i := range ctrlWeights {
		s := float32(100 + i*10)
		cw := &ctrlWeights[i]
		for j := 0; j < 4; j++ {
			cw.MsgQuery[j] = vec(s + float32(j))
			cw.Down[j] = makeColdStartF32QuantMatrix("cdown", 2, 2*hidden, s+float32(j))
			cw.Gain[j] = makeColdStartF32QuantMatrix("cgain", hidden, 2, s+float32(j)+5)
			cw.Forget[j] = makeColdStartF32QuantMatrix("cforget", hidden, 2, s+float32(j)+9)
		}
	}
	return &Inkling2LoopModel{
		HP: hp,
		Weights: &model.Inkling2LoopWeights{
			TokNorm:        vec(0.1),
			OutputNorm:     vec(0.2),
			OutputResQuery: vec(0.3),
			Output:         makeColdStartF32QuantMatrix("output", 8, hidden, 0.4),
			Layers:         layerWeights,
			Controllers:    ctrlWeights,
		},
	}
}

// fak-test:runtime fast est=1s
func TestInkling2ColdStartPipelinedUploadMatchesSequential(t *testing.T) {
	runWithWorkers := func(workers int) (*coldStartTraceDev, *inkling2Resident) {
		dev := &coldStartTraceDev{fakeInk2Budget: newFakeInk2Budget(1 << 30)}
		m := makeColdStartTestModel(6, 3, dev)
		s := &inkling2Resident{hp: m.HP, w: m.Weights, be: dev, dev: dev, q8: map[int]ink2Ref{}, arenaFloats: 1024}
		if err := s.uploadResidentWeights(workers); err != nil {
			t.Fatalf("uploadResidentWeights(%d): %v", workers, err)
		}
		return dev, s
	}

	seqDev, seqRes := runWithWorkers(1)
	defer seqRes.close()
	pipeDev, pipeRes := runWithWorkers(4)
	defer pipeRes.close()

	if len(pipeRes.owned) != len(seqRes.owned) {
		t.Fatalf("owned tensors = %d, want %d", len(pipeRes.owned), len(seqRes.owned))
	}
	if !slices.Equal(pipeDev.allocSizes, seqDev.allocSizes) {
		t.Fatalf("alloc sizes differ:\ngot:  %v\nwant: %v", pipeDev.allocSizes, seqDev.allocSizes)
	}
	if len(pipeDev.writes) != len(seqDev.writes) {
		t.Fatalf("write count = %d, want %d", len(pipeDev.writes), len(seqDev.writes))
	}
	for i := range seqDev.writes {
		if !slices.Equal(pipeDev.writes[i], seqDev.writes[i]) {
			t.Fatalf("write[%d] differs: got %v, want %v", i, pipeDev.writes[i], seqDev.writes[i])
		}
	}
	if len(pipeRes.layers) != len(seqRes.layers) || len(pipeRes.ctrls) != len(seqRes.ctrls) {
		t.Fatalf("layers/ctrls mismatch: (%d,%d) vs (%d,%d)", len(pipeRes.layers), len(pipeRes.ctrls), len(seqRes.layers), len(seqRes.ctrls))
	}
	for i := range seqRes.layers {
		if pipeRes.layers[i].resQRows != seqRes.layers[i].resQRows {
			t.Fatalf("layer %d resQRows = %d, want %d", i, pipeRes.layers[i].resQRows, seqRes.layers[i].resQRows)
		}
		if pipeRes.layers[i].wq.rows != seqRes.layers[i].wq.rows || pipeRes.layers[i].wq.cols != seqRes.layers[i].wq.cols {
			t.Fatalf("layer %d wq shape = [%d %d], want [%d %d]", i, pipeRes.layers[i].wq.rows, pipeRes.layers[i].wq.cols, seqRes.layers[i].wq.rows, seqRes.layers[i].wq.cols)
		}
	}
}

type badColdStartMatrix struct{}

func (badColdStartMatrix) Rows() int             { return 1 }
func (badColdStartMatrix) Cols() int             { return 1 }
func (badColdStartMatrix) MatVec(_, _ []float32) {}

// fak-test:runtime fast est=1s
func TestInkling2ColdStartPipelinedUploadErrorCleansUp(t *testing.T) {
	dev := &coldStartTraceDev{fakeInk2Budget: newFakeInk2Budget(1 << 30)}
	m := makeColdStartTestModel(4, 2, dev)
	m.Weights.Layers[1].Wq = badColdStartMatrix{}
	s := &inkling2Resident{hp: m.HP, w: m.Weights, be: dev, dev: dev, q8: map[int]ink2Ref{}, arenaFloats: 1024}
	err := s.uploadResidentWeights(4)
	if err == nil {
		t.Fatal("expected error for non-bindable matrix, got nil")
	}
	s.close()
	if dev.live != 0 || dev.frees != dev.allocs {
		t.Fatalf("error cleanup left %d floats live, %d frees of %d allocs", dev.live, dev.frees, dev.allocs)
	}
}

// fak-test:runtime fast est=1s
func TestInkling2ColdStartPipelinedUploadOverlapsHostPrep(t *testing.T) {
	var writes atomic.Int64
	dev := &coldStartTraceDev{
		fakeInk2Budget: newFakeInk2Budget(1 << 30),
		onWrite: func() {
			if writes.Add(1) <= 5 {
				time.Sleep(2 * time.Millisecond)
			}
		},
	}
	m := makeColdStartTestModel(8, 2, dev)
	s, err := newInkling2ResidentSized(m, dev, 64, 1024)
	if err != nil {
		t.Fatalf("newInkling2ResidentSized: %v", err)
	}
	defer s.close()
	if len(s.layers) != 8 || len(s.ctrls) != 2 {
		t.Fatalf("unexpected layers=%d ctrls=%d", len(s.layers), len(s.ctrls))
	}
}
