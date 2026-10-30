package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// The ordinary producer must prove ownership before its post-mix may recycle
// residuals. A later token failure may change private scratch, never committed
// streams, hidden state or carry. This is a host witness, not GPU qualification.
// fak-test:runtime fast est=250ms lane=default
func TestV41PrefillMHCPrivateResidualReuse(t *testing.T) {
	cloneRows := func(rows [][]float32) [][]float32 {
		out := make([][]float32, len(rows))
		for i, row := range rows {
			out[i] = append([]float32(nil), row...)
		}
		return out
	}
	checkBits := func(t *testing.T, got, want [][]float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("row count %d != %d", len(got), len(want))
		}
		for h := range want {
			if len(got[h]) != len(want[h]) {
				t.Fatalf("row %d width differs", h)
			}
			for d := range want[h] {
				if math.Float32bits(got[h][d]) != math.Float32bits(want[h][d]) {
					t.Fatalf("row %d coordinate %d differs: %g != %g", h, d, got[h][d], want[h][d])
				}
			}
		}
	}
	// Above the ordinary parallel threshold, with a non-divisible coordinate
	// tail and nontrivial cross-stream coefficients: an early source overwrite
	// would corrupt later destinations even though each worker owns coordinates.
	const width = 5123
	x := make([]float32, width)
	streams := make([][]float32, 4)
	mix := v41MHCMix{post: []float32{.7, -.4, .3, .6}, comb: make([]float32, 16)}
	for i := range mix.comb {
		mix.comb[i] = float32(i%7-3) / 13
	}
	for h := range streams {
		streams[h] = make([]float32, width)
		for d := range x {
			x[d] = float32(d%19-9) / 17
			streams[h][d] = float32((d+3*h)%23-11) / 29
		}
	}
	before := cloneRows(streams)
	want, err := v41MHCPostBF16On(nil, 0, x, streams, mix)
	if err != nil {
		t.Fatal(err)
	}
	borrowed, err := v41PrefillMHCPostBF16On(nil, 0, x, streams, mix, 2)
	if err != nil {
		t.Fatal(err)
	}
	checkBits(t, borrowed, want)
	checkBits(t, streams, before)
	got, err := v41PrefillMHCPostBF16Owned(nil, 0, x, streams, mix, 2)
	if err != nil {
		t.Fatal(err)
	}
	checkBits(t, got, want)
	for h := range streams {
		if &got[h][0] != &streams[h][0] {
			t.Fatal("owned multirow post allocated a replacement residual")
		}
	}
	decode := cloneRows(before)
	if _, err := v41PrefillMHCPostBF16Owned(nil, 0, x, decode, mix, 1); err != nil {
		t.Fatal(err)
	}
	checkBits(t, decode, before)

	m := v41RawFullFlattenedMHC(t)
	for _, mode := range []string{"owned", "borrowed", "late-fault"} {
		t.Run(mode, func(t *testing.T) {
			const rows = 2
			H := m.Cfg.HiddenSize
			in := v41LayerMoEInput{full: true, H: H, routeCfg: v41DefaultRouterConfig(),
				x: make([][]float32, rows), streams: make([][][]float32, rows),
				attnOut: make([][]float32, rows), hcByPos: make([]v41MHCMix, rows),
				scratch: &v41ProjScratch{mhcCarry: newV41MHCCarry(rows)}}
			for row := 0; row < rows; row++ {
				in.x[row] = make([]float32, H)
				in.attnOut[row] = make([]float32, H)
				for d := range in.x[row] {
					in.x[row][d] = float32((row+d)%9+1) / 16
					in.attnOut[row][d] = float32(d%5-2) / 32
				}
				in.streams[row], err = v41FullInitialStreams(in.x[row])
				if err != nil {
					t.Fatal(err)
				}
				in.hcByPos[row] = v41MHCMix{pre: []float32{.25, .25, .25, .25}, post: mix.post, comb: mix.comb}
			}
			panel, err := m.v41LayerFFNInputs(in)
			if err != nil || !panel.hostResidualOwned {
				t.Fatalf("ordinary private producer: owned=%v err=%v", panel.hostResidualOwned, err)
			}
			if mode == "borrowed" {
				// An ordinary caller supplying residuals carries no producer proof.
				panel.hostResidualOwned = false
			}
			if mode == "late-fault" {
				panel.mixes[1].comb[15] = float32(math.Inf(1))
			}
			originalX, originalCarry := cloneRows(in.x), cloneRows(in.scratch.mhcCarry.pre)
			originalStreams := make([][][]float32, rows)
			originalResidual := make([][][]float32, rows)
			wantNext := make([][][]float32, rows)
			routed := make([][]float32, rows)
			var wantErr error
			for row := 0; row < rows; row++ {
				originalStreams[row] = cloneRows(in.streams[row])
				originalResidual[row] = cloneRows(panel.residual[row])
				routed[row] = make([]float32, H)
				shared, e := m.v41SharedExpertSwiGLU(0, panel.inputs[row], m.Cfg)
				if e != nil {
					t.Fatal(e)
				}
				moe, e := v41SharedExpertAdd(routed[row], shared, in.routeCfg)
				if e != nil {
					t.Fatal(e)
				}
				wantNext[row], e = v41MHCPostBF16On(nil, 0, moe, panel.residual[row], panel.mixes[row])
				if e != nil {
					wantErr = e
				}
			}
			first := &panel.residual[0][0][0]
			err = m.v41LayerSharedExpertAndPost(in, panel, routed)
			if mode == "late-fault" {
				if wantErr == nil || err == nil || err.Error() != wantErr.Error() || !errors.Is(err, ErrV41ForwardStage) {
					t.Fatalf("late post failure=%v, want %v", err, wantErr)
				}
				if !reflect.DeepEqual(in.x, originalX) || !reflect.DeepEqual(in.streams, originalStreams) ||
					!reflect.DeepEqual(in.scratch.mhcCarry.pre, originalCarry) || in.scratch.mhcCarry.nextLayer != 0 {
					t.Fatal("late post failure published hidden state, streams or carry")
				}
			} else {
				if err != nil || in.scratch.mhcCarry.nextLayer != 1 {
					t.Fatalf("epilogue: nextLayer=%d err=%v", in.scratch.mhcCarry.nextLayer, err)
				}
				for row := range wantNext {
					checkBits(t, in.streams[row], wantNext[row])
				}
			}
			if mode == "borrowed" {
				if !reflect.DeepEqual(panel.residual, originalResidual) {
					t.Fatal("borrowed residual changed")
				}
			} else {
				checkBits(t, panel.residual[0], wantNext[0])
				if first != &panel.residual[0][0][0] || reflect.DeepEqual(panel.residual[0], originalResidual[0]) {
					t.Fatal("epilogue did not reuse the private first residual before publication")
				}
			}
		})
	}
}
