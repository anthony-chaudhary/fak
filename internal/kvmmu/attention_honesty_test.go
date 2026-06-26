package kvmmu_test

import (
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// attention_honesty_test.go is the cross-layer acceptance gate for #859. It uses
// the real model attention observer wired into the real kvmmu span ledger, then
// proves three things together: observer-on logits/tokens are bit-identical to
// observer-off, the recorded row stream replays to the same per-span accumulator,
// and every turn conserves emitted attention mass across the segment partition.

const witnessLambda = 0.75

type witnessChunk struct {
	id   string
	tool string
	ids  []int
}

type witnessRow struct {
	layer        int
	queryPos     int
	head         int
	keyPositions []int
	weights      []float32
}

type witnessTurn struct {
	emitted  float64
	residual float64
	rows     []witnessRow
}

type witnessRun struct {
	chunks   []witnessChunk
	logits   [][]float32
	tokens   []int
	turns    []witnessTurn
	snapshot []kvmmu.SpanAttention
}

func attentionWitnessModel() *model.Model {
	cfg := model.Config{
		HiddenSize: 16, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
	return model.NewSynthetic(cfg)
}

func TestAttentionWitnessHonestyReplayAndConservation(t *testing.T) {
	off := runAttentionWitnessTrace(t, false)
	on := runAttentionWitnessTrace(t, true)

	assertLogitsBitEqual(t, off.logits, on.logits)
	if len(off.tokens) != len(on.tokens) {
		t.Fatalf("generated token count differs: off=%d on=%d", len(off.tokens), len(on.tokens))
	}
	for i := range off.tokens {
		if off.tokens[i] != on.tokens[i] {
			t.Fatalf("generated token %d differs: observer off=%d on=%d", i, off.tokens[i], on.tokens[i])
		}
	}

	replayed := replayAttentionWitnessTrace(t, on.chunks, on.turns)
	assertAttentionSnapshotsEqual(t, on.snapshot, replayed)
}

func runAttentionWitnessTrace(t *testing.T, observe bool) witnessRun {
	t.Helper()

	m := attentionWitnessModel()
	c := kvmmu.New(m.NewSession())
	acc := kvmmu.NewAttentionAccumulator(witnessLambda, 0)

	var (
		out     witnessRun
		obsMu   sync.Mutex
		current *witnessTurn
	)
	if observe {
		m.SetAttnObserver(func(layer, queryPos, head int, keyPositions []int, weights []float32) {
			kp := append([]int(nil), keyPositions...)
			w := append([]float32(nil), weights...)
			var emitted float64
			for _, x := range w {
				emitted += float64(x)
			}

			obsMu.Lock()
			defer obsMu.Unlock()
			if current == nil {
				t.Errorf("attention observer fired outside an active turn")
				return
			}
			current.emitted += emitted
			current.residual += c.AttributeRow(kp, w)
			current.rows = append(current.rows, witnessRow{
				layer:        layer,
				queryPos:     queryPos,
				head:         head,
				keyPositions: kp,
				weights:      w,
			})
		})
	}

	appendChunk := func(ch witnessChunk) []float32 {
		if observe {
			out.turns = append(out.turns, witnessTurn{})
			obsMu.Lock()
			current = &out.turns[len(out.turns)-1]
			obsMu.Unlock()
		}

		logits, _ := c.Append(ch.id, ch.tool, ch.ids)
		logits = append([]float32(nil), logits...)
		out.chunks = append(out.chunks, cloneWitnessChunk(ch))
		out.logits = append(out.logits, logits)

		if observe {
			obsMu.Lock()
			turn := current
			current = nil
			obsMu.Unlock()
			if turn == nil {
				t.Fatalf("missing active witness turn for %s", ch.id)
			}
			if len(turn.rows) == 0 {
				t.Fatalf("observer emitted no rows for turn %s", ch.id)
			}
			if math.Abs(turn.residual) > 1e-6 {
				t.Fatalf("turn %s unattributed residual = %.9g, want 0", ch.id, turn.residual)
			}
			if d := math.Abs(c.AttendedMass() - turn.emitted); d > 1e-5 {
				t.Fatalf("turn %s attended mass = %.9g, emitted = %.9g (delta %.9g)", ch.id, c.AttendedMass(), turn.emitted, d)
			}
			acc.ObserveContext(c)
			c.ResetAttention()
		}
		return logits
	}

	base := []witnessChunk{
		{id: "sys", tool: "system", ids: []int{1, 2}},
		{id: "kb", tool: "search_kb", ids: []int{3, 4, 5}},
		{id: "usr", tool: "user", ids: []int{6, 7}},
	}
	var logits []float32
	for _, ch := range base {
		logits = appendChunk(ch)
	}
	for i := 0; i < 3; i++ {
		tok := argmaxWitness(logits)
		out.tokens = append(out.tokens, tok)
		logits = appendChunk(witnessChunk{
			id:   fmt.Sprintf("assistant-%d", i),
			tool: "assistant",
			ids:  []int{tok},
		})
	}
	if observe {
		out.snapshot = acc.Snapshot()
	}
	return out
}

func replayAttentionWitnessTrace(t *testing.T, chunks []witnessChunk, turns []witnessTurn) []kvmmu.SpanAttention {
	t.Helper()
	if len(chunks) != len(turns) {
		t.Fatalf("replay chunks=%d turns=%d, want equal", len(chunks), len(turns))
	}
	m := attentionWitnessModel()
	c := kvmmu.New(m.NewSession())
	acc := kvmmu.NewAttentionAccumulator(witnessLambda, 0)

	for i, ch := range chunks {
		c.Append(ch.id, ch.tool, ch.ids)
		var emitted, residual float64
		for _, row := range turns[i].rows {
			for _, w := range row.weights {
				emitted += float64(w)
			}
			residual += c.AttributeRow(row.keyPositions, row.weights)
		}
		if math.Abs(residual) > 1e-6 {
			t.Fatalf("replay turn %s unattributed residual = %.9g, want 0", ch.id, residual)
		}
		if d := math.Abs(c.AttendedMass() - emitted); d > 1e-5 {
			t.Fatalf("replay turn %s attended mass = %.9g, emitted = %.9g (delta %.9g)", ch.id, c.AttendedMass(), emitted, d)
		}
		acc.ObserveContext(c)
		c.ResetAttention()
	}
	return acc.Snapshot()
}

func cloneWitnessChunk(ch witnessChunk) witnessChunk {
	return witnessChunk{id: ch.id, tool: ch.tool, ids: append([]int(nil), ch.ids...)}
}

func assertLogitsBitEqual(t *testing.T, off, on [][]float32) {
	t.Helper()
	if len(off) != len(on) {
		t.Fatalf("logit turn count differs: off=%d on=%d", len(off), len(on))
	}
	for turn := range off {
		if len(off[turn]) != len(on[turn]) {
			t.Fatalf("turn %d logit length differs: off=%d on=%d", turn, len(off[turn]), len(on[turn]))
		}
		for i := range off[turn] {
			if math.Float32bits(off[turn][i]) != math.Float32bits(on[turn][i]) {
				t.Fatalf("turn %d logit[%d] differs with observer on: off=%v on=%v", turn, i, off[turn][i], on[turn][i])
			}
		}
	}
}

func assertAttentionSnapshotsEqual(t *testing.T, got, want []kvmmu.SpanAttention) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("snapshot length = %d, want %d: got=%+v want=%+v", len(got), len(want), got, want)
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.ID != w.ID || g.EMA != w.EMA || g.Cumulative != w.Cumulative {
			t.Fatalf("snapshot[%d] = %+v, want %+v", i, g, w)
		}
		if len(g.Trajectory) != len(w.Trajectory) {
			t.Fatalf("snapshot[%d] trajectory len = %d, want %d", i, len(g.Trajectory), len(w.Trajectory))
		}
		for j := range g.Trajectory {
			if g.Trajectory[j] != w.Trajectory[j] {
				t.Fatalf("snapshot[%d] trajectory[%d] = %+v, want %+v", i, j, g.Trajectory[j], w.Trajectory[j])
			}
		}
	}
}

func argmaxWitness(v []float32) int {
	best, bestV := 0, v[0]
	for i, x := range v {
		if x > bestV {
			best, bestV = i, x
		}
	}
	return best
}
