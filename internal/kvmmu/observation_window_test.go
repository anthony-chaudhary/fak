package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// refObservationMass is a dense float64 reference: full masked score matrix,
// softmax per row, column sum over the window rows restricted to the prefix.
func refObservationMass(q, k []float32, s kvmmu.ObservationWindowShape) [][]float64 {
	group := s.QueryHeads / s.KVHeads
	prefix := s.SeqLen - s.Window
	out := make([][]float64, s.QueryHeads)
	for h := 0; h < s.QueryHeads; h++ {
		out[h] = make([]float64, prefix)
		kvh := h / group
		for i := 0; i < s.Window; i++ {
			qp := prefix + i
			scores := make([]float64, s.SeqLen)
			for j := 0; j < s.SeqLen; j++ {
				if j >= prefix && j-prefix > i {
					scores[j] = math.Inf(-1)
					continue
				}
				var dot float64
				for d := 0; d < s.HeadDim; d++ {
					dot += float64(q[(h*s.SeqLen+qp)*s.HeadDim+d]) * float64(k[(kvh*s.SeqLen+j)*s.HeadDim+d])
				}
				scores[j] = dot / math.Sqrt(float64(s.HeadDim))
			}
			m := math.Inf(-1)
			for _, v := range scores {
				m = math.Max(m, v)
			}
			var z float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - m)
				z += scores[j]
			}
			for j := 0; j < prefix; j++ {
				out[h][j] += scores[j] / z
			}
		}
	}
	return out
}

// lcg gives deterministic pseudo-random inputs without math/rand seeding.
func lcg(n int, seed uint32) []float32 {
	out := make([]float32, n)
	x := seed
	for i := range out {
		x = x*1664525 + 1013904223
		out[i] = float32(int32(x>>8)%2000) / 1000
	}
	return out
}

func TestObservationWindowAttentionMass(t *testing.T) {
	t.Run("matches dense masked reference with GQA", func(t *testing.T) {
		s := kvmmu.ObservationWindowShape{QueryHeads: 4, KVHeads: 2, SeqLen: 11, HeadDim: 8, Window: 4}
		q := lcg(s.QueryHeads*s.SeqLen*s.HeadDim, 1)
		k := lcg(s.KVHeads*s.SeqLen*s.HeadDim, 2)
		got, err := kvmmu.ObservationWindowAttentionMass(q, k, s)
		if err != nil {
			t.Fatal(err)
		}
		want := refObservationMass(q, k, s)
		if len(got) != s.QueryHeads {
			t.Fatalf("heads = %d, want %d", len(got), s.QueryHeads)
		}
		for h := range want {
			if len(got[h]) != s.SeqLen-s.Window {
				t.Fatalf("head %d len = %d, want %d", h, len(got[h]), s.SeqLen-s.Window)
			}
			for p := range want[h] {
				if math.Abs(float64(got[h][p])-want[h][p]) > 1e-5 {
					t.Fatalf("head %d pos %d = %v, want %v", h, p, got[h][p], want[h][p])
				}
			}
		}
	})

	t.Run("zero queries give causal uniform mass", func(t *testing.T) {
		// Each window row i sees prefix+i+1 keys uniformly, so every prefix
		// key gets sum_i 1/(prefix+i+1); a missing causal mask would give
		// Window/SeqLen instead.
		s := kvmmu.ObservationWindowShape{QueryHeads: 1, KVHeads: 1, SeqLen: 6, HeadDim: 2, Window: 3}
		q := make([]float32, 12)
		k := lcg(12, 7)
		got, err := kvmmu.ObservationWindowAttentionMass(q, k, s)
		if err != nil {
			t.Fatal(err)
		}
		want := 1.0/4 + 1.0/5 + 1.0/6
		for p, v := range got[0] {
			if math.Abs(float64(v)-want) > 1e-6 {
				t.Fatalf("pos %d = %v, want %v", p, v, want)
			}
		}
	})

	t.Run("heads do not mix", func(t *testing.T) {
		// Head 0 attends hard to key 0, head 1 to key 1; they share a KV head.
		s := kvmmu.ObservationWindowShape{QueryHeads: 2, KVHeads: 1, SeqLen: 4, HeadDim: 2, Window: 1}
		k := []float32{10, 0, 0, 10, 0, 0, 0, 0}
		q := []float32{
			0, 0, 0, 0, 0, 0, 10, 0,
			0, 0, 0, 0, 0, 0, 0, 10,
		}
		got, err := kvmmu.ObservationWindowAttentionMass(q, k, s)
		if err != nil {
			t.Fatal(err)
		}
		if !(got[0][0] > 0.99 && got[0][1] < 0.01) {
			t.Fatalf("head 0 = %v, want mass on key 0", got[0])
		}
		if !(got[1][1] > 0.99 && got[1][0] < 0.01) {
			t.Fatalf("head 1 = %v, want mass on key 1", got[1])
		}
	})

	t.Run("prefix mass bounded by window", func(t *testing.T) {
		s := kvmmu.ObservationWindowShape{QueryHeads: 1, KVHeads: 1, SeqLen: 9, HeadDim: 4, Window: 3}
		got, err := kvmmu.ObservationWindowAttentionMass(lcg(36, 3), lcg(36, 4), s)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, v := range got[0] {
			if v < 0 {
				t.Fatalf("negative mass %v", v)
			}
			sum += float64(v)
		}
		if sum <= 0 || sum > float64(s.Window)+1e-5 {
			t.Fatalf("prefix mass %v outside (0, %d]", sum, s.Window)
		}
	})

	t.Run("window covering the whole prompt leaves no prefix", func(t *testing.T) {
		s := kvmmu.ObservationWindowShape{QueryHeads: 2, KVHeads: 2, SeqLen: 3, HeadDim: 2, Window: 3}
		got, err := kvmmu.ObservationWindowAttentionMass(lcg(12, 5), lcg(12, 6), s)
		if err != nil {
			t.Fatal(err)
		}
		for h, v := range got {
			if len(v) != 0 {
				t.Fatalf("head %d len = %d, want 0", h, len(v))
			}
		}
	})

	t.Run("rejects bad shapes", func(t *testing.T) {
		ok := kvmmu.ObservationWindowShape{QueryHeads: 2, KVHeads: 1, SeqLen: 4, HeadDim: 2, Window: 2}
		cases := map[string]struct {
			s    kvmmu.ObservationWindowShape
			q, k int
		}{
			"zero window":       {func() kvmmu.ObservationWindowShape { s := ok; s.Window = 0; return s }(), 16, 8},
			"window > seq":      {func() kvmmu.ObservationWindowShape { s := ok; s.Window = 5; return s }(), 16, 8},
			"heads not grouped": {func() kvmmu.ObservationWindowShape { s := ok; s.QueryHeads = 3; s.KVHeads = 2; return s }(), 24, 16},
			"zero head dim":     {func() kvmmu.ObservationWindowShape { s := ok; s.HeadDim = 0; return s }(), 0, 0},
			"short queries":     {ok, 15, 8},
			"short keys":        {ok, 16, 7},
		}
		for name, c := range cases {
			if _, err := kvmmu.ObservationWindowAttentionMass(make([]float32, c.q), make([]float32, c.k), c.s); err == nil {
				t.Errorf("%s: want error", name)
			}
		}
		if _, err := kvmmu.ObservationWindowAttentionMass(make([]float32, 16), make([]float32, 8), ok); err != nil {
			t.Fatalf("valid shape rejected: %v", err)
		}
	})
}
