//go:build cuda && cgo

package ggufload

import (
	"math"
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2SoftmaxKL is KL(p || q) between the softmaxes of two logit rows, in nats.
func inkling2SoftmaxKL(p, q []float32) float64 {
	lse := func(v []float32) float64 {
		mx := math.Inf(-1)
		for _, x := range v {
			mx = math.Max(mx, float64(x))
		}
		var s float64
		for _, x := range v {
			s += math.Exp(float64(x) - mx)
		}
		return mx + math.Log(s)
	}
	lp, lq := lse(p), lse(q)
	var kl float64
	for i := range p {
		a := float64(p[i]) - lp
		kl += math.Exp(a) * (a - (float64(q[i]) - lq))
	}
	return kl
}

// FAK_INKLING2_VERIFY_PARITY=1: verifies of every shape the spec loop can issue
// (full width, so graph-captured, and shorter eager tails) committed in full, in
// part and not at all, interleaved with plain decode steps, leave the session
// exactly where plain greedy decode is. Every verify feeds the true greedy tokens,
// so every row must return the next greedy token, and a wrong rewind shows up as a
// wrong row in the next verify or step. Runs with CUDA graphs and without, and the
// two runs, which rewind through different commit paths, must give every verify
// row the same softmax (KL under 1e-4 nats).
func TestInkling2CUDAVerifyCommitMatchesSerial(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_VERIFY_PARITY")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	prompt := inkling2OracleCases[0].prompt
	ref, _, err := inkling2ResidentGreedy(m, be, prompt, 48)
	if err != nil {
		t.Fatal(err)
	}
	const depth = 4
	// {T, keep}; T 0 is one plain decode step
	plan := [][2]int{
		{5, 0}, {5, 5}, {5, 2}, {0, 0}, {5, 0}, {5, 5}, {3, 1}, {3, 0}, {3, 3},
		{0, 0}, {0, 0}, {5, 4}, {1, 1}, {5, 1}, {1, 0}, {0, 0}, {5, 5},
	}
	var rows [2][][]float32 // per graph mode, every verify row's logits in plan order
	for gi, graph := range []string{"1", "0"} {
		t.Run("graph="+graph, func(t *testing.T) {
			t.Setenv("FAK_INKLING2_GRAPH", graph)
			s, err := newInkling2Resident(m, be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.close()
			if err := s.enableSpec(depth); err != nil {
				t.Fatal(err)
			}
			next, err := s.prefill(prompt)
			if err != nil {
				t.Fatal(err)
			}
			if next != ref[0] {
				t.Fatalf("prefill next %d, plain greedy %d", next, ref[0])
			}
			c := 0 // ref[c] is the token the next forward consumes
			for k, op := range plan {
				T, keep := op[0], op[1]
				pos := s.pos
				if T == 0 {
					got, err := s.step(ref[c])
					if err != nil {
						t.Fatal(err)
					}
					if got != ref[c+1] {
						t.Fatalf("op %d step at %d: next %d, plain greedy %d", k, pos, got, ref[c+1])
					}
					c++
					continue
				}
				g, err := s.verify(ref[c : c+T])
				if err != nil {
					t.Fatal(err)
				}
				for i := range g {
					l, err := s.dev.Ink2Read(at(s.verifyLogits, i*s.output.rows), s.hp.VocabUnpadded)
					if err != nil {
						t.Fatal(err)
					}
					rows[gi] = append(rows[gi], l)
				}
				for i, got := range g {
					if got != ref[c+i+1] {
						t.Fatalf("op %d verify T=%d at %d: row %d next %d, plain greedy %d (rows %v, want %v)",
							k, T, pos, i, got, ref[c+i+1], g, ref[c+1:c+T+1])
					}
				}
				if err := s.commitVerify(T, keep); err != nil {
					t.Fatal(err)
				}
				if s.pos != pos+keep {
					t.Fatalf("op %d commit %d of %d moved the position %d -> %d", k, keep, T, pos, s.pos)
				}
				c += keep
			}
			t.Logf("%d ops, %d tokens committed; decode graph ready %v, verify graph ready %v, commit graph ready %v",
				len(plan), c, s.graphReady, s.verifyReady, s.keepReady)
			if graph == "1" && !(s.graphReady && s.verifyReady && s.keepReady) {
				t.Fatalf("graph mode did not capture every graph (decode %v, verify %v, commit %v)", s.graphReady, s.verifyReady, s.keepReady)
			}
		})
	}
	if len(rows[0]) != len(rows[1]) {
		t.Fatalf("graph run read %d verify rows, eager run %d", len(rows[0]), len(rows[1]))
	}
	worst, same := 0.0, 0
	for i := range rows[0] {
		kl := inkling2SoftmaxKL(rows[0][i], rows[1][i])
		worst = math.Max(worst, kl)
		if kl > 1e-4 {
			t.Errorf("verify row %d: graph vs eager KL %.3g nats", i, kl)
		}
		eq := true
		for j := range rows[0][i] {
			if math.Float32bits(rows[0][i][j]) != math.Float32bits(rows[1][i][j]) {
				eq = false
				break
			}
		}
		if eq {
			same++
		}
	}
	t.Logf("graph vs eager: %d/%d verify rows bitwise equal, worst KL %.3g nats", same, len(rows[0]), worst)
}

// FAK_INKLING2_HOST_VERIFY=1: a full-width GPU verify of the greedy chain against
// the host reference Inkling2LoopHost.VerifyTree over the same chain (the f32 Go
// forward over the same GGUF). The kernels sum in a different order, so rows are
// held to the same greedy token and a KL bound
// (FAK_INKLING2_HOST_VERIFY_MAX_KL, default 1e-2 nats), and the bitwise-equal
// logit count is logged. The eager verify and its graph capture and replay must
// agree bit for bit.
func TestInkling2CUDAVerifyMatchesHostVerifyTree(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_HOST_VERIFY")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	maxKL := 1e-2
	if e := os.Getenv("FAK_INKLING2_HOST_VERIFY_MAX_KL"); e != "" {
		v, err := strconv.ParseFloat(e, 64)
		if err != nil {
			t.Fatal(err)
		}
		maxKL = v
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	const T = inkling2SpecMaxDraft + 1
	c := inkling2OracleCases[0]
	chain := c.greedy[:T]

	hm, err := OpenInkling2Loop(os.Getenv("FAK_INKLING2_GGUF"))
	if err != nil {
		t.Fatal(err)
	}
	defer hm.Close()
	h, err := model.NewInkling2LoopHost(hm.HP, hm.Weights)
	if err != nil {
		t.Fatal(err)
	}
	for _, tok := range c.prompt {
		h.Step(tok)
	}
	nodes := make([]model.Inkling2TreeNode, T)
	for i := range nodes {
		nodes[i] = model.Inkling2TreeNode{Token: chain[i], Parent: i - 1}
	}
	tree, err := h.VerifyTree(nodes)
	if err != nil {
		t.Fatal(err)
	}

	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if err := s.enableSpec(T - 1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.prefill(c.prompt); err != nil {
		t.Fatal(err)
	}
	vocab := s.hp.VocabUnpadded
	read := func() ([]int, [][]float32) {
		t.Helper()
		g, err := s.verify(chain)
		if err != nil {
			t.Fatal(err)
		}
		rows := make([][]float32, T)
		for i := range rows {
			if rows[i], err = s.dev.Ink2Read(at(s.verifyLogits, i*s.output.rows), vocab); err != nil {
				t.Fatal(err)
			}
		}
		if err := s.commitVerify(T, 0); err != nil {
			t.Fatal(err)
		}
		return g, rows
	}
	// eager (grows the scratch), then capture, then replay
	g, gpu := read()
	for run := 1; run < 3; run++ {
		g2, again := read()
		for i := range again {
			if g2[i] != g[i] {
				t.Errorf("run %d row %d: next %d, eager %d", run, i, g2[i], g[i])
			}
			for j := range again[i] {
				if math.Float32bits(again[i][j]) != math.Float32bits(gpu[i][j]) {
					t.Errorf("run %d row %d: logit %d = %v, eager %v", run, i, j, again[i][j], gpu[i][j])
					break
				}
			}
		}
	}
	t.Logf("verify graph ready %v", s.verifyReady)
	if s.graph && !s.verifyReady {
		t.Fatal("graph mode on but the verify graph was not captured, so eager was compared with itself")
	}

	for i := range nodes {
		hl := tree.Logits[i][:vocab]
		gl := gpu[i]
		same, maxAbs := 0, 0.0
		hostTop := 0
		for j := range hl {
			if math.Float32bits(hl[j]) == math.Float32bits(gl[j]) {
				same++
			}
			maxAbs = math.Max(maxAbs, math.Abs(float64(hl[j]-gl[j])))
			if hl[j] > hl[hostTop] {
				hostTop = j
			}
		}
		kl := inkling2SoftmaxKL(hl, gl)
		t.Logf("row %d: host top %d, gpu top %d, llama.cpp %d; KL %.3g nats; %d/%d logits bitwise equal, max |dlogit| %.3g",
			i, hostTop, g[i], c.greedy[i+1], kl, same, vocab, maxAbs)
		if hostTop != g[i] {
			t.Errorf("row %d: gpu greedy %d, host %d", i, g[i], hostTop)
		}
		if kl > maxKL {
			t.Errorf("row %d: KL %.3g exceeds %.3g", i, kl, maxKL)
		}
	}
}
