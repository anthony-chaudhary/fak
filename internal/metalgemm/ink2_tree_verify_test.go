//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math/rand"
	"testing"
)

type ink2TTestTree struct {
	name   string
	parent []int32
}

func (tr ink2TTestTree) depth(t int) int {
	d := 0
	for ; tr.parent[t] >= 0; t = int(tr.parent[t]) {
		d++
	}
	return d
}

// rootPath returns node t's root path, root first.
func (tr ink2TTestTree) rootPath(t int) []int {
	var up []int
	for ; t >= 0; t = int(tr.parent[t]) {
		up = append(up, t)
	}
	for i, j := 0, len(up)-1; i < j; i, j = i+1, j-1 {
		up[i], up[j] = up[j], up[i]
	}
	return up
}

func (tr ink2TTestTree) ints() []float32 {
	v := make([]int, len(tr.parent))
	for i, p := range tr.parent {
		v[i] = int(p)
	}
	return ink2VTestInts(v...)
}

func ink2TTestChain(T int) ink2TTestTree {
	p := make([]int32, T)
	for i := range p {
		p[i] = int32(i - 1)
	}
	return ink2TTestTree{fmt.Sprintf("chain of %d", T), p}
}

func ink2TTestTrees(rng *rand.Rand) []ink2TTestTree {
	// A width-4 tree as the builder lays one out: the chain, then siblings.
	wide := ink2TTestTree{"width 4 of 13", []int32{-1, 0, 1, 2, 0, 0, 1, 3, 4, 1, 2, 7, 5}}
	trees := []ink2TTestTree{wide}
	for _, T := range []int{5, 16, 32} {
		p := make([]int32, T)
		p[0] = -1
		for t := 1; t < T; t++ {
			p[t] = int32(rng.Intn(t))
		}
		trees = append(trees, ink2TTestTree{fmt.Sprintf("random of %d", T), p})
	}
	for _, tr := range trees {
		if err := TreeVerifyCheck(tr.parent); err != nil {
			panic(err)
		}
	}
	return trees
}

// fak-test:runtime integration est=6s lane=default
func TestTreeVerifyAttendMatchesChainAndEachRootPath(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(91))
	const maxT = TreeVerifyMaxNodes
	p := Ink2AttnParams{Heads: 6, KVHeads: 3, HeadDim: 32, RelLen: 16, Scale: .177}
	odd := Ink2AttnParams{Heads: 4, KVHeads: 2, HeadDim: 30, RelLen: 5, Scale: .2}
	trees := ink2TTestTrees(rng)
	for _, c := range []struct {
		name                      string
		p                         Ink2AttnParams
		cap, window, extent, pos0 int
	}{
		{"global", p, 200, 0, 64, 60},
		{"local ring that wraps", p, 48 + maxT, 48, 48, 130},
		{"first tokens", p, 64, 0, 16, 0},
		{"ragged head", odd, 40 + maxT, 12, 20, 17},
	} {
		t.Run(c.name, func(t *testing.T) {
			pp := c.p
			n, rn, kvDim := pp.Heads*pp.HeadDim, pp.Heads*pp.RelLen, pp.KVHeads*pp.HeadDim
			qv, rv := ink2OpsTestRand(rng, maxT*n, 1), ink2OpsTestRand(rng, maxT*rn, 1)
			sv := ink2OpsTestRand(rng, maxT*pp.Heads, 1)
			q, r, sinks := ink2VTestPut(t, d, qv), ink2VTestPut(t, d, rv), ink2VTestPut(t, d, sv)
			rel := ink2VTestPut(t, d, ink2OpsTestRand(rng, c.extent*pp.RelLen, .1))
			kc0, vc0 := ink2OpsTestRand(rng, c.cap*kvDim, .3), ink2OpsTestRand(rng, c.cap*kvDim, 1)
			kc, vc := ink2VTestPut(t, d, kc0), ink2VTestPut(t, d, vc0)
			posDev := ink2VTestPut(t, d, ink2VTestInts(c.pos0))
			attend := func(tr ink2TTestTree, devicePos bool) []float32 {
				T := len(tr.parent)
				var pd Ink2Ref
				pos0 := c.pos0
				if devicePos {
					pd, pos0 = posDev, 0
				}
				floats := T * pp.Heads * VerifyAttendSpan(devicePos, c.window, c.cap, pos0, T)
				scores, out := ink2TestAlloc(t, d, floats), ink2TestAlloc(t, d, T*n)
				parent := ink2VTestPut(t, d, tr.ints())
				if err := d.TreeVerifyAttend(q, r, sinks, rel, kc, vc, scores, floats, out, pd, parent, pp, c.extent, c.window, c.cap, pos0, T); err != nil {
					t.Fatal(err)
				}
				return ink2OpsTestRead(t, d, out, T*n)
			}
			for _, T := range []int{1, 2, 4, 8, 16} {
				scores, out := ink2TestAlloc(t, d, T*pp.Heads*c.cap), ink2TestAlloc(t, d, T*n)
				if err := d.VerifyAttend(q, r, sinks, rel, kc, vc, scores, T*pp.Heads*c.cap, out, Ink2Ref{}, pp, c.extent, c.window, c.cap, c.pos0, T); err != nil {
					t.Fatal(err)
				}
				want := ink2OpsTestRead(t, d, out, T*n)
				for _, dev := range []bool{false, true} {
					ink2OpsTestSame(t, fmt.Sprintf("chain of %d, device position %v", T, dev), attend(ink2TTestChain(T), dev), want)
				}
			}
			// Each row is the decode attend of the last token of its root path,
			// with that path's rows laid at consecutive positions.
			scratch, ref := ink2TestAlloc(t, d, pp.Heads*c.cap), ink2TestAlloc(t, d, n)
			for _, tr := range trees {
				got := attend(tr, true)
				for row := range tr.parent {
					kp, vp := append([]float32(nil), kc0...), append([]float32(nil), vc0...)
					for dd, a := range tr.rootPath(row) {
						to, from := (c.pos0+dd)%c.cap*kvDim, (c.pos0+a)%c.cap*kvDim
						copy(kp[to:to+kvDim], kc0[from:from+kvDim])
						copy(vp[to:to+kvDim], vc0[from:from+kvDim])
					}
					meta := ink2VTestPut(t, d, ink2VTestInts(c.pos0+tr.depth(row)))
					if err := d.Attend(q.At(row*n), r.At(row*rn), sinks.At(row*pp.Heads), rel, ink2VTestPut(t, d, kp), ink2VTestPut(t, d, vp),
						scratch, ref, meta, pp, c.extent, c.window, c.cap); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("%s row %d", tr.name, row), got[row*n:(row+1)*n], ink2OpsTestRead(t, d, ref, n))
				}
			}
		})
	}
}

// fak-test:runtime integration est=4s lane=default
func TestTreeVerifyQKVPostMatchesChainAndTreePositions(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(92))
	const maxT = TreeVerifyMaxNodes
	p := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 40, RelLen: 20, QKClip: 2, RelClip: 1.5, Eps: 1e-6, Global: true, Alpha: .35, Floor: 3}
	const cap, pos0 = 1100, 37
	qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
	qv, rv := ink2OpsTestRand(rng, maxT*qn, 1), ink2OpsTestRand(rng, maxT*rn, 1)
	for i := range qv {
		qv[i] *= .15 * float32(1+i/p.HeadDim%p.Heads)
	}
	kr, vr := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*kvDim, 1)), ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*kvDim, 1))
	qclip, rclip := ink2VTestPut(t, d, ink2OpsTestRand(rng, qn, 1.5)), ink2VTestPut(t, d, ink2OpsTestRand(rng, rn, 1.5))
	kc0, vc0 := ink2OpsTestRand(rng, cap*kvDim, 9), ink2OpsTestRand(rng, cap*kvDim, 9)
	type state struct{ q, r, kc, vc Ink2Ref }
	fresh := func() state {
		return state{ink2VTestPut(t, d, qv), ink2VTestPut(t, d, rv), ink2VTestPut(t, d, kc0), ink2VTestPut(t, d, vc0)}
	}
	read := func(s state) [4][]float32 {
		return [4][]float32{ink2OpsTestRead(t, d, s.q, maxT*qn), ink2OpsTestRead(t, d, s.r, maxT*rn),
			ink2OpsTestRead(t, d, s.kc, cap*kvDim), ink2OpsTestRead(t, d, s.vc, cap*kvDim)}
	}
	posDev := ink2VTestPut(t, d, ink2VTestInts(pos0))
	tree := func(tr ink2TTestTree) [4][]float32 {
		s := fresh()
		if err := d.TreeVerifyQKVPost(s.q, s.r, kr, vr, qclip, rclip, s.kc, s.vc, posDev, ink2VTestPut(t, d, tr.ints()), p, cap, 0, len(tr.parent)); err != nil {
			t.Fatal(err)
		}
		return read(s)
	}
	for _, T := range []int{1, 2, 4, 8} {
		s := fresh()
		if err := d.VerifyQKVPost(s.q, s.r, kr, vr, qclip, rclip, s.kc, s.vc, Ink2Ref{}, p, cap, pos0, T); err != nil {
			t.Fatal(err)
		}
		want, got := read(s), tree(ink2TTestChain(T))
		for i, name := range []string{"q", "r", "k cache", "v cache"} {
			ink2OpsTestSame(t, fmt.Sprintf("chain of %d %s", T, name), got[i], want[i])
		}
	}
	for _, tr := range ink2TTestTrees(rng) {
		got := tree(tr)
		for row := range tr.parent {
			s := fresh()
			pos := pos0 + tr.depth(row)
			if err := d.QKVPost(s.q.At(row*qn), s.r.At(row*rn), kr.At(row*kvDim), vr.At(row*kvDim), qclip, rclip, s.kc, s.vc,
				ink2VTestPut(t, d, ink2VTestInts(pos)), p, cap, nil); err != nil {
				t.Fatal(err)
			}
			want := read(s)
			what := fmt.Sprintf("%s row %d", tr.name, row)
			ink2OpsTestSame(t, what+" q", got[0][row*qn:(row+1)*qn], want[0][row*qn:(row+1)*qn])
			ink2OpsTestSame(t, what+" r", got[1][row*rn:(row+1)*rn], want[1][row*rn:(row+1)*rn])
			at, from := (pos0+row)%cap*kvDim, pos%cap*kvDim
			ink2OpsTestSame(t, what+" k row", got[2][at:at+kvDim], want[2][from:from+kvDim])
			ink2OpsTestSame(t, what+" v row", got[3][at:at+kvDim], want[3][from:from+kvDim])
		}
	}
}

// fak-test:runtime integration est=4s lane=default
func TestTreeVerifyConvMatchesChainAndEachRootPath(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(93))
	const maxT = TreeVerifyMaxNodes
	trees := ink2TTestTrees(rng)
	for _, c := range []ink2VTestConv{
		{name: "dynamic taps with a residual", n: 24, heads: 4, kn: 4, dynamic: true, resid: true},
		{name: "static taps", n: 300, heads: 1, kn: 4},
		{name: "kernel of two", n: 9, heads: 3, kn: 2, dynamic: true},
		{name: "widest kernel", n: 16, heads: 2, kn: 8, resid: true},
	} {
		t.Run(c.name, func(t *testing.T) {
			c.fill(rng, maxT)
			past := c.kn - 1
			x, w := ink2VTestPut(t, d, c.x), ink2VTestPut(t, d, c.w)
			var dyn, res Ink2Ref
			if c.dynamic {
				dyn = ink2VTestPut(t, d, c.dyn)
			}
			if c.resid {
				res = ink2VTestPut(t, d, c.res)
			}
			run := func(tr ink2TTestTree) (out, log []float32, hist, logRef Ink2Ref) {
				T := len(tr.parent)
				hist = ink2VTestPut(t, d, c.hist)
				o, l := ink2TestAlloc(t, d, T*c.n), ink2TestAlloc(t, d, c.n*(past+T))
				if err := d.TreeVerifyConv(x, w, dyn, hist, l, o, res, ink2VTestPut(t, d, tr.ints()), c.n, c.heads, c.kn, c.norm, T); err != nil {
					t.Fatal(err)
				}
				ink2OpsTestSame(t, "entry history", ink2OpsTestRead(t, d, hist, c.n*past), c.hist)
				return ink2OpsTestRead(t, d, o, T*c.n), ink2OpsTestRead(t, d, l, c.n*(past+T)), hist, l
			}
			for _, T := range []int{1, 2, 4, 8} {
				L := past + T
				hist, out, log := ink2VTestPut(t, d, c.hist), ink2TestAlloc(t, d, T*c.n), ink2TestAlloc(t, d, c.n*L)
				if err := d.VerifyConv(x, w, dyn, hist, log, out, res, c.n, c.heads, c.kn, c.norm, T); err != nil {
					t.Fatal(err)
				}
				gotOut, gotLog, thist, tlog := run(ink2TTestChain(T))
				ink2OpsTestSame(t, fmt.Sprintf("chain of %d outputs", T), gotOut, ink2OpsTestRead(t, d, out, T*c.n))
				ink2OpsTestSame(t, fmt.Sprintf("chain of %d log", T), gotLog, ink2OpsTestRead(t, d, log, c.n*L))
				for keep := 1; keep <= T; keep++ {
					path := ink2TTestChain(T).rootPath(keep - 1)
					if err := d.TreeVerifyConvCommit(thist, tlog, Ink2Ref{}, c.n, past, L, TreeConvCommitCols(path, past)); err != nil {
						t.Fatal(err)
					}
					if err := d.VerifyConvCommit(hist, log, Ink2Ref{}, c.n, past, L, keep); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("chain of %d commit of %d", T, keep),
						ink2OpsTestRead(t, d, thist, c.n*past), ink2OpsTestRead(t, d, hist, c.n*past))
				}
			}
			for _, tr := range trees {
				T := len(tr.parent)
				L := past + T
				got, _, thist, tlog := run(tr)
				for row := range tr.parent {
					// the decode conv along the row's root path
					h, o := ink2VTestPut(t, d, c.hist), ink2TestAlloc(t, d, c.n)
					for _, a := range tr.rootPath(row) {
						var dr, rr Ink2Ref
						if c.dynamic {
							dr = dyn.At(a * c.heads * c.kn)
						}
						if c.resid {
							rr = res.At(a * c.n)
						}
						if err := d.Conv(x.At(a*c.n), w, dr, h, o, rr, c.n, c.heads, c.kn, c.norm); err != nil {
							t.Fatal(err)
						}
					}
					what := fmt.Sprintf("%s row %d", tr.name, row)
					ink2OpsTestSame(t, what, got[row*c.n:(row+1)*c.n], ink2OpsTestRead(t, d, o, c.n))
					// Committing the path leaves the decode conv's history,
					// from host columns and from device ones.
					cols := TreeConvCommitCols(tr.rootPath(row), past)
					if err := d.TreeVerifyConvCommit(thist, tlog, Ink2Ref{}, c.n, past, L, cols); err != nil {
						t.Fatal(err)
					}
					want := ink2OpsTestRead(t, d, h, c.n*past)
					ink2OpsTestSame(t, what+" commit", ink2OpsTestRead(t, d, thist, c.n*past), want)
					if err := d.Zero(thist, c.n*past); err != nil {
						t.Fatal(err)
					}
					if err := d.TreeVerifyConvCommit(thist, tlog, ink2VTestPut(t, d, ink2VTestInts(cols...)), c.n, past, L, nil); err != nil {
						t.Fatal(err)
					}
					ink2OpsTestSame(t, what+" device commit", ink2OpsTestRead(t, d, thist, c.n*past), want)
				}
			}
		})
	}
}

func ink2TTestPathDev(path []int) []float32 {
	v := make([]int, 1+TreeVerifyMaxNodes)
	v[0] = len(path)
	copy(v[1:], path)
	return ink2VTestInts(v...)
}

// fak-test:runtime integration est=2s lane=default
func TestTreeVerifyKVCompactAndGatherMatchHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(94))
	const width = 70
	for _, tr := range ink2TTestTrees(rng) {
		T := len(tr.parent)
		for _, c := range []struct{ cap, pos0 int }{{T + 40, 3}, {T + 5, T + 3}, {T, 0}} {
			ring0 := ink2OpsTestRand(rng, c.cap*width, 1)
			ring1 := ink2OpsTestRand(rng, c.cap*width, 1)
			for trial := range 4 {
				path := tr.rootPath(rng.Intn(T))
				want := [2][]float32{append([]float32(nil), ring0...), append([]float32(nil), ring1...)}
				for _, w := range want {
					for i, src := range path {
						to, from := (c.pos0+i)%c.cap*width, (c.pos0+src)%c.cap*width
						copy(w[to:to+width], w[from:from+width])
					}
				}
				what := fmt.Sprintf("%s ring %d at %d path %v", tr.name, c.cap, c.pos0, path)
				a, b := ink2VTestPut(t, d, ring0), ink2VTestPut(t, d, ring1)
				var err error
				if trial%2 == 0 {
					err = d.TreeVerifyKVCompact(a, b, Ink2Ref{}, Ink2Ref{}, width, c.cap, c.pos0, path)
				} else {
					err = d.TreeVerifyKVCompact(a, b, ink2VTestPut(t, d, ink2TTestPathDev(path)), ink2VTestPut(t, d, ink2VTestInts(c.pos0)),
						width, c.cap, 0, nil)
				}
				if err != nil {
					t.Fatal(err)
				}
				ink2OpsTestSame(t, what+" k", ink2OpsTestRead(t, d, a, c.cap*width), want[0])
				ink2OpsTestSame(t, what+" v", ink2OpsTestRead(t, d, b, c.cap*width), want[1])
			}
		}
		// gather: rows of width out of a stride
		const stride = width + 6
		in := ink2OpsTestRand(rng, T*stride, 1)
		path := tr.rootPath(T - 1)
		want := make([]float32, 0, len(path)*width)
		for _, src := range path {
			want = append(want, in[src*stride:src*stride+width]...)
		}
		inRef := ink2VTestPut(t, d, in)
		for _, dev := range []bool{false, true} {
			out := ink2TestAlloc(t, d, T*width)
			var err error
			if dev {
				err = d.TreeVerifyGatherRows(inRef, out, ink2VTestPut(t, d, ink2TTestPathDev(path)), width, stride, T, nil)
			} else {
				err = d.TreeVerifyGatherRows(inRef, out, Ink2Ref{}, width, stride, T, path)
			}
			if err != nil {
				t.Fatal(err)
			}
			ink2OpsTestSame(t, fmt.Sprintf("%s gather, device path %v", tr.name, dev), ink2OpsTestRead(t, d, out, len(path)*width), want)
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestTreeVerifyRejectsBadOperands(t *testing.T) {
	d := ink2TestDevice(t)
	buf := ink2TestAlloc(t, d, 4096)
	if err := TreeVerifyCheck([]int32{-1, 1}); err == nil {
		t.Fatal("accepted a row that parents itself")
	}
	if err := TreeVerifyCheck([]int32{0}); err == nil {
		t.Fatal("accepted a root with a parent")
	}
	if err := TreeVerifyPathCheck([]int{0, 2, 1}, 4); err == nil {
		t.Fatal("accepted a decreasing path")
	}
	if err := d.TreeVerifyKVCompact(buf, Ink2Ref{}, Ink2Ref{}, Ink2Ref{}, 8, 16, 0, []int{1, 2}); err == nil {
		t.Fatal("compact accepted a path that skips the root's row order")
	}
	if err := d.TreeVerifyConv(buf, buf.At(100), Ink2Ref{}, buf.At(200), Ink2Ref{}, buf.At(300), Ink2Ref{}, Ink2Ref{}, 4, 1, 4, 1, 2); err == nil {
		t.Fatal("conv accepted no parent table")
	}
	if err := d.TreeVerifyConvCommit(buf, buf.At(100), Ink2Ref{}, 4, 3, 5, []int{0, 1}); err == nil {
		t.Fatal("commit accepted too few columns")
	}
	if err := d.TreeVerifyConvCommit(buf, buf.At(100), Ink2Ref{}, 4, 3, 5, []int{0, 1, 5}); err == nil {
		t.Fatal("commit accepted a column past the log row")
	}
	if got := TreeConvCommitCols([]int{0, 1, 2}, 3); fmt.Sprint(got) != "[3 4 5]" {
		t.Fatalf("chain columns %v", got)
	}
	if got := TreeConvCommitCols([]int{0, 4}, 3); fmt.Sprint(got) != "[2 3 7]" {
		t.Fatalf("branch columns %v", got)
	}
}
