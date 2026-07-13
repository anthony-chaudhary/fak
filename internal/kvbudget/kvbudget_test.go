package kvbudget

import (
	"math"
	"testing"
)

// TestKVBytesPerToken pins the closed-form per-token cache size against the
// triage doc §3.2: MLA latent+rope = 92×(512+64)×2 = 103.5 KiB/token, DSA index
// = 92×128×2 = 23.0 KiB/token, combined ≈ 126.5 KiB/token.
func TestKVBytesPerToken(t *testing.T) {
	s := GLM52DSA
	if got, want := s.MLAElemsPerToken(), 52992; got != want {
		t.Errorf("MLAElemsPerToken = %d, want %d", got, want)
	}
	if got, want := s.IndexElemsPerToken(), 11776; got != want {
		t.Errorf("IndexElemsPerToken = %d, want %d", got, want)
	}
	if got, want := s.KVElemsPerToken(), 64768; got != want {
		t.Errorf("KVElemsPerToken = %d, want %d", got, want)
	}
	if got, want := s.MLABytesPerToken(F16), 105984.0; got != want {
		t.Errorf("MLABytesPerToken(F16) = %v, want %v", got, want)
	}
	if got, want := s.KVBytesPerToken(F16), 129536.0; got != want {
		t.Errorf("KVBytesPerToken(F16) = %v, want %v", got, want)
	}
	// Doc §3.2 KiB/token readbacks.
	if got, want := s.KVBytesPerToken(F16)/1024, 126.5; got != want {
		t.Errorf("KV KiB/token = %v, want %v", got, want)
	}
	if got, want := s.MLABytesPerToken(F16)/1024, 103.5; got != want {
		t.Errorf("MLA KiB/token = %v, want %v", got, want)
	}
}

// TestKVGiBPerStreamExact proves the per-stream GiB is the exact rational
// ctx × bytes/token ÷ 1024³ with no floating drift, at the doc's three ctx.
func TestKVGiBPerStreamExact(t *testing.T) {
	s := GLM52DSA
	for _, ctx := range DocCtxLengths {
		wantKV := float64(ctx*s.KVElemsPerToken()*2) / float64(GiB)
		if got := s.KVGiBPerStream(ctx, F16); got != wantKV {
			t.Errorf("KVGiBPerStream(%d,F16) = %v, want %v", ctx, got, wantKV)
		}
		wantMLA := float64(ctx*s.MLAElemsPerToken()*2) / float64(GiB)
		if got := s.MLAGiBPerStream(ctx, F16); got != wantMLA {
			t.Errorf("MLAGiBPerStream(%d,F16) = %v, want %v", ctx, got, wantMLA)
		}
	}
}

// TestDocTable reproduces the landed triage doc §3.3 table cell-for-cell.
//
//	| ctx   | KV GiB/stream (MLA+idx) | KV GiB/stream (MLA only) | max @206 raw | max @~165 usable |
//	| 4096  | 0.494                   | 0.404                    | ~417         | ~334             |
//	| 8192  | 0.988                   | 0.809                    | ~208         | ~167             |
//	| 16384 | 1.977                   | 1.617                    | ~104         | ~83              |
func TestDocTable(t *testing.T) {
	want := []Row{
		{Ctx: 4096, KVGiBPerStream: 0.494, MLAGiBPerStream: 0.404, MaxStreamsRaw: 417, MaxStreamsUsable: 334},
		{Ctx: 8192, KVGiBPerStream: 0.988, MLAGiBPerStream: 0.809, MaxStreamsRaw: 208, MaxStreamsUsable: 167},
		{Ctx: 16384, KVGiBPerStream: 1.977, MLAGiBPerStream: 1.617, MaxStreamsRaw: 104, MaxStreamsUsable: 83},
	}
	got := DocTable()
	if len(got) != len(want) {
		t.Fatalf("DocTable len = %d, want %d", len(got), len(want))
	}
	const eps = 1e-9
	for i, w := range want {
		g := got[i]
		if g.Ctx != w.Ctx {
			t.Errorf("row %d ctx = %d, want %d", i, g.Ctx, w.Ctx)
		}
		if math.Abs(g.KVGiBPerStream-w.KVGiBPerStream) > eps {
			t.Errorf("row %d KVGiBPerStream = %v, want %v", i, g.KVGiBPerStream, w.KVGiBPerStream)
		}
		if math.Abs(g.MLAGiBPerStream-w.MLAGiBPerStream) > eps {
			t.Errorf("row %d MLAGiBPerStream = %v, want %v", i, g.MLAGiBPerStream, w.MLAGiBPerStream)
		}
		if g.MaxStreamsRaw != w.MaxStreamsRaw {
			t.Errorf("row %d MaxStreamsRaw = %d, want %d", i, g.MaxStreamsRaw, w.MaxStreamsRaw)
		}
		if g.MaxStreamsUsable != w.MaxStreamsUsable {
			t.Errorf("row %d MaxStreamsUsable = %d, want %d", i, g.MaxStreamsUsable, w.MaxStreamsUsable)
		}
	}
}

// TestMaxStreamsIsFloorDivision asserts the fit invariant the issue names:
// max streams = floor(budget / per-stream), and that each doc Row's stream
// counts equal that floor against the row's reported per-stream footprint.
func TestMaxStreamsIsFloorDivision(t *testing.T) {
	cases := []struct {
		budget, perStream float64
		want              int
	}{
		{206, 0.494, 417},
		{206, 0.988, 208},
		{206, 1.977, 104},
		{165, 0.494, 334},
		{165, 0.988, 167},
		{165, 1.977, 83},
		{206, 2.0, 103},
	}
	for _, c := range cases {
		want := int(math.Floor(c.budget / c.perStream))
		if want != c.want {
			t.Fatalf("test bug: floor(%v/%v)=%d, table says %d", c.budget, c.perStream, want, c.want)
		}
		if got := MaxStreams(c.budget, c.perStream); got != want {
			t.Errorf("MaxStreams(%v,%v) = %d, want %d", c.budget, c.perStream, got, want)
		}
	}
	// Non-positive per-stream => zero fit (no divide-by-zero).
	if got := MaxStreams(206, 0); got != 0 {
		t.Errorf("MaxStreams(206,0) = %d, want 0", got)
	}
	// Every doc Row's max-streams IS floor(budget / reported per-stream).
	for _, r := range DocTable() {
		if want := int(math.Floor(FreeVRAMGiB / r.KVGiBPerStream)); r.MaxStreamsRaw != want {
			t.Errorf("ctx %d: MaxStreamsRaw = %d, want floor(%v/%v) = %d",
				r.Ctx, r.MaxStreamsRaw, FreeVRAMGiB, r.KVGiBPerStream, want)
		}
		if want := int(math.Floor(UsableVRAMGiB / r.KVGiBPerStream)); r.MaxStreamsUsable != want {
			t.Errorf("ctx %d: MaxStreamsUsable = %d, want floor(%v/%v) = %d",
				r.Ctx, r.MaxStreamsUsable, UsableVRAMGiB, r.KVGiBPerStream, want)
		}
	}
}

// TestUsableBudget pins the headroom arithmetic: 206 × 0.8 = 164.8, reported as
// ~165 GiB (doc §3.3).
func TestUsableBudget(t *testing.T) {
	if got := math.Round(FreeVRAMGiB * HeadroomFactor); got != UsableVRAMGiB {
		t.Errorf("round(%v × %v) = %v, want UsableVRAMGiB = %v",
			FreeVRAMGiB, HeadroomFactor, got, UsableVRAMGiB)
	}
}

// TestQuantLever proves the §3.4 KV-quant lever: Q8_0 halves and Q4 quarters the
// per-token footprint (⇒ ~2× and ~4× the fit) vs the F16 default.
func TestQuantLever(t *testing.T) {
	s := GLM52DSA
	f16 := s.KVBytesPerToken(F16)
	if got, want := s.KVBytesPerToken(Q8_0), f16/2; got != want {
		t.Errorf("KVBytesPerToken(Q8_0) = %v, want %v (half F16)", got, want)
	}
	if got, want := s.KVBytesPerToken(Q4), f16/4; got != want {
		t.Errorf("KVBytesPerToken(Q4) = %v, want %v (quarter F16)", got, want)
	}
	// Halving the footprint doubles the fit: Q8_0 @8k matches F16 @4k (417).
	q8 := s.FitRow(8192, Q8_0)
	if q8.MaxStreamsRaw != 417 {
		t.Errorf("Q8_0 @8k MaxStreamsRaw = %d, want 417 (2× the F16 @8k of 208)", q8.MaxStreamsRaw)
	}
}

// TestMarkdownTable checks the emitter renders the doc §3.3 header and the 4k row.
func TestMarkdownTable(t *testing.T) {
	md := MarkdownTable(DocTable())
	wantHeader := "| ctx | KV GiB/stream (MLA+idx) | KV GiB/stream (MLA only) | max streams @206 GiB raw | max streams @~165 GiB usable |"
	if !containsLine(md, wantHeader) {
		t.Errorf("MarkdownTable missing header line %q\n---\n%s", wantHeader, md)
	}
	want4k := "| 4096 | 0.494 | 0.404 | 417 | 334 |"
	if !containsLine(md, want4k) {
		t.Errorf("MarkdownTable missing 4k row %q\n---\n%s", want4k, md)
	}
}

func containsLine(s, line string) bool {
	for _, l := range splitLines(s) {
		if l == line {
			return true
		}
	}
	return false
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
