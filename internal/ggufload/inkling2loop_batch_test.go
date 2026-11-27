package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeInk2SeqOps struct{}

func (fakeInk2SeqOps) Ink2BQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv, st compute.Ink2SeqStrides, kvHalf bool, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BAttend(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int, st compute.Ink2SeqStrides, kvHalf bool, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BConv(x, w, hist, out, resid compute.Ink2Ref, n, kn int, norm float32, st compute.Ink2SeqStrides, C int) error {
	return nil
}
func (fakeInk2SeqOps) Ink2BSoftcapArgmax(logits compute.Ink2Ref, n, ld int, softcap float32, outIdx compute.Ink2Ref, C int) error {
	return nil
}

func TestInkling2BatchSeqOpsSelection(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("", false)) == nil {
		t.Fatal("a backend with the batched ops must use them by default")
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("1", true)) == nil {
		t.Fatal("FAK_INKLING2_BATCH_SEQOPS=1 must keep the batched ops")
	}
	if inkling2BatchSeqOps(fakeInk2SeqOps{}, env("0", true)) != nil {
		t.Fatal("FAK_INKLING2_BATCH_SEQOPS=0 must force the per-sequence loop")
	}
	if inkling2BatchSeqOps(struct{}{}, env("", false)) != nil {
		t.Fatal("a backend without the batched ops must loop")
	}
}

// The per-sequence fallback addresses row i exactly where the batched kernels do
// (stride*i), and keeps an absent operand absent as the kernels keep null null.
func TestInkling2SeqRow(t *testing.T) {
	if r := inkling2SeqRow(compute.Ink2Ref{}, 7, 3); r.T.Buf() != nil || r.Off != 0 {
		t.Fatalf("absent operand became %+v", r)
	}
	st := compute.Ink2SeqStrides{Q: 1, R: 2, K: 3, V: 4, KDyn: 5, VDyn: 6, Sinks: 7, KCache: 8, VCache: 9, Hist: 10,
		Scores: 12, Out: 13, Meta: 14, X: 15, Resid: 16}
	f := st.Fields()
	for i, want := range []int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16} {
		if f[i] != want {
			t.Fatalf("stride field %d = %d, want %d (device launcher order)", i, f[i], want)
		}
	}
}

func TestInkling2BatchCaps(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 2,
		LocalLayer: []bool{true, false, true, false}}
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	local := make([]int, slots)
	for i := range local {
		local[i] = 160
	}
	got := inkling2BatchCaps(hp, local, 512, 0)
	// slots 4, 5 revisit core layers 1, 2
	want := []int{160, 512, 160, 512, 512, 160}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("caps %v, want %v", got, want)
		}
	}
	// a ring longer than ctx is cut to ctx: positions past ctx-1 never occur
	for i, c := range inkling2BatchCaps(hp, local, 100, 0) {
		if c != 100 {
			t.Fatalf("slot %d cap %d at ctx 100, want 100", i, c)
		}
	}
}

func TestInkling2BatchGroupedThreshold(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	unset := env(nil)
	for C := 1; C <= 16; C++ {
		if got, want := inkling2BatchGrouped(C, unset), C >= inkling2BatchGroupedMin; got != want {
			t.Fatalf("C=%d grouped %v by default, want %v", C, got, want)
		}
	}
	if inkling2BatchGroupedMin != 3 {
		t.Fatalf("default grouped minimum %d, want 3 (C<=2 runs the sorted decode GEMV)", inkling2BatchGroupedMin)
	}
	if !inkling2BatchGrouped(2, env(map[string]string{"FAK_INKLING2_BATCH_GROUPED_MIN": " 2 "})) {
		t.Fatal("FAK_INKLING2_BATCH_GROUPED_MIN=2 must group a C=2 batch")
	}
	if inkling2BatchGrouped(8, env(map[string]string{"FAK_INKLING2_BATCH_GROUPED_MIN": "9"})) {
		t.Fatal("FAK_INKLING2_BATCH_GROUPED_MIN=9 must keep a C=8 batch sorted")
	}
	for _, bad := range []string{"", "x", "0", "-3"} {
		if inkling2BatchGrouped(2, env(map[string]string{"FAK_INKLING2_BATCH_GROUPED_MIN": bad})) ||
			!inkling2BatchGrouped(3, env(map[string]string{"FAK_INKLING2_BATCH_GROUPED_MIN": bad})) {
			t.Fatalf("FAK_INKLING2_BATCH_GROUPED_MIN=%q must fall back to the default", bad)
		}
	}
	forced := env(map[string]string{"FAK_INKLING2_PREFILL_GROUPED": "0", "FAK_INKLING2_BATCH_GROUPED_MIN": "1"})
	if inkling2BatchGrouped(16, forced) {
		t.Fatal("FAK_INKLING2_PREFILL_GROUPED=0 must force sorted experts at every C")
	}
	if !inkling2BatchGrouped(8, env(map[string]string{"FAK_INKLING2_PREFILL_GROUPED": "1"})) {
		t.Fatal("FAK_INKLING2_PREFILL_GROUPED=1 must keep the C threshold")
	}
}

func TestInkling2BatchQ8RowTileSelection(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	if !inkling2BatchQ8RowTile(env("", false)) || !inkling2BatchQ8RowTile(env("1", true)) {
		t.Fatal("the batch must default to the Q8 row-tile panel GEMM")
	}
	if inkling2BatchQ8RowTile(env("0", true)) {
		t.Fatal("FAK_INKLING2_BATCH_Q8_ROWTILE=0 must opt out")
	}
	if inkling2BatchMem(struct{}{}) != nil {
		t.Fatal("a backend without the strict allocator must keep the resident allocator")
	}
}

func TestInkling2BatchAdmissionArithmetic(t *testing.T) {
	// the Inkling-2-Mini shape: 24 layers plus two passes over core 6..17
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3, KVHeads: 8, HeadDim: 128,
		Hidden: 2304, Heads: 16, ConvKernel: 4, LocalLayer: make([]bool, 24)}
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	if slots != 48 {
		t.Fatalf("%d slots, want 48", slots)
	}
	caps := make([]int, slots)
	for i := range caps {
		caps[i] = 512
	}
	// 48 slots x (2 x 512 x 1024 KV floats + (2048+4608) x 3 history floats) x 4 bytes
	perSeq := int64(48) * (2*512*1024 + (2*1024+2*2304)*3) * 4
	if got := inkling2BatchStateBytes(hp, caps, 1, false); got != perSeq {
		t.Fatalf("state %d bytes per sequence, want %d", got, perSeq)
	}
	if mib := float64(perSeq) / (1 << 20); mib < 195 || mib > 196.5 {
		t.Fatalf("state %.1f MiB per sequence, want the measured 195.7", mib)
	}
	if got := inkling2BatchStateBytes(hp, caps, 8, false); got != 8*perSeq {
		t.Fatalf("state is not linear in C: %d vs %d", got, 8*perSeq)
	}
	keep, scr := 1000, 3000
	extra := int64(4) * (int64(keep+scr) + 8*int64(hp.Hidden+1+4+2*hp.Heads*512))
	if got := inkling2BatchBytes(hp, caps, 8, keep, scr, false); got != 8*perSeq+extra {
		t.Fatalf("need %d bytes, want state %d + arenas and rows %d", got, 8*perSeq, extra)
	}

	need := int64(1 << 30)
	if err := inkling2BatchAdmit(8, need, need+inkling2BatchHeadroom, true); err != nil {
		t.Fatalf("an exact fit with headroom must admit: %v", err)
	}
	err := inkling2BatchAdmit(12, need, need+inkling2BatchHeadroom-1, true)
	var refused *Inkling2BatchMemoryError
	if !errors.As(err, &refused) || refused.Sequences != 12 || refused.Need != need {
		t.Fatalf("one byte short must refuse with a typed error, got %v", err)
	}
	if err := inkling2BatchAdmit(16, need, compute.FreeUnknown, true); err != nil {
		t.Fatalf("an unknown free must fail open: %v", err)
	}
	if err := inkling2BatchAdmit(16, need, 0, false); err != nil {
		t.Fatalf("an unprobeable device must fail open: %v", err)
	}
}

func TestInkling2BatchKVHalfStateBytes(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3, KVHeads: 8, HeadDim: 128,
		Hidden: 2304, Heads: 16, ConvKernel: 4, LocalLayer: make([]bool, 24)}
	caps := make([]int, 48)
	for i := range caps {
		caps[i] = 512
	}
	full := inkling2BatchStateBytes(hp, caps, 1, false)
	half := inkling2BatchStateBytes(hp, caps, 1, true)
	// the KV is 2 bytes per element; the conv histories stay fp32
	if saved := full - half; saved != 48*2*512*1024*2 {
		t.Fatalf("f16 KV saves %d bytes per sequence, want %d", saved, 48*2*512*1024*2)
	}
	if mib := float64(full-half) / (1 << 20); mib != 96 {
		t.Fatalf("f16 KV saves %.1f MiB per sequence, want 96", mib)
	}
	if got := inkling2BatchStateBytes(hp, caps, 16, true); got != 16*half {
		t.Fatalf("f16 state is not linear in C: %d vs %d", got, 16*half)
	}
	keep, scr := 1000, 3000
	if d := inkling2BatchBytes(hp, caps, 16, keep, scr, false) - inkling2BatchBytes(hp, caps, 16, keep, scr, true); d != 16*(full-half) {
		t.Fatalf("the admission estimate drops %d bytes with f16 KV, want the state's %d", d, 16*(full-half))
	}
	// an odd element count rounds the arena up to whole floats
	if got := inkling2BatchKVFloats(7, true); got != 4 {
		t.Fatalf("7 halves take %d floats, want 4", got)
	}
	if got := inkling2BatchKVFloats(7, false); got != 7 {
		t.Fatalf("7 floats take %d floats", got)
	}
}

func TestInkling2BatchKVHalfSelection(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	for _, c := range []struct {
		v    string
		set  bool
		half bool
	}{{"", false, false}, {"", true, false}, {"fp32", true, false}, {"f32", true, false}, {"f16", true, true}, {"FP16", true, true}} {
		got, err := inkling2BatchKVHalf(env(c.v, c.set))
		if err != nil || got != c.half {
			t.Fatalf("FAK_INKLING2_BATCH_KV=%q (set %v): half %v err %v, want %v", c.v, c.set, got, err, c.half)
		}
	}
	if _, err := inkling2BatchKVHalf(env("bf16", true)); err == nil {
		t.Fatal("an unknown cache type must refuse, not fall back silently")
	}
}

type fakeInk2FewRows struct {
	compute.Inkling2VerifyDevice
	compute.Inkling2PrefillDevice
	compute.Inkling2BatchMemory
	calls []string
}

func (f *fakeInk2FewRows) Ink2VQ8Rows(w compute.Tensor, x, gate, q, y compute.Ink2Ref, rows int, quantize bool) (bool, error) {
	f.calls = append(f.calls, map[bool]string{true: "q8rows+quant", false: "q8rows"}[quantize])
	return true, nil
}
func (f *fakeInk2FewRows) Ink2VRows(w compute.Tensor, x, y compute.Ink2Ref, rows int) (bool, error) {
	f.calls = append(f.calls, "krows")
	return true, nil
}
func (f *fakeInk2FewRows) Ink2PMatMul(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
	f.calls = append(f.calls, "pmatmul")
	return nil
}
func (f *fakeInk2FewRows) Ink2PMatMulRows(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
	f.calls = append(f.calls, "rowtile")
	return nil
}

func TestInkling2BatchFewRowsSelection(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	vd := &fakeInk2FewRows{}
	if inkling2BatchFewRows(struct{}{}, 2, env("", false)) != nil {
		t.Fatal("a backend without the few-row kernels must keep the prefill kernels")
	}
	if inkling2BatchFewRows(vd, 1, env("", false)) == nil || inkling2BatchFewRows(vd, 8, env("", false)) == nil {
		t.Fatal("C <= 8 must run the few-row kernels by default")
	}
	if inkling2BatchFewRows(vd, 9, env("", false)) != nil {
		t.Fatal("C = 9 is past the default FAK_INKLING2_BATCH_ROWS_MAX")
	}
	if inkling2BatchFewRows(vd, 1, env("0", true)) != nil {
		t.Fatal("FAK_INKLING2_BATCH_ROWS_MAX=0 must disable the few-row kernels")
	}
	if inkling2BatchFewRows(vd, 4, env("3", true)) != nil || inkling2BatchFewRows(vd, 3, env("3", true)) == nil {
		t.Fatal("FAK_INKLING2_BATCH_ROWS_MAX=3 must bound C at 3")
	}
	if inkling2BatchFewRows(vd, 9, env("64", true)) == nil || inkling2BatchFewRows(vd, 10, env("64", true)) != nil {
		t.Fatal("a raised FAK_INKLING2_BATCH_ROWS_MAX must stop at the kernels' 9-row limit")
	}
	if inkling2BatchFewRows(vd, 8, env("x", true)) == nil {
		t.Fatal("an unparsable FAK_INKLING2_BATCH_ROWS_MAX must keep the default")
	}
}

// With fewRows set the batch's dense matmuls, dense F32 included, take the verify
// chunk's few-row kernels, quantizing a shared activation once per visit, without
// setting verify; unset, the Q8_0 row-tile and prefill GEMMs stay as they were.
func TestInkling2BatchFewRowsMatMul(t *testing.T) {
	f := &fakeInk2FewRows{}
	p := &ink2Prefill{s: &inkling2Resident{}, pd: f, vd: f, rows: f, T: 4, scrFloats: 1 << 20, fewRows: true}
	q8 := ink2Mat{t: compute.Tensor{Dtype: compute.Q8_0}, rows: 64, cols: 64}
	q5 := ink2Mat{t: compute.Tensor{Dtype: compute.Q5_K}, rows: 64, cols: 256}
	f32 := ink2Mat{t: compute.Tensor{Dtype: compute.F32}, rows: 64, cols: 64}
	x := p.tmp(p.T * 256)
	p.mm(q8, x)
	p.mm(q8, x)
	p.mm(q5, x)
	p.mm(f32, x)
	p.scrOff = 0
	clear(p.q8)
	p.mm(q8, x)
	want := []string{"q8rows+quant", "q8rows", "krows", "krows", "q8rows+quant"}
	if p.verify || len(f.calls) != len(want) {
		t.Fatalf("calls %v, want %v with verify unset", f.calls, want)
	}
	for i := range want {
		if f.calls[i] != want[i] {
			t.Fatalf("calls %v, want %v", f.calls, want)
		}
	}
	f.calls = nil
	p.fewRows = false
	p.mm(q8, x)
	p.mm(q5, x)
	if len(f.calls) != 2 || f.calls[0] != "rowtile" || f.calls[1] != "pmatmul" {
		t.Fatalf("without fewRows calls %v, want [rowtile pmatmul]", f.calls)
	}
	if p.s.err != nil {
		t.Fatal(p.s.err)
	}
}

func TestInkling2BatchFewRowsScratch(t *testing.T) {
	hp := model.Inkling2LoopHParams{Hidden: 2304, Heads: 32, HeadDim: 128, ExpertsUsed: 8}
	// codes are in/4 + in/32 floats per row, each buffer padded to 64 floats
	pad := func(n int) int { return (n + 63) &^ 63 }
	C := 8
	want := 4*pad(C*(2304/4+2304/32)) + pad(C*(4096/4+4096/32)) + pad(C*8*2304)
	if got := inkling2BatchFewRowsScratch(hp, C); got != want {
		t.Fatalf("few-row scratch %d floats, want %d", got, want)
	}
}

// A reused state re-selects the C-dependent expert path at every begin, so a
// narrow cohort after a wide one runs as if built for its own width.
func TestInkling2BatchBeginReselectsPerCohort(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_GROUPED_MIN", "3")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	run := func(width int) {
		t.Helper()
		reqs := make([]Inkling2LoopBatchRequest, width)
		for i := range reqs {
			reqs[i] = Inkling2LoopBatchRequest{Prompt: []int{i + 1}, MaxTokens: 2, Emit: func(int) bool { return true }}
		}
		if _, err := bt.Generate(reqs); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		width   int
		grouped bool
	}{{2, false}, {5, true}, {2, false}} {
		run(c.width)
		if bt.b == nil || bt.b.C != c.width || bt.b.p.grouped != c.grouped {
			t.Fatalf("cohort of %d: state missing, narrower than the cohort, or grouped != %v", c.width, c.grouped)
		}
	}
}

// f16 caches need the sequence-batched ops; a backend without them is refused
// before any arena is allocated.
func TestInkling2BatchStateRefusesHalfKVWithoutSeqOps(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_KV", "f16")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	emit := func(int) bool { return true }
	reqs := []Inkling2LoopBatchRequest{{Prompt: []int{1}, MaxTokens: 1, Emit: emit}, {Prompt: []int{2}, MaxTokens: 1, Emit: emit}}
	if _, err := bt.Generate(reqs); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_BATCH_KV") {
		t.Fatal("f16 cohort on a backend without sequence ops succeeded")
	}
	if bt.b != nil {
		t.Fatal("refused f16 cohort cached a state")
	}
}

func TestInkling2BatchKVHalfForAutoDefault(t *testing.T) {
	env := func(v string, set bool) func(string) (string, bool) {
		return func(string) (string, bool) { return v, set }
	}
	ready := fakeInk2HalfPrefill{}
	notReady := fakeInk2HalfPrefill{notReady: errors.New("no f16")}
	seq := fakeInk2SeqOps{}

	// Unset/blank auto-enables f16 when seq != nil and backend has ready f16 prefill.
	for _, tc := range []struct {
		name string
		be   any
		seq  compute.Inkling2BatchDevice
		v    string
		set  bool
		want bool
	}{
		{"unset capable backend defaults f16", ready, seq, "", false, true},
		{"blank capable backend defaults f16", ready, seq, "  ", true, true},
		{"explicit f32 overrides capable backend", ready, seq, "f32", true, false},
		{"explicit fp32 overrides capable backend", ready, seq, "FP32", true, false},
		{"explicit f16 on seq-only backend", struct{}{}, seq, "f16", true, true},
		{"unset without seq stays f32", ready, nil, "", false, false},
		{"unset without half prefill stays f32", struct{}{}, seq, "", false, false},
		{"unset when half prefill not ready stays f32", notReady, seq, "", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := inkling2BatchKVHalfFor(tc.be, tc.seq, env(tc.v, tc.set))
			if err != nil || got != tc.want {
				t.Fatalf("got (%v, %v), want (%v, nil)", got, err, tc.want)
			}
		})
	}
	if _, err := inkling2BatchKVHalfFor(ready, nil, env("f16", true)); err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_BATCH_KV") {
		t.Fatalf("explicit f16 without seq must error, got %v", err)
	}
}

func TestInkling2BatchChunkStepDown(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "1")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if err := bt.bind(); err != nil {
		t.Fatal(err)
	}
	sh512, err := bt.s.batchShapeChunk(2, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	sh256, err := bt.s.batchShapeChunk(2, 512, 256)
	if err != nil {
		t.Fatal(err)
	}
	sh128, err := bt.s.batchShapeChunk(2, 512, 128)
	if err != nil {
		t.Fatal(err)
	}
	if !(sh512.need > sh256.need && sh256.need > sh128.need) {
		t.Fatalf("expected strictly shrinking need: 512=%d 256=%d 128=%d", sh512.need, sh256.need, sh128.need)
	}
	// Cap VRAM so 512 fails but 256 fits.
	d.capFloats = d.live + int((sh256.need+inkling2BatchHeadroom+3)/4)
	b256, err := newInkling2BatchState(bt.s, 2, 512, 512)
	if err != nil {
		t.Fatalf("step-down 512->256 failed: %v", err)
	}
	if b256.capT != 256 || b256.prefillT != 256 {
		t.Fatalf("got capT=%d prefillT=%d, want 256", b256.capT, b256.prefillT)
	}
	b256.release()

	// Cap VRAM so 512 and 256 fail, but 128 fits.
	d.capFloats = d.live + int((sh128.need+inkling2BatchHeadroom+3)/4)
	b128, err := newInkling2BatchState(bt.s, 2, 512, 512)
	if err != nil {
		t.Fatalf("step-down 512->128 failed: %v", err)
	}
	if b128.capT != 128 || b128.prefillT != 128 {
		t.Fatalf("got capT=%d prefillT=%d, want 128", b128.capT, b128.prefillT)
	}
	b128.release()

	// Local ring caps also shrink cleanly at chunk=256 and chunk=128.
	hp := model.Inkling2LoopHParams{Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 2, SlidingWindow: 128,
		LocalLayer: []bool{true, false, true, false}}
	base := inkling2ResidentCaps(hp, 0, true, 512)
	if got := inkling2BatchCaps(hp, base, 4096, 256); got[0] != 384 || got[1] != 4096 {
		t.Fatalf("chunk=256 caps = %v, want local=384 global=4096", got)
	}
	if got := inkling2BatchCaps(hp, base, 4096, 128); got[0] != 256 || got[1] != 4096 {
		t.Fatalf("chunk=128 caps = %v, want local=256 global=4096", got)
	}
}

// Back-to-back cohorts after a 512 -> 256 chunk step-down reuse the cached 256-chunk
// arenas during cooldown instead of discarding and reallocating on every cohort.
func TestInkling2BatchChunkStepDownReusesArena(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "1")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if err := bt.bind(); err != nil {
		t.Fatal(err)
	}
	sh256, err := bt.s.batchShapeChunk(2, 512, 256)
	if err != nil {
		t.Fatal(err)
	}
	// Fit 256-chunk arenas, refuse 512-chunk arenas.
	d.capFloats = d.live + int((sh256.need+inkling2BatchHeadroom+3)/4)

	run := func() {
		t.Helper()
		reqs := []Inkling2LoopBatchRequest{
			{Prompt: make([]int, 300), MaxTokens: 2, Emit: func(int) bool { return true }},
			{Prompt: make([]int, 300), MaxTokens: 2, Emit: func(int) bool { return true }},
		}
		if _, err := bt.Generate(reqs); err != nil {
			t.Fatal(err)
		}
	}
	run()
	if bt.b == nil || bt.b.capT != 256 {
		t.Fatalf("first cohort capT = %v, want 256", bt.b)
	}
	held, liveAfterFirst := bt.b, d.live
	run()
	run()
	if bt.b != held || bt.b.capT != 256 || d.live != liveAfterFirst {
		t.Fatalf("stepped-down 256-chunk decode arena reallocated on reuse: reused=%v capT=%d live %d -> %d",
			bt.b == held, bt.b.capT, liveAfterFirst, d.live)
	}
}

type fakeInk2GraphCohort struct {
	*fakeInk2Cohort
	trims         int
	graphBegins   int
	graphEnds     int
	graphLaunches int
	graphResets   int
}

func newFakeInk2GraphCohort() *fakeInk2GraphCohort {
	return &fakeInk2GraphCohort{fakeInk2Cohort: newFakeInk2Cohort()}
}

func (g *fakeInk2GraphCohort) TrimLarge(int)          { g.trims++ }
func (g *fakeInk2GraphCohort) Ink2GraphBegin() error  { g.graphBegins++; return nil }
func (g *fakeInk2GraphCohort) Ink2GraphEnd() error    { g.graphEnds++; return nil }
func (g *fakeInk2GraphCohort) Ink2GraphLaunch() error { g.graphLaunches++; return nil }
func (g *fakeInk2GraphCohort) Ink2GraphReset()        { g.graphResets++ }

func TestInkling2BatchDecodeArenasSizedAtC(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "1")
	d := newFakeInk2GraphCohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()

	b, err := bt.acquireFor(2, 512, 300)
	if err != nil {
		t.Fatal(err)
	}
	wantKeep, wantScr := bt.s.prefillArenas(2)
	wantScr += 2*bt.s.output.rows + 64
	if b.p.keepFloats != wantKeep || b.p.scrFloats != wantScr {
		t.Fatalf("decode arenas = (%d, %d), want T=C (%d, %d)", b.p.keepFloats, b.p.scrFloats, wantKeep, wantScr)
	}
	if len(b.prefillOwned) != 0 {
		t.Fatalf("prefill arenas allocated before prefill: %d", len(b.prefillOwned))
	}
	if b.decodeNeed >= b.need {
		t.Fatalf("expected decodeNeed (%d) < peak need (%d) when prefillT=%d > C=%d", b.decodeNeed, b.need, b.prefillT, b.C)
	}
	if got := bt.ArenaBytes(); got != b.decodeNeed {
		t.Fatalf("ArenaBytes before prefill = %d, want decodeNeed %d", got, b.decodeNeed)
	}

	// Split credits steady-state bt.b.arenaBytes() (decodeNeed), not peak b.need, while
	// still enforcing peak <= free - headroom for the next cohort.
	sh, err := bt.s.batchShapeChunk(2, 512, 128)
	if err != nil {
		t.Fatal(err)
	}
	// Cap device so free + decodeNeed fits sh.need + headroom, but free + decodeNeed - 1 does not.
	// Since b.need > b.decodeNeed, crediting b.need instead of b.decodeNeed would falsely pass.
	d.capFloats = d.live - int((b.decodeNeed+3)/4) + int((sh.need+inkling2BatchHeadroom+3)/4) - 1
	groups, err := bt.Split([]int{300, 300})
	if err != nil {
		t.Fatal(err)
	}
	// Over-budget at C=2 forces Split into 1-row groups.
	if len(groups) != 2 {
		t.Fatalf("Split with steady-state credit = %v, want 2 single-row groups when peak exceeds free", groups)
	}
}

func TestInkling2BatchPrefillArenasReleasedAfterPrefill(t *testing.T) {
	for _, tc := range []struct {
		name    string
		longCap string
	}{
		{name: "non-mixed", longCap: "0"},
		{name: "mixed", longCap: "64"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
			t.Setenv("FAK_INKLING2_BATCH_ADMIT", "1")
			t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", tc.longCap)
			d := newFakeInk2GraphCohort()
			bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
			defer bt.Close()

			reqs := []Inkling2LoopBatchRequest{
				{Prompt: make([]int, 160), MaxTokens: 4, Emit: func(int) bool { return true }},
				{Prompt: make([]int, 160), MaxTokens: 4, Emit: func(int) bool { return true }},
			}
			peakObserved := false
			var peakBytes int64
			d.onWrite = func(dst compute.Ink2Ref) {
				if bt.b != nil && len(bt.b.prefillOwned) > 0 {
					peakObserved = true
					peakBytes = bt.ArenaBytes()
				}
			}
			res, err := bt.Generate(reqs)
			d.onWrite = nil
			if err != nil {
				t.Fatal(err)
			}
			for i, r := range res {
				if r.Err != nil || r.Generated != 4 {
					t.Fatalf("row %d: %+v", i, r)
				}
			}
			if !peakObserved || peakBytes <= bt.b.decodeNeed {
				t.Fatalf("prefill arenas never observed active during prefill (observed=%v peak=%d decode=%d)",
					peakObserved, peakBytes, bt.b.decodeNeed)
			}
			if len(bt.b.prefillOwned) != 0 {
				t.Fatalf("prefill arenas still held after decode: %d", len(bt.b.prefillOwned))
			}
			if bt.ArenaBytes() != bt.b.decodeNeed {
				t.Fatalf("ArenaBytes after decode = %d, want decodeNeed %d", bt.ArenaBytes(), bt.b.decodeNeed)
			}
			if d.trims == 0 {
				t.Fatal("expected inkling2TrimIdle to call TrimLarge after releasing prefill arenas")
			}
		})
	}
}

func TestInkling2BatchKeepPrefillArenasFallback(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_INKLING2_KEEP_PREFILL_ARENAS", "1")
	d := newFakeInk2GraphCohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()

	reqs := []Inkling2LoopBatchRequest{
		{Prompt: make([]int, 160), MaxTokens: 4, Emit: func(int) bool { return true }},
		{Prompt: make([]int, 160), MaxTokens: 4, Emit: func(int) bool { return true }},
	}
	if _, err := bt.Generate(reqs); err != nil {
		t.Fatal(err)
	}
	if len(bt.b.prefillOwned) != 2 {
		t.Fatalf("FAK_INKLING2_KEEP_PREFILL_ARENAS=1: prefillOwned = %d, want 2", len(bt.b.prefillOwned))
	}
	if bt.ArenaBytes() <= bt.b.decodeNeed {
		t.Fatalf("FAK_INKLING2_KEEP_PREFILL_ARENAS=1: ArenaBytes = %d, want > decodeNeed %d", bt.ArenaBytes(), bt.b.decodeNeed)
	}
}

func TestInkling2BatchGraphStabilityAcrossPureDecode(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_INKLING2_GRAPH", "1")
	t.Setenv("FAK_INKLING2_BATCH_GRAPH", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	d := newFakeInk2GraphCohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()

	// Prompt 100 tokens, MaxTokens 6 -> 1st token from prefill, then 5 pure-decode steps:
	// step 0 warms, step 1 captures + launches, steps 2..4 replay via Ink2GraphLaunch.
	reqs := []Inkling2LoopBatchRequest{
		{Prompt: make([]int, 100), MaxTokens: 6, Emit: func(int) bool { return true }},
		{Prompt: make([]int, 100), MaxTokens: 6, Emit: func(int) bool { return true }},
	}
	if _, err := bt.Generate(reqs); err != nil {
		t.Fatal(err)
	}
	if !bt.b.graphReady {
		t.Fatal("expected graphReady=true after multi-step pure decode")
	}
	if d.graphBegins != 1 || d.graphEnds != 1 {
		t.Fatalf("graph captures = (%d begins, %d ends), want (1, 1)", d.graphBegins, d.graphEnds)
	}
	if d.graphLaunches < 4 {
		t.Fatalf("graphLaunches = %d, want >= 4 across pure-decode steps", d.graphLaunches)
	}
	if len(bt.b.prefillOwned) != 0 {
		t.Fatalf("prefill arenas still held after pure decode: %d", len(bt.b.prefillOwned))
	}
	// begin() resets once; releasing prefill arenas must not reset the captured graph.
	if d.graphResets != 1 {
		t.Fatalf("graphResets = %d, want 1 (only from begin)", d.graphResets)
	}
}
