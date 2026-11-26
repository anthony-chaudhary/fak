//go:build darwin && arm64 && cgo

package ggufload

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2PipeEmbedCheck embeds rows lo..hi of the device table chunk at a time
// and requires every value to equal the host row bit for bit.
func inkling2PipeEmbedCheck(t *testing.T, be compute.Backend, emb inkling2DeviceEmbedding, lo, hi int) {
	t.Helper()
	dev := be.(compute.Inkling2Device)
	pipe, ok := be.(ink2Pipeliner)
	if !ok || !pipe.Ink2CanEmbed(emb.table) {
		t.Fatalf("backend %T cannot embed a %s table", be, emb.m.typ)
	}
	cols := emb.m.cols
	const chunk = 512
	alloc := func(n int) ink2Ref {
		tt, err := dev.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { be.Free(tt) })
		return compute.Ink2At(tt, 0)
	}
	idx, out := alloc(chunk), alloc(chunk*(cols+1))
	host := make([]float32, cols)
	ids := make([]float32, chunk)
	for base := lo; base < hi; base += chunk {
		n := min(chunk, hi-base)
		for i := 0; i < n; i++ {
			ids[i] = math.Float32frombits(uint32(base + i))
		}
		if err := dev.Ink2Write(idx, ids[:n]); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			if err := pipe.Ink2EmbedRow(emb.table, at(idx, i), base+i, at(out, i*(cols+1))); err != nil {
				t.Fatal(err)
			}
		}
		got, err := dev.Ink2Read(out, n*(cols+1))
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			emb.Row(base+i, host)
			row := got[i*(cols+1) : (i+1)*(cols+1)]
			for j, v := range host {
				if math.Float32bits(row[j]) != math.Float32bits(v) {
					t.Fatalf("row %d value %d: device %v (%#x), host %v (%#x)", base+i, j, row[j], math.Float32bits(row[j]), v, math.Float32bits(v))
				}
			}
			if p := int(int32(math.Float32bits(row[cols]))); p != base+i {
				t.Fatalf("row %d position word %d", base+i, p)
			}
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInkling2DeviceEmbeddingMatchesHostRows(t *testing.T) {
	be, ok := compute.MetalInkling2Backend()
	if !ok {
		t.Skip("requires the Metal inkling2 device")
	}
	const rows, cols, lead = 1500, 1024, 4 * 777
	rowBytes := cols / 256 * 176
	page := os.Getpagesize()
	mapped, err := syscall.Mmap(-1, 0, (lead+rows*rowBytes+page-1)&^(page-1), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mapped)
	raw := mapped[lead : lead+rows*rowBytes]
	rng := rand.New(rand.NewSource(11))
	rng.Read(raw)
	for b := 0; b < len(raw)/176; b++ {
		for _, o := range []int{0, 2} {
			// any finite half scale, subnormals included
			for raw[b*176+o+1]&0x7c == 0x7c {
				raw[b*176+o+1] = byte(rng.Intn(256))
			}
		}
	}
	store, err := metalgemm.NewPackedGGUFStore(mapped)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	set := &inkling2MetalDeviceSet{be: be, store: store, data: mapped}
	host := inkling2QuantEmbedding{&inkling2QuantMatrix{name: "token_embd.weight", typ: TensorQ5_K, raw: raw, rows: rows, cols: cols, rowBytes: rowBytes}}
	emb, ok := set.embedding(host)
	if !ok {
		t.Fatal("a Q5_K table was not given a device view")
	}
	inkling2PipeEmbedCheck(t, be, emb, 0, rows)
	// a type the device cannot read stays host-only
	q6 := inkling2QuantEmbedding{&inkling2QuantMatrix{typ: TensorQ6_K, raw: raw, rows: 1, cols: 256, rowBytes: 210}}
	if _, ok := set.embedding(q6); ok {
		t.Fatal("a Q6_K table was given a device view the embed kernel cannot read")
	}
}

func inkling2PipeTestModel(t *testing.T) (*Inkling2LoopModel, compute.Backend, []int) {
	t.Helper()
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	// pipelining runs only on plain decode, which spec decode replaces by default
	t.Setenv("FAK_INKLING2_SPEC", "0")
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	table, ok := f.GGMLTokenizer()
	if !ok {
		t.Fatal("checkpoint has no embedded BPE tokenizer")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		t.Fatal(err)
	}
	control := func(s string) tokenizer.Segment { return tokenizer.Segment{Text: s, Trusted: true} }
	prompt, err := tok.EncodeSegments([]tokenizer.Segment{
		control("<|message_system|><|content_text|>"), {Text: "Thinking effort level: 0.9"}, control("<|end_message|>"),
		control("<|message_user|><|content_text|>"), {Text: "Write a short story about a lighthouse keeper."}, control("<|end_message|>"),
	})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 6 * time.Minute, Logf: t.Logf})
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lease.Release)
	m, be, err := OpenInkling2LoopMetalDevice(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m, be, prompt
}

// fak-test:runtime slow est=60s
func TestInkling2MetalPipelineEmbedsEveryRowAsHost(t *testing.T) {
	m, be, _ := inkling2PipeTestModel(t)
	emb, ok := m.Weights.TokEmbd.(inkling2DeviceEmbedding)
	if !ok {
		t.Fatalf("token table %T has no device view", m.Weights.TokEmbd)
	}
	inkling2PipeEmbedCheck(t, be, emb, 0, emb.m.rows)
}

// inkling2PipeScript drives one resident session, pipelined or not, through
// script and returns every token it produced and how many steps left a
// speculative forward in flight.
func inkling2PipeScript(t *testing.T, m *Inkling2LoopModel, be compute.Backend, pipe bool, script func(step func(int) int, r *inkling2ResidentSession) []int) (out []int, live int) {
	t.Helper()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	if pipe && s.pipe == nil {
		s.close()
		t.Fatal("decode pipelining is not enabled on this session")
	}
	if !pipe {
		s.pipe = nil
	}
	r := &inkling2ResidentSession{s: s}
	defer r.Close()
	step := func(tok int) int {
		next, err := r.Step(tok)
		if err != nil {
			t.Fatal(err)
		}
		if s.pipe != nil && s.pipe.live {
			live++
		}
		return next
	}
	return script(step, r), live
}

// fak-test:runtime slow est=240s
func TestInkling2MetalPipelineGreedyMatchesPlain(t *testing.T) {
	m, be, prompt := inkling2PipeTestModel(t)
	const n = 64
	greedy := func(step func(int) int, r *inkling2ResidentSession) []int {
		next, err := r.Prefill(context.Background(), prompt)
		if err != nil {
			t.Fatal(err)
		}
		var out []int
		for len(out) < n {
			out = append(out, next)
			next = step(next)
		}
		return out
	}
	plain, _ := inkling2PipeScript(t, m, be, false, greedy)
	start := time.Now()
	piped, live := inkling2PipeScript(t, m, be, true, greedy)
	t.Logf("pipelined: %d tokens in %s, %d steps left a forward in flight; ids %v", n, time.Since(start), live, piped)
	if !slices.Equal(piped, plain) {
		t.Fatalf("pipelined decode %v, plain %v", piped, plain)
	}
	if live < n-1 {
		t.Fatalf("only %d of %d steps were pipelined", live, n)
	}
}

// fak-test:runtime slow est=300s
func TestInkling2MetalPipelineForcedTokensRewind(t *testing.T) {
	m, be, prompt := inkling2PipeTestModel(t)
	vocab := m.HP.VocabUnpadded
	script := func(stepper func(int) int, r *inkling2ResidentSession) []int {
		var out []int
		step := func(tok int) int {
			next := stepper(tok)
			out = append(out, next)
			return next
		}
		greedy := func(next, k int) int {
			for range k {
				next = step(next)
			}
			return next
		}
		next, err := r.Prefill(context.Background(), prompt)
		if err != nil {
			t.Fatal(err)
		}
		next = greedy(next, 10)
		// a forced token discards the forward already running on the argmax
		next = step((next + 17) % vocab)
		next = greedy(next, 6)
		// two forced tokens in a row, then back to greedy
		step((next + 101) % vocab)
		next = step((next + 202) % vocab)
		next = greedy(next, 6)
		// checkpoint while a speculative forward is in flight, run on, restore
		if err := r.Checkpoint(16); err != nil {
			t.Fatal(err)
		}
		saved := next
		greedy(next, 8)
		if err := r.Restore(); err != nil {
			t.Fatal(err)
		}
		next = greedy(saved, 8)
		// append a prompt mid-stream, then start over
		if next, err = r.Prefill(context.Background(), prompt[:5]); err != nil {
			t.Fatal(err)
		}
		out = append(out, next)
		next = greedy(next, 6)
		if err := r.Reset(); err != nil {
			t.Fatal(err)
		}
		if next, err = r.Prefill(context.Background(), prompt); err != nil {
			t.Fatal(err)
		}
		out = append(out, next)
		greedy(next, 6)
		return out
	}
	plain, _ := inkling2PipeScript(t, m, be, false, script)
	piped, live := inkling2PipeScript(t, m, be, true, script)
	t.Logf("%d steps, %d left a forward in flight; ids %v", len(piped), live, piped)
	if !slices.Equal(piped, plain) {
		t.Fatalf("pipelined session %v, plain %v", piped, plain)
	}
	if live == 0 {
		t.Fatal("no step was pipelined")
	}
}

// fak-test:runtime slow est=60s
func TestInkling2MetalPipelineOffUnderDefaultSpec(t *testing.T) {
	m, be, prompt := inkling2PipeTestModel(t)
	t.Setenv("FAK_INKLING2_SPEC", "")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	s, err := newInkling2Resident(m, be)
	if err != nil {
		t.Fatal(err)
	}
	r := &inkling2ResidentSession{s: s}
	defer r.Close()
	if s.pipe != nil {
		t.Fatal("decode pipelining is enabled while spec decode is on by default")
	}
	next, err := r.Prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	for range 4 {
		if next, err = r.Step(next); err != nil {
			t.Fatal(err)
		}
	}
}
