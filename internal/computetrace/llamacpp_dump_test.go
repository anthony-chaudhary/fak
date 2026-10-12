package computetrace

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

type dumpTensor struct {
	name   string
	ne     [4]int64
	dumped bool
	typ    string
}

// writeDump writes a llama.cpp-style dump where element k of each tensor holds
// float32(seq*10000 + k), so every sampled value names its own flat offset.
func writeDump(t *testing.T, tokens []int, tensors []dumpTensor) string {
	t.Helper()
	dir := t.TempDir()
	tok, _ := json.Marshal(map[string]any{"prompt": "hi", "tokens": tokens})
	if err := os.WriteFile(filepath.Join(dir, "tokens.json"), tok, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, "index.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for seq, d := range tensors {
		file := fmt.Sprintf("%s.%d.bin", d.name, seq)
		n := d.ne[0] * d.ne[1] * d.ne[2] * d.ne[3]
		buf := make([]byte, 4*n)
		for k := int64(0); k < n; k++ {
			binary.LittleEndian.PutUint32(buf[4*k:], math.Float32bits(float32(int64(seq)*10000+k)))
		}
		if err := os.WriteFile(filepath.Join(dir, file), buf, 0o644); err != nil {
			t.Fatal(err)
		}
		typ := d.typ
		if typ == "" {
			typ = "f32"
		}
		line, _ := json.Marshal(map[string]any{
			"seq": seq, "name": d.name, "op": "MUL_MAT", "type": typ, "ne": d.ne,
			"dumped": d.dumped, "file": file, "sum": 0, "sumsq": 0, "maxabs": 0, "nonfinite": 0,
		})
		if _, err := f.Write(append(line, '\n')); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestActivationLlamaCppDump(t *testing.T) {
	const H = 16
	dir := writeDump(t, []int{7, 8}, []dumpTensor{
		{name: "qr-0", ne: [4]int64{32, 2, 1, 1}, dumped: true},
		{name: "kv-0", ne: [4]int64{512, 1, 2, 1}, dumped: true},
		{name: "attn_norm-0", ne: [4]int64{H, 2, 1, 1}, dumped: true},
		{name: "attn_out-0", ne: [4]int64{H, 2, 1, 1}, dumped: false},
		{name: "ffn_out-0", ne: [4]int64{H, 2, 1, 1}, dumped: true},
		{name: "l_last-0", ne: [4]int64{H, 4, 2, 1}, dumped: true},
		{name: "l_last-0", ne: [4]int64{H, 4, 1, 1}, dumped: true},
	})
	a, err := ConvertLlamaCppDump(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(a.InputTokens, []int{7, 8}) || a.Schema != Schema {
		t.Fatalf("tokens=%v schema=%q", a.InputTokens, a.Schema)
	}
	type key struct {
		stage string
		token int
	}
	got := map[key]Event{}
	for _, e := range a.Events {
		got[key{e.Phase, e.Token}] = e
	}
	// qr x2 + kv x2 + ffn_out x2 + l_last x3; attn_norm unknown, attn_out not dumped.
	if len(a.Events) != 9 || a.RetainedSampleValues != 72 {
		t.Fatalf("events=%d retained=%d, want 9/72", len(a.Events), a.RetainedSampleValues)
	}
	if _, ok := got[key{"attention_projected", 0}]; ok {
		t.Fatal("an undumped tensor produced an event")
	}

	kv := got[key{"kv_latent", 1}]
	if !slices.Equal(kv.SampleIndices, ActivationSampleIndices(512, 8)) || !slices.Equal(kv.ExcludedSampleIndices, []int{448}) {
		t.Fatalf("kv indices=%v excluded=%v", kv.SampleIndices, kv.ExcludedSampleIndices)
	}
	if kv.SampleValues[1] != float32(1*10000+512+64) || kv.Shapes[0][0] != 512 {
		t.Fatalf("kv token 1 sample[1]=%v shape=%v", kv.SampleValues[1], kv.Shapes)
	}
	if q := got[key{"q_latent", 0}]; len(q.ExcludedSampleIndices) != 0 || q.Shapes[0][0] != 32 {
		t.Fatalf("q_latent excluded=%v shape=%v", q.ExcludedSampleIndices, q.Shapes)
	}

	// l_last [H, hc=4, tokens] is [t][hc][i]: flat index j of token t is t*4H+j.
	res := got[key{"residual_streams", 1}]
	wantIdx := ActivationSampleIndices(4*H, 8)
	if !slices.Equal(res.SampleIndices, wantIdx) || res.Shapes[0][0] != 4*H || res.Layer != 0 {
		t.Fatalf("residual indices=%v shape=%v layer=%d", res.SampleIndices, res.Shapes, res.Layer)
	}
	for i, j := range wantIdx {
		if want := float32(5*10000 + 4*H + j); res.SampleValues[i] != want {
			t.Fatalf("residual token 1 sample %d (j=%d) = %v, want %v", i, j, res.SampleValues[i], want)
		}
	}
	// A repeated name is the next ubatch: its token numbering continues.
	if next, ok := got[key{"residual_streams", 2}]; !ok || next.SampleValues[0] != float32(6*10000) {
		t.Fatalf("second l_last-0 occurrence token 2 missing or wrong: %+v", next)
	}

	t.Run("adopted reference profiles against an equal fak-shaped trace", func(t *testing.T) {
		fak := Artifact{Schema: Schema, InputTokens: []int{7, 8}}
		for _, e := range a.Events {
			e.InputDigest, e.WeightDigest = Digest("fak-input"), Digest("fak-weights")
			e.ExcludedSampleIndices = nil
			fak.Events = append(fak.Events, e)
		}
		fak.Events[3].SampleValues = append([]float32(nil), fak.Events[3].SampleValues...)
		fak.Events[3].SampleValues[7] += 1 // the RoPE-tail kv sample: excluded, not a divergence
		ref := a
		if p := ActivationErrorProfile(ref, fak, ActivationTolerance{}); p.Verdict != TraceIncomparable {
			t.Fatalf("unadopted profile verdict=%s, want INCOMPARABLE", p.Verdict)
		}
		ref.Events = slices.Clone(a.Events)
		if err := AdoptActivationDigests(&ref, fak); err != nil {
			t.Fatal(err)
		}
		if p := ActivationErrorProfile(ref, fak, ActivationTolerance{}); p.Verdict != TraceEqual {
			t.Fatalf("adopted profile verdict=%s reason=%s rows=%+v", p.Verdict, p.Reason, p.Rows)
		}
	})

	t.Run("malformed dumps refuse", func(t *testing.T) {
		bad := writeDump(t, []int{1}, []dumpTensor{{name: "qr-0", ne: [4]int64{8, 1, 1, 1}, dumped: true, typ: "f16"}})
		if _, err := ConvertLlamaCppDump(bad); !errors.Is(err, ErrLlamaCppDump) {
			t.Fatalf("f16 err=%v", err)
		}
		short := writeDump(t, []int{1}, []dumpTensor{{name: "qr-0", ne: [4]int64{8, 1, 1, 1}, dumped: true}})
		if err := os.WriteFile(filepath.Join(short, "qr-0.0.bin"), make([]byte, 8), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ConvertLlamaCppDump(short); !errors.Is(err, ErrLlamaCppDump) {
			t.Fatalf("short file err=%v", err)
		}
	})
}
