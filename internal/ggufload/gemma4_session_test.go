package ggufload

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestGemma4ResidentQuantSessionPrefillsOnDedicatedForward is the #5495 contract: a
// model.Session over a RESIDENT-QUANT gemma4 checkpoint (every projection stored Q8_0,
// no f32 copy) must prefill and decode through the DEDICATED gemma4 forward instead of
// refusing with *model.ResidentQuantUnsupportedError.
//
// Before the wiring, Session.Prefill fell through to the generic uniform-geometry
// resident-quant lane (tokenHiddenQ -> blockStep), whose qk-norm band passes the SCALAR
// cfg.HeadDim to applyQKNormCfg — so a per-layer-head_dim checkpoint was fail-closed to
// the f32 forward, i.e. unservable at any real size. The assertions below are what make
// "wired" checkable rather than asserted:
//
//  1. Residency premise — the projections really are Q8_0 (HasQ8), so a green here cannot
//     be a secretly-f32 session.
//  2. Path identity — the session's logits are BIT-EXACT with the dedicated cacheless
//     gemma4 forward (Model.Forward) over the same ids on the same resident-quant model.
//     The generic block path cannot produce those numbers; it panics on this geometry.
//  3. History bookkeeping — Prefill(ids[:k]) + Step(...) reproduces Prefill(ids) exactly.
//  4. Correctness witness — logit cosine of the resident-quant forward against the f32
//     reference load of the SAME checkpoint (the bar the #4274 refusal comment set), with
//     the lossy-quant premise pinned (the two must not be the identical f32 numbers).
func TestGemma4ResidentQuantSessionPrefillsOnDedicatedForward(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma4.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	qm, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}
	// (1) Residency premise: the attention/MLP projections are resident Q8_0, not f32.
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.3.self_attn.k_proj.weight",
		"model.layers.3.mlp.down_proj.weight",
	} {
		if !qm.HasQ8(name) {
			t.Fatalf("premise broken: %s is not resident Q8_0, so this is not the resident-quant path", name)
		}
	}
	if !qm.Cfg.QKNorm || len(qm.Cfg.HeadDimPerLayer) == 0 {
		t.Fatalf("premise broken: fixture must carry q/k norms and a per-layer head_dim, got QKNorm=%v HeadDimPerLayer=%v",
			qm.Cfg.QKNorm, qm.Cfg.HeadDimPerLayer)
	}

	// Seven positions over a fixture whose sliding window is 2: the last position's sliding
	// layers must DROP most of the prefix while its global layer keeps all of it, so the two
	// regimes are both live and disagree about range — the geometry the generic block path
	// cannot express.
	ids := []int{0, 1, 2, 3, 4, 0, 1}

	// (2) Path identity: the resident-quant Session must reproduce the dedicated cacheless
	// gemma4 forward BIT-EXACTLY. Run the cacheless reference first so a panic in the
	// session below cannot be blamed on the reference.
	want := qm.Forward(ids).Logits[len(ids)-1]

	s := qm.NewSession()
	s.Quant = true
	got := s.Prefill(ids)
	if len(got) != len(want) {
		t.Fatalf("Prefill logits len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("resident-quant Session logit[%d] = %v, dedicated gemma4 Forward = %v (the session is not on the gemma4 path)", i, got[i], want[i])
		}
	}

	// (3) History bookkeeping: a split ingest must land on the same distribution as one
	// Prefill of the whole prompt — the property a recompute bridge can only satisfy by
	// carrying the full token history forward. The last token is fed through Step, so this
	// covers the decode entry as well as the prefill one.
	last := len(ids) - 1
	split := qm.NewSession()
	split.Quant = true
	split.Prefill(ids[:last])
	stepped := split.Step(ids[last])
	for i := range want {
		if stepped[i] != want[i] {
			t.Fatalf("Prefill(ids[:%d])+Step(ids[%d]) logit[%d] = %v, want %v (history bookkeeping is wrong)", last, last, i, stepped[i], want[i])
		}
	}
	// PrefillNoLogits must advance the same history (it is the KV-advancing twin).
	quiet := qm.NewSession()
	quiet.Quant = true
	quiet.PrefillNoLogits(ids[:last])
	quietStep := quiet.Step(ids[last])
	for i := range want {
		if quietStep[i] != want[i] {
			t.Fatalf("PrefillNoLogits(ids[:%d])+Step(ids[%d]) logit[%d] = %v, want %v", last, last, i, quietStep[i], want[i])
		}
	}

	// (3b) The equalities above would ALSO hold for a bridge that ignored its history and
	// scored only the newest token. Pin context sensitivity: the same final token after a
	// different prefix must produce a different distribution.
	blind := qm.NewSession()
	blind.Quant = true
	short := blind.Prefill(ids[last:])
	differs := false
	for i := range want {
		if math.IsInf(float64(want[i]), -1) {
			continue
		}
		if short[i] != want[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("context-blind: Prefill(last token alone) == Prefill(whole prompt); the bridge is not attending its history")
	}

	// (4) Correctness witness: cosine vs the f32 reference load of the same checkpoint.
	fm, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel (f32 reference): %v", err)
	}
	ref := fm.Forward(ids).Logits[len(ids)-1]
	cos, n, maxAbsDelta := finiteCosine(got, ref)
	if n < 2 {
		t.Fatalf("witness needs at least 2 finite logits, got %d", n)
	}
	t.Logf("gemma4 resident-Q8_0 session vs f32 reference: logit cosine = %.12f over %d finite logits, max|delta| = %g",
		cos, n, maxAbsDelta)
	if cos < 0.999 {
		t.Fatalf("resident-quant logit cosine vs f32 reference = %.12f over %d logits, want >= 0.999", cos, n)
	}
	// The decision-relevant property a cosine over few logits can hide: greedy decode must
	// pick the same token on both loads.
	if a, b := finiteArgmax(got), finiteArgmax(ref); a != b {
		t.Fatalf("resident-quant argmax = %d, f32 reference argmax = %d (greedy decode diverges)", a, b)
	}
	// Lossy-quant premise: if the "quantized" model were secretly the f32 one, the cosine
	// above would be 1 by construction and would witness nothing.
	if maxAbsDelta == 0 {
		t.Fatal("resident-quant logits are bit-identical to the f32 reference: the cosine witness is vacuous")
	}

	// (5) Format independence. The dispatch is placed ahead of EVERY generic resident-quant
	// lane, not just the Q8_0 one, so a session flagged for any resident format must land on
	// the same dedicated forward. Note what this does and does not witness: the fixture GGUF
	// holds f32 tensors, so LoadModelQ4K's raw-Q4_K admission (which needs Q4_K source blocks)
	// finds nothing eligible and the weights land in the resident Q8_0 store. What is pinned
	// here is therefore the SESSION FLAG the #4274 refusal fired on — Q4K, and its Q4/GPTQ
	// siblings — no longer diverting a gemma4 checkpoint into the uniform-geometry band. The
	// store behind the flag is name-dispatched by residentMatRows, so a real q4_k_m checkpoint
	// reaches this same forward through its q4kw entry.
	q4km, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}
	q4kmWant := q4km.Forward(ids).Logits[len(ids)-1]
	for flag, set := range map[string]func(*model.Session){
		"Q4K":  func(s *model.Session) { s.Q4K = true },
		"Q4":   func(s *model.Session) { s.Q4 = true },
		"GPTQ": func(s *model.Session) { s.GPTQ = true },
	} {
		fs := q4km.NewSession()
		set(fs)
		out := fs.Prefill(ids)
		if len(out) != len(q4kmWant) {
			t.Fatalf("%s session Prefill logits len = %d, want %d", flag, len(out), len(q4kmWant))
		}
		for i := range q4kmWant {
			if out[i] != q4kmWant[i] {
				t.Fatalf("%s session logit[%d] = %v, dedicated gemma4 Forward = %v (this resident lane is not routed)",
					flag, i, out[i], q4kmWant[i])
			}
		}
	}
}

// TestGemma4StillUnsupportedSessionModesRefuseByName pins the other half of the #5495
// contract: shrinking the refusal's domain must not delete it. A gemma4 session mode the
// dedicated forward is NOT wired for (a device/Metal/dynamic-precision session) must still
// fail closed by name rather than silently running the wrong arithmetic.
func TestGemma4StillUnsupportedSessionModesRefuseByName(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma4.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	qm, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}
	s := qm.NewSession()
	s.Quant = true
	s.Metal = true // a session mode the gemma4 bridge does not wire

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("a Metal gemma4 session must fail closed, got no panic")
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("panic value = %T (%v), want an error", r, r)
		}
		var rqe *model.ResidentQuantUnsupportedError
		if !errors.As(err, &rqe) {
			t.Fatalf("panic error = %T (%v), want *model.ResidentQuantUnsupportedError", err, err)
		}
		if rqe.Arch == "" || rqe.Format == "" {
			t.Fatalf("refusal must name the arch and format, got %+v", rqe)
		}
	}()
	s.Prefill([]int{0, 1, 2})
}

// finiteCosine returns the cosine similarity of a and b over the indices where BOTH are
// finite (gemma4 forces its suppressed placeholder tokens to -inf, which no similarity
// measure can consume), the number of indices compared, and the largest absolute
// element-wise difference over those indices.
func finiteCosine(a, b []float32) (cos float64, n int, maxAbsDelta float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			continue
		}
		dot += x * y
		na += x * x
		nb += y * y
		if d := math.Abs(x - y); d > maxAbsDelta {
			maxAbsDelta = d
		}
		n++
	}
	if na == 0 || nb == 0 {
		return 0, n, maxAbsDelta
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), n, maxAbsDelta
}

// finiteArgmax is the greedy pick over the finite logits (the suppressed -inf placeholders
// can never be selected), returning -1 when none are finite.
func finiteArgmax(a []float32) int {
	best := -1
	for i, v := range a {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		if best < 0 || v > a[best] {
			best = i
		}
	}
	return best
}
