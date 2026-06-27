package model

// lora.go — LoRA (Low-Rank Adaptation) adapters for the in-kernel model (#291, A-011).
//
// A fine-tuned variant rarely ships a whole new checkpoint; it ships a small LoRA
// adapter — two low-rank factor matrices per patched linear weight. For a base
// projection W of shape [Out, In], PEFT stores a down-projection A of shape
// [Rank, In] and an up-projection B of shape [Out, Rank]; the effective weight
// delta is
//
//	ΔW = (alpha/Rank) · B · A          (shape [Out, In])
//
// applied to the layer as y = W·x + ΔW·x. Because Rank ≪ min(In,Out), the adapter
// is a few MB next to a multi-GB base, and there are two honest ways to use it:
//
//   - MERGE: fold ΔW into the resident f32 weight once at load (MergeInto /
//     Model.MergeLoRA). Decode is then the unmodified matRows over W' — exactly
//     ZERO per-token overhead. This is the path the "within 5% overhead"
//     acceptance criterion is met by (0% < 5%); the cost is a per-adapter merged
//     copy of the weights, so it suits one pinned adapter.
//   - DYNAMIC: keep the factors separate and add the small delta each decode step
//     (Delta). No merged weight copy, so several adapters can be switched per
//     request, at a bounded compute overhead of Rank·(In+Out)/(In·Out) MACs
//     (OverheadFraction) — ~1.6% for a rank-16 adapter on a 2048×2048 projection.
//
// Multiple adapters that target the SAME weight COMPOSE additively
// (ΔW_total = Σ ΔW_k), the standard multi-LoRA semantics, so a LoRASet supports
// 2+ adapters simultaneously and can switch bundles on/off without reloading.
//
// The math here is the same matRows reduction the base forward uses, so a merged
// forward and the dynamic Delta agree to f32 reassociation tolerance
// (TestLoRAAdapterDeltaMatchesMerge), and a merged synthetic forward is proven to
// differ from the base while an empty/nil set leaves it byte-identical
// (TestMergeLoRA*). What remains for the full epic — wiring Delta into every
// production forward path (q/k/v/o/gate/up/down across the f32/Q8/Q4_K/AWQ/GPTQ
// and CUDA seams) and a measured end-to-end <5% wall-clock on a GPU node — is
// tracked on #291; this file is the load/merge/apply/compose/switch core those
// steps build on.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
)

// LoRAAdapter is one low-rank adapter for a single base weight of shape [Out, In].
// A is the down-projection [Rank, In] (row-major), B the up-projection [Out, Rank]
// (row-major) — the layout matRows expects, so applying either factor is a matRows
// call. Adapter names the bundle it came from (for dynamic switching); Target names
// the base weight it patches (e.g. "model.layers.0.self_attn.q_proj.weight").
type LoRAAdapter struct {
	Adapter string  // bundle name, e.g. "math-lora" — the unit of dynamic switching
	Target  string  // base weight patched, e.g. "model.layers.0.self_attn.q_proj.weight"
	In      int     // base input features
	Out     int     // base output features
	Rank    int     // low-rank inner dimension
	Alpha   float64 // PEFT lora_alpha; the applied scaling is Alpha/Rank
	A       []float32
	B       []float32
}

// Scaling is the PEFT scaling factor alpha/Rank that ΔW is multiplied by.
func (a *LoRAAdapter) Scaling() float32 { return float32(a.Alpha / float64(a.Rank)) }

// validate checks the factor shapes are internally consistent before the adapter
// is admitted to a set — a bad shape is named here, not as an out-of-range panic
// deep in a matvec.
func (a *LoRAAdapter) validate() error {
	if a.Rank <= 0 || a.In <= 0 || a.Out <= 0 {
		return fmt.Errorf("lora %q/%q: non-positive dim (rank=%d in=%d out=%d)", a.Adapter, a.Target, a.Rank, a.In, a.Out)
	}
	if len(a.A) != a.Rank*a.In {
		return fmt.Errorf("lora %q/%q: A has %d values, want rank*in=%d", a.Adapter, a.Target, len(a.A), a.Rank*a.In)
	}
	if len(a.B) != a.Out*a.Rank {
		return fmt.Errorf("lora %q/%q: B has %d values, want out*rank=%d", a.Adapter, a.Target, len(a.B), a.Out*a.Rank)
	}
	if a.Alpha == 0 {
		return fmt.Errorf("lora %q/%q: zero alpha", a.Adapter, a.Target)
	}
	return nil
}

// Delta returns the decode-time additive term ΔW·x = scaling·B·(A·x) for an input
// activation x of length In. It computes the two small matvecs (A·x is [Rank],
// B·that is [Out]) through matRows so the reduction matches the base path, and it
// never materializes the dense [Out,In] ΔW — this is the "apply at decode time"
// primitive. The returned slice is freshly allocated and owned by the caller.
func (a *LoRAAdapter) Delta(x []float32) []float32 {
	if len(x) != a.In {
		panic(fmt.Sprintf("lora %q/%q: Delta input len %d != in %d", a.Adapter, a.Target, len(x), a.In))
	}
	ax := matRows(a.A, x, a.Rank, a.In)    // [Rank]
	bax := matRows(a.B, ax, a.Out, a.Rank) // [Out]
	s := a.Scaling()
	for o := range bax {
		bax[o] *= s
	}
	return bax
}

// MergeInto folds this adapter's ΔW into a base weight w (row-major [Out, In]) in
// place: w[o*In+i] += scaling · Σ_r B[o*Rank+r]·A[r*In+i]. After the merge the
// decode path is the unmodified matRows over w, so the per-token overhead is
// exactly zero. matRows over the merged weight equals matRows over the base plus
// Delta(x) up to f32 summation order — the equivalence the tests pin.
func (a *LoRAAdapter) MergeInto(w []float32) error {
	if len(w) != a.Out*a.In {
		return fmt.Errorf("lora %q/%q: base weight has %d values, want out*in=%d", a.Adapter, a.Target, len(w), a.Out*a.In)
	}
	s := a.Scaling()
	for o := 0; o < a.Out; o++ {
		brow := a.B[o*a.Rank : o*a.Rank+a.Rank]
		wrow := w[o*a.In : o*a.In+a.In]
		for r := 0; r < a.Rank; r++ {
			c := s * brow[r]
			if c == 0 {
				continue
			}
			arow := a.A[r*a.In : r*a.In+a.In]
			for i := 0; i < a.In; i++ {
				wrow[i] += c * arow[i]
			}
		}
	}
	return nil
}

// OverheadFraction is the dynamic-apply compute overhead of this adapter relative
// to the base matmul: the two factor matvecs add Rank·(In+Out) MACs on top of the
// base In·Out, so the fraction is Rank·(In+Out)/(In·Out). It is the number a caller
// (or the "within 5% overhead" gate) checks to decide dynamic-vs-merge; the merge
// path is 0 by construction. Pure arithmetic, no allocation.
func (a *LoRAAdapter) OverheadFraction() float64 {
	return float64(a.Rank) * float64(a.In+a.Out) / (float64(a.In) * float64(a.Out))
}

// LoRASet is a switchable collection of loaded adapters keyed by the base weight
// they patch. More than one adapter may target the same weight — they compose
// additively — so the set is how "2+ adapters simultaneously" is expressed.
// Adapters belong to a named bundle (LoRAAdapter.Adapter); a bundle can be toggled
// with Activate/Deactivate for dynamic switching without reloading, and only
// active bundles contribute to a Delta or a merge. The zero value is not usable;
// build one with NewLoRASet.
type LoRASet struct {
	byTarget map[string][]*LoRAAdapter // base weight name -> adapters, in add order
	active   map[string]bool           // bundle name -> on (absent/false => off)
}

// NewLoRASet returns an empty, ready set.
func NewLoRASet() *LoRASet {
	return &LoRASet{byTarget: map[string][]*LoRAAdapter{}, active: map[string]bool{}}
}

// Add validates an adapter and registers it, activating its bundle by default so a
// freshly loaded adapter is live. Adding a second adapter to the same target stacks
// it (composition); adding another adapter of an already-active bundle keeps it on.
func (s *LoRASet) Add(a *LoRAAdapter) error {
	if a == nil {
		return fmt.Errorf("lora: nil adapter")
	}
	if err := a.validate(); err != nil {
		return err
	}
	s.byTarget[a.Target] = append(s.byTarget[a.Target], a)
	if _, seen := s.active[a.Adapter]; !seen {
		s.active[a.Adapter] = true
	}
	return nil
}

// Activate / Deactivate toggle a bundle. Deactivating a bundle removes its
// contribution from every Delta and merge without dropping the loaded factors, so
// it can be switched back on. Toggling an unknown bundle is a no-op that records
// the desired state (a later Add of that bundle honors it).
func (s *LoRASet) Activate(bundle string)   { s.active[bundle] = true }
func (s *LoRASet) Deactivate(bundle string) { s.active[bundle] = false }

// IsActive reports whether a bundle currently contributes.
func (s *LoRASet) IsActive(bundle string) bool { return s.active[bundle] }

// Bundles returns the loaded bundle names, sorted.
func (s *LoRASet) Bundles() []string {
	seen := map[string]bool{}
	for _, as := range s.byTarget {
		for _, a := range as {
			seen[a.Adapter] = true
		}
	}
	out := make([]string, 0, len(seen))
	for b := range seen {
		out = append(out, b)
	}
	sort.Strings(out)
	return out
}

// activeFor returns the active adapters patching one target, in add order.
func (s *LoRASet) activeFor(target string) []*LoRAAdapter {
	var out []*LoRAAdapter
	for _, a := range s.byTarget[target] {
		if s.active[a.Adapter] {
			out = append(out, a)
		}
	}
	return out
}

// Targets returns the base weight names that have at least one ACTIVE adapter,
// sorted — the set of weights a merge or a dynamic apply must touch.
func (s *LoRASet) Targets() []string {
	var out []string
	for target := range s.byTarget {
		if len(s.activeFor(target)) > 0 {
			out = append(out, target)
		}
	}
	sort.Strings(out)
	return out
}

// Delta returns the summed decode-time term Σ_k ΔW_k·x over every active adapter
// patching target, or nil when none is active — so the no-adapter path costs one
// map lookup and allocates nothing. This is the dynamic multi-adapter apply.
func (s *LoRASet) Delta(target string, x []float32) []float32 {
	active := s.activeFor(target)
	if len(active) == 0 {
		return nil
	}
	sum := active[0].Delta(x)
	for _, a := range active[1:] {
		d := a.Delta(x)
		for o := range sum {
			sum[o] += d[o]
		}
	}
	return sum
}

// MergeInto folds every active adapter for target into base weight w in place.
func (s *LoRASet) MergeInto(target string, w []float32) error {
	for _, a := range s.activeFor(target) {
		if err := a.MergeInto(w); err != nil {
			return err
		}
	}
	return nil
}

// MergeLoRA folds a set's active adapters into this model's resident f32 weights
// in place, so the unmodified forward/decode path then reflects the adapter at zero
// per-token overhead. It reads each target's [Out,In] geometry from the manifest
// and mutates the zero-copy f32 view, validating that every adapter's dims match
// the base weight. A nil/empty set is a no-op, so the proven forward is
// byte-identical unless an adapter is explicitly merged.
//
// It operates on the f32-resident weights (the path NewSynthetic, Load, and the
// oracle use). A target that was quantized away from f32 residency (Q8/Q4_K/AWQ/…)
// is reported, not silently skipped — merging into a quantized store is a later
// step of #291.
func (m *Model) MergeLoRA(set *LoRASet) error {
	if set == nil {
		return nil
	}
	for _, target := range set.Targets() {
		meta, ok := m.manifest[target]
		if !ok {
			return fmt.Errorf("lora: base weight %q is not f32-resident; merge needs the f32 weight (quantized-store merge is not yet supported)", target)
		}
		if len(meta.Shape) != 2 {
			return fmt.Errorf("lora: base weight %q has shape %v, want 2-D [out,in]", target, meta.Shape)
		}
		out, in := meta.Shape[0], meta.Shape[1]
		for _, a := range set.activeFor(target) {
			if a.Out != out || a.In != in {
				return fmt.Errorf("lora %q: adapter dims [out=%d,in=%d] != base %q [out=%d,in=%d]", a.Adapter, a.Out, a.In, target, out, in)
			}
		}
		w := m.tensor(target) // zero-copy mutable view into m.raw
		if err := set.MergeInto(target, w); err != nil {
			return err
		}
	}
	return nil
}

// LoRAConfig is the subset of PEFT's adapter_config.json this loader reads.
type LoRAConfig struct {
	R             int      `json:"r"`
	Alpha         float64  `json:"lora_alpha"`
	TargetModules []string `json:"target_modules"`
	PeftType      string   `json:"peft_type"`
}

// LoadLoRAAdapter loads one PEFT adapter directory (adapter_config.json +
// adapter_model.safetensors) into a fresh, active LoRASet. The bundle is named for
// the directory. This is the "load single LoRA adapter" path; load more adapters
// into the same set with AddLoRAAdapter to compose/switch them.
func LoadLoRAAdapter(dir string) (*LoRASet, error) {
	set := NewLoRASet()
	if err := AddLoRAAdapter(set, dir); err != nil {
		return nil, err
	}
	return set, nil
}

// AddLoRAAdapter loads a PEFT adapter directory into an existing set as one named
// bundle, so several adapters can be composed and switched. The bundle name is the
// directory's base name.
func AddLoRAAdapter(set *LoRASet, dir string) error {
	var cfg LoRAConfig
	if err := readJSON(filepath.Join(dir, "adapter_config.json"), &cfg); err != nil {
		return fmt.Errorf("lora: adapter_config.json: %w", err)
	}
	if cfg.R <= 0 {
		return fmt.Errorf("lora: adapter_config.json has non-positive r=%d", cfg.R)
	}
	if cfg.PeftType != "" && !strings.EqualFold(cfg.PeftType, "LORA") {
		return fmt.Errorf("lora: unsupported peft_type %q (only LORA)", cfg.PeftType)
	}
	tensors, err := readLoRASafetensors(filepath.Join(dir, "adapter_model.safetensors"))
	if err != nil {
		return err
	}
	bundle := filepath.Base(filepath.Clean(dir))
	adapters, err := parseLoRATensors(bundle, cfg, tensors)
	if err != nil {
		return err
	}
	for _, a := range adapters {
		if err := set.Add(a); err != nil {
			return err
		}
	}
	return nil
}

// readLoRASafetensors decodes every f32/bf16/f16 tensor of an adapter safetensors
// file into loader-neutral NamedTensorF32 payloads, reusing the model's safetensors
// reader and decoder. Adapter files are small (a few MB), so a full decode is fine.
func readLoRASafetensors(path string) ([]NamedTensorF32, error) {
	sf, err := openSafetensorsFile(path)
	if err != nil {
		return nil, fmt.Errorf("lora: %w", err)
	}
	defer func() { _ = sf.Close() }()

	names := safetensorsTensorNames(sf.hdr)
	out := make([]NamedTensorF32, 0, len(names))
	for _, name := range names {
		var e stEntry
		if err := json.Unmarshal(sf.hdr[name], &e); err != nil {
			return nil, fmt.Errorf("lora: tensor %q header: %w", name, err)
		}
		raw, err := sf.tensorBytes(e)
		if err != nil {
			return nil, fmt.Errorf("lora: tensor %q bytes: %w", name, err)
		}
		f32b, err := decodeSafetensorF32(name, e, raw)
		if err != nil {
			return nil, fmt.Errorf("lora: %w", err)
		}
		out = append(out, NamedTensorF32{Name: name, Shape: e.Shape, Data: f32sFromLE(f32b)})
	}
	return out, nil
}

// f32sFromLE copies little-endian f32 bytes into a freshly owned []float32 (the
// source may be a transient or mmap-backed buffer).
func f32sFromLE(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// parseLoRATensors groups a bundle's lora_A/lora_B tensors by the base weight they
// patch and pairs them into adapters. PEFT names a factor
// "[base_model.model.]<base-weight-path>.lora_{A,B}.weight"; the target base weight
// is that path with ".weight" re-appended. Rank/In/Out are read from the factor
// shapes (A is [Rank,In], B is [Out,Rank]) and cross-checked, so a shape that does
// not pair is named rather than silently dropped.
func parseLoRATensors(bundle string, cfg LoRAConfig, tensors []NamedTensorF32) ([]*LoRAAdapter, error) {
	type pair struct{ a, b *NamedTensorF32 }
	pairs := map[string]*pair{}
	order := []string{}
	get := func(target string) *pair {
		p, ok := pairs[target]
		if !ok {
			p = &pair{}
			pairs[target] = p
			order = append(order, target)
		}
		return p
	}
	for i := range tensors {
		t := &tensors[i]
		name := strings.TrimPrefix(t.Name, "base_model.model.")
		switch {
		case strings.HasSuffix(name, ".lora_A.weight"):
			get(strings.TrimSuffix(name, ".lora_A.weight") + ".weight").a = t
		case strings.HasSuffix(name, ".lora_B.weight"):
			get(strings.TrimSuffix(name, ".lora_B.weight") + ".weight").b = t
		default:
			// Non-factor tensors (e.g. modules_to_save copies) are not LoRA factors.
		}
	}
	if len(order) == 0 {
		return nil, fmt.Errorf("lora %q: no lora_A/lora_B factor tensors found", bundle)
	}
	alpha := cfg.Alpha
	if alpha == 0 {
		alpha = float64(cfg.R) // PEFT default scaling 1.0 when alpha is unset
	}
	out := make([]*LoRAAdapter, 0, len(order))
	for _, target := range order {
		p := pairs[target]
		if p.a == nil || p.b == nil {
			return nil, fmt.Errorf("lora %q: target %q missing %s factor", bundle, target, missingFactor(p.a, p.b))
		}
		if len(p.a.Shape) != 2 || len(p.b.Shape) != 2 {
			return nil, fmt.Errorf("lora %q: target %q factors must be 2-D (A=%v B=%v)", bundle, target, p.a.Shape, p.b.Shape)
		}
		rank, in := p.a.Shape[0], p.a.Shape[1]
		out2, rankB := p.b.Shape[0], p.b.Shape[1]
		if rank != rankB {
			return nil, fmt.Errorf("lora %q: target %q rank mismatch (A rank=%d, B rank=%d)", bundle, target, rank, rankB)
		}
		a := &LoRAAdapter{
			Adapter: bundle, Target: target,
			In: in, Out: out2, Rank: rank, Alpha: alpha,
			A: p.a.Data, B: p.b.Data,
		}
		if err := a.validate(); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Target < out[j].Target })
	return out, nil
}

func missingFactor(a, b *NamedTensorF32) string {
	if a == nil {
		return "lora_A"
	}
	if b == nil {
		return "lora_B"
	}
	return "?"
}
