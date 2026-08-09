package model

// vision.go — the vision-tower substrate for the VLM epic (#4029).
//
// A VLM's image encoder (llama.cpp's CLIP tower: patch-embed → ViT blocks →
// projector) is a SEPARATE weight stack from the text decoder. Its tensors ship
// either in a companion mmproj GGUF (v.* / mm.*, loaded via ggufload.OpenMMProj,
// #4028) or inline in an HF safetensors checkpoint (model.visual.*). Both the text
// forward and every existing loader DROP these tensors today (materializeQwen35Tensors
// here; isGLMMoeDsaVisionTensor in ggufload). This file gives them a home: a
// VisionTower held on the Model in a dedicated field (mirroring Model.MLA), so the
// decoder path is byte-for-byte unchanged (Vision is nil for every text-only model)
// while the encoder slice (#4030) has a resident, resolvable set of vision weights to
// read. Nothing is retained unless RetainVision is set — the unchanged default drops
// the vision tower exactly as before.

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
)

// RetainVision is the "an mmproj / vision source is present" gate, mirroring
// RetainMTP (safetensors.go). When false (the default) the loaders drop the vision
// tower exactly as before, so a text-only load is byte-for-byte unchanged. When
// true, the vision tensors (model.visual.* in safetensors, v.*/mm.* in GGUF) are
// retained and segregated into Model.Vision instead of being discarded. It is a
// package var — not a Config field — because it is a load-time operator choice
// (the --mmproj flag, wired in #4032), exactly like RetainMTP.
var RetainVision bool

// VisionConfig is the CLIP/ViT image-tower geometry the encoder (#4030) needs,
// parsed from an mmproj GGUF's clip.* metadata (or an HF vision_config). Only the
// fields the tower's forward reads live here; the projector output dim must equal
// the decoder hidden size so image vectors splice into the prompt sequence. Zero
// fields mean "unknown from this source" — the encoder slice pins any it requires.
type VisionConfig struct {
	HiddenSize    int     // ViT hidden width (clip.vision.embedding_length)
	NumLayers     int     // ViT block count (clip.vision.block_count); drives the resolver's per-layer template
	NumHeads      int     // ViT attention heads (clip.vision.attention.head_count)
	FFNLength     int     // ViT MLP inner size (clip.vision.feed_forward_length)
	PatchSize     int     // conv patch edge in pixels (clip.vision.patch_size)
	ImageSize     int     // native square input edge (clip.vision.image_size)
	ProjOutDim    int     // projector output width = decoder hidden size (mm.* out features)
	ProjectorType string  // llama.cpp clip.projector_type (e.g. "mlp", "qwen2vl_merger")
	LNEps         float32 // ViT layernorm epsilon (clip.vision.attention.layer_norm_epsilon)
	MergeSize     int     // spatial patch-merge factor for Qwen2/2.5-VL (1 when absent)
}

// VisionTower is a resident vision weight stack: its parsed geometry plus the same
// zero-copy (manifest, raw f32 LE) representation Model uses, so a tensor view is a
// reinterpretation of the blob, not a copy. It is owned by Model.Vision and read
// only by the vision encoder (#4030); the text decoder never touches it.
type VisionTower struct {
	Cfg      VisionConfig
	manifest map[string]tensorMeta
	raw      []byte
}

// NewVisionTower packs decoded f32 vision tensors into the (manifest, raw) layout,
// exactly as NewFromF32Tensors does for the decoder. It is the single construction
// point both source paths funnel through: ggufload.OpenMMProj's v.*/mm.* reader
// (#4028) for the GGUF companion file, and the safetensors model.visual.* extractor
// for inline checkpoints. Names are stored verbatim (the source scheme), so the
// vision resolver's aliases map them onto the canonical v.* set.
func NewVisionTower(cfg VisionConfig, tensors []NamedTensorF32) (*VisionTower, error) {
	man := make(map[string]tensorMeta, len(tensors))
	var raw []byte
	off := 0
	for _, t := range tensors {
		if t.Name == "" {
			return nil, fmt.Errorf("model: vision tower: empty tensor name")
		}
		if _, ok := man[t.Name]; ok {
			return nil, fmt.Errorf("model: vision tower: duplicate tensor %s", t.Name)
		}
		elems, err := tensorShapeElems(t.Name, t.Shape)
		if err != nil {
			return nil, err
		}
		if elems != len(t.Data) {
			return nil, fmt.Errorf("model: vision tensor %s has %d values, shape wants %d", t.Name, len(t.Data), elems)
		}
		nbytes := len(t.Data) * 4
		if nbytes/4 != len(t.Data) || off > math.MaxInt-nbytes {
			return nil, fmt.Errorf("model: vision tensor %s byte size overflows int", t.Name)
		}
		start := len(raw)
		raw = append(raw, make([]byte, nbytes)...)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint32(raw[start+i*4:], math.Float32bits(v))
		}
		shape := append([]int(nil), t.Shape...)
		man[t.Name] = tensorMeta{Dtype: "f32", Shape: shape, Offset: off, Nbytes: nbytes}
		off += nbytes
	}
	return &VisionTower{Cfg: cfg, manifest: man, raw: raw}, nil
}

// Config returns the tower's parsed geometry.
func (t *VisionTower) Config() VisionConfig { return t.Cfg }

// Bytes is the resident size of the vision weights (the f32 blob length) — the
// figure the load estimator adds when the tower is retained (#4029 byte-accounting).
func (t *VisionTower) Bytes() int64 { return int64(len(t.raw)) }

// TensorNames returns the tower's tensor names in sorted order (deterministic for
// tests and byte-accounting; the map is otherwise unordered).
func (t *VisionTower) TensorNames() []string {
	names := make([]string, 0, len(t.manifest))
	for n := range t.manifest {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// has / tensor / tensorOptional mirror the Model accessors so the encoder (#4030)
// reads vision weights the same zero-copy way the decoder reads its own.
func (t *VisionTower) has(name string) bool {
	_, ok := t.manifest[name]
	return ok
}

func (t *VisionTower) tensor(name string) []float32 {
	return manifestTensor(t.manifest, t.raw, "vision tower ", name)
}

func (t *VisionTower) tensorOptional(name string) []float32 {
	if t.has(name) {
		return t.tensor(name)
	}
	return nil
}

// AttachVisionTower joins a SEPARATELY-LOADED image tower to this text model — the
// companion-mmproj half of a VLM load (#4875).
//
// The inline path already worked: a safetensors checkpoint carries model.visual.* beside
// the decoder, so newModel segregates it into Model.Vision during the load
// (extractQwen35VisionTower, below). The CANONICAL llama.cpp VLM deployment does not look
// like that — it ships TWO files, a text GGUF plus a companion mmproj GGUF, and the mmproj
// half is read by ggufload.OpenMMProj and turned into a *VisionTower by
// WeightSource.VisionTower(). Nothing joined that tower to a Model. Model.Vision had
// exactly one assignment in the tree (weights.go, the inline path), so a GGUF VLM could
// only ever be loaded as two halves that could never meet, and EncodeImagePrompt reported
// "no vision tower retained" no matter what mmproj the operator supplied. This is that
// join, and it is why the seam needs no RetainVision: that flag gates whether a DECODER
// load keeps vision tensors it would otherwise drop, whereas an mmproj is opened
// explicitly and has nothing to drop.
//
// Admission is the encoder's OWN construction rather than a second opinion: the tower is
// accepted only if newVisionEncoder resolves it against this decoder's hidden size, so
// anything AttachVisionTower admits is something NewVisionEncoder can build. That moves a
// mismatched mmproj — wrong projector width, missing required tensor, incomplete clip.*
// geometry — from a failure at the first image request to a refusal at load, while the
// operator can still act on it, and makes it impossible for attach to admit a tower the
// encoder would then reject. The trial encoder is discarded; it is a validation, not a
// cache, so NewVisionEncoder stays the single construction point.
//
// Attaching is one-shot. An already-attached model is refused rather than silently
// re-pointed, so a second mmproj cannot quietly swap the tower a running server is already
// encoding against. The tower is shared, not copied, matching the zero-copy weight
// contract the rest of the model uses.
func (m *Model) AttachVisionTower(tower *VisionTower) error {
	if tower == nil {
		return fmt.Errorf("model: attach vision tower: nil tower")
	}
	if m.Vision != nil {
		return fmt.Errorf("model: attach vision tower: this model already carries a vision tower (%d tensor(s), %d bytes); attaching is one-shot",
			len(m.Vision.manifest), m.Vision.Bytes())
	}
	if _, err := newVisionEncoder(tower, m.Cfg.HiddenSize); err != nil {
		return fmt.Errorf("model: attach vision tower: %w", err)
	}
	m.Vision = tower
	return nil
}

// extractQwen35VisionTower segregates the retained model.visual.* tensors out of a
// decoder manifest+raw into a standalone VisionTower — the safetensors twin of
// ggufload's mmproj VisionTower(). materializeQwen35Tensors leaves model.visual.* in
// man when RetainVision is set; newModel calls this immediately after (before any
// other pass sees them) so the vision stack lives on Model.Vision and the decoder
// manifest holds only text weights. Returns (nil, nil) when the checkpoint carries no
// vision tensors. The tower's geometry is derived from the tensor names (block count);
// finer vision_config geometry is pinned by the encoder slice (#4030).
func extractQwen35VisionTower(man map[string]tensorMeta, raw []byte) (*VisionTower, error) {
	const prefix = "model.visual."
	var names []string
	for n := range man {
		if strings.HasPrefix(n, prefix) {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		return nil, nil
	}
	sort.Strings(names)
	tensors := make([]NamedTensorF32, 0, len(names))
	for _, n := range names {
		meta := man[n]
		if meta.Nbytes%4 != 0 {
			return nil, fmt.Errorf("model: vision tensor %s is %d bytes, not an f32 multiple", n, meta.Nbytes)
		}
		nf := meta.Nbytes / 4
		if meta.Offset < 0 || meta.Offset+meta.Nbytes > len(raw) {
			return nil, fmt.Errorf("model: vision tensor %s [%d,%d) out of raw bounds %d", n, meta.Offset, meta.Offset+meta.Nbytes, len(raw))
		}
		data := make([]float32, nf)
		for i := 0; i < nf; i++ {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[meta.Offset+i*4:]))
		}
		shape := append([]int(nil), meta.Shape...)
		tensors = append(tensors, NamedTensorF32{Name: n, Shape: shape, Data: data})
		delete(man, n)
	}
	cfg := VisionConfig{NumLayers: countIndexedChildren(names, prefix+"blocks."), MergeSize: 1}
	return NewVisionTower(cfg, tensors)
}

// countIndexedChildren returns one past the highest <n> across names of the form
// "<prefix><n>." — i.e. the block/layer count of an indexed tensor family. Names that
// do not carry a parseable index under prefix are ignored.
func countIndexedChildren(names []string, prefix string) int {
	max := -1
	for _, n := range names {
		if !strings.HasPrefix(n, prefix) {
			continue
		}
		rest := n[len(prefix):]
		dot := strings.IndexByte(rest, '.')
		if dot <= 0 {
			continue
		}
		idx, ok := atoiStrict(rest[:dot])
		if !ok {
			continue
		}
		if idx > max {
			max = idx
		}
	}
	return max + 1
}

// atoiStrict parses a non-negative base-10 integer, rejecting empty/non-digit input
// (so "blocks.<n>" indices are read without pulling strconv into this hot package).
func atoiStrict(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	n := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

// ---- vision tensor-name resolver (the #4029 proof) ---------------------------------
//
// The vision tower is NOT one of the decoder families resolveSpecFor routes, so it
// gets its own entry point rather than a branch there — keeping the decoder resolver
// (issue #473) untouched. The canonical scheme is llama.cpp's mmproj GGUF naming
// (v.* globals + per-layer v.blk.<l>.* + the mm.* projector); an HF safetensors
// source is mapped on via aliases.

// ResolveVisionTensorNames proves a vision manifest carries every required tower
// tensor, mapping each canonical v.* name onto the source name that provides it. A
// required tensor with no source is a precise error naming the missing canonical
// tensor and the candidates searched — the same contract ResolveTensorNames gives
// the decoder families. It inspects manifest KEYS only, never bytes.
func ResolveVisionTensorNames(cfg VisionConfig, manifest map[string]tensorMeta) (*Resolution, error) {
	return resolveAgainstSpec(visionSpec(cfg), cfg.NumLayers, manifest)
}

// visionSpec is the CLIP/ViT tower's required-tensor table under the canonical mmproj
// v.* scheme. Globals: the patch-embed conv, the position embedding, and the mm.*
// projector (the seam that maps ViT hidden states into the decoder's embedding
// space). Per layer v.blk.<l>.*: split q/k/v/out attention, the two block layernorms,
// and the MLP up/down. Biases are optional (presence-driven — CLIP carries attention
// bias; some variants omit projector bias). Names an HF model.visual.* source can
// also satisfy are supplied as aliases. Finer per-variant differences (Qwen2-VL's
// merger, windowed attention) are pinned by the encoder forward (#4030), not asserted
// here — the same "core required, variant gated on a real checkpoint" discipline the
// deepseek/gptoss specs use.
func visionSpec(cfg VisionConfig) resolverSpec {
	return resolverSpec{
		family: "vision-clip",
		globals: []tensorReq{
			{canonical: "v.patch_embd.weight", aliases: []string{"model.visual.patch_embed.proj.weight"}},
			{canonical: "v.patch_embd.bias", aliases: []string{"model.visual.patch_embed.proj.bias"}, optional: true},
			{canonical: "v.position_embd.weight", aliases: []string{"model.visual.pos_embed", "model.visual.position_embedding.weight"}, optional: true},
			{canonical: "v.post_ln.weight", aliases: []string{"model.visual.post_layernorm.weight", "model.visual.merger.ln_q.weight"}, optional: true},
			{canonical: "v.post_ln.bias", aliases: []string{"model.visual.post_layernorm.bias"}, optional: true},
			{canonical: "mm.0.weight", aliases: []string{"model.visual.merger.mlp.0.weight", "mm.mlp.0.weight"}},
			{canonical: "mm.0.bias", aliases: []string{"model.visual.merger.mlp.0.bias"}, optional: true},
			{canonical: "mm.2.weight", aliases: []string{"model.visual.merger.mlp.2.weight"}, optional: true},
			{canonical: "mm.2.bias", aliases: []string{"model.visual.merger.mlp.2.bias"}, optional: true},
		},
		perLayer: func(l int) []tensorReq {
			p := "v.blk." + itoa(l) + "."
			hf := "model.visual.blocks." + itoa(l) + "."
			return []tensorReq{
				{canonical: p + "attn_q.weight", aliases: []string{hf + "attn.q.weight"}},
				{canonical: p + "attn_q.bias", aliases: []string{hf + "attn.q.bias"}, optional: true},
				{canonical: p + "attn_k.weight", aliases: []string{hf + "attn.k.weight"}},
				{canonical: p + "attn_k.bias", aliases: []string{hf + "attn.k.bias"}, optional: true},
				{canonical: p + "attn_v.weight", aliases: []string{hf + "attn.v.weight"}},
				{canonical: p + "attn_v.bias", aliases: []string{hf + "attn.v.bias"}, optional: true},
				{canonical: p + "attn_out.weight", aliases: []string{hf + "attn.proj.weight"}},
				{canonical: p + "attn_out.bias", aliases: []string{hf + "attn.proj.bias"}, optional: true},
				{canonical: p + "ln1.weight", aliases: []string{hf + "norm1.weight"}},
				{canonical: p + "ln1.bias", aliases: []string{hf + "norm1.bias"}, optional: true},
				{canonical: p + "ln2.weight", aliases: []string{hf + "norm2.weight"}},
				{canonical: p + "ln2.bias", aliases: []string{hf + "norm2.bias"}, optional: true},
				{canonical: p + "ffn_up.weight", aliases: []string{hf + "mlp.fc1.weight"}},
				{canonical: p + "ffn_up.bias", aliases: []string{hf + "mlp.fc1.bias"}, optional: true},
				{canonical: p + "ffn_down.weight", aliases: []string{hf + "mlp.fc2.weight"}},
				{canonical: p + "ffn_down.bias", aliases: []string{hf + "mlp.fc2.bias"}, optional: true},
			}
		},
	}
}
