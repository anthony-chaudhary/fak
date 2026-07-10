package ggufload

import (
	"fmt"
	"io"
	"strings"
)

// gguf_mmproj.go — the companion-mmproj load primitive for the vision epic (#4028).
//
// A VLM ships as TWO GGUFs: the text checkpoint and a standalone `mmproj` vision
// file (llama.cpp's CLIP tower — general.architecture "clip", metadata under
// clip.*, tensors under v.* / mm.*). OpenWeights (gguf_open.go) merges the split
// SHARDS of a single checkpoint, keyed on general.architecture; it has no notion
// of a second, independent weight source. OpenMMProj is that second source: a
// single-file open (an mmproj is never sharded) that hands back a *WeightSource so
// the vision tensor family (#4029) can read clip.* metadata via the embedded *File
// accessors and pull v.* / mm.* tensors through TensorF32, exactly as the text
// path reads its weights. Text-only loading is untouched — nothing calls this
// unless a --mmproj path is supplied (#4032).

// mmProjVisionTensorPrefixes are the GGUF namespaces llama.cpp's CLIP-tower
// converter emits: v.* for the vision transformer (v.patch_embd, v.position_embd,
// v.blk.<l>.*) and mm.* for the projector/merger that maps vision hidden states
// into the decoder's embedding space. These mirror isGLMMoeDsaVisionTensor's drop
// predicate (gguf_glm_tensors.go) — recognized there only to discard, recognized
// here to load.
var mmProjVisionTensorPrefixes = []string{"v.", "mm."}

// OpenMMProj opens a companion mmproj (CLIP vision-tower) GGUF as a standalone
// weight source. Unlike OpenWeights it does NOT stitch -N-of-M shards: an mmproj
// is a single self-contained file, and it must never be merged with the text
// checkpoint's shard set. The returned *WeightSource embeds the parsed *File, so
// its String/Uint64/Float64/Bool metadata accessors and Tensor/TensorF32 readers
// are immediately available to the model build; the caller owns it and must Close.
//
// It fails loud when the file carries no vision tensors — the common misuse is
// pointing --mmproj at the text model, which would otherwise load an empty tower
// and silently produce garbage image embeddings.
func OpenMMProj(path string) (*WeightSource, error) {
	f, gg, size, err := openAndRead(path)
	if err != nil {
		return nil, err
	}
	if err := validateMMProj(gg); err != nil {
		_ = f.Close()
		return nil, err
	}
	ws, err := NewWeightSource(gg, f, size)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	ws.closers = []io.Closer{f}
	return ws, nil
}

// validateMMProj rejects a GGUF that does not look like a CLIP mmproj: it must
// carry at least one v.* / mm.* vision tensor. (A clip.* metadata key alone is not
// enough — a header can declare vision metadata with no tower to back it.) The
// error names the file's architecture to make "you passed the text model" obvious.
func validateMMProj(f *File) error {
	for _, t := range f.Tensors {
		if isMMProjVisionTensor(t.Name) {
			return nil
		}
	}
	arch, _ := f.String("general.architecture")
	return fmt.Errorf("gguf: mmproj file has no vision tensors (no %s prefix); architecture=%q — is this the text model rather than the mmproj?",
		strings.Join(mmProjVisionTensorPrefixes, "/"), arch)
}

// isMMProjVisionTensor reports whether a GGUF tensor name lives in the mmproj
// vision namespace (v.* or mm.*). It is the load-side twin of the drop predicate
// isGLMMoeDsaVisionTensor and the anchor the vision-retain slice (#4029) keys on.
func isMMProjVisionTensor(name string) bool {
	for _, p := range mmProjVisionTensorPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}
