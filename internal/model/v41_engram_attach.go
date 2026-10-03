package model

// v41_engram_attach.go is the model-level, config-carrying attachment seam for
// the V4.1 Engram stage (public #13662, leaf of parent #12640). The reduced
// forward already knows how to execute a wired Engram stage
// (v41_forward_engram.go), but its only entry points are package-private
// (wireV41Engram) and require m.Cfg.DeepSeekV41 != nil, which the ordinary GGUF
// load path does NOT populate (a "deepseek41" GGUF is identified by ModelType;
// see IsDeepSeekV41). This file is the exported seam a loader calls to bind a
// verified Engram layout + row sources to an already-materialized model.
//
// Two invariants are enforced here that the package-private seam does not:
//   - every row source must carry a verified artifact binding
//     (V41EngramBinding(src) ok), so an unverified prepared-reader source can
//     never reach the forward through this path; and
//   - the GGUF identity's nil DeepSeekV41 config is populated from the supplied
//     geometry before wiring, so the returned model actually carries the stage
//     instead of silently omitting a declared one.
//
// It makes no correctness, parity, or throughput claim: the attachment is the
// admission seam only.

import (
	"fmt"
	"io"
)

// V41EngramAttachSpec is the Engram geometry a loader binds to a loaded model.
// The hash layout (TokenMap / Primes / Multipliers / Rows) travels separately in
// V41EngramLayout; this spec only carries the axes the reduced forward reads off
// m.Cfg.DeepSeekV41 (v41_forward_engram.go), which the GGUF identity does not
// populate on its own.
type V41EngramAttachSpec struct {
	LayerIDs            []int
	MaxNgramSize        int
	VocabSize           int
	NHeads              int
	HeadDim             int
	PadTokenID          int
	CompressedVocabSize int
}

// V41EngramDigestExtent returns the canonical "sha256:<hex>" content address
// over reader's [offset,offset+size) extent, the same digest form
// V41EngramArtifactBinding carries. It lets a loader bind an artifact it opened
// (and whose rows it dequantizes) without re-implementing the hashing contract
// the verified constructor enforces.
func V41EngramDigestExtent(reader io.ReaderAt, offset, size int64) (string, error) {
	return v41EngramDigestExtent(reader, offset, size)
}

// AttachV41Engram binds the Engram stage for spec + layout to this model, using
// one row source per declared layer in declaration order. The model config's
// V4.1 Engram axes are populated from spec (creating DeepSeekV41 for the GGUF
// "deepseek41" identity), then wireV41Engram builds the stage.
//
// Every source must be a *verified* row source (V41EngramBinding ok); a nil or
// unverified source, a malformed layout, or a geometry disagreement is refused
// with ErrV41NativeUnsupported before anything is wired, so a declared stage is
// never silently omitted and an unverified reader never reaches the forward.
func (m *Model) AttachV41Engram(spec V41EngramAttachSpec, layout V41EngramLayout, srcs []V41EngramRowSource, budgetBytes int64) error {
	if m == nil {
		return fmt.Errorf("%w: Engram attach needs a model", ErrV41NativeUnsupported)
	}
	if len(spec.LayerIDs) == 0 {
		return fmt.Errorf("%w: Engram attach declares no layers", ErrV41NativeUnsupported)
	}
	if len(spec.LayerIDs) != len(layout.Rows) || len(srcs) != len(layout.Rows) {
		return fmt.Errorf("%w: Engram declares %d layers, layout has %d row counts, %d sources",
			ErrV41NativeUnsupported, len(spec.LayerIDs), len(layout.Rows), len(srcs))
	}
	if err := layout.validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err)
	}
	for i, src := range srcs {
		if src == nil {
			return fmt.Errorf("%w: nil Engram row source for layer %d", ErrV41NativeUnsupported, spec.LayerIDs[i])
		}
		if _, ok := V41EngramBinding(src); !ok {
			return fmt.Errorf("%w: Engram source for layer %d carries no verified artifact binding", ErrV41NativeUnsupported, spec.LayerIDs[i])
		}
	}

	// Populate the config axes wireV41Engram reads. The GGUF identity reaches
	// this seam with DeepSeekV41 == nil, so create it rather than refuse.
	if m.Cfg.DeepSeekV41 == nil {
		m.Cfg.DeepSeekV41 = &DeepSeekV41Config{}
	}
	d41 := m.Cfg.DeepSeekV41
	d41.EngramLayerIDs = append([]int(nil), spec.LayerIDs...)
	d41.EngramNumEmbeddings = make([]int, len(layout.Rows))
	for i, rows := range layout.Rows {
		d41.EngramNumEmbeddings[i] = int(rows)
	}
	d41.EngramMaxNgramSize = spec.MaxNgramSize
	d41.EngramVocabSize = spec.VocabSize
	d41.EngramNHeads = spec.NHeads
	d41.EngramHeadDim = spec.HeadDim
	d41.EngramPadTokenID = spec.PadTokenID
	d41.EngramCompressedVocabSize = spec.CompressedVocabSize

	if err := m.wireV41Engram(layout, srcs, budgetBytes); err != nil {
		return fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err)
	}
	return nil
}

// V41EngramAttached reports whether this model carries a wired Engram stage whose
// declared layers agree with its V4.1 config. It is the model-level receipt a
// loader's witness asserts so a successful header/row read is not mistaken for
// an attached serving stage.
func (m *Model) V41EngramAttached() bool {
	stage := m.v41EngramStageFor()
	if stage == nil || m.Cfg.DeepSeekV41 == nil {
		return false
	}
	ids := m.Cfg.DeepSeekV41.EngramLayerIDs
	if len(ids) == 0 || len(ids) != len(stage.layerIDs) {
		return false
	}
	for i, id := range ids {
		if stage.layerIDs[i] != id {
			return false
		}
	}
	return true
}

// V41EngramStageBindings returns a copy of the artifact bindings the wired
// sources were admitted under (zero binding for an unverified source), and
// whether a stage exists at all. A loader uses it to prove the verified route was
// taken, not merely that some reader was wired.
func (m *Model) V41EngramStageBindings() ([]V41EngramArtifactBinding, bool) {
	stage := m.v41EngramStageFor()
	if stage == nil || len(stage.bindings) == 0 {
		return nil, false
	}
	out := make([]V41EngramArtifactBinding, len(stage.bindings))
	copy(out, stage.bindings)
	return out, true
}
