package model

import "fmt"

// The published DeepSeek-V4.1-Flash shared-attention topology
// (DeepSeekV41FlashModelID@DeepSeekV41FlashRevision). admitDeepSeekV41Published
// checks a parsed config against it; NewDeepSeekV41GGUFConfig cross-checks a
// GGUF's tensor-derived topology against it.
var (
	v41PublishedCompressRatios = []int{
		0, 0,
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0,
	}
	v41PublishedKVSourceLayerIDs    = []int{2, 8, 14, 20}
	v41PublishedIndexSourceLayerIDs = []int{2, 8, 14, 20, 24, 28, 32, 36}
)

const (
	v41PublishedNumLayers              = 40
	v41PublishedCompressRopeTheta      = 160000
	v41PublishedCandidateSourceLayerID = 20
	v41PublishedCandidateTopKBlocks    = 2048
	v41PublishedCandidateBlockSize     = 8
)

// V41GGUFTopology is what a GGUF loader observes about a "deepseek41" artifact's
// shared-attention schedule. The GGUF carries no kv/index source or candidate
// keys, so sources come from which blocks ship the owning tensors. Nil source
// lists mean no tensor directory was observed (a header-only file).
type V41GGUFTopology struct {
	KVSourceLayerIDs    []int
	IndexSourceLayerIDs []int
}

// NewDeepSeekV41GGUFConfig synthesizes the V4.1 metadata the forward reads for a
// GGUF load, from the flat config axes (v41ConfigFromFlat) and the observed
// topology. A file whose decoder schedule is the published one must also carry
// the published sources (adopted when unobserved); its candidate geometry is the
// published one because the GGUF has no key for it. A mismatch is refused with ErrV41ConfigAdmission
// rather than run as a different attention topology.
func NewDeepSeekV41GGUFConfig(c Config, t V41GGUFTopology) (*DeepSeekV41Config, error) {
	d41 := v41ConfigFromFlat(c)
	if t.KVSourceLayerIDs != nil || t.IndexSourceLayerIDs != nil {
		d41.KVSourceLayerIDs = append([]int(nil), t.KVSourceLayerIDs...)
		d41.IndexSourceLayerIDs = append([]int(nil), t.IndexSourceLayerIDs...)
	}
	if !v41PublishedDecoderSchedule(c) {
		return d41, nil
	}
	for _, src := range []struct {
		name      string
		got       *[]int
		published []int
	}{
		{"kv sources", &d41.KVSourceLayerIDs, v41PublishedKVSourceLayerIDs},
		{"index sources", &d41.IndexSourceLayerIDs, v41PublishedIndexSourceLayerIDs},
	} {
		if t.KVSourceLayerIDs == nil && t.IndexSourceLayerIDs == nil {
			*src.got = append([]int(nil), src.published...)
			continue
		}
		if !v41EqualInts(*src.got, src.published) {
			return nil, fmt.Errorf("%w: GGUF %s=%v disagree with the published schedule %v", ErrV41ConfigAdmission, src.name, *src.got, src.published)
		}
	}
	d41.CandidateSourceLayerID = v41PublishedCandidateSourceLayerID
	d41.CandidateTopKBlocks = v41PublishedCandidateTopKBlocks
	d41.CandidateBlockSize = v41PublishedCandidateBlockSize
	return d41, nil
}

// v41PublishedDecoderSchedule reports a 40-layer decoder whose compress schedule
// matches the published one on every decoder layer.
func v41PublishedDecoderSchedule(c Config) bool {
	n := v41PublishedNumLayers
	return c.NumLayers == n && len(c.CompressRatios) >= n && v41EqualInts(c.CompressRatios[:n], v41PublishedCompressRatios[:n])
}
