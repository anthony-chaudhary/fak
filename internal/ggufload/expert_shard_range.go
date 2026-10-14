package ggufload

import (
	"fmt"
	"io"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ownedExpertRange validates the entire slab before admitting any owner-local
// allocation or read. FileOffset is relative to the retained tensor reader's
// shard, not the primary file in a split checkpoint.
func (s *WeightSource) ownedExpertRange(info TensorInfo, cfg model.Config, band ExpertShard) (io.ReaderAt, int, int, int, error) {
	if !archIsDeepSeek41(cfg.ModelType) || (info.Type != TensorQ2_K && info.Type != TensorQ3_K) {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s requires DeepSeek-V4.1 Q2_K or Q3_K", info.Name)
	}
	layer, proj, ok := glmMoeDsaBatchedExpert(info.Name)
	if !ok || layer < 0 || layer >= cfg.NumLayers {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s has invalid routed projection or layer", info.Name)
	}
	shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	experts, out, in, err := parseGLMMoeDsaExpertShape(shape)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	if experts != cfg.NumExperts || band.Lo < 0 || band.Hi <= band.Lo || band.Hi > experts {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s has %d experts, config %d, invalid band [%d,%d)", info.Name, experts, cfg.NumExperts, band.Lo, band.Hi)
	}
	wantOut, wantIn := cfg.MoEIntermediateSize, cfg.HiddenSize
	if proj == "down_proj" {
		wantOut, wantIn = wantIn, wantOut
	}
	if out != wantOut || in != wantIn {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s shape %v disagrees with config [%d,%d,%d]", info.Name, shape, cfg.NumExperts, wantOut, wantIn)
	}
	blockWeights, blockBytes, _ := residentExpertBlockGeometry(info.Type)
	if in%blockWeights != 0 || !model.ResidentKQuantEligible(cfg, batchedExpertCanonicalName(cfg.ModelType, layer, band.Lo, proj)) {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s is not row-aligned resident raw quant", info.Name)
	}
	// The resident builder also multiplies out*in using int, even though the
	// compressed payload may fit. Refuse before either multiplication can wrap.
	if out > math.MaxInt/in {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s expert element count overflows int", info.Name)
	}
	blocks := out * (in / blockWeights)
	if blocks > math.MaxInt/blockBytes {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s expert payload overflows int", info.Name)
	}
	perBytes := blocks * blockBytes
	if uint64(experts) > uint64(math.MaxInt64)/uint64(perBytes) {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s full payload overflows int64", info.Name)
	}
	fullBytes := int64(experts) * int64(perBytes)
	payloadBytes, err := tensorPayloadBytes(info)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	if payloadBytes != uint64(fullBytes) {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s payload %d differs from geometry %d", info.Name, payloadBytes, fullBytes)
	}
	r, size, err := s.tensorReader(info)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	// Validate the WHOLE payload, including the nonowned tail, against its own
	// shard. Subtraction avoids overflowing FileOffset+fullBytes.
	if info.FileOffset < 0 || info.FileOffset > size || fullBytes > size-info.FileOffset {
		return nil, 0, 0, 0, fmt.Errorf("gguf: owner-range tensor %s overruns file", info.Name)
	}
	return r, out, in, perBytes, nil
}

// computeOwnedExpertRange never calls TensorBytes or a dequantizing fallback.
// Each owned expert has one exact-size allocation adopted by the Q2/Q3 resident
// builder; neither a full slab nor any nonowned expert is materialized.
func (s *WeightSource) computeOwnedExpertRange(info TensorInfo, cfg model.Config, band ExpertShard) tensorWork {
	var tw tensorWork
	r, out, in, perBytes, err := s.ownedExpertRange(info, cfg, band)
	if err != nil {
		tw.err = err
		return tw
	}
	layer, proj, _ := glmMoeDsaBatchedExpert(info.Name)
	pending := make([]pendingTensor, 0, band.Hi-band.Lo)
	for expert := band.Lo; expert < band.Hi; expert++ {
		raw := make([]byte, perBytes)
		// The complete slab/band checks above prove both products and the sum
		// lie in the file's signed offset domain.
		offset := info.FileOffset + int64(expert)*int64(perBytes)
		n, err := r.ReadAt(raw, offset)
		if err == nil && n != len(raw) {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			tw.err = fmt.Errorf("gguf: read owner-range tensor %s expert %d: %w", info.Name, expert, err)
			return tw
		}
		pending = append(pending, pendingTensor{
			resident: true, residentType: info.Type,
			name:  batchedExpertCanonicalName(cfg.ModelType, layer, expert, proj),
			shape: []int{out, in}, raw: raw,
		})
	}
	tw.pending = pending
	tw.acctType, tw.acctExpert, tw.acctResident = info.Type.String(), true, true
	tw.acctBytes, tw.acctTensors = int64(band.Hi-band.Lo)*int64(perBytes), len(pending)
	tw.tickBytes = tw.acctBytes
	return tw
}
