package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41EPOwnerDeviceBinding admits the resident mixed Q2_K/Q3_K owner path once
// per Session binding, before any forward math. A host or cpu-ref EP session
// keeps the software transport's existing fallback contract. A DeviceMemory EP
// session instead promises that every owned pick uses the fused routed seam;
// its decline may not turn into a host contraction after admission.
//
// This inspects actual packed stores and backend capabilities without uploading
// or reading weights. It is neither checkpoint authentication nor evidence of
// physical device execution/residency, and does not qualify production serve.
// Model weights and quant descriptors must remain immutable after binding.
func (s *Session) v41EPOwnerDeviceBinding(owners TPPlan, routed v41RoutedExpertFunc) (v41RoutedExpertFunc, error) {
	if s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return routed, nil
	}
	refuse := func(reason string) (v41RoutedExpertFunc, error) {
		return nil, fmt.Errorf("%w: owner-device admission: %s", ErrV41ExpertParallel, reason)
	}
	if !s.Backend.Caps().UploadDtype || s.CPUOffloadExperts || s.M.Cfg.ActGeluTanh || s.M.Cfg.ActGeluErf || routed == nil || s.v41RoutedExpertFunc() == nil {
		return refuse("requires native packed uploads and the fused routed expert operation")
	}
	for _, dtype := range []compute.Dtype{compute.Q2_K, compute.Q3_K} {
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, dtype) {
			return refuse(fmt.Sprintf("backend %s cannot execute %v", s.Backend.Name(), dtype))
		}
	}
	cfg := s.M.Cfg
	if err := owners.Validate(); err != nil || owners.Dim != cfg.NumExperts || cfg.NumLayers <= 0 || s.M.epRank < 0 || s.M.epRank >= len(owners.Shards) {
		return refuse("invalid owner band")
	}
	band := owners.Shards[s.M.epRank]
	for layer := 0; layer < cfg.NumLayers; layer++ {
		for expert := 0; expert < cfg.NumExperts; expert++ {
			for _, projection := range []struct {
				leaf       string
				kind       kQuantKind
				dtype      compute.Dtype
				blockBytes int
				out, in    int
			}{
				{"w1", kindQ2K, compute.Q2_K, q2kBlockBytes, cfg.MoEIntermediateSize, cfg.HiddenSize},
				{"w3", kindQ2K, compute.Q2_K, q2kBlockBytes, cfg.MoEIntermediateSize, cfg.HiddenSize},
				{"w2", kindQ3K, compute.Q3_K, q3kBlockBytes, cfg.HiddenSize, cfg.MoEIntermediateSize},
			} {
				name := layerName(layer, "ffn.experts."+itoa(expert)+"."+projection.leaf+".weight")
				if expert < band.Lo || expert >= band.Hi {
					if s.M.hasWeight(name) || s.M.q4w[name] != nil || s.M.expertCheckpoint.Has(name) {
						return refuse("nonowned projection " + name + " is present")
					}
					continue
				}
				weight, ok := s.resolveExpertWeight(name)
				qt := weight.kq
				if !ok || qt == nil || qt.kind != projection.kind || qt.out != projection.out || qt.in != projection.in || qt.lazy != nil {
					return refuse("requires resident mixed Q2_K/Q3_K projection " + name + " at the configured shape")
				}
				desc, ok := LookupQuantDescriptor(qt.kind)
				if !ok || !desc.SupportsHAL() || desc.Dtype() != projection.dtype || desc.BlockWeights() != qkK || desc.BlockBytes() != projection.blockBytes || qt.in <= 0 || qt.out <= 0 || qt.in%qkK != 0 || qt.nblk != qt.in/qkK {
					return refuse("invalid packed row geometry for " + name)
				}
				blocks, fits := checkedMulInt(qt.out, qt.nblk)
				bytes, sized := checkedMulInt(blocks, desc.BlockBytes())
				if !fits || !sized || bytes != len(qt.raw) {
					return refuse("invalid packed byte span for " + name)
				}
				if s.M.has(layerName(layer, "ffn.experts."+itoa(expert)+"."+projection.leaf+".bias")) {
					return refuse("biased projection " + name)
				}
			}
		}
	}
	return func(layer int, stem string, input []float32, weight float32) ([]float32, v41RoutedExpertOutcome, error) {
		output, outcome, err := routed(layer, stem, input, weight)
		if outcome == v41RoutedExpertDeclined {
			return nil, v41RoutedExpertError, fmt.Errorf("%w: owner-device layer %d %s declined after admission", ErrV41ExpertParallel, layer, stem)
		}
		return output, outcome, err
	}, nil
}
