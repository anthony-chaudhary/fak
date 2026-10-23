package ggufload

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// EmbedStorage is how the selected load arm keeps the input-embedding table resident. It names
// the arm's weights estimator, so the resident bytes decode only row-gathers match that plan.
type EmbedStorage int

const (
	EmbedPayload EmbedStorage = iota // raw GGUF payload: EstimateLoadMemoryPlan and the cpu-offload estimators
	EmbedF32                         // EstimateF32LoadMemoryPlan
	EmbedQ8                          // EstimateQ8LoadMemoryPlan
	EmbedQ4K                         // EstimateQ4KLoadMemoryPlan under the same options
)

const inputEmbedTensor = "token_embd.weight"

// FitPlanModel derives the planner's model facts from the header alone, plus the resident bytes
// of the row-gathered input embedding under the arm's storage (opts apply to EmbedQ4K only). A
// routed MoE with a known expert_used_count reuses RoutedExpertActiveSet; anything else is a
// dense sweep of the loaded tensors, and an MoE without a derivable active set is flagged
// ActiveSetUnknown.
func (s *WeightSource) FitPlanModel(embed EmbedStorage, opts ...Q4KLoadOption) (fitplan.Model, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return fitplan.Model{}, err
	}
	m, err := s.fitPlanSweep(cfg)
	if err != nil {
		return fitplan.Model{}, err
	}
	if m.HeadParams, err = s.headParams(cfg); err != nil {
		return fitplan.Model{}, err
	}
	m.QuantFamily = s.fitPlanQuantFamily()
	// The gather table is excluded from the swept scale only where it is excluded from the sweep.
	if cfg.TieWordEmbeddings || m.GatherPayloadBytes > 0 {
		if m.GatherResidentBytes, err = s.inputEmbedGatherResident(cfg, embed, opts); err != nil {
			return fitplan.Model{}, err
		}
	}
	return m, nil
}

func (s *WeightSource) fitPlanSweep(cfg model.Config) (fitplan.Model, error) {
	csc := cfg.ContextSizeConfig()
	m := fitplan.Model{
		Arch:             cfg.ModelType,
		MaxContextTokens: csc.MaxContext,
		AttnLayers:       csc.KV.NumLayers,
		NumHeads:         cfg.NumHeads,
		HeadDim:          csc.KV.HeadDim,
	}
	as, ok, err := s.RoutedExpertActiveSet()
	if err != nil {
		return fitplan.Model{}, err
	}
	if ok && as.ExpertsUsed > 0 {
		m.PayloadWeightBytes = as.RoutedResident + as.NonExpertResident
		m.StreamBytesPerToken = as.ActiveBytesPerTokenSwept
		m.GatherPayloadBytes = as.InputEmbedGather
		m.ActiveParamsPerToken = as.ActiveParamsPerToken
		m.RoutedExpertBytes = as.RoutedResident
		m.ActiveRoutedBytesPerToken = as.ActivePerToken
		m.NumExperts = as.NumExperts
		m.ExpertsUsed = as.ExpertsUsed
		if as.InputEmbedGather > 0 {
			embElems, err := s.tensorElemsByName(inputEmbedTensor)
			if err != nil {
				return fitplan.Model{}, err
			}
			m.ActiveParamsPerToken = max(m.ActiveParamsPerToken-embElems, 0)
		}
		return m, nil
	}

	var bytes, elems, embBytes, embElems uint64
	for _, info := range s.File.Tensors {
		if glmMoeDsaSkipGGUFTensorForType(cfg.ModelType, info.Name) {
			continue
		}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return fitplan.Model{}, fmt.Errorf("gguf: fit-plan tensor %s: %w", info.Name, err)
		}
		e, err := tensorElems(info)
		if err != nil {
			return fitplan.Model{}, fmt.Errorf("gguf: fit-plan tensor %s: %w", info.Name, err)
		}
		if bytes > math.MaxUint64-n || elems > math.MaxUint64-e {
			return fitplan.Model{}, fmt.Errorf("gguf: fit-plan totals overflow uint64")
		}
		bytes += n
		elems += e
		if strings.EqualFold(info.Name, inputEmbedTensor) {
			embBytes, embElems = n, e
		}
	}
	if bytes > math.MaxInt64 || elems > math.MaxInt64 {
		return fitplan.Model{}, fmt.Errorf("gguf: fit-plan totals %d/%d overflow int64", bytes, elems)
	}
	m.PayloadWeightBytes = int64(bytes)
	m.StreamBytesPerToken = int64(bytes)
	m.ActiveParamsPerToken = int64(elems)
	// Untied input embeddings are a per-token row gather, not a sweep (as in RoutedExpertActiveSet).
	if !cfg.TieWordEmbeddings && embBytes > 0 && embBytes < bytes {
		m.StreamBytesPerToken -= int64(embBytes)
		m.GatherPayloadBytes = int64(embBytes)
		m.ActiveParamsPerToken -= int64(embElems)
	}
	m.ActiveSetUnknown = cfg.NumExperts > 0
	return m, nil
}

// fitPlanQuantFamily is the quant family carrying the most payload bytes, the one a decode's
// dequant cost tracks; float tensors name the family only when nothing is quantized.
func (s *WeightSource) fitPlanQuantFamily() string {
	byFamily := map[string]uint64{}
	for _, info := range s.File.Tensors {
		if n, err := tensorPayloadBytes(info); err == nil {
			byFamily[fitplan.QuantFamily(info.Type.String())] += n
		}
	}
	best, bestN := "", uint64(0)
	for family, n := range byFamily {
		if family != "float" && (n > bestN || (n == bestN && family < best)) {
			best, bestN = family, n
		}
	}
	if best == "" && byFamily["float"] > 0 {
		return "float"
	}
	return best
}

func (s *WeightSource) tensorElemsByName(name string) (int64, error) {
	info, ok := s.Tensor(name)
	if !ok {
		return 0, nil
	}
	e, err := tensorElems(info)
	if err != nil {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s: %w", name, err)
	}
	if e > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s elements %d overflow int64", name, e)
	}
	return int64(e), nil
}

// headParams is the LM head's element count: output.weight, or the input table when tied.
func (s *WeightSource) headParams(cfg model.Config) (int64, error) {
	if cfg.TieWordEmbeddings {
		return s.tensorElemsByName(inputEmbedTensor)
	}
	return s.tensorElemsByName("output.weight")
}

// inputEmbedGatherResident is the resident bytes of the input-embedding table that decode only
// row-gathers under embed: the whole table when untied; when tied, the gather copy an arm keeps
// beside a separate head, else 0 because the table is the swept head itself.
func (s *WeightSource) inputEmbedGatherResident(cfg model.Config, embed EmbedStorage, opts []Q4KLoadOption) (int64, error) {
	info, ok := s.Tensor(inputEmbedTensor)
	if !ok {
		return 0, nil
	}
	tied := cfg.TieWordEmbeddings
	elems, err := tensorElems(info)
	if err != nil {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s: %w", info.Name, err)
	}
	if elems > math.MaxInt64/4 {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s f32 bytes overflow int64", info.Name)
	}
	f32 := int64(elems) * 4
	switch embed {
	case EmbedF32:
		if tied {
			return 0, nil
		}
		return f32, nil
	case EmbedQ8:
		// EstimateQ8LoadBytes keeps the table f32 and adds a native-Q8 head only for a 2-D tied table.
		if tied && len(info.Dims) != 2 {
			return 0, nil
		}
		return f32, nil
	case EmbedQ4K:
		var gather uint64
		_, _, err := s.walkQ4KLoadStorage(opts, func(st q4kTensorStorage) error {
			if st.info.Name == inputEmbedTensor && !st.streamedBounded && (!tied || st.tiedHeadQ8 > 0) {
				gather = st.bytes
			}
			return nil
		})
		switch {
		case errors.Is(err, ErrQ4KLoadEstimateUnsupported):
			// The arm then admits the raw payload plan, so price the gather in payload bytes.
		case err != nil:
			return 0, err
		case gather > math.MaxInt64:
			return 0, fmt.Errorf("gguf: fit-plan gather bytes %d overflow int64", gather)
		default:
			return int64(gather), nil
		}
	}
	if tied {
		return 0, nil
	}
	n, err := tensorPayloadBytes(info)
	if err != nil {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s: %w", info.Name, err)
	}
	if n > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: fit-plan tensor %s payload %d overflow int64", info.Name, n)
	}
	return int64(n), nil
}
