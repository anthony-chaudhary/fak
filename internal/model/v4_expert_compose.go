package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var ErrV4ExpertCompose = errors.New("invalid v4 expert composition")

type v4RoutedExpert struct {
	Expert int
	Weight float32
}

// newV4ExpertQuantStager joins the production selective source to the exact
// pinned MXFP4 decoder. Companion scale reads are charged to the same staging
// statistics as weight reads, while only decoded projection weights enter the
// bounded ring.
func newV4ExpertQuantStager(source *v4ExpertSource, ring *pagedRing, plan v4ExpertBatchPlan) (*v4ExpertStager, error) {
	stager, err := newV4ExpertStager(source, ring, plan, compute.F32, func(v4ExpertTensor) (compute.Tensor, error) {
		return compute.Tensor{}, fmt.Errorf("%w: quant decoder not initialized", ErrV4ExpertCompose)
	})
	if err != nil {
		return nil, err
	}
	stager.decode = func(weight v4ExpertTensor) (compute.Tensor, error) {
		scaleName := strings.TrimSuffix(weight.Name, ".weight") + ".scale"
		if _, selected := stager.selected[scaleName]; !selected {
			return compute.Tensor{}, fmt.Errorf("%w: scale %s not selected", ErrV4ExpertCompose, scaleName)
		}
		scale, err := source.read(scaleName)
		if err != nil {
			return compute.Tensor{}, err
		}
		stager.stats.SourceReads++
		stager.stats.SourceBytes += int64(len(scale.Bytes))
		weightEntry, weightOK := source.entries[weight.Name]
		scaleEntry, scaleOK := source.entries[scaleName]
		if !weightOK || !scaleOK {
			return compute.Tensor{}, fmt.Errorf("%w: missing quant metadata for %s", ErrV4ExpertCompose, weight.Name)
		}
		decoded, shape, err := decodeV4ExpertQuant(weight.Name, scaleName, weightEntry, scaleEntry, weight.Bytes, scale.Bytes)
		if err != nil {
			return compute.Tensor{}, err
		}
		values := make([]float32, len(decoded)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(decoded[i*4:]))
		}
		return compute.NewF32(ring.be, shape, values), nil
	}
	return stager, nil
}

func newV4ShardedExpertQuantStager(source *v4ShardedExpertSource, ring *pagedRing, plan v4ExpertBatchPlan) (*v4ExpertStager, error) {
	stager, err := newV4ShardedExpertStager(source, ring, plan, compute.F32, func(v4ExpertTensor) (compute.Tensor, error) {
		return compute.Tensor{}, fmt.Errorf("%w: quant decoder not initialized", ErrV4ExpertCompose)
	})
	if err != nil {
		return nil, err
	}
	stager.decode = func(weight v4ExpertTensor) (compute.Tensor, error) {
		scaleName := strings.TrimSuffix(weight.Name, ".weight") + ".scale"
		if _, selected := stager.selected[scaleName]; !selected {
			return compute.Tensor{}, fmt.Errorf("%w: scale %s not selected", ErrV4ExpertCompose, scaleName)
		}
		scale, err := source.read(scaleName)
		if err != nil {
			return compute.Tensor{}, err
		}
		stager.stats.SourceReads++
		stager.stats.SourceBytes += int64(len(scale.Bytes))
		weightEntry, weightOK := source.entry(weight.Name)
		scaleEntry, scaleOK := source.entry(scaleName)
		if !weightOK || !scaleOK {
			return compute.Tensor{}, fmt.Errorf("%w: missing quant metadata for %s", ErrV4ExpertCompose, weight.Name)
		}
		decoded, shape, err := decodeV4ExpertQuant(weight.Name, scaleName, weightEntry, scaleEntry, weight.Bytes, scale.Bytes)
		if err != nil {
			return compute.Tensor{}, err
		}
		values := make([]float32, len(decoded)/4)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(decoded[i*4:]))
		}
		return compute.NewF32(ring.be, shape, values), nil
	}
	return stager, nil
}

// composeV4RoutedExperts executes the pinned V4 Expert.forward arithmetic for
// one router selection. The stager owns all tensor IO, decode, and bounded-ring
// admission; this function only composes the three projections:
//
//	w2(router_weight * (SiLU(w1(x)) * w3(x)))
//
// Results are accumulated in router order. Floating-point accumulation order is
// therefore explicit and reproducible rather than dependent on map iteration.
func composeV4RoutedExperts(layer int, selected []v4RoutedExpert, x compute.Tensor, swigluLimit float32, stager *v4ExpertStager) ([]float32, error) {
	if layer < 0 || len(selected) == 0 || stager == nil || stager.ring == nil {
		return nil, fmt.Errorf("%w: layer=%d selected=%d or nil stager", ErrV4ExpertCompose, layer, len(selected))
	}
	if len(x.Shape) != 1 || x.Shape[0] <= 0 {
		return nil, fmt.Errorf("%w: input shape %v is not a non-empty vector", ErrV4ExpertCompose, x.Shape)
	}
	if swigluLimit < 0 || math.IsNaN(float64(swigluLimit)) || math.IsInf(float64(swigluLimit), 0) {
		return nil, fmt.Errorf("%w: swiglu limit=%v", ErrV4ExpertCompose, swigluLimit)
	}
	groups := make(map[int]v4ExpertProjectionNames, len(stager.selected))
	for _, group := range stagerPlanGroups(stager) {
		if group.Layer != layer {
			continue
		}
		names, err := v4ProjectionNames(group)
		if err != nil {
			return nil, err
		}
		groups[group.Expert] = names
	}

	seen := make(map[int]struct{}, len(selected))
	orderedNames := make([]v4ExpertProjectionNames, len(selected))
	// Validate the complete router selection before the first source read. A bad
	// later route therefore cannot leave a partially paged composition behind.
	for i, route := range selected {
		if route.Expert < 0 || route.Weight < 0 || math.IsNaN(float64(route.Weight)) || math.IsInf(float64(route.Weight), 0) {
			return nil, fmt.Errorf("%w: expert=%d router weight=%v", ErrV4ExpertCompose, route.Expert, route.Weight)
		}
		if _, duplicate := seen[route.Expert]; duplicate {
			return nil, fmt.Errorf("%w: duplicate expert %d", ErrV4ExpertCompose, route.Expert)
		}
		seen[route.Expert] = struct{}{}
		names, ok := groups[route.Expert]
		if !ok {
			return nil, fmt.Errorf("%w: layer %d expert %d not selected by plan", ErrV4ExpertCompose, layer, route.Expert)
		}
		if err := validateV4CompositionSourceShapes(names, stager.source, x.Shape[0]); err != nil {
			return nil, fmt.Errorf("%w: expert %d: %v", ErrV4ExpertCompose, route.Expert, err)
		}
		orderedNames[i] = names
	}

	var output []float32
	for routeIndex, route := range selected {
		names := orderedNames[routeIndex]
		gate, err := stager.matMul(names.w1, x)
		if err != nil {
			return nil, err
		}
		up, err := stager.matMul(names.w3, x)
		if err != nil {
			return nil, err
		}
		if len(gate) == 0 || len(gate) != len(up) {
			return nil, fmt.Errorf("%w: expert %d projection dimensions w1=%d w3=%d", ErrV4ExpertCompose, route.Expert, len(gate), len(up))
		}
		activated := make([]float32, len(gate))
		for i := range gate {
			gateValue, upValue := gate[i], up[i]
			if swigluLimit > 0 {
				// Pinned Expert.forward clamps the gate only from above, but the
				// up projection symmetrically. Keeping the asymmetry is observable
				// for large negative gate values and is required for parity.
				gateValue = min(gateValue, swigluLimit)
				upValue = max(-swigluLimit, min(upValue, swigluLimit))
			}
			activated[i] = route.Weight * v4SiLU(gateValue) * upValue
		}
		activation := stager.ring.be.Upload(compute.NewF32(stager.ring.be, []int{len(activated)}, activated), compute.F32)
		projected, err := stager.matMul(names.w2, activation)
		stager.ring.be.Free(activation)
		if err != nil {
			return nil, err
		}
		if len(projected) != x.Shape[0] {
			return nil, fmt.Errorf("%w: expert %d w2 output=%d input=%d", ErrV4ExpertCompose, route.Expert, len(projected), x.Shape[0])
		}
		if output == nil {
			output = make([]float32, len(projected))
		}
		for i := range output {
			output[i] += projected[i]
		}
	}
	return output, nil
}

type v4ExpertProjectionNames struct{ w1, w2, w3 string }

// stagerPlanGroups reconstructs complete selected groups from the stager's
// admitted identities. It intentionally ignores scale tensors: they are decode
// companions, not executable projections.
func stagerPlanGroups(stager *v4ExpertStager) []v4ExpertGroup {
	byIdentity := make(map[[2]int][]string)
	for name := range stager.selected {
		layer, expert, err := parseV4ExpertIdentity(name)
		if err == nil {
			key := [2]int{layer, expert}
			byIdentity[key] = append(byIdentity[key], name)
		}
	}
	keys := make([][2]int, 0, len(byIdentity))
	for key := range byIdentity {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i][0] != keys[j][0] {
			return keys[i][0] < keys[j][0]
		}
		return keys[i][1] < keys[j][1]
	})
	groups := make([]v4ExpertGroup, 0, len(keys))
	for _, key := range keys {
		sort.Strings(byIdentity[key])
		groups = append(groups, v4ExpertGroup{Layer: key[0], Expert: key[1], TensorNames: byIdentity[key]})
	}
	return groups
}

func v4ProjectionNames(group v4ExpertGroup) (v4ExpertProjectionNames, error) {
	var out v4ExpertProjectionNames
	for _, name := range group.TensorNames {
		switch {
		case strings.HasSuffix(name, ".w1.weight"):
			if out.w1 != "" {
				return out, fmt.Errorf("%w: duplicate w1 for expert %d", ErrV4ExpertCompose, group.Expert)
			}
			out.w1 = name
		case strings.HasSuffix(name, ".w2.weight"):
			if out.w2 != "" {
				return out, fmt.Errorf("%w: duplicate w2 for expert %d", ErrV4ExpertCompose, group.Expert)
			}
			out.w2 = name
		case strings.HasSuffix(name, ".w3.weight"):
			if out.w3 != "" {
				return out, fmt.Errorf("%w: duplicate w3 for expert %d", ErrV4ExpertCompose, group.Expert)
			}
			out.w3 = name
		}
	}
	if out.w1 == "" || out.w2 == "" || out.w3 == "" {
		return out, fmt.Errorf("%w: layer %d expert %d lacks exact w1/w2/w3 weights", ErrV4ExpertCompose, group.Layer, group.Expert)
	}
	return out, nil
}

func validateV4CompositionShapes(names v4ExpertProjectionNames, entries map[string]stEntry, inputDim int) error {
	w1, ok1 := entries[names.w1]
	w2, ok2 := entries[names.w2]
	w3, ok3 := entries[names.w3]
	if !ok1 || !ok2 || !ok3 || len(w1.Shape) != 2 || len(w2.Shape) != 2 || len(w3.Shape) != 2 {
		return fmt.Errorf("missing or non-matrix projection metadata")
	}
	// Packed V4 MXFP4 weights have half the logical input width. F32 fixtures
	// carry their full width. Normalize only the exact admitted packed dtype.
	logicalCols := func(entry stEntry) int {
		if entry.Dtype == "I8" {
			return entry.Shape[1] * 2
		}
		return entry.Shape[1]
	}
	if logicalCols(w1) != inputDim || logicalCols(w3) != inputDim || w1.Shape[0] != w3.Shape[0] {
		return fmt.Errorf("w1=%v/%s w3=%v/%s input=%d", w1.Shape, w1.Dtype, w3.Shape, w3.Dtype, inputDim)
	}
	if w2.Shape[0] != inputDim || logicalCols(w2) != w1.Shape[0] {
		return fmt.Errorf("w2=%v/%s wants output=%d intermediate=%d", w2.Shape, w2.Dtype, inputDim, w1.Shape[0])
	}
	return nil
}

func v4SiLU(x float32) float32 {
	return float32(float64(x) / (1 + math.Exp(-float64(x))))
}

func validateV4CompositionSourceShapes(names v4ExpertProjectionNames, source v4ExpertTensorSource, input int) error {
	entries := make(map[string]stEntry, 3)
	for _, name := range []string{names.w1, names.w2, names.w3} {
		entry, ok := source.entry(name)
		if !ok {
			return fmt.Errorf("missing tensor %s", name)
		}
		entries[name] = entry
	}
	return validateV4CompositionShapes(names, entries, input)
}
