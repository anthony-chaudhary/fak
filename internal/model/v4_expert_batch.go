package model

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

var (
	ErrV4ExpertSelection = errors.New("invalid v4 expert selection")
	ErrV4ExpertBatchCap  = errors.New("v4 selected expert batch exceeds byte cap")
)

type v4ExpertGroup struct {
	Layer       int
	Expert      int
	TensorNames []string
	Bytes       int64
}

type v4ExpertBatchPlan struct {
	Groups      []v4ExpertGroup
	TensorCount int
	Bytes       int64
}

type v4ExpertBatch struct {
	Plan    v4ExpertBatchPlan
	Tensors []v4ExpertTensor
}

// planV4ExpertBatch resolves a router selection to complete expert tensor groups
// and performs all identity, completeness, and byte-cap checks before payload IO.
func (s *v4ExpertSource) planV4ExpertBatch(layer int, selected []int, byteCap int64) (v4ExpertBatchPlan, error) {
	if layer < 0 || len(selected) == 0 || byteCap < 0 {
		return v4ExpertBatchPlan{}, fmt.Errorf("%w: layer=%d selected=%d cap=%d", ErrV4ExpertSelection, layer, len(selected), byteCap)
	}
	byExpert := make(map[int][]string)
	for name := range s.entries {
		entryLayer, expert, err := parseV4ExpertIdentity(name)
		if err != nil {
			return v4ExpertBatchPlan{}, err
		}
		if entryLayer == layer {
			byExpert[expert] = append(byExpert[expert], name)
		}
	}

	seen := make(map[int]struct{}, len(selected))
	plan := v4ExpertBatchPlan{Groups: make([]v4ExpertGroup, 0, len(selected))}
	for _, expert := range selected {
		if expert < 0 {
			return v4ExpertBatchPlan{}, fmt.Errorf("%w: negative expert %d", ErrV4ExpertSelection, expert)
		}
		if _, duplicate := seen[expert]; duplicate {
			return v4ExpertBatchPlan{}, fmt.Errorf("%w: duplicate expert %d", ErrV4ExpertSelection, expert)
		}
		seen[expert] = struct{}{}
		names := append([]string(nil), byExpert[expert]...)
		if len(names) == 0 {
			return v4ExpertBatchPlan{}, fmt.Errorf("%w: layer %d expert %d missing", ErrV4ExpertSelection, layer, expert)
		}
		sort.Strings(names)
		group := v4ExpertGroup{Layer: layer, Expert: expert, TensorNames: names}
		for _, name := range names {
			entry := s.entries[name]
			start, end, err := safetensorsDataBounds(s.file.dataBase, s.file.size, entry)
			if err != nil {
				return v4ExpertBatchPlan{}, fmt.Errorf("%w: %s: %v", ErrV4ExpertMetadata, name, err)
			}
			group.Bytes += end - start
		}
		plan.TensorCount += len(names)
		plan.Bytes += group.Bytes
		plan.Groups = append(plan.Groups, group)
	}
	if plan.Bytes > byteCap {
		return v4ExpertBatchPlan{}, fmt.Errorf("%w: planned=%d cap=%d", ErrV4ExpertBatchCap, plan.Bytes, byteCap)
	}
	return plan, nil
}

func (s *v4ExpertSource) readV4ExpertBatch(layer int, selected []int, byteCap int64) (v4ExpertBatch, error) {
	plan, err := s.planV4ExpertBatch(layer, selected, byteCap)
	if err != nil {
		return v4ExpertBatch{}, err
	}
	batch := v4ExpertBatch{Plan: plan, Tensors: make([]v4ExpertTensor, 0, plan.TensorCount)}
	for _, group := range plan.Groups {
		for _, name := range group.TensorNames {
			tensor, err := s.read(name)
			if err != nil {
				return v4ExpertBatch{}, err
			}
			batch.Tensors = append(batch.Tensors, tensor)
		}
	}
	return batch, nil
}

func parseV4ExpertIdentity(name string) (layer, expert int, err error) {
	parts := strings.Split(name, ".")
	if len(parts) > 0 && parts[0] == "model" {
		parts = parts[1:]
	}
	if len(parts) < 7 || parts[0] != "layers" || (parts[2] != "ffn" && parts[2] != "mlp") || parts[3] != "experts" {
		return 0, 0, fmt.Errorf("%w: malformed routed-expert name %q", ErrV4ExpertMetadata, name)
	}
	layer, layerErr := strconv.Atoi(parts[1])
	expert, expertErr := strconv.Atoi(parts[4])
	if layerErr != nil || expertErr != nil || layer < 0 || expert < 0 {
		return 0, 0, fmt.Errorf("%w: malformed routed-expert identity %q", ErrV4ExpertMetadata, name)
	}
	return layer, expert, nil
}
