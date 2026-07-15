package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var ErrV4ShardedSource = errors.New("invalid v4 sharded expert source")

type v4ShardHandle struct {
	name string
	src  *v4ExpertSource
}

// v4ShardedExpertSource routes V4 expert tensors through a safetensors index while
// retaining only a bounded number of open shard/header handles. Payloads remain ReadAt-only.
type v4ShardedExpertSource struct {
	dir       string
	weightMap map[string]string
	maxOpen   int
	open      map[string]*v4ShardHandle
	lru       []string
	openCount int
	readCount int
}

func newV4ShardedExpertSource(dir string, maxOpen int) (*v4ShardedExpertSource, error) {
	if maxOpen <= 0 {
		return nil, fmt.Errorf("%w: maxOpen=%d", ErrV4ShardedSource, maxOpen)
	}
	b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, fmt.Errorf("%w: index: %v", ErrV4ShardedSource, err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil || len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("%w: malformed or empty weight_map", ErrV4ShardedSource)
	}
	clean := make(map[string]string, len(idx.WeightMap))
	for tensor, shard := range idx.WeightMap {
		if tensor == "" || shard == "" || filepath.IsAbs(shard) || filepath.Base(shard) != shard || filepath.Ext(shard) != ".safetensors" || strings.Contains(shard, "..") {
			return nil, fmt.Errorf("%w: unsafe mapping %q -> %q", ErrV4ShardedSource, tensor, shard)
		}
		if _, _, err := parseV4ExpertIdentity(tensor); err != nil {
			// Non-expert tensors legitimately share the model index; only retain routed experts.
			continue
		}
		clean[tensor] = shard
	}
	if len(clean) == 0 {
		return nil, fmt.Errorf("%w: no routed experts", ErrV4ShardedSource)
	}
	return &v4ShardedExpertSource{dir: dir, weightMap: clean, maxOpen: maxOpen, open: make(map[string]*v4ShardHandle)}, nil
}

func (s *v4ShardedExpertSource) Close() error {
	var first error
	for _, h := range s.open {
		if err := h.src.file.Close(); err != nil && first == nil {
			first = err
		}
	}
	s.open = make(map[string]*v4ShardHandle)
	s.lru = nil
	return first
}

func (s *v4ShardedExpertSource) shard(name string) (*v4ExpertSource, error) {
	if h := s.open[name]; h != nil {
		s.touch(name)
		return h.src, nil
	}
	if len(s.open) == s.maxOpen {
		victim := s.lru[0]
		s.lru = s.lru[1:]
		_ = s.open[victim].src.file.Close()
		delete(s.open, victim)
	}
	sf, err := openSafetensorsFileReadAt(filepath.Join(s.dir, name))
	if err != nil {
		return nil, fmt.Errorf("%w: open shard %q: %v", ErrV4ShardedSource, name, err)
	}
	src, err := newV4ExpertSource(sf)
	if err != nil {
		_ = sf.Close()
		return nil, err
	}
	s.open[name] = &v4ShardHandle{name: name, src: src}
	s.lru = append(s.lru, name)
	s.openCount++
	return src, nil
}

func (s *v4ShardedExpertSource) touch(name string) {
	for i, n := range s.lru {
		if n == name {
			s.lru = append(append(s.lru[:i], s.lru[i+1:]...), name)
			return
		}
	}
}

func (s *v4ShardedExpertSource) planV4ExpertBatch(layer int, selected []int, byteCap int64) (v4ExpertBatchPlan, error) {
	if layer < 0 || len(selected) == 0 || byteCap < 0 {
		return v4ExpertBatchPlan{}, ErrV4ExpertSelection
	}
	byExpert := make(map[int][]string)
	for name := range s.weightMap {
		l, e, err := parseV4ExpertIdentity(name)
		if err != nil {
			return v4ExpertBatchPlan{}, err
		}
		if l == layer {
			byExpert[e] = append(byExpert[e], name)
		}
	}
	seen := map[int]bool{}
	plan := v4ExpertBatchPlan{Groups: make([]v4ExpertGroup, 0, len(selected))}
	for _, expert := range selected {
		if expert < 0 || seen[expert] {
			return v4ExpertBatchPlan{}, fmt.Errorf("%w: expert %d", ErrV4ExpertSelection, expert)
		}
		seen[expert] = true
		names := append([]string(nil), byExpert[expert]...)
		sort.Strings(names)
		if len(names) == 0 {
			return v4ExpertBatchPlan{}, fmt.Errorf("%w: layer %d expert %d missing", ErrV4ExpertSelection, layer, expert)
		}
		g := v4ExpertGroup{Layer: layer, Expert: expert, TensorNames: names}
		for _, name := range names {
			src, err := s.shard(s.weightMap[name])
			if err != nil {
				return v4ExpertBatchPlan{}, err
			}
			e, ok := src.entries[name]
			if !ok {
				return v4ExpertBatchPlan{}, fmt.Errorf("%w: index tensor %q absent from shard", ErrV4ShardedSource, name)
			}
			start, end, err := safetensorsDataBounds(src.file.dataBase, src.file.size, e)
			if err != nil {
				return v4ExpertBatchPlan{}, err
			}
			g.Bytes += end - start
		}
		plan.Groups = append(plan.Groups, g)
		plan.TensorCount += len(names)
		plan.Bytes += g.Bytes
	}
	if plan.Bytes > byteCap {
		return v4ExpertBatchPlan{}, fmt.Errorf("%w: planned=%d cap=%d", ErrV4ExpertBatchCap, plan.Bytes, byteCap)
	}
	return plan, nil
}

func (s *v4ShardedExpertSource) readV4ExpertBatch(layer int, selected []int, byteCap int64) (v4ExpertBatch, error) {
	plan, err := s.planV4ExpertBatch(layer, selected, byteCap)
	if err != nil {
		return v4ExpertBatch{}, err
	}
	out := v4ExpertBatch{Plan: plan, Tensors: make([]v4ExpertTensor, 0, plan.TensorCount)}
	for _, g := range plan.Groups {
		for _, name := range g.TensorNames {
			t, err := s.read(name)
			if err != nil {
				return v4ExpertBatch{}, err
			}
			out.Tensors = append(out.Tensors, t)
		}
	}
	return out, nil
}

func (s *v4ShardedExpertSource) read(name string) (v4ExpertTensor, error) {
	shard, ok := s.weightMap[name]
	if !ok {
		return v4ExpertTensor{}, fmt.Errorf("%w: %s", ErrV4ExpertNotFound, name)
	}
	src, err := s.shard(shard)
	if err != nil {
		return v4ExpertTensor{}, err
	}
	tensor, err := src.read(name)
	if err == nil {
		s.readCount++
	}
	return tensor, err
}

func (s *v4ShardedExpertSource) entry(name string) (stEntry, bool) {
	shard, ok := s.weightMap[name]
	if !ok {
		return stEntry{}, false
	}
	src, err := s.shard(shard)
	if err != nil {
		return stEntry{}, false
	}
	return src.entry(name)
}
