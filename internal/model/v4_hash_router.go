package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	v4HashLayers  = 3
	v4HashVocab   = 129280
	v4HashTopK    = 6
	v4HashExperts = 384
)

type v4HashRouterSource struct {
	dir       string
	weightMap map[string]string
	openName  string
	openFile  *safetensorsFile
	openCount int
	readCount int
	readBytes int64
}

func newV4HashRouterSource(dir string) (*v4HashRouterSource, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, &v4RouteError{Field: "hash_index", Reason: err.Error()}
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return nil, &v4RouteError{Field: "hash_index", Reason: "malformed weight_map: " + err.Error()}
	}
	m := make(map[string]string, v4HashLayers)
	for layer := 0; layer < v4HashLayers; layer++ {
		name := v4HashTensorName(layer)
		shard, ok := idx.WeightMap[name]
		if !ok {
			return nil, &v4RouteError{Field: "hash_index", Reason: "missing " + name}
		}
		if shard == "" || filepath.IsAbs(shard) || filepath.Base(shard) != shard || filepath.Ext(shard) != ".safetensors" || strings.Contains(shard, "..") {
			return nil, &v4RouteError{Field: "hash_index", Reason: fmt.Sprintf("unsafe mapping %q -> %q", name, shard)}
		}
		m[name] = shard
	}
	return &v4HashRouterSource{dir: dir, weightMap: m}, nil
}

func v4HashTensorName(layer int) string {
	return fmt.Sprintf("layers.%d.ffn.gate.tid2eid", layer)
}

func (s *v4HashRouterSource) Close() error {
	if s == nil || s.openFile == nil {
		return nil
	}
	err := s.openFile.Close()
	s.openFile = nil
	s.openName = ""
	return err
}

func (s *v4HashRouterSource) shard(name string) (*safetensorsFile, error) {
	if s.openFile != nil && s.openName == name {
		return s.openFile, nil
	}
	if s.openFile != nil {
		if err := s.openFile.Close(); err != nil {
			return nil, &v4RouteError{Field: "hash_shard", Reason: err.Error()}
		}
		s.openFile = nil
		s.openName = ""
	}
	sf, err := openSafetensorsFileReadAt(filepath.Join(s.dir, name))
	if err != nil {
		return nil, &v4RouteError{Field: "hash_shard", Reason: err.Error()}
	}
	s.openFile, s.openName = sf, name
	s.openCount++
	return sf, nil
}

func (s *v4HashRouterSource) lookup(layer, tokenID int) ([]int, error) {
	if layer < 0 || layer >= v4HashLayers {
		return nil, &v4RouteError{Field: "hash_layer", Reason: fmt.Sprintf("%d outside [0,%d)", layer, v4HashLayers)}
	}
	if tokenID < 0 || tokenID >= v4HashVocab {
		return nil, &v4RouteError{Field: "token_id", Reason: fmt.Sprintf("%d outside [0,%d)", tokenID, v4HashVocab)}
	}
	name := v4HashTensorName(layer)
	sf, err := s.shard(s.weightMap[name])
	if err != nil {
		return nil, err
	}
	raw, ok := sf.hdr[name]
	if !ok {
		return nil, &v4RouteError{Field: "hash_tensor", Reason: "missing " + name}
	}
	var entry stEntry
	if err := json.Unmarshal(raw, &entry); err != nil {
		return nil, &v4RouteError{Field: "hash_tensor", Reason: "malformed " + name}
	}
	if entry.Dtype != "I64" || len(entry.Shape) != 2 || entry.Shape[0] != v4HashVocab || entry.Shape[1] != v4HashTopK {
		return nil, &v4RouteError{Field: "hash_tensor", Reason: fmt.Sprintf("%s must be I64 [%d,%d], got %s %v", name, v4HashVocab, v4HashTopK, entry.Dtype, entry.Shape)}
	}
	const rowBytes = int64(v4HashTopK * 8)
	if len(entry.DataOffsets) != 2 || entry.DataOffsets[0] < 0 || int64(entry.DataOffsets[1]-entry.DataOffsets[0]) != int64(v4HashVocab)*rowBytes {
		return nil, &v4RouteError{Field: "hash_tensor", Reason: name + " has invalid data range"}
	}
	buf := make([]byte, rowBytes)
	off := sf.dataBase + int64(entry.DataOffsets[0]) + int64(tokenID)*rowBytes
	if _, err := sf.r.ReadAt(buf, off); err != nil {
		return nil, &v4RouteError{Field: "hash_row", Reason: err.Error()}
	}
	s.readCount++
	s.readBytes += rowBytes
	ids := make([]int, v4HashTopK)
	seen := make(map[int]bool, v4HashTopK)
	for i := range ids {
		u := binary.LittleEndian.Uint64(buf[i*8:])
		if u >= v4HashExperts {
			return nil, &v4RouteError{Field: "expert_id", Reason: fmt.Sprintf("%d at slot %d outside [0,%d)", u, i, v4HashExperts)}
		}
		ids[i] = int(u)
		if seen[ids[i]] {
			return nil, &v4RouteError{Field: "expert_id", Reason: fmt.Sprintf("duplicate %d", ids[i])}
		}
		seen[ids[i]] = true
	}
	return ids, nil
}

func v4HashRoute(logits []float32, expertIDs []int, routeScale float32) ([]routePick, error) {
	if len(logits) != v4HashExperts {
		return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("width %d, want %d", len(logits), v4HashExperts)}
	}
	if len(expertIDs) != v4HashTopK {
		return nil, &v4RouteError{Field: "expert_ids", Reason: fmt.Sprintf("width %d, want %d", len(expertIDs), v4HashTopK)}
	}
	if !finite32(routeScale) || routeScale <= 0 {
		return nil, &v4RouteError{Field: "route_scale", Reason: fmt.Sprintf("must be finite and positive, got %g", routeScale)}
	}
	scores := make([]float32, len(logits))
	for i, z := range logits {
		if !finite32(z) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
		}
		zf := float64(z)
		scores[i] = float32(math.Sqrt(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))))
	}
	picks := make([]routePick, len(expertIDs))
	seen := make(map[int]bool, len(expertIDs))
	var sum float32
	for i, id := range expertIDs {
		if id < 0 || id >= len(logits) || seen[id] {
			return nil, &v4RouteError{Field: "expert_ids", Reason: fmt.Sprintf("invalid or duplicate expert %d at slot %d", id, i)}
		}
		seen[id] = true
		picks[i] = routePick{expert: id, weight: scores[id]}
		sum += scores[id]
	}
	if !finite32(sum) || sum <= 0 {
		return nil, &v4RouteError{Field: "normalization", Reason: fmt.Sprintf("selected score sum must be finite and positive, got %g", sum)}
	}
	for i := range picks {
		picks[i].weight = picks[i].weight / sum * routeScale
	}
	return picks, nil
}
