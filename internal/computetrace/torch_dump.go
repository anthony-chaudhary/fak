package computetrace

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

var (
	ErrTorchDump              = errors.New("computetrace: malformed Torch dump")
	ErrTorchRankDisagree      = errors.New("RANK_DISAGREE")
	ErrTorchDeferredResidual  = errors.New("DEFERRED_RESIDUAL")
	ErrTorchUnsupportedLayout = errors.New("UNSUPPORTED_LAYOUT")
)

// TorchDumpOptions selects exactly one forward. PromptTokens, when supplied,
// are checked against every rank's exact int64 input_ids before sampling.
type TorchDumpOptions struct {
	Engine, Architecture string
	Step                 *int
	Rank                 int
	PromptTokens         []int
}

// TorchDumpResult includes an explicit reason for each deliberately skipped path.
type TorchDumpResult struct {
	Artifact Artifact
	Unmapped map[string]string
	Step     int
}
type torchDumpEntry struct {
	step, rank, index int
	name, path        string
}
type torchDumpKey struct {
	layer int
	stage string
}

// Keep exact full-tensor identity while retaining only bounded token samples.
type torchDumpSummary struct {
	shape   []int
	kind    string
	digest  [32]byte
	width   int
	indices []int
	values  []float32
}

func torchSummarize(t TorchTensor, tokens int) torchDumpSummary {
	s := torchDumpSummary{shape: append([]int(nil), t.Shape...), kind: t.kind, width: len(t.Values) / tokens}
	h := sha256.New()
	var raw [4]byte
	for _, v := range t.Values {
		binary.LittleEndian.PutUint32(raw[:], math.Float32bits(v))
		h.Write(raw[:])
	}
	copy(s.digest[:], h.Sum(nil))
	s.indices = ActivationSampleIndices(s.width, LlamaCppActivationSamples)
	s.values = make([]float32, 0, len(s.indices)*tokens)
	for token := 0; token < tokens; token++ {
		for _, index := range s.indices {
			s.values = append(s.values, t.Values[token*s.width+index])
		}
	}
	return s
}

func torchDumpErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrTorchDump}, args...)...)
}

func torchDumpTag(name string) (torchDumpEntry, error) {
	e := torchDumpEntry{}
	tags := map[string]string{}
	for _, tag := range strings.Split(strings.TrimSuffix(name, ".pt"), "___") {
		k, v, ok := strings.Cut(tag, "=")
		if !ok || tags[k] != "" {
			return e, fmt.Errorf("%w: filename tags", ErrTorchUnsupportedLayout)
		}
		tags[k] = v
	}
	for k, target := range map[string]*int{"step": &e.step, "rank": &e.rank, "dump_index": &e.index} {
		n, err := strconv.Atoi(tags[k])
		if err != nil || n < 0 {
			return e, fmt.Errorf("%w: missing tag %s", ErrTorchUnsupportedLayout, k)
		}
		*target = n
	}
	e.name = tags["name"]
	if e.name == "" {
		return e, fmt.Errorf("%w: missing name", ErrTorchUnsupportedLayout)
	}
	return e, nil
}

// ConvertTorchDump converts the DUMPER_* non-intrusive layout entirely offline.
// Each mapped stage must agree bit-for-bit across all observed ranks.
func ConvertTorchDump(dir string, opts TorchDumpOptions) (TorchDumpResult, error) {
	result := TorchDumpResult{Unmapped: map[string]string{}}
	m, ok := TorchStageMapFor(opts.Engine, opts.Architecture)
	if !ok {
		return result, fmt.Errorf("%w: %s:%s", ErrUnknownStageMap, opts.Engine, opts.Architecture)
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		return result, err
	}
	var entries []torchDumpEntry
	var materialized int
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if !strings.HasSuffix(f.Name(), ".pt") {
			continue
		}
		if f.Type()&os.ModeSymlink != 0 {
			return result, torchDumpErr("symlink dump")
		}
		e, err := torchDumpTag(f.Name())
		if err != nil {
			return result, err
		}
		e.path = filepath.Join(dir, f.Name())
		entries = append(entries, e)
	}
	if len(entries) == 0 {
		return result, ErrTorchUnsupportedLayout
	}
	if len(entries) > 100000 {
		return result, torchDumpErr("file limit")
	}
	sort.Slice(entries, func(i, j int) bool {
		a, b := entries[i], entries[j]
		if a.step != b.step {
			return a.step < b.step
		}
		if a.index != b.index {
			return a.index < b.index
		}
		return a.rank < b.rank
	})
	ids := map[int]map[int][]int{}
	ranks := map[int]map[int]bool{}
	readValue := func(e torchDumpEntry) (any, error) {
		v, err := ReadTorchPickle(e.path)
		if err != nil {
			return nil, err
		}
		d, ok := v.(map[string]any)
		if !ok {
			return nil, torchDumpErr("payload is not a dictionary")
		}
		value, ok := d["value"]
		if !ok {
			return nil, torchDumpErr("missing value")
		}
		if _, ok := d["meta"].(map[string]any); !ok {
			return nil, torchDumpErr("missing meta")
		}
		return value, nil
	}
	for _, e := range entries {
		if ranks[e.step] == nil {
			ranks[e.step] = map[int]bool{}
		}
		ranks[e.step][e.rank] = true
		if e.name != "input_ids" {
			continue
		}
		v, err := readValue(e)
		if err != nil {
			return result, err
		}
		tensor, ok := v.(TorchTensor)
		if !ok || len(tensor.IDs) == 0 {
			return result, torchDumpErr("input_ids must be int64")
		}
		materialized += len(tensor.IDs)
		if materialized > torchMaxItems {
			return result, torchDumpErr("dump element limit")
		}
		tokens := make([]int, len(tensor.IDs))
		for i, n := range tensor.IDs {
			if n < 0 || int64(int(n)) != n {
				return result, torchDumpErr("token range")
			}
			tokens[i] = int(n)
		}
		if ids[e.step] == nil {
			ids[e.step] = map[int][]int{}
		}
		if _, exists := ids[e.step][e.rank]; exists {
			return result, torchDumpErr("duplicate input_ids")
		}
		ids[e.step][e.rank] = tokens
	}
	var candidates []int
	for step, byRank := range ids {
		if opts.Step != nil && step != *opts.Step {
			continue
		}
		tokens, ok := byRank[opts.Rank]
		if !ok {
			continue
		}
		if len(opts.PromptTokens) > 0 && !reflect.DeepEqual(tokens, opts.PromptTokens) {
			continue
		}
		for rank := range ranks[step] {
			if !reflect.DeepEqual(tokens, byRank[rank]) {
				return result, fmt.Errorf("%w: input_ids", ErrTorchRankDisagree)
			}
		}
		candidates = append(candidates, step)
	}
	if len(candidates) != 1 {
		return result, torchDumpErr("expected exactly one matching step, found %d; supply --step or prompt ids", len(candidates))
	}
	result.Step = candidates[0]
	tokens := ids[result.Step][opts.Rank]
	a := Artifact{Schema: Schema, InputTokens: append([]int(nil), tokens...)}
	group := map[torchDumpKey]map[int]torchDumpSummary{}
	retainedSamples := 0
	sources := map[torchDumpKey]string{}
	maxLayer := -1
	var add func(string, any, int) error
	add = func(name string, v any, rank int) error {
		if list, ok := v.([]any); ok {
			for i, part := range list {
				if err := add(name+"."+strconv.Itoa(i), part, rank); err != nil {
					return err
				}
			}
			return nil
		}
		if strings.Contains(opts.Architecture, "DeepseekV4") && strings.HasPrefix(name, "non_intrusive__model.layers.") {
			parts := strings.Split(strings.TrimPrefix(name, "non_intrusive__model.layers."), ".")
			if len(parts) == 3 && parts[1] == "output" && parts[2] != "0" {
				if v != nil {
					return ErrTorchDeferredResidual
				}
			}
		}
		stage, layer, reason := m.classify(name)
		if stage == "" {
			result.Unmapped[name] = reason
			return nil
		}
		tensor, ok := v.(TorchTensor)
		if !ok || len(tensor.Values) == 0 {
			return torchDumpErr("mapped output is not a floating tensor: %s", name)
		}
		if len(tensor.Shape) < 2 || tensor.Shape[0] != len(tokens) {
			return torchDumpErr("mapped output token axis: %s", name)
		}
		summary := torchSummarize(tensor, len(tokens))
		retainedSamples += len(summary.values)
		if retainedSamples > torchMaxItems {
			return torchDumpErr("retained sample limit")
		}
		key := torchDumpKey{layer, stage}
		if group[key] == nil {
			group[key] = map[int]torchDumpSummary{}
			sources[key] = name
		}
		if _, exists := group[key][rank]; exists {
			return torchDumpErr("duplicate mapped stage: %s", name)
		}
		group[key][rank] = summary
		maxLayer = max(maxLayer, layer)
		return nil
	}
	for _, e := range entries {
		if e.step != result.Step || e.name == "input_ids" {
			continue
		}
		v, err := readValue(e)
		if err != nil {
			return result, err
		}
		if err = add(e.name, v, e.rank); err != nil {
			return result, err
		}
	}
	keys := make([]torchDumpKey, 0, len(group))
	for key := range group {
		keys = append(keys, key)
	}
	if len(keys) > (1<<16)/len(tokens) {
		return result, torchDumpErr("retained event limit")
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].layer != keys[j].layer {
			return keys[i].layer < keys[j].layer
		}
		return keys[i].stage < keys[j].stage
	})
	for _, key := range keys {
		byRank := group[key]
		tensor, ok := byRank[opts.Rank]
		if !ok {
			return result, fmt.Errorf("%w: missing selected rank", ErrTorchRankDisagree)
		}
		for rank := range ranks[result.Step] {
			other, ok := byRank[rank]
			if !ok || tensor.kind != other.kind || !reflect.DeepEqual(tensor.shape, other.shape) || tensor.digest != other.digest {
				return result, fmt.Errorf("%w: %s rank %d", ErrTorchRankDisagree, sources[key], rank)
			}
		}
		width := tensor.width
		indices := tensor.indices
		for token := range tokens {
			values := make([]float32, len(indices))
			copy(values, tensor.values[token*len(indices):(token+1)*len(indices)])
			a.Events = append(a.Events, Event{Sequence: int64(len(a.Events) + 1), Operation: opts.Engine + ".dump", Phase: key.stage, Backend: opts.Engine, Device: "reference", Layer: key.layer, Token: token, InputDigest: Digest("torch-dump-input", fmt.Sprint(tokens)), Shapes: [][]int{{width}}, SampleIndices: append([]int(nil), indices...), SampleValues: values, ProvenanceDigest: Digest(opts.Engine+"-dump", sources[key], strconv.Itoa(result.Step)), Status: "ok"})
			a.RetainedSampleValues += uint64(len(values))
		}
	}
	if len(a.Events) == 0 {
		return result, torchDumpErr("no mapped activations")
	}
	for i := range a.Events {
		if a.Events[i].Layer == llamaCppPostLayer {
			a.Events[i].Layer = maxLayer + 1
		}
	}
	result.Artifact = a
	return result, nil
}
