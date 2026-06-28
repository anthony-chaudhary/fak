package modelladder

import (
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Spec describes one rung of the auto-detected model ladder.
type Spec struct {
	Name    string // display label
	Dir     string // weights dir
	Kind    string // "dir" (fak export) or "hf" (safetensors snapshot, lean Q8 load)
	Params  string // human param count, for the curve x-axis
	Order   int    // ladder order
	Present bool
}

// LadderSpecs returns the candidate model table (135m → 0.5B → 1.5B → 3B),
// auto-detected from disk: each rung's Present is set by stat-ing its directory
// (resolved against the working directory when the path is relative), and the
// rungs are returned sorted by Order.
func LadderSpecs() []Spec {
	home, _ := os.UserHomeDir()
	hf := filepath.Join(home, ".cache", "fak-models", "hf")
	candidates := []Spec{
		{Name: "SmolLM2-135M", Params: "0.14B", Order: 0, Kind: "dir", Dir: "internal/model/.cache/smollm2-135m"},
		{Name: "Qwen2.5-0.5B", Params: "0.5B", Order: 1, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-0.5B-Instruct")},
		{Name: "Qwen2.5-1.5B", Params: "1.5B", Order: 2, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-1.5B-Instruct")},
		{Name: "Qwen2.5-3B", Params: "3B", Order: 3, Kind: "hf", Dir: filepath.Join(hf, "Qwen2.5-3B-Instruct")},
	}
	wd, _ := os.Getwd()
	for i := range candidates {
		dir := candidates[i].Dir
		if !filepath.IsAbs(dir) {
			dir = filepath.Join(wd, dir)
		}
		_, err := os.Stat(dir)
		candidates[i].Present = err == nil
		candidates[i].Dir = dir
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Order < candidates[j].Order })
	return candidates
}

// SmallestPresent returns the first present rung in ladder order (the smallest
// model on disk), or false if no rung is present.
func SmallestPresent(specs []Spec) (Spec, bool) {
	for _, s := range specs {
		if s.Present {
			return s, true
		}
	}
	return Spec{}, false
}

// Loaded is a model loaded + quantized once and memoized by the Registry.
type Loaded struct {
	M     *model.Model
	Vocab int
	Name  string
}

// Registry lazily loads + quantizes each model once, memoizing by name.
type Registry struct {
	mu    sync.Mutex
	cache map[string]*Loaded
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry { return &Registry{cache: map[string]*Loaded{}} }

// Get returns the loaded + quantized model for s, loading and memoizing it on
// first use.
func (r *Registry) Get(s Spec) (*Loaded, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if l, ok := r.cache[s.Name]; ok {
		return l, nil
	}
	m, err := LoadSpec(s)
	if err != nil {
		return nil, err
	}
	m.Quantize()
	l := &Loaded{M: m, Vocab: m.Cfg.VocabSize, Name: s.Name}
	r.cache[s.Name] = l
	return l, nil
}

// LoadSpec loads a model from a spec: "hf" reads the HF config and loads the
// (sharded or single) safetensors with a lean Q8 quantization; anything else
// loads a fak export directory.
func LoadSpec(s Spec) (*model.Model, error) {
	switch s.Kind {
	case "hf":
		cfg, err := benchcli.ReadHFConfig(s.Dir)
		if err != nil {
			return nil, err
		}
		// sharded vs single safetensors
		if _, err := os.Stat(filepath.Join(s.Dir, "model.safetensors.index.json")); err == nil {
			return model.LoadSafetensorsQuantDir(s.Dir, cfg)
		}
		return model.LoadSafetensorsQuant(filepath.Join(s.Dir, "model.safetensors"), cfg)
	default:
		return model.Load(s.Dir)
	}
}
