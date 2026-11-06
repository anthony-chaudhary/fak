package model

import (
	"fmt"
	"path/filepath"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// nvfp4Weight is one resident ModelOpt NVFP4 projection: the packed weight is
// owned by be, which must implement compute.NVFP4MatMulBackend.
type nvfp4Weight struct {
	be      compute.Backend
	w       compute.Tensor
	out, in int
}

// LoadModelOptNVFP4Dir loads a ModelOpt NVFP4 safetensors directory (sharded or
// single-file). Float tensors land in the f32 manifest; every packed NVFP4
// triple becomes a resident projection owned by be (compute.Default() when nil)
// and is served through compute.MatMulNVFP4 by residentMatRows. A backend
// without the native W4A4 capability, or one whose policy refuses it (e.g. an
// sm_120 device below cuBLAS 12.8.4), is refused at load with
// compute.ErrNVFP4GEMMUnsupported rather than silently widened.
func LoadModelOptNVFP4Dir(dir string, be compute.Backend) (*Model, error) {
	var cfg Config
	if err := readJSON(filepath.Join(dir, "config.json"), &cfg); err != nil {
		return nil, fmt.Errorf("modelopt load config: %w", err)
	}
	if err := refuseDeepSeekV41Native(cfg); err != nil {
		return nil, err
	}
	if _, ok, err := ReadModelOptNVFP4Config(dir); err != nil {
		return nil, err
	} else if !ok {
		return nil, fmt.Errorf("%w: %s declares no ModelOpt NVFP4 quantization", ErrModelOptNVFP4Config, dir)
	}
	if be == nil {
		be = compute.Default()
	}
	if err := admitNVFP4Projection(be); err != nil {
		return nil, err
	}
	ck, err := OpenModelOptNVFP4Checkpoint(dir, cfg, be)
	if err != nil {
		return nil, err
	}
	defer ck.Close()

	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	for _, shard := range sortedShardKeys(ck.sources) {
		sf := ck.sources[shard].file
		skip := map[string]bool{}
		for _, name := range safetensorsTensorNames(sf.hdr) {
			if ck.ownsCheckpointTensor(name) {
				skip[name] = true
			}
		}
		if err := appendGPTQF32FileInto(sf, skip, man, &raw, &off, cfg); err != nil {
			return nil, fmt.Errorf("modelopt shard %s: %w", shard, err)
		}
	}
	m, err := newHFCheckpointModel(cfg, man, raw)
	if err != nil {
		return nil, err
	}
	m.nvfp4w = make(map[string]*nvfp4Weight, len(ck.byName))
	for _, name := range ck.Names() {
		host, err := ck.Weight(name, compute.NVFP4)
		if err != nil {
			m.freeNVFP4()
			return nil, err
		}
		w := be.Upload(host, compute.NVFP4)
		m.nvfp4w[name] = &nvfp4Weight{be: be, w: w, out: host.Shape[0], in: host.Shape[1]}
	}
	return m, nil
}

func sortedShardKeys(m map[string]*ModelOptNVFP4Source) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// admitNVFP4Projection proves be executes a W4A4 product before any weight is
// uploaded, so an unsupported device fails at load with a typed error.
func admitNVFP4Projection(be compute.Backend) error {
	if _, ok := be.(compute.NVFP4MatMulBackend); !ok {
		return fmt.Errorf("%w: backend %q has no native NVFP4 W4A4 capability", compute.ErrNVFP4GEMMUnsupported, be.Name())
	}
	const out, k = 4, 32
	wHost, err := compute.QuantizeNVFP4(be, []int{out, k}, make([]float32, out*k))
	if err != nil {
		return err
	}
	aHost, err := compute.QuantizeNVFP4(be, []int{1, k}, make([]float32, k))
	if err != nil {
		return err
	}
	w, a := be.Upload(wHost, compute.NVFP4), be.Upload(aHost, compute.NVFP4)
	defer be.Free(w)
	defer be.Free(a)
	y, err := compute.MatMulNVFP4(be, w, a)
	if err != nil {
		return fmt.Errorf("model: NVFP4 projection admission on %q: %w", be.Name(), err)
	}
	be.Free(y)
	return nil
}

// nvfp4MatRows computes y = W·x for one token: x is quantized to NVFP4 with a
// dynamic per-tensor global scale (ModelOpt's static input_scale is not yet
// consumed) and multiplied through compute.MatMulNVFP4. Errors, including
// ErrNVFP4GEMMUnsupported, panic like every other resident-dispatch failure.
func nvfp4MatRows(name string, qt *nvfp4Weight, x []float32) []float32 {
	host, err := compute.QuantizeNVFP4(qt.be, []int{1, qt.in}, x[:qt.in])
	if err != nil {
		panic(fmt.Errorf("model: NVFP4 activation for %s: %w", name, err))
	}
	a := qt.be.Upload(host, compute.NVFP4)
	defer qt.be.Free(a)
	y, err := compute.MatMulNVFP4(qt.be, qt.w, a)
	if err != nil {
		panic(fmt.Errorf("model: NVFP4 projection %s: %w", name, err))
	}
	defer qt.be.Free(y)
	return append([]float32(nil), qt.be.Read(y)...)
}

// NVFP4Count reports how many projections are resident as NVFP4.
func (m *Model) NVFP4Count() int { return len(m.nvfp4w) }

func (m *Model) freeNVFP4() {
	for _, qt := range m.nvfp4w {
		qt.be.Free(qt.w)
	}
	m.nvfp4w = nil
}
