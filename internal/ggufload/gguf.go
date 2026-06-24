// Package ggufload parses GGUF metadata and tensor directories for off-path model loading.
package ggufload

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	Magic          = "GGUF"
	Version        = 3
	defaultAlign   = 32
	maxStringBytes = 1 << 30
	qk4            = 32
	qk5            = 32
	qk8_0          = 32
	qkMXFP4        = 32
	qkK            = 256
	kScaleSize     = 12
	blockQ4_0Bytes = 2 + qk4/2
	blockQ4_1Bytes = 2 + 2 + qk4/2
	blockQ5_0Bytes = 2 + 4 + qk5/2
	blockQ5_1Bytes = 2 + 2 + 4 + qk5/2
	blockQ8_0Bytes = 2 + qk8_0
	blockQ2KBytes  = qkK/16 + qkK/4 + 2 + 2
	blockQ3KBytes  = qkK/8 + qkK/4 + kScaleSize + 2
	blockQ4KBytes  = 2 + 2 + kScaleSize + qkK/2
	blockQ5KBytes  = 2 + 2 + kScaleSize + qkK/8 + qkK/2
	blockQ6KBytes  = qkK/2 + qkK/4 + qkK/16 + 2
	// MXFP4 (gpt-oss): a 1-byte E8M0 shared scale + qkMXFP4/2 bytes of packed
	// 4-bit E2M1 codes (two per byte) = 17 bytes per 32-element block.
	blockMXFP4Bytes = 1 + qkMXFP4/2
)

type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

type TensorType uint32

const (
	TensorF32   TensorType = 0
	TensorF16   TensorType = 1
	TensorQ4_0  TensorType = 2
	TensorQ4_1  TensorType = 3
	TensorQ5_0  TensorType = 6
	TensorQ5_1  TensorType = 7
	TensorQ8_0  TensorType = 8
	TensorQ2_K  TensorType = 10
	TensorQ3_K  TensorType = 11
	TensorQ4_K  TensorType = 12
	TensorQ5_K  TensorType = 13
	TensorQ6_K  TensorType = 14
	TensorBF16  TensorType = 30
	TensorMXFP4 TensorType = 39
)

type Value struct {
	Type  ValueType
	Value any
}

type TensorInfo struct {
	Name       string
	Dims       []uint64
	Type       TensorType
	Offset     uint64
	FileOffset int64
}

func (t TensorType) String() string {
	switch t {
	case TensorF32:
		return "F32"
	case TensorF16:
		return "F16"
	case TensorQ4_0:
		return "Q4_0"
	case TensorQ4_1:
		return "Q4_1"
	case TensorQ5_0:
		return "Q5_0"
	case TensorQ5_1:
		return "Q5_1"
	case TensorQ8_0:
		return "Q8_0"
	case TensorQ2_K:
		return "Q2_K"
	case TensorQ3_K:
		return "Q3_K"
	case TensorQ4_K:
		return "Q4_K"
	case TensorQ5_K:
		return "Q5_K"
	case TensorQ6_K:
		return "Q6_K"
	case TensorBF16:
		return "BF16"
	case TensorMXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("TensorType(%d)", t)
	}
}

type File struct {
	Version          uint32
	Metadata         map[string]Value
	Tensors          []TensorInfo
	Alignment        uint64
	TensorDataOffset int64
}

type WeightSource struct {
	File *File
	r    io.ReaderAt
	// readerFor (parallel to File.Tensors) routes tensor i's bytes to the shard
	// file that actually holds them. A nil entry falls back to r, which preserves
	// the original single-file behaviour. sizeFor[i] is readerFor[i]'s file size,
	// used for the overrun bounds check.
	readerFor []io.ReaderAt
	sizeFor   []int64
	// closers holds every open shard file; Close shuts them all. For a single-file
	// checkpoint this is exactly one entry.
	closers []io.Closer
	size    int64
	byName  map[string]int
}

// LoadPhaseStat is one aggregate phase in the GGUF quant-on-load path.
type LoadPhaseStat struct {
	Phase   string  `json:"phase"`
	Calls   int     `json:"calls"`
	Tensors int     `json:"tensors,omitempty"`
	Bytes   int64   `json:"bytes,omitempty"`
	Nanos   int64   `json:"nanos"`
	MS      float64 `json:"ms"`
	TimePct float64 `json:"time_pct"`
}

// LoadTensorStat records per-tensor timing so a 27B load profile can identify the
// specific tensor(s) causing page churn or allocator pressure.
type LoadTensorStat struct {
	Name           string  `json:"name"`
	CanonicalName  string  `json:"canonical_name"`
	Type           string  `json:"type"`
	Shape          []int   `json:"shape,omitempty"`
	PayloadBytes   int64   `json:"payload_bytes,omitempty"`
	Values         int     `json:"values,omitempty"`
	ReadNanos      int64   `json:"read_nanos,omitempty"`
	DequantNanos   int64   `json:"dequant_nanos,omitempty"`
	NormalizeNanos int64   `json:"normalize_nanos,omitempty"`
	AddNanos       int64   `json:"add_nanos,omitempty"`
	TotalNanos     int64   `json:"total_nanos"`
	TotalMS        float64 `json:"total_ms"`
}

// LoadProfile is a machine-readable load-phase report for modelbench. It is scoped
// to the pure GGUF->resident-model path, not tokenizer or inference.
type LoadProfile struct {
	Mode        string           `json:"mode"`
	Source      string           `json:"source,omitempty"`
	TensorCount int              `json:"tensor_count"`
	TotalNanos  int64            `json:"total_nanos"`
	TotalMS     float64          `json:"total_ms"`
	Phases      []LoadPhaseStat  `json:"phases"`
	TopTensors  []LoadTensorStat `json:"top_tensors,omitempty"`
	Bottleneck  string           `json:"bottleneck"`
}

// LoadProfiler records opt-in GGUF load timings. Nil keeps the loader on its
// existing behavior with no timing or per-tensor bookkeeping.
type LoadProfiler struct {
	stat    map[string]*LoadPhaseStat
	order   []string
	tensors []LoadTensorStat
	TopN    int
	Trace   io.Writer
	Every   int
}

func NewLoadProfiler() *LoadProfiler {
	return &LoadProfiler{stat: map[string]*LoadPhaseStat{}, TopN: 16}
}

func loadProfileStart(p *LoadProfiler) time.Time {
	if p == nil {
		return time.Time{}
	}
	return time.Now()
}

func loadProfileEnd(p *LoadProfiler, phase string, start time.Time, bytes int64, tensors int) int64 {
	if p == nil || start.IsZero() {
		return 0
	}
	nanos := time.Since(start).Nanoseconds()
	p.record(phase, nanos, bytes, tensors)
	return nanos
}

func (p *LoadProfiler) record(phase string, nanos, bytes int64, tensors int) {
	if p == nil {
		return
	}
	st := p.stat[phase]
	if st == nil {
		st = &LoadPhaseStat{Phase: phase}
		p.stat[phase] = st
		p.order = append(p.order, phase)
	}
	st.Calls++
	st.Tensors += tensors
	st.Bytes += bytes
	st.Nanos += nanos
}

func (p *LoadProfiler) recordTensor(st LoadTensorStat) {
	if p == nil {
		return
	}
	st.TotalMS = float64(st.TotalNanos) / 1e6
	p.tensors = append(p.tensors, st)
	if p.Trace != nil {
		n := len(p.tensors)
		every := p.Every
		if every <= 0 {
			every = 1
		}
		if n == 1 || n%every == 0 {
			fmt.Fprintf(p.Trace, "[gguf load] tensor %d %s -> %s type=%s payload=%.1fMB total=%.1fms read=%.1fms dequant=%.1fms normalize=%.1fms add=%.1fms\n",
				n, st.Name, st.CanonicalName, st.Type, float64(st.PayloadBytes)/1e6,
				st.TotalMS, float64(st.ReadNanos)/1e6, float64(st.DequantNanos)/1e6,
				float64(st.NormalizeNanos)/1e6, float64(st.AddNanos)/1e6)
		}
	}
}

func (p *LoadProfiler) Snapshot(mode, source string, totalNanos int64) *LoadProfile {
	if p == nil {
		return nil
	}
	denom := totalNanos
	if denom <= 0 {
		for _, st := range p.stat {
			denom += st.Nanos
		}
	}
	out := &LoadProfile{
		Mode:        mode,
		Source:      source,
		TensorCount: len(p.tensors),
		TotalNanos:  totalNanos,
		TotalMS:     float64(totalNanos) / 1e6,
	}
	for _, key := range p.order {
		src := p.stat[key]
		st := *src
		st.MS = float64(st.Nanos) / 1e6
		if denom > 0 {
			st.TimePct = 100 * float64(st.Nanos) / float64(denom)
		}
		out.Phases = append(out.Phases, st)
	}
	sort.Slice(out.Phases, func(i, j int) bool { return out.Phases[i].Nanos > out.Phases[j].Nanos })
	if len(out.Phases) > 0 {
		out.Bottleneck = out.Phases[0].Phase
	}
	top := append([]LoadTensorStat(nil), p.tensors...)
	sort.Slice(top, func(i, j int) bool { return top[i].TotalNanos > top[j].TotalNanos })
	n := p.TopN
	if n <= 0 || n > len(top) {
		n = len(top)
	}
	out.TopTensors = top[:n]
	return out
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

func OpenWeights(path string) (*WeightSource, error) {
	f, gg, size, err := openAndRead(path)
	if err != nil {
		return nil, err
	}

	splitCount, hasSplit := gg.Uint64("split.count")
	if !hasSplit || splitCount <= 1 {
		// Single-file checkpoint.
		ws, err := NewWeightSource(gg, f, size)
		if err != nil {
			_ = f.Close()
			return nil, err
		}
		ws.closers = []io.Closer{f}
		return ws, nil
	}

	// Split checkpoint: shard 1 carries the model config (general.architecture
	// and friends); later shards carry only split.* metadata plus their tensor
	// subset. The config-carrying shard is identified by general.architecture
	// presence — NOT by split.no, which is 0-indexed in HuggingFace's split
	// writer and 1-indexed in llama.cpp's. If the caller handed us a later shard
	// (no architecture), close it and rebuild from shard 1's path.
	if _, hasArch := gg.String("general.architecture"); !hasArch {
		_ = f.Close()
		shard1, err := firstShardPath(path)
		if err != nil {
			return nil, fmt.Errorf("gguf: %s is a split shard but its name is not a shard path: %w", filepath.Base(path), err)
		}
		return openWeightsSplit(shard1, int(splitCount))
	}
	return openWeightsSplitFromFirst(path, int(splitCount), f, gg, size)
}

// openAndRead opens a GGUF file, parses its header, and returns the still-open
// file, the parsed File, and the file size. The caller owns the returned file
// and must Close it.
func openAndRead(path string) (*os.File, *File, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, 0, err
	}
	gg, err := Read(f)
	if err != nil {
		_ = f.Close()
		return nil, nil, 0, err
	}
	return f, gg, st.Size(), nil
}

// shardSuffixRe matches the "-NNNNN-of-MMMMM.gguf" shard suffix that the
// HuggingFace GGUF split writer produces (zero-padded to a fixed width). The
// two capture groups are the shard number and the total count, used to derive
// sibling shard paths while preserving the original padding width.
var shardSuffixRe = regexp.MustCompile(`-(\d+)-of-(\d+)\.gguf$`)

// firstShardPath rewrites a shard path's -N-of-M.gguf suffix to -1-of-M.gguf so
// the caller can open the config-carrying first shard.
func firstShardPath(path string) (string, error) {
	loc := shardSuffixRe.FindStringSubmatchIndex(path)
	if loc == nil {
		return "", fmt.Errorf("no -N-of-M.gguf shard suffix in %q", path)
	}
	// Rebuild the suffix with shard number 1, preserving the original width.
	numStr := path[loc[2]:loc[3]]
	countStr := path[loc[4]:loc[5]]
	width := len(numStr)
	prefix := path[:loc[2]]
	return fmt.Sprintf("%s%0*d-of-%s.gguf", prefix, width, 1, countStr), nil
}

// shardPaths expands shard 1's path into the full ordered shard-1..shard-N list,
// preserving the zero-padding width and total count encoded in the filename.
func shardPaths(shard1Path string, count int) ([]string, error) {
	loc := shardSuffixRe.FindStringSubmatchIndex(shard1Path)
	if loc == nil {
		return nil, fmt.Errorf("no -N-of-M.gguf shard suffix in %q", shard1Path)
	}
	numStr := shard1Path[loc[2]:loc[3]]
	countStr := shard1Path[loc[4]:loc[5]]
	width := len(numStr)
	prefix := shard1Path[:loc[2]]
	paths := make([]string, count)
	for i := 0; i < count; i++ {
		paths[i] = fmt.Sprintf("%s%0*d-of-%s.gguf", prefix, width, i+1, countStr)
	}
	return paths, nil
}

// openWeightsSplit opens shard 1 fresh and assembles the merged WeightSource.
func openWeightsSplit(shard1Path string, count int) (*WeightSource, error) {
	f, gg, size, err := openAndRead(shard1Path)
	if err != nil {
		return nil, err
	}
	// The config carrier is identified by general.architecture, not split.no
	// (whose index base differs between HuggingFace and llama.cpp writers).
	if _, ok := gg.String("general.architecture"); !ok {
		_ = f.Close()
		return nil, fmt.Errorf("gguf: %s expected to be the config-carrying shard 1, but general.architecture is absent", shard1Path)
	}
	return openWeightsSplitFromFirst(shard1Path, count, f, gg, size)
}

// validShardNo reports whether declared split.no is consistent with shard i
// (1-indexed by filename) under either convention: HuggingFace writes a 0-indexed
// split.no (so shard i => i-1) and llama.cpp writes a 1-indexed one (shard i => i).
// An absent split.no is treated as consistent (older/unknown writers).
func validShardNo(declared uint64, present bool, i int) bool {
	if !present {
		return true
	}
	return int(declared) == i-1 || int(declared) == i
}

// openWeightsSplitFromFirst assembles the merged WeightSource given an
// already-open shard 1. It opens shards 2..N, merges their tensor directories,
// and records which shard reader serves each tensor.
func openWeightsSplitFromFirst(shard1Path string, count int, shard1File *os.File, shard1GG *File, shard1Size int64) (*WeightSource, error) {
	paths, err := shardPaths(shard1Path, count)
	if err != nil {
		_ = shard1File.Close()
		return nil, err
	}
	if len(paths) == 0 || paths[0] != shard1Path {
		_ = shard1File.Close()
		return nil, fmt.Errorf("gguf: shard path derivation mismatch (%s vs %s)", paths[0], shard1Path)
	}

	// Merge view: shard 1's metadata/config + tensors from every shard, in order.
	tensors := make([]TensorInfo, 0, len(shard1GG.Tensors))
	readerFor := make([]io.ReaderAt, 0, len(shard1GG.Tensors))
	sizeFor := make([]int64, 0, len(shard1GG.Tensors))
	seen := make(map[string]bool, len(shard1GG.Tensors))
	for _, t := range shard1GG.Tensors {
		if seen[t.Name] {
			_ = shard1File.Close()
			return nil, fmt.Errorf("gguf: duplicate tensor %s within shard 1", t.Name)
		}
		seen[t.Name] = true
		tensors = append(tensors, t)
		readerFor = append(readerFor, shard1File)
		sizeFor = append(sizeFor, shard1Size)
	}
	closers := []io.Closer{shard1File}

	for i := 2; i <= count; i++ {
		p := paths[i-1]
		f, gg, sz, err := openAndRead(p)
		if err != nil {
			closeAll(closers)
			return nil, fmt.Errorf("gguf: open shard %d (%s): %w", i, p, err)
		}
		closers = append(closers, f)
		if no, ok := gg.Uint64("split.no"); !validShardNo(no, ok, i) {
			closeAll(closers)
			return nil, fmt.Errorf("gguf: shard %s declares split.no=%d, want %d or %d", p, no, i-1, i)
		}
		for _, t := range gg.Tensors {
			if seen[t.Name] {
				closeAll(closers)
				return nil, fmt.Errorf("gguf: duplicate tensor %s across shards", t.Name)
			}
			seen[t.Name] = true
			tensors = append(tensors, t)
			readerFor = append(readerFor, f)
			sizeFor = append(sizeFor, sz)
		}
	}

	merged := *shard1GG
	merged.Tensors = tensors
	ws, err := NewWeightSource(&merged, shard1File, shard1Size)
	if err != nil {
		closeAll(closers)
		return nil, err
	}
	ws.readerFor = readerFor
	ws.sizeFor = sizeFor
	ws.closers = closers
	return ws, nil
}

func closeAll(closers []io.Closer) {
	for _, c := range closers {
		_ = c.Close()
	}
}

// LoadModel loads a GGUF checkpoint through the default dequant-to-f32 path and returns a
// regular in-kernel model.Model. GGUF tensor names are normalized to the canonical HF-Llama
// names that internal/model already consumes.
func LoadModel(path string) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.Model()
}

// LoadModelQuant loads a GGUF checkpoint through the memory-lean quant-on-load path:
// each tensor is dequantized only long enough to normalize/quantize it, resident matmul
// weights are kept as Q8_0, and only small non-matmul tensors remain f32.
func LoadModelQuant(path string) (*model.Model, error) {
	return LoadModelQuantProfile(path, nil)
}

func LoadModelQuantProfile(path string, p *LoadProfiler) (*model.Model, error) {
	t := loadProfileStart(p)
	ws, err := OpenWeights(path)
	loadProfileEnd(p, "gguf_open_index", t, 0, 0)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.QuantModelProfile(p)
}

func NewWeightSource(f *File, r io.ReaderAt, size int64) (*WeightSource, error) {
	byName := make(map[string]int, len(f.Tensors))
	for i, t := range f.Tensors {
		if _, ok := byName[t.Name]; ok {
			return nil, fmt.Errorf("gguf: duplicate tensor %s", t.Name)
		}
		byName[t.Name] = i
	}
	return &WeightSource{File: f, r: r, size: size, byName: byName}, nil
}

func (s *WeightSource) Close() error {
	if len(s.closers) == 0 {
		return nil
	}
	var firstErr error
	for _, c := range s.closers {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.closers = nil
	return firstErr
}

func (s *WeightSource) Tensor(name string) (TensorInfo, bool) {
	i, ok := s.byName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return s.File.Tensors[i], true
}

func (s *WeightSource) TensorBytes(name string) ([]byte, TensorInfo, error) {
	info, ok := s.Tensor(name)
	if !ok {
		return nil, TensorInfo{}, fmt.Errorf("gguf: missing tensor %s", name)
	}
	n, err := tensorPayloadBytes(info)
	if err != nil {
		return nil, info, err
	}
	if n > uint64(math.MaxInt) || n > uint64(math.MaxInt64) {
		return nil, info, fmt.Errorf("gguf: tensor %s payload is too large", name)
	}
	// Route to the shard reader that holds this tensor's bytes. For a single-file
	// checkpoint readerFor is nil and we read from the primary reader, as before.
	r, sz := s.r, s.size
	if idx, ok := s.byName[name]; ok && idx < len(s.readerFor) && s.readerFor[idx] != nil {
		r = s.readerFor[idx]
		sz = s.sizeFor[idx]
	}
	if info.FileOffset < 0 || info.FileOffset > math.MaxInt64-int64(n) || info.FileOffset+int64(n) > sz {
		return nil, info, fmt.Errorf("gguf: tensor %s overruns file", name)
	}
	buf := make([]byte, int(n))
	if _, err := r.ReadAt(buf, info.FileOffset); err != nil {
		return nil, info, fmt.Errorf("gguf: read tensor %s: %w", name, err)
	}
	return buf, info, nil
}

func (s *WeightSource) TensorF32(name string) ([]float32, TensorInfo, error) {
	raw, info, err := s.TensorBytes(name)
	if err != nil {
		return nil, info, err
	}
	out, err := dequantF32(info, raw)
	return out, info, err
}

func (s *WeightSource) Model() (*model.Model, error) {
	cfg, tensors, err := s.F32Tensors()
	if err != nil {
		return nil, err
	}
	return model.NewFromF32Tensors(cfg, tensors)
}

func (s *WeightSource) QuantModel() (*model.Model, error) {
	return s.QuantModelProfile(nil)
}

func (s *WeightSource) QuantModelProfile(p *LoadProfiler) (*model.Model, error) {
	t := loadProfileStart(p)
	cfg, err := s.File.Config()
	loadProfileEnd(p, "gguf_config", t, 0, 0)
	if err != nil {
		return nil, err
	}
	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	for _, info := range s.File.Tensors {
		var tensorStart time.Time
		var tt LoadTensorStat
		if p != nil {
			tensorStart = time.Now()
			tt = LoadTensorStat{Name: info.Name, Type: info.Type.String()}
		}

		t = loadProfileStart(p)
		name, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		}
		if p != nil {
			tt.CanonicalName = name
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
		}
		if p != nil {
			tt.Shape = append([]int(nil), shape...)
		}
		loadProfileEnd(p, "gguf_map_shape", t, 0, 1)

		t = loadProfileStart(p)
		raw, _, err := s.TensorBytes(info.Name)
		readNanos := loadProfileEnd(p, "gguf_read", t, int64(len(raw)), 1)
		if p != nil {
			tt.ReadNanos = readNanos
			tt.PayloadBytes = int64(len(raw))
		}
		if err != nil {
			return nil, err
		}

		t = loadProfileStart(p)
		data, err := dequantF32(info, raw)
		dequantNanos := loadProfileEnd(p, "gguf_dequant", t, int64(len(data))*4, 1)
		if p != nil {
			tt.DequantNanos = dequantNanos
			tt.Values = len(data)
		}
		if err != nil {
			return nil, err
		}

		t = loadProfileStart(p)
		data, err = normalizeCanonicalTensorData(name, data, cfg)
		normalizeNanos := loadProfileEnd(p, "gguf_normalize", t, int64(len(data))*4, 1)
		if p != nil {
			tt.NormalizeNanos = normalizeNanos
		}
		if err != nil {
			return nil, err
		}

		t = loadProfileStart(p)
		if err := builder.AddF32Tensor(name, shape, data); err != nil {
			addNanos := loadProfileEnd(p, "quant_builder_add", t, int64(len(data))*4, 1)
			if p != nil {
				tt.AddNanos = addNanos
			}
			return nil, err
		}
		addNanos := loadProfileEnd(p, "quant_builder_add", t, int64(len(data))*4, 1)
		if p != nil {
			tt.AddNanos = addNanos
			tt.TotalNanos = time.Since(tensorStart).Nanoseconds()
			p.recordTensor(tt)
		}
	}
	t = loadProfileStart(p)
	m, err := builder.Build()
	loadProfileEnd(p, "quant_builder_finalize", t, 0, 0)
	if err != nil {
		return nil, err
	}
	return m, nil
}

func (s *WeightSource) F32Tensors() (model.Config, []model.NamedTensorF32, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return model.Config{}, nil, err
	}
	tensors := make([]model.NamedTensorF32, 0, len(s.File.Tensors))
	for _, info := range s.File.Tensors {
		name, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return model.Config{}, nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return model.Config{}, nil, err
		}
		data, _, err := s.TensorF32(info.Name)
		if err != nil {
			return model.Config{}, nil, err
		}
		data, err = normalizeCanonicalTensorData(name, data, cfg)
		if err != nil {
			return model.Config{}, nil, err
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	return cfg, tensors, nil
}

func normalizeCanonicalTensorData(name string, data []float32, cfg model.Config) ([]float32, error) {
	if cfg.IsQwen35Hybrid() {
		if out, handled, err := normalizeQwen35OrdinaryNormTensor(name, data, cfg); handled || err != nil {
			return out, err
		}
		if out, handled, err := normalizeQwen35LinearTensor(name, data, cfg); handled || err != nil {
			return out, err
		}
	}
	switch {
	case strings.HasSuffix(name, ".self_attn.q_proj.weight"):
		if cfg.IsQwen35Hybrid() && cfg.AttnOutputGate {
			return unpermuteQwen35GatedQTensor(name, data, cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
		}
		if ggufArchStoresHFRotaryLayout(cfg.ModelType) {
			return data, nil
		}
		return unpermuteRotaryTensor(name, data, cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
	case strings.HasSuffix(name, ".self_attn.k_proj.weight"):
		if ggufArchStoresHFRotaryLayout(cfg.ModelType) {
			return data, nil
		}
		return unpermuteRotaryTensor(name, data, cfg.NumKVHeads, cfg.HeadDim, cfg.HiddenSize)
	default:
		return data, nil
	}
}

func normalizeQwen35OrdinaryNormTensor(name string, src []float32, cfg model.Config) ([]float32, bool, error) {
	want := 0
	switch {
	case name == "model.norm.weight":
		want = cfg.HiddenSize
	case strings.HasSuffix(name, ".input_layernorm.weight"),
		strings.HasSuffix(name, ".post_attention_layernorm.weight"):
		want = cfg.HiddenSize
	case strings.HasSuffix(name, ".self_attn.q_norm.weight"),
		strings.HasSuffix(name, ".self_attn.k_norm.weight"):
		want = cfg.HeadDim
	default:
		return nil, false, nil
	}
	if want > 0 && len(src) != want {
		return nil, true, fmt.Errorf("gguf: tensor %s has %d values, qwen35 norm wants %d", name, len(src), want)
	}
	return subtractOneFromTensor(src), true, nil
}

func subtractOneFromTensor(src []float32) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = v - 1
	}
	return dst
}

func normalizeQwen35LinearTensor(name string, src []float32, cfg model.Config) ([]float32, bool, error) {
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	if nK <= 0 || nV <= 0 || kHd <= 0 || vHd <= 0 {
		return nil, false, nil
	}
	if nV%nK != 0 {
		return nil, true, fmt.Errorf("gguf: qwen35 linear heads are not divisible: value=%d key=%d", nV, nK)
	}
	if nV == nK {
		return nil, false, nil
	}
	if kHd != vHd {
		return nil, true, fmt.Errorf("gguf: qwen35 linear key/value head dims differ: key=%d value=%d", kHd, vHd)
	}
	keyDim := nK * kHd
	switch {
	case strings.HasSuffix(name, ".linear_attn.in_proj_qkv.weight"),
		strings.HasSuffix(name, ".self_attn.qkv_proj.weight"):
		out, err := reorderQwen35LinearQKVRows(name, src, keyDim, nK, nV, vHd, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.conv1d.weight"):
		out, err := reorderQwen35LinearQKVRows(name, src, keyDim, nK, nV, vHd, cfg.LinearConvKernelDim)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.in_proj_z.weight"),
		strings.HasSuffix(name, ".self_attn.q_gate_proj.weight"):
		out, err := reorderQwen35InterleavedValueRows(name, src, nK, nV, vHd, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"):
		out, err := reorderQwen35InterleavedValueRows(name, src, nK, nV, 1, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.out_proj.weight"):
		out, err := reorderQwen35InterleavedValueCols(name, src, cfg.HiddenSize, nK, nV, vHd)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.A_log"),
		strings.HasSuffix(name, ".linear_attn.dt_bias"):
		out, err := reorderQwen35InterleavedValueVector(name, src, nK, nV)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.norm.weight"):
		if len(src) != vHd {
			return nil, true, fmt.Errorf("gguf: tensor %s has %d values, qwen35 linear norm wants %d", name, len(src), vHd)
		}
		return src, true, nil
	default:
		return nil, false, nil
	}
}

func reorderQwen35LinearQKVRows(name string, src []float32, keyDim, nK, nV, headDim, rowWidth int) ([]float32, error) {
	valDim := nV * headDim
	want := (2*keyDim + valDim) * rowWidth
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 qkv shape wants %d", name, len(src), want)
	}
	dst := append([]float32(nil), src...)
	vOff := 2 * keyDim * rowWidth
	v, err := reorderQwen35InterleavedValueRows(name, src[vOff:], nK, nV, headDim, rowWidth)
	if err != nil {
		return nil, err
	}
	copy(dst[vOff:], v)
	return dst, nil
}

func reorderQwen35InterleavedValueRows(name string, src []float32, nK, nV, headSpan, rowWidth int) ([]float32, error) {
	if nK <= 0 || nV <= 0 || headSpan <= 0 || rowWidth <= 0 || nV%nK != 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid qwen35 value layout nK=%d nV=%d span=%d width=%d", name, nK, nV, headSpan, rowWidth)
	}
	want := nV * headSpan * rowWidth
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value rows want %d", name, len(src), want)
	}
	ratio := nV / nK
	dst := make([]float32, len(src))
	rowBlock := headSpan * rowWidth
	for k := 0; k < nK; k++ {
		for r := 0; r < ratio; r++ {
			dstHead := k*ratio + r
			srcHead := r*nK + k
			copy(dst[dstHead*rowBlock:(dstHead+1)*rowBlock], src[srcHead*rowBlock:(srcHead+1)*rowBlock])
		}
	}
	return dst, nil
}

func reorderQwen35InterleavedValueCols(name string, src []float32, rows, nK, nV, headDim int) ([]float32, error) {
	if rows <= 0 || nK <= 0 || nV <= 0 || headDim <= 0 || nV%nK != 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid qwen35 value column layout rows=%d nK=%d nV=%d headDim=%d", name, rows, nK, nV, headDim)
	}
	cols := nV * headDim
	want := rows * cols
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value columns want %d", name, len(src), want)
	}
	ratio := nV / nK
	dst := make([]float32, len(src))
	for row := 0; row < rows; row++ {
		rowOff := row * cols
		for k := 0; k < nK; k++ {
			for r := 0; r < ratio; r++ {
				dstHead := k*ratio + r
				srcHead := r*nK + k
				copy(dst[rowOff+dstHead*headDim:rowOff+(dstHead+1)*headDim], src[rowOff+srcHead*headDim:rowOff+(srcHead+1)*headDim])
			}
		}
	}
	return dst, nil
}

func reorderQwen35InterleavedValueVector(name string, src []float32, nK, nV int) ([]float32, error) {
	if len(src) != nV {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value vector wants %d", name, len(src), nV)
	}
	return reorderQwen35InterleavedValueRows(name, src, nK, nV, 1, 1)
}

// ggufArchStoresHFRotaryLayout reports whether a GGUF of this architecture stores its
// q/k projection weights already in the HF "rotate_half" (NEOX) RoPE layout — i.e.
// convert_hf_to_gguf.py did NOT permute them on the way out. fak's forward pass always
// applies the rotate_half convention (forward.go), so for these models the q/k weights
// must be consumed exactly as stored: running them through unpermuteRotaryTensor scrambles
// every head's rotary pairs and yields incoherent output.
//
// Only the llama-family NORM-rope architectures (llama — which also carries Mistral/Mixtral
// exports — baichuan, command-r, …) are permuted by the converter and therefore still need
// the unpermute. Anything not on this NEOX allow-list keeps the historical unpermute, so no
// currently-covered architecture (the "llama" rotary test, the qwen35 hybrid test) regresses.
func ggufArchStoresHFRotaryLayout(arch string) bool {
	switch arch {
	case "qwen2", "qwen2moe", "qwen3", "qwen3moe",
		"gemma", "gemma2", "gemma3", "gemma4", "gemma4-assistant",
		"phi2", "phi3",
		"stablelm", "gptneox", "gpt2", "starcoder2",
		"falcon", "mpt", "olmo2", "gptoss":
		return true
	}
	return false
}

func unpermuteRotaryTensor(name string, src []float32, heads, headDim, in int) ([]float32, error) {
	if heads <= 0 || headDim <= 0 || in <= 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid rotary shape heads=%d head_dim=%d in=%d", name, heads, headDim, in)
	}
	if headDim%2 != 0 {
		return nil, fmt.Errorf("gguf: tensor %s head_dim %d is not even", name, headDim)
	}
	if heads > math.MaxInt/headDim || heads*headDim > math.MaxInt/in {
		return nil, fmt.Errorf("gguf: tensor %s rotary shape overflows int", name)
	}
	want := heads * headDim * in
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, rotary shape wants %d", name, len(src), want)
	}
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[((h*2+p)*half+j)*in+c] = src[((h*half+j)*2+p)*in+c]
				}
			}
		}
	}
	return dst, nil
}

func unpermuteQwen35GatedQTensor(name string, src []float32, heads, headDim, in int) ([]float32, error) {
	if heads <= 0 || headDim <= 0 || in <= 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid gated rotary shape heads=%d head_dim=%d in=%d", name, heads, headDim, in)
	}
	if headDim%2 != 0 {
		return nil, fmt.Errorf("gguf: tensor %s head_dim %d is not even", name, headDim)
	}
	if heads > math.MaxInt/(2*headDim) || heads*2*headDim > math.MaxInt/in {
		return nil, fmt.Errorf("gguf: tensor %s gated rotary shape overflows int", name)
	}
	want := heads * 2 * headDim * in
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, gated rotary shape wants %d", name, len(src), want)
	}
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		srcHead := h * 2 * headDim
		dstHead := h * 2 * headDim
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[(dstHead+p*half+j)*in+c] = src[(srcHead+j*2+p)*in+c]
				}
			}
		}
		copy(dst[(dstHead+headDim)*in:(dstHead+2*headDim)*in], src[(srcHead+headDim)*in:(srcHead+2*headDim)*in])
	}
	return dst, nil
}

// CanonicalTensorName maps a GGUF tensor name to fak's canonical HF-Llama name with
// the Llama-family norm convention (the historical, arch-blind behavior). Loaders that
// know the file architecture call CanonicalTensorNameArch instead, so a family whose
// norm tensors carry different roles (Gemma's sandwich norm) maps correctly.
func CanonicalTensorName(name string) (string, bool) {
	return CanonicalTensorNameArch(name, "")
}

// archIsGemma reports whether arch is a Gemma family whose GGUF carries sandwich-norm
// tensors (a distinct pre-feedforward norm plus post-attention and post-feedforward
// norms) rather than the single Llama post-attention norm. For these, blk.*.ffn_norm is
// the PRE-feedforward norm — not the post-attention norm it is for Llama — so the
// canonical mapping must branch on the family or two tensors collide on one name. Gemma
// v1 is intentionally excluded: it has no post-feedforward norm and its ffn_norm is the
// pre-MLP norm in the Llama role, so it keeps the default mapping.
func archIsGemma(arch string) bool {
	switch arch {
	case "gemma2", "gemma3", "gemma4", "gemma4-assistant":
		return true
	}
	return false
}

func archIsGemma4(arch string) bool {
	return arch == "gemma4" || arch == "gemma4-assistant"
}

// CanonicalTensorNameArch maps a GGUF tensor name to fak's canonical HF name, honoring
// arch-specific tensor roles. arch=="" preserves the Llama-family mapping.
func CanonicalTensorNameArch(name, arch string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight", true
	case "output_norm.weight":
		return "model.norm.weight", true
	case "output.weight":
		return "lm_head.weight", true
	case "rope_freqs.weight":
		// Gemma4's global (full-attention) layers carry a single shared rope frequency-
		// factor vector (proportional/NTK rope). It is a model-global tensor, read by the
		// forward when rotating the long-context layers.
		return "model.rope_freqs.weight", true
	}
	if !strings.HasPrefix(name, "blk.") {
		return "", false
	}
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return "", false
	}
	layer := rest[:dot]
	if _, err := strconv.Atoi(layer); err != nil {
		return "", false
	}
	suffix := rest[dot+1:]
	// Gemma sandwich norm: ffn_norm is the PRE-feedforward norm and post_ffw_norm the
	// POST-feedforward norm, distinct from the post-attention norm. The Llama default
	// keeps ffn_norm == post_attention_layernorm (the single pre-MLP norm).
	ffnNormCanon := "post_attention_layernorm.weight"
	if archIsGemma(arch) {
		ffnNormCanon = "pre_feedforward_layernorm.weight"
	}
	mapped, ok := map[string]string{
		"attn_norm.weight":           "input_layernorm.weight",
		"ffn_norm.weight":            ffnNormCanon,
		"post_attention_norm.weight": "post_attention_layernorm.weight",
		"post_ffw_norm.weight":       "post_feedforward_layernorm.weight",
		"layer_output_scale.weight":  "layer_output_scale.weight",
		"attn_q.weight":              "self_attn.q_proj.weight",
		"attn_k.weight":              "self_attn.k_proj.weight",
		"attn_v.weight":              "self_attn.v_proj.weight",
		"attn_qkv.weight":            "self_attn.qkv_proj.weight",
		"attn_gate.weight":           "self_attn.q_gate_proj.weight",
		"attn_output.weight":         "self_attn.o_proj.weight",
		"attn_q.bias":                "self_attn.q_proj.bias",
		"attn_k.bias":                "self_attn.k_proj.bias",
		"attn_v.bias":                "self_attn.v_proj.bias",
		"attn_q_norm.weight":         "self_attn.q_norm.weight",
		"attn_k_norm.weight":         "self_attn.k_norm.weight",
		"ffn_gate.weight":            "mlp.gate_proj.weight",
		"ffn_up.weight":              "mlp.up_proj.weight",
		"ffn_down.weight":            "mlp.down_proj.weight",
		"ssm_a":                      "linear_attn.A_log",
		"ssm_alpha.weight":           "linear_attn.in_proj_a.weight",
		"ssm_beta.weight":            "linear_attn.in_proj_b.weight",
		"ssm_conv1d.weight":          "linear_attn.conv1d.weight",
		"ssm_dt.bias":                "linear_attn.dt_bias",
		"ssm_norm.weight":            "linear_attn.norm.weight",
		"ssm_out.weight":             "linear_attn.out_proj.weight",
	}[suffix]
	if !ok {
		return "", false
	}
	return "model.layers." + layer + "." + mapped, true
}

func modelShapeFromGGUFDims(name string, dims []uint64) ([]int, error) {
	if len(dims) == 0 {
		return nil, fmt.Errorf("gguf: tensor %s has no dimensions", name)
	}
	shape := make([]int, len(dims))
	for i := range dims {
		d := dims[len(dims)-1-i]
		if d == 0 || d > uint64(math.MaxInt) {
			return nil, fmt.Errorf("gguf: tensor %s dimension %d overflows int", name, d)
		}
		shape[i] = int(d)
	}
	return shape, nil
}

func Read(r io.Reader) (*File, error) {
	rr := &countingReader{r: r}
	magic := make([]byte, 4)
	if err := rr.readFull(magic); err != nil {
		return nil, err
	}
	if string(magic) != Magic {
		return nil, fmt.Errorf("gguf: bad magic %q", string(magic))
	}
	ver, err := rr.u32()
	if err != nil {
		return nil, err
	}
	if ver != Version {
		return nil, fmt.Errorf("gguf: unsupported version %d", ver)
	}
	tensorCount, err := rr.u64()
	if err != nil {
		return nil, err
	}
	kvCount, err := rr.u64()
	if err != nil {
		return nil, err
	}

	meta := make(map[string]Value, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := rr.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata key %d: %w", i, err)
		}
		typ, err := rr.valueType()
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata %s type: %w", key, err)
		}
		v, err := rr.value(typ)
		if err != nil {
			return nil, fmt.Errorf("gguf: metadata %s: %w", key, err)
		}
		meta[key] = v
	}

	tensors := make([]TensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := rr.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %d name: %w", i, err)
		}
		nd, err := rr.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s dims: %w", name, err)
		}
		if nd > 4 {
			return nil, fmt.Errorf("gguf: tensor %s has %d dimensions", name, nd)
		}
		dims := make([]uint64, nd)
		for j := range dims {
			dims[j], err = rr.u64()
			if err != nil {
				return nil, fmt.Errorf("gguf: tensor %s dim %d: %w", name, j, err)
			}
		}
		typ, err := rr.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s type: %w", name, err)
		}
		off, err := rr.u64()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %s offset: %w", name, err)
		}
		tensors = append(tensors, TensorInfo{Name: name, Dims: dims, Type: TensorType(typ), Offset: off})
	}

	align, err := alignment(meta)
	if err != nil {
		return nil, err
	}
	data := alignOffset(uint64(rr.n), align)
	if data > uint64(math.MaxInt64) {
		return nil, fmt.Errorf("gguf: tensor data offset overflows int64")
	}
	for i := range tensors {
		if tensors[i].Offset%align != 0 {
			return nil, fmt.Errorf("gguf: tensor %s offset %d is not %d-byte aligned", tensors[i].Name, tensors[i].Offset, align)
		}
		if data+tensors[i].Offset > uint64(math.MaxInt64) {
			return nil, fmt.Errorf("gguf: tensor %s file offset overflows int64", tensors[i].Name)
		}
		tensors[i].FileOffset = int64(data + tensors[i].Offset)
	}

	return &File{
		Version:          ver,
		Metadata:         meta,
		Tensors:          tensors,
		Alignment:        align,
		TensorDataOffset: int64(data),
	}, nil
}

func (f *File) Config() (model.Config, error) {
	arch, ok := f.String("general.architecture")
	if !ok || arch == "" {
		return model.Config{}, fmt.Errorf("gguf: missing general.architecture")
	}
	p := arch + "."
	hidden, err := f.requiredInt(p + "embedding_length")
	if err != nil {
		return model.Config{}, err
	}
	layers, err := f.requiredInt(p + "block_count")
	if err != nil {
		return model.Config{}, err
	}
	heads, err := f.requiredInt(p + "attention.head_count")
	if err != nil {
		return model.Config{}, err
	}
	ffn, err := f.requiredInt(p + "feed_forward_length")
	if err != nil {
		return model.Config{}, err
	}
	headDim := hidden / heads
	if v, ok := f.Uint64(p + "attention.key_length"); ok {
		headDim = int(v)
	}
	kvHeads := heads
	if v, ok := f.Uint64(p + "attention.head_count_kv"); ok {
		kvHeads = int(v)
	}
	rms, err := f.requiredFloat(p + "attention.layer_norm_rms_epsilon")
	if err != nil {
		return model.Config{}, err
	}
	theta := 10000.0
	if v, ok := f.Float64(p + "rope.freq_base"); ok {
		theta = v
	}
	ropeDim := headDim
	if v, ok := f.Uint64(p + "rope.dimension_count"); ok {
		ropeDim = int(v)
	}
	vocab := 0
	if toks, ok := f.StringArray("tokenizer.ggml.tokens"); ok {
		vocab = len(toks)
	}
	eos := -1
	if v, ok := f.Uint64("tokenizer.ggml.eos_token_id"); ok {
		eos = int(v)
	}
	cfg := model.Config{
		HiddenSize:            hidden,
		NumLayers:             layers,
		NumHeads:              heads,
		NumKVHeads:            kvHeads,
		HeadDim:               headDim,
		IntermediateSize:      ffn,
		VocabSize:             vocab,
		RMSNormEps:            rms,
		RopeTheta:             theta,
		TieWordEmbeddings:     !f.hasTensor("output.weight") && !f.hasTensor("lm_head.weight"),
		AttentionBias:         f.hasTensor("blk.0.attn_q.bias") || f.hasTensor("blk.0.attn_k.bias") || f.hasTensor("blk.0.attn_v.bias"),
		ModelType:             arch,
		EOSTokenID:            eos,
		MaxPositionEmbeddings: intValueOrZero(f, p+"context_length"),
		HiddenAct:             "silu",
	}
	if ropeDim > 0 && ropeDim < headDim {
		cfg.PartialRotaryFactor = float64(ropeDim) / float64(headDim)
	}
	if arch == "qwen35" || arch == "qwen35moe" {
		if interval, ok := f.Uint64(p + "full_attention_interval"); ok {
			cfg.FullAttentionInterval = int(interval)
		}
		if conv, ok := f.Uint64(p + "ssm.conv_kernel"); ok {
			cfg.LinearConvKernelDim = int(conv)
		}
		if state, ok := f.Uint64(p + "ssm.state_size"); ok {
			cfg.LinearKeyHeadDim = int(state)
			cfg.LinearValueHeadDim = int(state)
		}
		if groups, ok := f.Uint64(p + "ssm.group_count"); ok {
			cfg.LinearNumKeyHeads = int(groups)
		}
		if rank, ok := f.Uint64(p + "ssm.time_step_rank"); ok {
			cfg.LinearNumValueHeads = int(rank)
		} else if inner, ok := f.Uint64(p + "ssm.inner_size"); ok && cfg.LinearValueHeadDim > 0 {
			cfg.LinearNumValueHeads = int(inner) / cfg.LinearValueHeadDim
		}
		cfg.AttnOutputGate = true
		cfg.NormGain1p = true
		cfg.QKNorm = true
		if cfg.FullAttentionInterval > 0 && len(cfg.LayerTypes) == 0 {
			cfg.LayerTypes = make([]string, cfg.NumLayers)
			for l := range cfg.LayerTypes {
				if (l+1)%cfg.FullAttentionInterval == 0 {
					cfg.LayerTypes[l] = "full_attention"
				} else {
					cfg.LayerTypes[l] = "linear_attention"
				}
			}
		}
	}
	if archIsGemma4(arch) {
		if err := applyGemma4Config(f, p, &cfg); err != nil {
			return model.Config{}, err
		}
	}
	if arch == "glm_moe_dsa" {
		applyGLMMoeDsaConfig(f, p, &cfg, ropeDim)
	}
	return cfg, nil
}

// applyGemma4Config derives Google Gemma 4's architecture axes from GGUF metadata into
// cfg. Gemma 4 is GeGLU + sandwich-norm + sqrt(hidden) embed scale + a final-logit
// soft-cap, atop a HETEROGENEOUS per-layer attention geometry: local (sliding) layers
// and global (full) layers carry different head_dim, kv-head counts, RoPE bases, and
// windows, encoded as GGUF arrays. The norm weights are baked (+1) at convert time and
// consumed with plain RMSNorm, so NormGain1p stays false (the safetensors path, which
// reads raw HF weights, is the one that sets it true).
func applyGemma4Config(f *File, p string, cfg *model.Config) error {
	cfg.ActGeluTanh = true
	cfg.BlockTopology = model.SandwichNorm
	cfg.NormGain1p = false
	if cfg.HiddenSize > 0 {
		cfg.EmbedScale = math.Sqrt(float64(cfg.HiddenSize))
	}
	if v, ok := f.Float64(p + "final_logit_softcapping"); ok {
		cfg.LogitSoftcap = v
	}
	if v, ok := f.Float64(p + "attn_logit_softcapping"); ok {
		cfg.AttnSoftcap = v
	}
	if f.hasTensor("blk.0.attn_q_norm.weight") {
		cfg.QKNorm = true
	}
	// Gemma 4 masks image/audio placeholder tokens (a known checkpoint issue) via a
	// final-logit -inf bias; the ids live in the tokenizer metadata.
	if sup, ok := f.IntArray("tokenizer.ggml.suppress_tokens"); ok {
		cfg.SuppressTokens = sup
	}

	n := cfg.NumLayers
	if n <= 0 {
		return fmt.Errorf("gguf: gemma4 has no layers")
	}
	pattern, ok := f.BoolArray(p + "attention.sliding_window_pattern")
	if !ok || len(pattern) < n {
		return fmt.Errorf("gguf: gemma4 attention.sliding_window_pattern missing or short (have %d, want %d)", len(pattern), n)
	}
	kvArr, ok := f.IntArray(p + "attention.head_count_kv")
	if !ok || len(kvArr) < n {
		return fmt.Errorf("gguf: gemma4 attention.head_count_kv missing or short (have %d, want %d)", len(kvArr), n)
	}
	keyLenFull := intValueOrZero(f, p+"attention.key_length")     // global head_dim
	keyLenSWA := intValueOrZero(f, p+"attention.key_length_swa")  // local head_dim
	ropeDimFull := intValueOrZero(f, p+"rope.dimension_count")    // global rotary width
	ropeDimSWA := intValueOrZero(f, p+"rope.dimension_count_swa") // local rotary width
	swaWindow := intValueOrZero(f, p+"attention.sliding_window")
	thetaFull := cfg.RopeTheta // base read rope.freq_base
	if v, ok := f.Float64(p + "rope.freq_base"); ok {
		thetaFull = v
	}
	thetaSWA := 10000.0
	if v, ok := f.Float64(p + "rope.freq_base_swa"); ok {
		thetaSWA = v
	}
	if keyLenSWA == 0 {
		keyLenSWA = keyLenFull
	}
	if ropeDimFull == 0 {
		ropeDimFull = keyLenFull
	}
	if ropeDimSWA == 0 {
		ropeDimSWA = keyLenSWA
	}

	cfg.LayerTypes = make([]string, n)
	cfg.NumKVHeadsPerLayer = make([]int, n)
	cfg.HeadDimPerLayer = make([]int, n)
	cfg.RopeDimPerLayer = make([]int, n)
	cfg.RopeThetaPerLayer = make([]float64, n)
	cfg.Window = make([]int, n)
	for l := 0; l < n; l++ {
		cfg.NumKVHeadsPerLayer[l] = kvArr[l]
		if pattern[l] { // true == sliding / local
			cfg.LayerTypes[l] = "sliding_attention"
			cfg.HeadDimPerLayer[l] = keyLenSWA
			cfg.RopeDimPerLayer[l] = ropeDimSWA
			cfg.RopeThetaPerLayer[l] = thetaSWA
			cfg.Window[l] = swaWindow
		} else { // false == full / global
			cfg.LayerTypes[l] = "full_attention"
			cfg.HeadDimPerLayer[l] = keyLenFull
			cfg.RopeDimPerLayer[l] = ropeDimFull
			cfg.RopeThetaPerLayer[l] = thetaFull
			cfg.Window[l] = -1
		}
	}

	// Representative scalars: the dedicated gemma4 forward uses the per-layer slices, but
	// keep HeadDim/NumKVHeads/GroupSize sane for any shared code that still reads them.
	cfg.HeadDim = keyLenFull
	if kvArr[0] > 0 {
		cfg.NumKVHeads = kvArr[0]
	}
	cfg.RopeTheta = thetaFull
	return nil
}

// GLM-5.2 (model_type "glm_moe_dsa") GGUF metadata keys.
//
// GLM-5.2's architecture is a Mixture-of-Experts FFN over DeepSeek-style
// Multi-head Latent Attention (MLA) plus a learned Dynamic Sparse Attention
// (DSA) indexer. The MoE and MLA metadata mirror llama.cpp's deepseek2.*
// convention (GLM-DSA attention IS DeepSeek MLA + an indexer), so a real
// converter is most likely to spell them this way; the indexer scalars are
// GLM-5.2-specific and have no upstream llama.cpp analogue yet.
//
// PROVISIONAL: no real GLM-5.2 GGUF exists on disk to pin these against, and
// upstream llama.cpp may not yet ship a glm_moe_dsa converter. The spellings
// are collected here as the single source of truth so the deliberate follow-on
// — a golden against a REAL GLM-5.2 GGUF header — only has to re-pin this one
// block. Every key is read relative to the "<arch>." metadata prefix.
const (
	glmKeyExpertCount        = "expert_count"
	glmKeyExpertUsedCount    = "expert_used_count"
	glmKeyExpertFFNLength    = "expert_feed_forward_length"
	glmKeyExpertSharedCount  = "expert_shared_count"
	glmKeyExpertSharedFFNLen = "expert_shared_feed_forward_length"
	glmKeyLeadingDenseBlocks = "leading_dense_block_count"
	glmKeyExpertGroupCount   = "expert_group_count"
	glmKeyExpertGroupUsed    = "expert_group_used_count"
	glmKeyExpertWeightsScale = "expert_weights_scale"
	glmKeyExpertWeightsNorm  = "expert_weights_norm"

	glmKeyQLoraRank   = "attention.q_lora_rank"
	glmKeyKVLoraRank  = "attention.kv_lora_rank"
	glmKeyQKNopeDim   = "attention.qk_nope_head_dim"
	glmKeyQKRopeDim   = "attention.qk_rope_head_dim"
	glmKeyVHeadDim    = "attention.v_head_dim"
	glmKeyKeyLength   = "attention.key_length"
	glmKeyValueLength = "attention.value_length"

	glmKeyIndexNHeads  = "index_n_heads"
	glmKeyIndexHeadDim = "index_head_dim"
	glmKeyIndexTopK    = "index_topk"
	glmKeyIndexerTypes = "indexer_types"
)

// applyGLMMoeDsaConfig derives GLM-5.2's MoE + MLA + DSA-indexer axes from GGUF
// metadata into the model.Config the generic block already populated. It reads
// every key only if present, so it never overwrites a generic value with zero:
// a MoE GLM-5.2 (expert_count>0) and a dense glm_moe_dsa variant (NumExperts==0,
// the synthetic/pipelinegen form) both load correctly. The result mirrors, field
// for field, the model.Config the JSON/safetensors loader already produces for
// the same model (config_test.go TestConfigDerives...), so cfg.isGLMMoeDsa() and
// cfg.IsMoE() fire and the existing native glm_dsa.go forward consumes it.
//
// ropeDim is the already-resolved rope.dimension_count; it is reused as the
// qk_rope_head_dim fallback under the deepseek2 convention (where the rotary
// portion of each latent head equals the global rope dimension).
//
// Scope (deliberate, per the staged native-753B plan): this is config parsing
// ONLY. The GGUF MoE/MLA/indexer TENSOR-name mapping (CanonicalTensorNameArch)
// and the batched-expert splitter are the next two slices; HeadDim semantics for
// MLA are reconciled when the forward wiring lands.
func applyGLMMoeDsaConfig(f *File, p string, cfg *model.Config, ropeDim int) {
	// ---- MoE FFN axis -------------------------------------------------------
	if v := intValueOrZero(f, p+glmKeyExpertCount); v > 0 {
		cfg.NumExperts = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertUsedCount); v > 0 {
		cfg.NumExpertsPerTok = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertFFNLength); v > 0 {
		cfg.MoEIntermediateSize = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertSharedCount); v > 0 {
		cfg.NSharedExperts = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertSharedFFNLen); v > 0 {
		cfg.SharedIntermediateSize = v
	}
	if v := intValueOrZero(f, p+glmKeyLeadingDenseBlocks); v > 0 {
		cfg.FirstKDenseReplace = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertGroupCount); v > 0 {
		cfg.NGroup = v
	}
	if v := intValueOrZero(f, p+glmKeyExpertGroupUsed); v > 0 {
		cfg.TopKGroup = v
	}
	if v, ok := f.Float64(p + glmKeyExpertWeightsScale); ok {
		cfg.RoutedScalingFactor = v
	}
	if v, ok := f.Bool(p + glmKeyExpertWeightsNorm); ok {
		cfg.NormTopKProb = v
	}

	// ---- MLA (DeepSeek latent attention) axis -------------------------------
	if v := intValueOrZero(f, p+glmKeyQLoraRank); v > 0 {
		cfg.QLoraRank = v
	}
	if v := intValueOrZero(f, p+glmKeyKVLoraRank); v > 0 {
		cfg.KVLoraRank = v
	}
	// qk_rope_head_dim: explicit key, else the resolved rope.dimension_count.
	cfg.QKRopeHeadDim = intValueOrZero(f, p+glmKeyQKRopeDim)
	if cfg.QKRopeHeadDim == 0 {
		cfg.QKRopeHeadDim = ropeDim
	}
	// qk_nope_head_dim: explicit key, else attention.key_length - qk_rope_head_dim
	// (deepseek2 stores n_embd_head_k = nope + rope under attention.key_length).
	cfg.QKNopeHeadDim = intValueOrZero(f, p+glmKeyQKNopeDim)
	if cfg.QKNopeHeadDim == 0 {
		if kl := intValueOrZero(f, p+glmKeyKeyLength); kl > cfg.QKRopeHeadDim {
			cfg.QKNopeHeadDim = kl - cfg.QKRopeHeadDim
		}
	}
	// v_head_dim: explicit key, else attention.value_length.
	cfg.VHeadDim = intValueOrZero(f, p+glmKeyVHeadDim)
	if cfg.VHeadDim == 0 {
		cfg.VHeadDim = intValueOrZero(f, p+glmKeyValueLength)
	}

	// ---- DSA learned-indexer axis (GLM-5.2-specific) ------------------------
	if v := intValueOrZero(f, p+glmKeyIndexNHeads); v > 0 {
		cfg.IndexNHeads = v
	}
	if v := intValueOrZero(f, p+glmKeyIndexHeadDim); v > 0 {
		cfg.IndexHeadDim = v
	}
	if v := intValueOrZero(f, p+glmKeyIndexTopK); v > 0 {
		cfg.IndexTopK = v
	}
	if types, ok := f.StringArray(p + glmKeyIndexerTypes); ok {
		cfg.IndexerTypes = types
	}
}

func intValueOrZero(f *File, key string) int {
	if v, ok := f.Uint64(key); ok && v <= uint64(math.MaxInt) {
		return int(v)
	}
	return 0
}

func (f *File) String(key string) (string, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeString {
		return "", false
	}
	s, ok := v.Value.(string)
	return s, ok
}

func (f *File) Uint64(key string) (uint64, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	return valueUint64(v)
}

func (f *File) Float64(key string) (float64, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch v.Type {
	case TypeFloat32:
		return float64(v.Value.(float32)), true
	case TypeFloat64:
		return v.Value.(float64), true
	default:
		return 0, false
	}
}

// Bool reads a scalar GGUF boolean metadata value (TypeBool, one byte). GLM-5.2
// encodes expert_weights_norm (the MoE top-k renormalization flag, HF's
// norm_topk_prob) this way. Returns (false,false) when the key is absent or not
// a scalar bool.
func (f *File) Bool(key string) (bool, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeBool {
		return false, false
	}
	b, ok := v.Value.(bool)
	return b, ok
}

func (f *File) StringArray(key string) ([]string, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]string, len(items))
	for i, item := range items {
		if item.Type != TypeString {
			return nil, false
		}
		out[i] = item.Value.(string)
	}
	return out, true
}

// IntArray reads a GGUF metadata array of integers into []int. Gemma4 encodes a
// per-layer head_count_kv as such an array (one entry per decoder layer). Returns
// (nil,false) when the key is absent, not an array, or carries a non-integer item.
func (f *File) IntArray(key string) ([]int, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]int, len(items))
	for i, item := range items {
		u, ok := valueUint64(item)
		if !ok || u > uint64(math.MaxInt) {
			return nil, false
		}
		out[i] = int(u)
	}
	return out, true
}

// BoolArray reads a GGUF metadata array of bools into []bool. Gemma4 encodes its
// per-layer local/global cadence as sliding_window_pattern (true = sliding/local,
// false = full/global). Returns (nil,false) on any non-bool item.
func (f *File) BoolArray(key string) ([]bool, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]bool, len(items))
	for i, item := range items {
		if item.Type != TypeBool {
			return nil, false
		}
		b, ok := item.Value.(bool)
		if !ok {
			return nil, false
		}
		out[i] = b
	}
	return out, true
}

func (f *File) requiredInt(key string) (int, error) {
	v, ok := f.Uint64(key)
	if !ok {
		return 0, fmt.Errorf("gguf: missing %s", key)
	}
	if v > uint64(math.MaxInt) {
		return 0, fmt.Errorf("gguf: %s overflows int", key)
	}
	return int(v), nil
}

func (f *File) requiredFloat(key string) (float64, error) {
	v, ok := f.Float64(key)
	if !ok {
		return 0, fmt.Errorf("gguf: missing %s", key)
	}
	return v, nil
}

func (f *File) hasTensor(name string) bool {
	for _, t := range f.Tensors {
		if t.Name == name {
			return true
		}
	}
	return false
}

func valueUint64(v Value) (uint64, bool) {
	switch v.Type {
	case TypeUint8:
		return uint64(v.Value.(uint8)), true
	case TypeUint16:
		return uint64(v.Value.(uint16)), true
	case TypeUint32:
		return uint64(v.Value.(uint32)), true
	case TypeUint64:
		return v.Value.(uint64), true
	case TypeInt8:
		x := v.Value.(int8)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt16:
		x := v.Value.(int16)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt32:
		x := v.Value.(int32)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt64:
		x := v.Value.(int64)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	default:
		return 0, false
	}
}

func alignment(meta map[string]Value) (uint64, error) {
	align := uint64(defaultAlign)
	if v, ok := meta["general.alignment"]; ok {
		got, ok := valueUint64(v)
		if !ok {
			return 0, fmt.Errorf("gguf: general.alignment is not an unsigned integer")
		}
		align = got
	}
	if align == 0 || align%8 != 0 {
		return 0, fmt.Errorf("gguf: invalid alignment %d", align)
	}
	return align, nil
}

func alignOffset(off, align uint64) uint64 {
	return off + (align-(off%align))%align
}

func tensorPayloadBytes(t TensorInfo) (uint64, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return 0, err
	}
	switch t.Type {
	case TensorF32:
		return elems * 4, nil
	case TensorF16, TensorBF16:
		return elems * 2, nil
	case TensorQ4_0:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_0 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return elems / qk4 * blockQ4_0Bytes, nil
	case TensorQ4_1:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_1 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return elems / qk4 * blockQ4_1Bytes, nil
	case TensorQ5_0:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_0 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return elems / qk5 * blockQ5_0Bytes, nil
	case TensorQ5_1:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_1 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return elems / qk5 * blockQ5_1Bytes, nil
	case TensorQ8_0:
		if elems%qk8_0 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q8_0 element count %d is not a multiple of %d", t.Name, elems, qk8_0)
		}
		return elems / qk8_0 * blockQ8_0Bytes, nil
	case TensorQ2_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q2_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ2KBytes, nil
	case TensorQ3_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q3_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ3KBytes, nil
	case TensorQ4_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ4KBytes, nil
	case TensorQ5_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ5KBytes, nil
	case TensorQ6_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q6_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ6KBytes, nil
	case TensorMXFP4:
		if elems%qkMXFP4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s MXFP4 element count %d is not a multiple of %d", t.Name, elems, qkMXFP4)
		}
		return elems / qkMXFP4 * blockMXFP4Bytes, nil
	default:
		return 0, fmt.Errorf("gguf: tensor %s type %d does not have a simple f32 payload", t.Name, t.Type)
	}
}

func tensorElems(t TensorInfo) (uint64, error) {
	if len(t.Dims) == 0 {
		return 0, fmt.Errorf("gguf: tensor %s has no dimensions", t.Name)
	}
	n := uint64(1)
	for _, d := range t.Dims {
		if d == 0 {
			return 0, fmt.Errorf("gguf: tensor %s has zero dimension", t.Name)
		}
		if n > math.MaxUint64/d {
			return 0, fmt.Errorf("gguf: tensor %s element count overflows uint64", t.Name)
		}
		n *= d
	}
	return n, nil
}

func dequantF32(t TensorInfo, raw []byte) ([]float32, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return nil, err
	}
	if elems > uint64(math.MaxInt) {
		return nil, fmt.Errorf("gguf: tensor %s element count overflows int", t.Name)
	}
	out := make([]float32, int(elems))
	switch t.Type {
	case TensorF32:
		if len(raw) != len(out)*4 {
			return nil, fmt.Errorf("gguf: tensor %s f32 payload has %d bytes, want %d", t.Name, len(raw), len(out)*4)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case TensorF16:
		if len(raw) != len(out)*2 {
			return nil, fmt.Errorf("gguf: tensor %s f16 payload has %d bytes, want %d", t.Name, len(raw), len(out)*2)
		}
		for i := range out {
			out[i] = math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[i*2:])))
		}
	case TensorBF16:
		if len(raw) != len(out)*2 {
			return nil, fmt.Errorf("gguf: tensor %s bf16 payload has %d bytes, want %d", t.Name, len(raw), len(out)*2)
		}
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case TensorQ4_0:
		if elems%qk4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_0 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		want := int(elems / qk4 * blockQ4_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4_0(out, raw)
	case TensorQ4_1:
		if elems%qk4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_1 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		want := int(elems / qk4 * blockQ4_1Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_1 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4_1(out, raw)
	case TensorQ5_0:
		if elems%qk5 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_0 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		want := int(elems / qk5 * blockQ5_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5_0(out, raw)
	case TensorQ5_1:
		if elems%qk5 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_1 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		want := int(elems / qk5 * blockQ5_1Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_1 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5_1(out, raw)
	case TensorQ8_0:
		if elems%qk8_0 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q8_0 element count %d is not a multiple of %d", t.Name, elems, qk8_0)
		}
		want := int(elems / qk8_0 * blockQ8_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q8_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		for block := 0; block < int(elems)/qk8_0; block++ {
			base := block * blockQ8_0Bytes
			d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
			for j := 0; j < qk8_0; j++ {
				out[block*qk8_0+j] = float32(int8(raw[base+2+j])) * d
			}
		}
	case TensorQ2_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q2_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ2KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q2_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ2K(out, raw)
	case TensorQ3_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q3_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ3KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q3_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ3K(out, raw)
	case TensorQ4_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ4KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4K(out, raw)
	case TensorQ5_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ5KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5K(out, raw)
	case TensorQ6_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q6_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ6KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q6_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ6K(out, raw)
	case TensorMXFP4:
		if elems%qkMXFP4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s MXFP4 element count %d is not a multiple of %d", t.Name, elems, qkMXFP4)
		}
		want := int(elems / qkMXFP4 * blockMXFP4Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s MXFP4 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantMXFP4(out, raw)
	default:
		return nil, fmt.Errorf("gguf: tensor %s type %d cannot dequantize to f32 yet", t.Name, t.Type)
	}
	return out, nil
}

// dequantQ4_0 expands the legacy GGML Q4_0 32-element block. Each block is a
// little-endian f16 scale d followed by qk4/2 bytes of packed 4-bit codes (two
// nibbles per byte). The GGML layout (dequantize_row_q4_0) is interleaved: the low
// nibble of byte j is element j, the high nibble is element j+qk4/2, and each code is
// re-centered by -8 before scaling: y = (nibble-8)*d. This is the 4-bit sibling of
// dequantQ5_0 with no 5th high bit.
func dequantQ4_0(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk4; block++ {
		base := block * blockQ4_0Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		qs := raw[base+2 : base+blockQ4_0Bytes]
		yi := block * qk4
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j]&0x0f) - 8
			x1 := int(qs[j]>>4) - 8
			out[yi+j] = float32(x0) * d
			out[yi+j+qk4/2] = float32(x1) * d
		}
	}
}

// kvaluesMXFP4 maps a 4-bit E2M1 (FP4) code to its value, stored as 2x the real
// FP4 magnitude so the table is exact integers; the ×0.5 that restores the true
// E2M1 values {0,.5,1,1.5,2,3,4,6} is folded into the E8M0 scale by e8m0ToF32Half
// (which yields 2^(e-128) rather than 2^(e-127)). This matches GGML's
// kvalues_mxfp4 + GGML_E8M0_TO_FP32_HALF pairing for gpt-oss weights.
var kvaluesMXFP4 = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half decodes an E8M0 shared-exponent scale byte to 2^(e-128) — the
// half-scaled power that pairs with the doubled kvaluesMXFP4 table so that
// kvaluesMXFP4[code] * e8m0ToF32Half(e) == fp4(code) * 2^(e-127).
func e8m0ToF32Half(e uint8) float32 {
	return float32(math.Ldexp(1, int(e)-128))
}

// dequantMXFP4 expands the MXFP4 (gpt-oss) 32-element block: a 1-byte E8M0 shared
// scale followed by qkMXFP4/2 bytes of packed 4-bit E2M1 codes. The GGML layout
// (dequantize_row_mxfp4) interleaves like Q4_0 — the low nibble of byte j is
// element j, the high nibble is element j+qkMXFP4/2 — and each code indexes the
// E2M1 value table scaled by the block's half-scaled E8M0 exponent.
func dequantMXFP4(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkMXFP4; block++ {
		base := block * blockMXFP4Bytes
		d := e8m0ToF32Half(raw[base])
		qs := raw[base+1 : base+blockMXFP4Bytes]
		yi := block * qkMXFP4
		for j := 0; j < qkMXFP4/2; j++ {
			out[yi+j] = kvaluesMXFP4[qs[j]&0x0f] * d
			out[yi+j+qkMXFP4/2] = kvaluesMXFP4[qs[j]>>4] * d
		}
	}
}

// dequantQ4_1 expands the legacy GGML Q4_1 32-element block: a little-endian f16
// scale d, then a little-endian f16 min m, then qk4/2 bytes of packed 4-bit codes.
// The GGML layout (dequantize_row_q4_1) keeps the same low/high-nibble interleave as
// Q4_0 but the codes are NOT re-centered — they carry an affine min: y = nibble*d + m.
func dequantQ4_1(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk4; block++ {
		base := block * blockQ4_1Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		m := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		qs := raw[base+4 : base+blockQ4_1Bytes]
		yi := block * qk4
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j] & 0x0f)
			x1 := int(qs[j] >> 4)
			out[yi+j] = float32(x0)*d + m
			out[yi+j+qk4/2] = float32(x1)*d + m
		}
	}
}

func dequantQ5_0(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk5; block++ {
		base := block * blockQ5_0Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		qh := binary.LittleEndian.Uint32(raw[base+2:])
		qs := raw[base+6 : base+blockQ5_0Bytes]
		yi := block * qk5
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j]&0x0f)|xh0) - 16
			x1 := int((qs[j]>>4)|xh1) - 16
			out[yi+j] = float32(x0) * d
			out[yi+j+qk5/2] = float32(x1) * d
		}
	}
}

func dequantQ5_1(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk5; block++ {
		base := block * blockQ5_1Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		m := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		qh := binary.LittleEndian.Uint32(raw[base+4:])
		qs := raw[base+8 : base+blockQ5_1Bytes]
		yi := block * qk5
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j] & 0x0f) | xh0)
			x1 := int((qs[j] >> 4) | xh1)
			out[yi+j] = float32(x0)*d + m
			out[yi+j+qk5/2] = float32(x1)*d + m
		}
	}
}

func dequantQ2K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ2KBytes
		scales := raw[base : base+qkK/16]
		q := raw[base+qkK/16 : base+qkK/16+qkK/4]
		dm := base + qkK/16 + qkK/4
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[dm:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[dm+2:])))
		yi := block * qkK
		qi := 0
		is := 0
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl, ml := d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[yi+n+j*32+l] = dl*float32((q[qi+l]>>shift)&3) - ml
				}

				sc = scales[is]
				is++
				dl, ml = d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[yi+n+j*32+16+l] = dl*float32((q[qi+16+l]>>shift)&3) - ml
				}
				shift += 2
			}
			qi += 32
		}
	}
}

func dequantQ3K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ3KBytes
		hmask := raw[base : base+qkK/8]
		q := raw[base+qkK/8 : base+qkK/8+qkK/4]
		scales := unpackQ3KScales(raw[base+qkK/8+qkK/4 : base+qkK/8+qkK/4+kScaleSize])
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+blockQ3KBytes-2:])))
		yi := block * qkK
		qi := 0
		is := 0
		mask := byte(1)
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+l] >> shift) & 3)
					if hmask[l]&mask == 0 {
						code -= 4
					}
					out[yi+n+j*32+l] = dl * float32(code)
				}

				dl = d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+16+l] >> shift) & 3)
					if hmask[16+l]&mask == 0 {
						code -= 4
					}
					out[yi+n+j*32+16+l] = dl * float32(code)
				}
				shift += 2
				mask <<= 1
			}
			qi += 32
		}
	}
}

func unpackQ3KScales(raw []byte) [16]int8 {
	const (
		kmask1 = uint32(0x03030303)
		kmask2 = uint32(0x0f0f0f0f)
	)
	aux0 := binary.LittleEndian.Uint32(raw[0:4])
	aux1 := binary.LittleEndian.Uint32(raw[4:8])
	aux2 := binary.LittleEndian.Uint32(raw[8:12])
	tmp := aux2
	words := [4]uint32{
		(aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var scales [16]int8
	for i, word := range words {
		for j := 0; j < 4; j++ {
			scales[i*4+j] = int8(byte(word >> (8 * j)))
		}
	}
	return scales
}

func dequantQ4K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ4KBytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		scales := raw[base+4 : base+4+kScaleSize]
		q := raw[base+4+kScaleSize : base+blockQ4KBytes]
		qi := 0
		is := 0
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float32(sc), min*float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), min*float32(m)
			for l := 0; l < 32; l++ {
				out[yi+j+l] = d1*float32(q[qi+l]&0x0f) - m1
			}
			for l := 0; l < 32; l++ {
				out[yi+j+32+l] = d2*float32(q[qi+l]>>4) - m2
			}
			qi += 32
			is += 2
		}
	}
}

func getScaleMinK4(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func dequantQ5K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ5KBytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		scales := raw[base+4 : base+4+kScaleSize]
		qh := raw[base+4+kScaleSize : base+4+kScaleSize+qkK/8]
		ql := raw[base+4+kScaleSize+qkK/8 : base+blockQ5KBytes]
		qi := 0
		is := 0
		u1, u2 := byte(1), byte(2)
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float32(sc), min*float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), min*float32(m)
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u1 != 0 {
					hi = 16
				}
				out[yi+j+l] = d1*float32((ql[qi+l]&0x0f)+hi) - m1
			}
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u2 != 0 {
					hi = 16
				}
				out[yi+j+32+l] = d2*float32((ql[qi+l]>>4)+hi) - m2
			}
			qi += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ6KBytes
		ql := raw[base : base+qkK/2]
		qh := raw[base+qkK/2 : base+qkK/2+qkK/4]
		scales := raw[base+qkK/2+qkK/4 : base+qkK/2+qkK/4+qkK/16]
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+blockQ6KBytes-2:])))
		yi := block * qkK
		qlOff, qhOff, scOff := 0, 0, 0
		for n := 0; n < qkK; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				out[yi+n+l+0] = d * float32(int8(scales[scOff+is+0])) * float32(q1)
				out[yi+n+l+32] = d * float32(int8(scales[scOff+is+2])) * float32(q2)
				out[yi+n+l+64] = d * float32(int8(scales[scOff+is+4])) * float32(q3)
				out[yi+n+l+96] = d * float32(int8(scales[scOff+is+6])) * float32(q4)
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
}

func f16bitsToF32bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		exp = -14
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x03ff
		return sign | uint32(exp+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(exp-15+127)<<23 | frac<<13
	}
}

type countingReader struct {
	r io.Reader
	n int64
}

func (r *countingReader) readFull(b []byte) error {
	if _, err := io.ReadFull(r.r, b); err != nil {
		return err
	}
	r.n += int64(len(b))
	return nil
}

func (r *countingReader) u32() (uint32, error) {
	var b [4]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (r *countingReader) u64() (uint64, error) {
	var b [8]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (r *countingReader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringBytes {
		return "", fmt.Errorf("string too large: %d bytes", n)
	}
	b := make([]byte, int(n))
	if err := r.readFull(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *countingReader) valueType() (ValueType, error) {
	u, err := r.u32()
	return ValueType(u), err
}

func (r *countingReader) value(typ ValueType) (Value, error) {
	switch typ {
	case TypeUint8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: b[0]}, nil
	case TypeInt8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int8(b[0])}, nil
	case TypeUint16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: binary.LittleEndian.Uint16(b[:])}, nil
	case TypeInt16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int16(binary.LittleEndian.Uint16(b[:]))}, nil
	case TypeUint32:
		v, err := r.u32()
		return Value{Type: typ, Value: v}, err
	case TypeInt32:
		v, err := r.u32()
		return Value{Type: typ, Value: int32(v)}, err
	case TypeFloat32:
		v, err := r.u32()
		return Value{Type: typ, Value: math.Float32frombits(v)}, err
	case TypeBool:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		if b[0] > 1 {
			return Value{}, fmt.Errorf("invalid bool byte %d", b[0])
		}
		return Value{Type: typ, Value: b[0] == 1}, nil
	case TypeString:
		s, err := r.str()
		return Value{Type: typ, Value: s}, err
	case TypeArray:
		elem, err := r.valueType()
		if err != nil {
			return Value{}, err
		}
		n, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		if n > uint64(math.MaxInt) {
			return Value{}, fmt.Errorf("array too large: %d elements", n)
		}
		items := make([]Value, int(n))
		for i := range items {
			items[i], err = r.value(elem)
			if err != nil {
				return Value{}, fmt.Errorf("array element %d: %w", i, err)
			}
		}
		return Value{Type: typ, Value: items}, nil
	case TypeUint64:
		v, err := r.u64()
		return Value{Type: typ, Value: v}, err
	case TypeInt64:
		v, err := r.u64()
		return Value{Type: typ, Value: int64(v)}, err
	case TypeFloat64:
		v, err := r.u64()
		return Value{Type: typ, Value: math.Float64frombits(v)}, err
	default:
		return Value{}, fmt.Errorf("unsupported value type %d", typ)
	}
}
