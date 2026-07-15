package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	qwen35GDNCorpusFormat  = "fak.qwen35-gdn-corpus.v1"
	qwen35GDNCorpusPathEnv = "FAK_CUDA_GDN_CORPUS"
	qwen35GDNCorpusModeEnv = "FAK_QWEN35_GDN_CORPUS_MODE"

	qwen35GDNCorpusMetadataFile = "corpus.json"
	qwen35GDNCorpusManifestFile = "SHA256SUMS"
	qwen35GDNCorpusStepCount    = 4
	qwen35GDNCorpusMaxFiles     = 128
	qwen35GDNCorpusMaxBytes     = 64 << 20
	qwen35GDNCorpusMaxElements  = qwen35GDNCorpusMaxBytes / 4

	qwen35GDNWeightInQKV = "weight.in_proj_qkv"
	qwen35GDNWeightInZ   = "weight.in_proj_z"
	qwen35GDNWeightInB   = "weight.in_proj_b"
	qwen35GDNWeightInA   = "weight.in_proj_a"
	qwen35GDNWeightConv  = "weight.conv1d"
	qwen35GDNWeightALog  = "weight.a_log"
	qwen35GDNWeightDT    = "weight.dt_bias"
	qwen35GDNWeightNorm  = "weight.norm"
	qwen35GDNWeightOut   = "weight.out_proj"

	qwen35GDNInitialConvState      = "state.initial.conv"
	qwen35GDNInitialRecurrentState = "state.initial.recurrent"
)

var qwen35GDNCorpusProducer = qwen35GDNProducerIdentity{
	Name:   "TestQwen35GDNCorpusProcess/produce",
	Module: "github.com/anthony-chaudhary/fak/internal/model",
	Oracle: "Session.linearAttnStep",
}

type qwen35GDNProducerIdentity struct {
	Name   string `json:"name"`
	Module string `json:"module"`
	Oracle string `json:"oracle"`
}

type qwen35GDNCorpusGeometry struct {
	HiddenSize    int `json:"hidden_size"`
	NumKeyHeads   int `json:"num_key_heads"`
	NumValueHeads int `json:"num_value_heads"`
	KeyHeadDim    int `json:"key_head_dim"`
	ValueHeadDim  int `json:"value_head_dim"`
	KeyDim        int `json:"key_dim"`
	ValueDim      int `json:"value_dim"`
	ConvDim       int `json:"conv_dim"`
	ConvKernel    int `json:"conv_kernel"`
	StepCount     int `json:"step_count"`
}

type qwen35GDNTensorMetadata struct {
	Name           string  `json:"name"`
	File           string  `json:"file"`
	Role           string  `json:"role"`
	DType          string  `json:"dtype"`
	Shape          []int   `json:"shape"`
	Elements       int     `json:"elements"`
	Norm           float64 `json:"norm"`
	RequireNonzero bool    `json:"require_nonzero"`
	MustBeZero     bool    `json:"must_be_zero"`
}

type qwen35GDNReferenceMetadata struct {
	Tensor      string  `json:"tensor"`
	Norm        float64 `json:"norm"`
	MaxAbsError float64 `json:"max_abs_error"`
}

type qwen35GDNStepMetadata struct {
	Index          int                        `json:"index"`
	Input          string                     `json:"input"`
	Output         qwen35GDNReferenceMetadata `json:"output"`
	ConvState      qwen35GDNReferenceMetadata `json:"conv_state"`
	RecurrentState qwen35GDNReferenceMetadata `json:"recurrent_state"`
}

type qwen35GDNCorpusMetadata struct {
	Format   string                    `json:"format"`
	Producer qwen35GDNProducerIdentity `json:"producer"`
	Geometry qwen35GDNCorpusGeometry   `json:"geometry"`
	Epsilon  float64                   `json:"epsilon"`
	Steps    []qwen35GDNStepMetadata   `json:"steps"`
	Tensors  []qwen35GDNTensorMetadata `json:"tensors"`
}

type qwen35GDNTensorContract struct {
	Name           string
	File           string
	Role           string
	Shape          []int
	RequireNonzero bool
	MustBeZero     bool
}

type qwen35GDNCorpus struct {
	Metadata       qwen35GDNCorpusMetadata
	Tensors        map[string][]float32
	ManifestSHA256 string
}

type qwen35GDNManifestEntry struct {
	Name   string
	Digest string
}

func qwen35GDNStepTensor(step int, kind string) string {
	return fmt.Sprintf("step.%03d.%s", step, kind)
}

func qwen35GDNTensorFile(name string) string { return name + ".f32le" }

func qwen35GDNFixtureInput(step, hidden int) []float32 {
	x := make([]float32, hidden)
	for i := range x {
		// Four deterministic, distinct, well-scaled normalized-input vectors.
		v := ((step+1)*(i+5)*17 + i*i*3 + 11) % 101
		x[i] = (float32(v) - 50) / 53
	}
	return x
}

func qwen35GDNCPUConvState(state linearAttnLayerState, keep, convDim int) []float32 {
	out := make([]float32, keep*convDim)
	start := keep - len(state.conv)
	if start < 0 {
		start = 0
	}
	for i, row := range state.conv {
		if i+start >= keep {
			break
		}
		copy(out[(i+start)*convDim:(i+start+1)*convDim], row)
	}
	return out
}

func qwen35GDNCPURecurrentState(state linearAttnLayerState) []float32 {
	var out []float32
	for _, head := range state.recurrent {
		out = append(out, head...)
	}
	return out
}

func qwen35GDNCorpusContracts(g qwen35GDNCorpusGeometry) ([]qwen35GDNTensorContract, error) {
	if err := validateQwen35GDNGeometry(g); err != nil {
		return nil, err
	}
	var contracts []qwen35GDNTensorContract
	add := func(name, role string, nonzero, zero bool, shape ...int) {
		contracts = append(contracts, qwen35GDNTensorContract{
			Name: name, File: qwen35GDNTensorFile(name), Role: role,
			Shape: append([]int(nil), shape...), RequireNonzero: nonzero, MustBeZero: zero,
		})
	}
	add(qwen35GDNWeightInQKV, "weight", true, false, g.ConvDim, g.HiddenSize)
	add(qwen35GDNWeightInZ, "weight", true, false, g.ValueDim, g.HiddenSize)
	add(qwen35GDNWeightInB, "weight", true, false, g.NumValueHeads, g.HiddenSize)
	add(qwen35GDNWeightInA, "weight", true, false, g.NumValueHeads, g.HiddenSize)
	add(qwen35GDNWeightConv, "weight", true, false, g.ConvDim, 1, g.ConvKernel)
	add(qwen35GDNWeightALog, "weight", true, false, g.NumValueHeads)
	add(qwen35GDNWeightDT, "weight", true, false, g.NumValueHeads)
	add(qwen35GDNWeightNorm, "weight", true, false, g.ValueHeadDim)
	add(qwen35GDNWeightOut, "weight", true, false, g.HiddenSize, g.ValueDim)
	add(qwen35GDNInitialConvState, "initial_state", false, true, g.ConvKernel-1, g.ConvDim)
	add(qwen35GDNInitialRecurrentState, "initial_state", false, true, g.NumValueHeads, g.KeyHeadDim, g.ValueHeadDim)
	for step := 0; step < g.StepCount; step++ {
		add(qwen35GDNStepTensor(step, "input"), "step_input", true, false, g.HiddenSize)
		add(qwen35GDNStepTensor(step, "output"), "cpu_reference", true, false, g.HiddenSize)
		add(qwen35GDNStepTensor(step, "conv_state"), "cpu_reference", true, false, g.ConvKernel-1, g.ConvDim)
		add(qwen35GDNStepTensor(step, "recurrent_state"), "cpu_reference", true, false, g.NumValueHeads, g.KeyHeadDim, g.ValueHeadDim)
	}
	sort.Slice(contracts, func(i, j int) bool { return contracts[i].Name < contracts[j].Name })
	return contracts, nil
}

func validateQwen35GDNGeometry(g qwen35GDNCorpusGeometry) error {
	values := []int{g.HiddenSize, g.NumKeyHeads, g.NumValueHeads, g.KeyHeadDim, g.ValueHeadDim, g.KeyDim, g.ValueDim, g.ConvDim, g.ConvKernel, g.StepCount}
	for _, value := range values {
		if value <= 0 {
			return fmt.Errorf("corpus geometry contains non-positive value %d", value)
		}
	}
	if g.StepCount != qwen35GDNCorpusStepCount {
		return fmt.Errorf("corpus step_count=%d, want %d for %s", g.StepCount, qwen35GDNCorpusStepCount, qwen35GDNCorpusFormat)
	}
	if g.ConvKernel < 2 {
		return fmt.Errorf("corpus conv_kernel=%d, want >=2", g.ConvKernel)
	}
	if g.KeyDim != g.NumKeyHeads*g.KeyHeadDim || g.ValueDim != g.NumValueHeads*g.ValueHeadDim || g.ConvDim != 2*g.KeyDim+g.ValueDim {
		return fmt.Errorf("corpus geometry derivation mismatch: key=%d/%d value=%d/%d conv=%d/%d",
			g.KeyDim, g.NumKeyHeads*g.KeyHeadDim, g.ValueDim, g.NumValueHeads*g.ValueHeadDim, g.ConvDim, 2*g.KeyDim+g.ValueDim)
	}
	if g.NumValueHeads%g.NumKeyHeads != 0 {
		return fmt.Errorf("corpus value heads %d are not divisible by key heads %d", g.NumValueHeads, g.NumKeyHeads)
	}
	return nil
}

func qwen35GDNCorpusData() (qwen35GDNCorpusMetadata, map[string][]float32, error) {
	cfg := qwen35HybridTestCfg()
	nK, nV, kHd, vHd, keyDim, valueDim, convDim := cfg.linearAttnDims()
	g := qwen35GDNCorpusGeometry{
		HiddenSize: cfg.HiddenSize, NumKeyHeads: nK, NumValueHeads: nV,
		KeyHeadDim: kHd, ValueHeadDim: vHd, KeyDim: keyDim, ValueDim: valueDim,
		ConvDim: convDim, ConvKernel: cfg.LinearConvKernelDim, StepCount: qwen35GDNCorpusStepCount,
	}
	contracts, err := qwen35GDNCorpusContracts(g)
	if err != nil {
		return qwen35GDNCorpusMetadata{}, nil, err
	}
	m := NewSynthetic(cfg)
	cpu := m.NewSession()
	p := func(suffix string) string { return layerName(0, suffix) }
	cloneTensor := func(suffix string) []float32 { return append([]float32(nil), m.tensor(p(suffix))...) }
	tensors := map[string][]float32{
		qwen35GDNWeightInQKV:           cloneTensor("linear_attn.in_proj_qkv.weight"),
		qwen35GDNWeightInZ:             cloneTensor("linear_attn.in_proj_z.weight"),
		qwen35GDNWeightInB:             cloneTensor("linear_attn.in_proj_b.weight"),
		qwen35GDNWeightInA:             cloneTensor("linear_attn.in_proj_a.weight"),
		qwen35GDNWeightConv:            cloneTensor("linear_attn.conv1d.weight"),
		qwen35GDNWeightALog:            cloneTensor("linear_attn.A_log"),
		qwen35GDNWeightDT:              cloneTensor("linear_attn.dt_bias"),
		qwen35GDNWeightNorm:            cloneTensor("linear_attn.norm.weight"),
		qwen35GDNWeightOut:             cloneTensor("linear_attn.out_proj.weight"),
		qwen35GDNInitialConvState:      make([]float32, (g.ConvKernel-1)*g.ConvDim),
		qwen35GDNInitialRecurrentState: make([]float32, g.NumValueHeads*g.KeyHeadDim*g.ValueHeadDim),
	}
	for step := 0; step < g.StepCount; step++ {
		input := qwen35GDNFixtureInput(step, g.HiddenSize)
		tensors[qwen35GDNStepTensor(step, "input")] = input
		tensors[qwen35GDNStepTensor(step, "output")] = append([]float32(nil), cpu.linearAttnStep(0, input, residentKernel{m})...)
		state := cpu.Cache.linear.layers[0]
		tensors[qwen35GDNStepTensor(step, "conv_state")] = qwen35GDNCPUConvState(state, g.ConvKernel-1, g.ConvDim)
		tensors[qwen35GDNStepTensor(step, "recurrent_state")] = qwen35GDNCPURecurrentState(state)
	}

	metadata := qwen35GDNCorpusMetadata{
		Format: qwen35GDNCorpusFormat, Producer: qwen35GDNCorpusProducer,
		Geometry: g, Epsilon: cfg.RMSNormEps,
	}
	byName := make(map[string]qwen35GDNTensorMetadata, len(contracts))
	for _, contract := range contracts {
		data, ok := tensors[contract.Name]
		if !ok {
			return qwen35GDNCorpusMetadata{}, nil, fmt.Errorf("producer omitted tensor %q", contract.Name)
		}
		elements, err := qwen35GDNShapeElements(contract.Shape)
		if err != nil || elements != len(data) {
			return qwen35GDNCorpusMetadata{}, nil, fmt.Errorf("producer tensor %q length=%d shape=%v: %v", contract.Name, len(data), contract.Shape, err)
		}
		norm, err := qwen35GDNVectorNorm(data)
		if err != nil {
			return qwen35GDNCorpusMetadata{}, nil, fmt.Errorf("producer tensor %q: %w", contract.Name, err)
		}
		if contract.RequireNonzero && norm == 0 {
			return qwen35GDNCorpusMetadata{}, nil, fmt.Errorf("producer tensor %q has zero norm", contract.Name)
		}
		if contract.MustBeZero && norm != 0 {
			return qwen35GDNCorpusMetadata{}, nil, fmt.Errorf("producer tensor %q must start at zero", contract.Name)
		}
		tensorMeta := qwen35GDNTensorMetadata{
			Name: contract.Name, File: contract.File, Role: contract.Role, DType: "float32-le",
			Shape: append([]int(nil), contract.Shape...), Elements: elements, Norm: norm,
			RequireNonzero: contract.RequireNonzero, MustBeZero: contract.MustBeZero,
		}
		metadata.Tensors = append(metadata.Tensors, tensorMeta)
		byName[contract.Name] = tensorMeta
	}
	for step := 0; step < g.StepCount; step++ {
		ref := func(kind string) qwen35GDNReferenceMetadata {
			name := qwen35GDNStepTensor(step, kind)
			return qwen35GDNReferenceMetadata{Tensor: name, Norm: byName[name].Norm, MaxAbsError: 0}
		}
		metadata.Steps = append(metadata.Steps, qwen35GDNStepMetadata{
			Index: step, Input: qwen35GDNStepTensor(step, "input"),
			Output: ref("output"), ConvState: ref("conv_state"), RecurrentState: ref("recurrent_state"),
		})
	}
	return metadata, tensors, nil
}

func writeQwen35GDNCorpus(dir string) (string, error) {
	if strings.TrimSpace(dir) == "" {
		return "", fmt.Errorf("%s is empty", qwen35GDNCorpusPathEnv)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create corpus directory: %w", err)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return "", fmt.Errorf("stat corpus directory: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("corpus path must be a real directory: %s", dir)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("read corpus directory: %w", err)
	}
	if len(entries) != 0 {
		return "", fmt.Errorf("corpus directory must be empty, found %d entries", len(entries))
	}
	metadata, tensors, err := qwen35GDNCorpusData()
	if err != nil {
		return "", err
	}
	for _, tensor := range metadata.Tensors {
		if err := writeQwen35GDNF32(filepath.Join(dir, tensor.File), tensors[tensor.Name]); err != nil {
			return "", fmt.Errorf("write tensor %q: %w", tensor.Name, err)
		}
	}
	metadataBytes, err := marshalQwen35GDNMetadata(metadata)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, qwen35GDNCorpusMetadataFile), metadataBytes, 0o644); err != nil {
		return "", fmt.Errorf("write corpus metadata: %w", err)
	}
	return writeQwen35GDNManifest(dir)
}

func marshalQwen35GDNMetadata(metadata qwen35GDNCorpusMetadata) ([]byte, error) {
	b, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal corpus metadata: %w", err)
	}
	return append(b, '\n'), nil
}

func writeQwen35GDNF32(path string, values []float32) error {
	b := make([]byte, len(values)*4)
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("value[%d] is non-finite", i)
		}
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(value))
	}
	return os.WriteFile(path, b, 0o644)
}

func writeQwen35GDNManifest(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("read corpus for manifest: %w", err)
	}
	var names []string
	for _, entry := range entries {
		if entry.Name() == qwen35GDNCorpusManifestFile {
			continue
		}
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("manifest input %q is not a regular file", entry.Name())
		}
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	var manifest strings.Builder
	for _, name := range names {
		digest, _, err := qwen35GDNHashFile(filepath.Join(dir, name))
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&manifest, "%s  %s\n", digest, name)
	}
	manifestBytes := []byte(manifest.String())
	if err := os.WriteFile(filepath.Join(dir, qwen35GDNCorpusManifestFile), manifestBytes, 0o644); err != nil {
		return "", fmt.Errorf("write corpus manifest: %w", err)
	}
	digest := sha256.Sum256(manifestBytes)
	return hex.EncodeToString(digest[:]), nil
}

func loadQwen35GDNCorpus(dir string) (*qwen35GDNCorpus, error) {
	manifestBytes, manifestEntries, err := readQwen35GDNManifest(dir)
	if err != nil {
		return nil, err
	}
	metadataBytes, err := os.ReadFile(filepath.Join(dir, qwen35GDNCorpusMetadataFile))
	if err != nil {
		return nil, fmt.Errorf("read corpus metadata: %w", err)
	}
	if len(metadataBytes) > 1<<20 {
		return nil, fmt.Errorf("corpus metadata is %d bytes, limit is %d", len(metadataBytes), 1<<20)
	}
	var metadata qwen35GDNCorpusMetadata
	decoder := json.NewDecoder(bytes.NewReader(metadataBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metadata); err != nil {
		return nil, fmt.Errorf("parse corpus metadata: %w", err)
	}
	if err := ensureQwen35GDNJSONEOF(decoder); err != nil {
		return nil, err
	}
	contracts, err := validateQwen35GDNMetadata(metadata, manifestEntries)
	if err != nil {
		return nil, err
	}
	tensors := make(map[string][]float32, len(contracts))
	tensorMeta := make(map[string]qwen35GDNTensorMetadata, len(metadata.Tensors))
	for _, tensor := range metadata.Tensors {
		tensorMeta[tensor.Name] = tensor
	}
	var totalElements int
	for _, contract := range contracts {
		meta := tensorMeta[contract.Name]
		if meta.Elements > qwen35GDNCorpusMaxElements-totalElements {
			return nil, fmt.Errorf("corpus tensor elements exceed bounded limit %d", qwen35GDNCorpusMaxElements)
		}
		totalElements += meta.Elements
		values, err := readQwen35GDNF32(filepath.Join(dir, meta.File), meta.Elements)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", meta.Name, err)
		}
		norm, err := qwen35GDNVectorNorm(values)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", meta.Name, err)
		}
		if math.Float64bits(norm) != math.Float64bits(meta.Norm) {
			return nil, fmt.Errorf("tensor %q norm=%g, metadata=%g", meta.Name, norm, meta.Norm)
		}
		if meta.RequireNonzero && norm == 0 {
			return nil, fmt.Errorf("tensor %q violates nonzero requirement", meta.Name)
		}
		if meta.MustBeZero && norm != 0 {
			return nil, fmt.Errorf("tensor %q violates zero-initial-state requirement", meta.Name)
		}
		tensors[meta.Name] = values
	}
	if err := validateQwen35GDNSteps(metadata, tensors, tensorMeta); err != nil {
		return nil, err
	}
	manifestDigest := sha256.Sum256(manifestBytes)
	return &qwen35GDNCorpus{
		Metadata: metadata, Tensors: tensors, ManifestSHA256: hex.EncodeToString(manifestDigest[:]),
	}, nil
}

func readQwen35GDNManifest(dir string) ([]byte, []qwen35GDNManifestEntry, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, nil, fmt.Errorf("%s is empty", qwen35GDNCorpusPathEnv)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("read corpus directory: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return nil, nil, fmt.Errorf("corpus path must be a real directory: %s", dir)
	}
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("read corpus directory: %w", err)
	}
	if len(dirEntries) > qwen35GDNCorpusMaxFiles {
		return nil, nil, fmt.Errorf("corpus has %d files, limit is %d", len(dirEntries), qwen35GDNCorpusMaxFiles)
	}
	var actual []string
	var totalBytes int64
	for _, entry := range dirEntries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			return nil, nil, fmt.Errorf("corpus entry %q is not a regular file", entry.Name())
		}
		entryInfo, err := entry.Info()
		if err != nil || !entryInfo.Mode().IsRegular() {
			return nil, nil, fmt.Errorf("corpus entry %q is not a regular file", entry.Name())
		}
		totalBytes += entryInfo.Size()
		if totalBytes > qwen35GDNCorpusMaxBytes {
			return nil, nil, fmt.Errorf("corpus is %d bytes, limit is %d", totalBytes, qwen35GDNCorpusMaxBytes)
		}
		actual = append(actual, entry.Name())
	}
	sort.Strings(actual)
	manifestBytes, err := os.ReadFile(filepath.Join(dir, qwen35GDNCorpusManifestFile))
	if err != nil {
		return nil, nil, fmt.Errorf("read corpus manifest: %w", err)
	}
	if len(manifestBytes) == 0 || len(manifestBytes) > 1<<20 || manifestBytes[len(manifestBytes)-1] != '\n' || bytes.Contains(manifestBytes, []byte{'\r'}) {
		return nil, nil, fmt.Errorf("corpus manifest is malformed or exceeds the 1 MiB limit")
	}
	lines := strings.Split(string(manifestBytes[:len(manifestBytes)-1]), "\n")
	entries := make([]qwen35GDNManifestEntry, 0, len(lines))
	previous := ""
	for lineNo, line := range lines {
		digest, name, ok := strings.Cut(line, "  ")
		if !ok || len(digest) != sha256.Size*2 || name == "" || strings.Contains(name, "  ") || filepath.Base(name) != name || name == qwen35GDNCorpusManifestFile {
			return nil, nil, fmt.Errorf("corpus manifest line %d is malformed", lineNo+1)
		}
		decoded, err := hex.DecodeString(digest)
		if err != nil || hex.EncodeToString(decoded) != digest {
			return nil, nil, fmt.Errorf("corpus manifest line %d has invalid lowercase SHA-256", lineNo+1)
		}
		if previous != "" && name <= previous {
			return nil, nil, fmt.Errorf("corpus manifest is not strictly sorted at %q", name)
		}
		previous = name
		entries = append(entries, qwen35GDNManifestEntry{Name: name, Digest: digest})
	}
	expectedActual := []string{qwen35GDNCorpusManifestFile}
	for _, entry := range entries {
		expectedActual = append(expectedActual, entry.Name)
	}
	sort.Strings(expectedActual)
	if !equalQwen35GDNStrings(actual, expectedActual) {
		return nil, nil, fmt.Errorf("corpus directory entries mismatch: have=%v manifest=%v", actual, expectedActual)
	}
	for _, entry := range entries {
		digest, _, err := qwen35GDNHashFile(filepath.Join(dir, entry.Name))
		if err != nil {
			return nil, nil, err
		}
		if digest != entry.Digest {
			return nil, nil, fmt.Errorf("corpus hash mismatch for %q: have=%s want=%s", entry.Name, digest, entry.Digest)
		}
	}
	return manifestBytes, entries, nil
}

func validateQwen35GDNMetadata(metadata qwen35GDNCorpusMetadata, manifest []qwen35GDNManifestEntry) ([]qwen35GDNTensorContract, error) {
	if metadata.Format != qwen35GDNCorpusFormat {
		return nil, fmt.Errorf("corpus schema=%q, want %q", metadata.Format, qwen35GDNCorpusFormat)
	}
	if metadata.Producer != qwen35GDNCorpusProducer {
		return nil, fmt.Errorf("corpus producer identity=%+v, want %+v", metadata.Producer, qwen35GDNCorpusProducer)
	}
	if math.IsNaN(metadata.Epsilon) || math.IsInf(metadata.Epsilon, 0) || metadata.Epsilon <= 0 {
		return nil, fmt.Errorf("corpus epsilon=%g is not finite and positive", metadata.Epsilon)
	}
	contracts, err := qwen35GDNCorpusContracts(metadata.Geometry)
	if err != nil {
		return nil, err
	}
	if len(metadata.Tensors) != len(contracts) {
		return nil, fmt.Errorf("corpus tensor metadata count=%d, want %d", len(metadata.Tensors), len(contracts))
	}
	manifestNames := make([]string, 0, len(manifest))
	for _, entry := range manifest {
		manifestNames = append(manifestNames, entry.Name)
	}
	expectedManifest := []string{qwen35GDNCorpusMetadataFile}
	for i, contract := range contracts {
		meta := metadata.Tensors[i]
		if meta.Name != contract.Name {
			return nil, fmt.Errorf("corpus tensor metadata is not canonical/sorted at %d: have=%q want=%q", i, meta.Name, contract.Name)
		}
		elements, shapeErr := qwen35GDNShapeElements(contract.Shape)
		if shapeErr != nil {
			return nil, shapeErr
		}
		if meta.File != contract.File || meta.Role != contract.Role || meta.DType != "float32-le" ||
			!equalQwen35GDNInts(meta.Shape, contract.Shape) || meta.Elements != elements ||
			meta.RequireNonzero != contract.RequireNonzero || meta.MustBeZero != contract.MustBeZero {
			return nil, fmt.Errorf("corpus tensor %q contract mismatch: metadata=%+v expected=%+v", meta.Name, meta, contract)
		}
		if math.IsNaN(meta.Norm) || math.IsInf(meta.Norm, 0) || meta.Norm < 0 {
			return nil, fmt.Errorf("corpus tensor %q has invalid norm metadata %g", meta.Name, meta.Norm)
		}
		expectedManifest = append(expectedManifest, contract.File)
	}
	sort.Strings(expectedManifest)
	if !equalQwen35GDNStrings(manifestNames, expectedManifest) {
		return nil, fmt.Errorf("corpus manifest entries do not match schema: have=%v want=%v", manifestNames, expectedManifest)
	}
	return contracts, nil
}

func validateQwen35GDNSteps(metadata qwen35GDNCorpusMetadata, tensors map[string][]float32, tensorMeta map[string]qwen35GDNTensorMetadata) error {
	if len(metadata.Steps) != metadata.Geometry.StepCount {
		return fmt.Errorf("corpus step metadata count=%d, want %d", len(metadata.Steps), metadata.Geometry.StepCount)
	}
	var previousInput []float32
	for index, step := range metadata.Steps {
		if step.Index != index || step.Input != qwen35GDNStepTensor(index, "input") {
			return fmt.Errorf("corpus step %d identity mismatch: %+v", index, step)
		}
		input := tensors[step.Input]
		if index > 0 && equalQwen35GDNFloat32Bits(input, previousInput) {
			return fmt.Errorf("corpus step %d input duplicates step %d", index, index-1)
		}
		previousInput = input
		refs := []struct {
			kind string
			ref  qwen35GDNReferenceMetadata
		}{
			{"output", step.Output},
			{"conv_state", step.ConvState},
			{"recurrent_state", step.RecurrentState},
		}
		for _, item := range refs {
			wantName := qwen35GDNStepTensor(index, item.kind)
			meta, ok := tensorMeta[item.ref.Tensor]
			if !ok || item.ref.Tensor != wantName {
				return fmt.Errorf("corpus step %d %s reference=%q, want %q", index, item.kind, item.ref.Tensor, wantName)
			}
			if math.Float64bits(item.ref.Norm) != math.Float64bits(meta.Norm) || item.ref.Norm <= 0 {
				return fmt.Errorf("corpus step %d %s norm metadata=%g, tensor norm=%g", index, item.kind, item.ref.Norm, meta.Norm)
			}
			if math.IsNaN(item.ref.MaxAbsError) || math.IsInf(item.ref.MaxAbsError, 0) || item.ref.MaxAbsError != 0 {
				return fmt.Errorf("corpus step %d %s max_abs_error=%g, want exact producer value 0", index, item.kind, item.ref.MaxAbsError)
			}
		}
	}
	return nil
}

func selectQwen35GDNCorpus(required bool, path string) (string, bool, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		if required {
			return "", false, fmt.Errorf("required CUDA GDN acceptance needs explicit %s=<corpus-dir>; refusing in-process oracle regeneration", qwen35GDNCorpusPathEnv)
		}
		return "", true, nil
	}
	return path, false, nil
}

func ensureQwen35GDNJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if err == io.EOF {
		return nil
	}
	if err == nil {
		return fmt.Errorf("parse corpus metadata: multiple JSON values")
	}
	return fmt.Errorf("parse corpus metadata trailer: %w", err)
}

func readQwen35GDNF32(path string, elements int) ([]float32, error) {
	if elements < 0 || elements > qwen35GDNCorpusMaxElements {
		return nil, fmt.Errorf("element count %d exceeds bounded limit", elements)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	if len(b) != elements*4 {
		return nil, fmt.Errorf("byte length=%d, want %d", len(b), elements*4)
	}
	values := make([]float32, elements)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
			return nil, fmt.Errorf("value[%d] is non-finite", i)
		}
	}
	return values, nil
}

func qwen35GDNVectorNorm(values []float32) (float64, error) {
	var norm2 float64
	for i, value := range values {
		v := float64(value)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("value[%d] is non-finite", i)
		}
		norm2 += v * v
	}
	norm := math.Sqrt(norm2)
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		return 0, fmt.Errorf("norm is non-finite")
	}
	return norm, nil
}

func qwen35GDNShapeElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty tensor shape")
	}
	elements := 1
	for _, dim := range shape {
		if dim <= 0 || elements > qwen35GDNCorpusMaxElements/dim {
			return 0, fmt.Errorf("tensor shape %v exceeds bounded element limit", shape)
		}
		elements *= dim
	}
	return elements, nil
}

func qwen35GDNHashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, fmt.Errorf("open corpus file %q: %w", filepath.Base(path), err)
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, qwen35GDNCorpusMaxBytes+1))
	if err != nil {
		return "", n, fmt.Errorf("hash corpus file %q: %w", filepath.Base(path), err)
	}
	if n > qwen35GDNCorpusMaxBytes {
		return "", n, fmt.Errorf("corpus file %q exceeds %d-byte bound", filepath.Base(path), qwen35GDNCorpusMaxBytes)
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func equalQwen35GDNInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalQwen35GDNStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalQwen35GDNFloat32Bits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
