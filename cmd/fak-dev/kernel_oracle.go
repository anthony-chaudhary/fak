package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// kernel-oracle joins llama.cpp `test-backend-ops perf` rows with fak's Vulkan
// kernel timings at the same op, dtype and shape, and ranks where fak is slower.
// ggml is used as an external binary only; no ggml code enters fak.

const (
	kernelOracleShapesSchema  = "fak.kernel-oracle.shapes/v1"
	kernelOracleReceiptSchema = "fak.kernel-oracle/v1"
	kernelOraclePortRatio     = 1.25

	oracleStatusBoth        = "both"
	oracleStatusFakMissing  = "fak_missing"
	oracleStatusGGMLMissing = "ggml_missing"
	oracleStatusBothMissing = "both_missing"
)

var errKernelOracleUsage = errors.New("kernel-oracle usage")

type oracleShape struct {
	Op      string   `json:"op"`
	Dtype   string   `json:"dtype"`
	M       int64    `json:"m"`
	K       int64    `json:"k"`
	N       int64    `json:"n"`
	Count   int      `json:"count"`
	Tensors []string `json:"tensors,omitempty"`
}

type oracleShapeList struct {
	Schema   string        `json:"schema"`
	Model    string        `json:"model"`
	Arch     string        `json:"arch,omitempty"`
	Ns       []int64       `json:"ns"`
	Shapes   []oracleShape `json:"shapes"`
	Ops      []string      `json:"ops"`
	Commands []string      `json:"commands"`
}

type oracleMeasurement struct {
	Op      string
	Dtype   string
	M, K, N int64
	Params  string
	US      float64
	GBps    float64
	TFLOPS  float64
	Samples int
}

func (m oracleMeasurement) key() string {
	if m.Op == "MUL_MAT" && m.M > 0 {
		return oracleMatKey(m.Dtype, m.M, m.K, m.N)
	}
	return m.Op + "|" + m.Params
}

func oracleMatKey(dtype string, m, k, n int64) string {
	return fmt.Sprintf("MUL_MAT|%s|%d|%d|%d", dtype, m, k, n)
}

type oracleRow struct {
	Op          string   `json:"op"`
	Shape       string   `json:"shape"`
	Dtype       string   `json:"dtype,omitempty"`
	N           int64    `json:"n,omitempty"`
	Params      string   `json:"params,omitempty"`
	Count       int      `json:"count,omitempty"`
	Status      string   `json:"status"`
	FakUS       *float64 `json:"fak_us,omitempty"`
	FakSource   string   `json:"fak_source,omitempty"`
	FakInsituUS *float64 `json:"fak_insitu_us,omitempty"`
	GGMLUS      *float64 `json:"ggml_us,omitempty"`
	Ratio       *float64 `json:"ratio,omitempty"`
	FakGBps     *float64 `json:"fak_gbps,omitempty"`
	GGMLGBps    *float64 `json:"ggml_gbps,omitempty"`
	FakTFLOPS   *float64 `json:"fak_tflops,omitempty"`
	GGMLTFLOPS  *float64 `json:"ggml_tflops,omitempty"`
	SharePct    *float64 `json:"share_pct,omitempty"`
	Score       *float64 `json:"score,omitempty"`
	PortOrAdapt bool     `json:"port_or_adapt,omitempty"`
	m, k        int64
}

type oracleStage struct {
	Kernel      string  `json:"kernel"`
	Groups      string  `json:"groups"`
	CallsBatch  float64 `json:"calls_per_batch"`
	USPerCall   float64 `json:"us_per_call"`
	SharePct    float64 `json:"share_pct"`
	JoinedShape string  `json:"joined_shape,omitempty"`
	Unjoined    string  `json:"unjoined,omitempty"`
	gx, gy      int64
}

type oracleSummary struct {
	Rows            int `json:"rows"`
	Both            int `json:"both"`
	FakMissing      int `json:"fak_missing"`
	GGMLMissing     int `json:"ggml_missing"`
	PortOrAdapt     int `json:"port_or_adapt"`
	GGMLUnsupported int `json:"ggml_unsupported"`
}

type oracleReceipt struct {
	Schema        string            `json:"schema"`
	EvidenceClass string            `json:"evidence_class"`
	Node          string            `json:"node,omitempty"`
	Model         string            `json:"model,omitempty"`
	Rule          string            `json:"rule"`
	GGMLBackend   string            `json:"ggml_backend,omitempty"`
	GGMLDevice    string            `json:"ggml_device,omitempty"`
	Inputs        map[string]string `json:"inputs"`
	Notes         []string          `json:"notes,omitempty"`
	Summary       oracleSummary     `json:"summary"`
	Rows          []oracleRow       `json:"rows"`
	FakStages     []oracleStage     `json:"fak_stages,omitempty"`
}

func runKernelOracle(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeKernelOracleUsage(stderr)
		return 2
	}
	var err error
	switch argv[0] {
	case "shapes":
		err = runKernelOracleShapes(stdout, stderr, argv[1:])
	case "rank":
		err = runKernelOracleRank(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		writeKernelOracleUsage(stdout)
		return 0
	default:
		writeKernelOracleUsage(stderr)
		return 2
	}
	if errors.Is(err, errKernelOracleUsage) || errors.Is(err, flag.ErrHelp) {
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, "fak-dev kernel-oracle:", err)
		return 1
	}
	return 0
}

func writeKernelOracleUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak-dev kernel-oracle shapes (--gguf F | --metadata-json F) [--n 1,2,4,8,512] [--out F]")
	fmt.Fprintln(w, "       fak-dev kernel-oracle rank --shapes F [--ggml F] [--fak-bench F] [--fak-profile F]")
	fmt.Fprintln(w, "                                  [--ggml-backend Vulkan0] [--evidence-class C] [--node N] [--out F]")
	fmt.Fprintln(w, "ggml input: console output of `test-backend-ops perf -b <backend> [--test-file F]`")
	fmt.Fprintln(w, "fak input:  `go test -bench` lines and/or FAK_VULKAN_TIMESTAMP_PROFILE=1 ts-stage lines")
}

func runKernelOracleShapes(stdout, stderr io.Writer, argv []string) error {
	fs := flag.NewFlagSet("kernel-oracle shapes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ggufPath := fs.String("gguf", "", "GGUF checkpoint to read the tensor directory from")
	metaPath := fs.String("metadata-json", "", "`ggufmeta -json` export instead of --gguf")
	nsFlag := fs.String("n", "1,2,4,8,512", "comma-separated token counts")
	out := fs.String("out", "", "write JSON here instead of stdout")
	if err := fs.Parse(argv); err != nil {
		return err
	}
	if (*ggufPath == "") == (*metaPath == "") {
		fmt.Fprintln(stderr, "kernel-oracle shapes: exactly one of --gguf or --metadata-json is required")
		return errKernelOracleUsage
	}
	ns, err := parseOracleNs(*nsFlag)
	if err != nil {
		return err
	}
	var meta ggufload.Metadata
	model := *ggufPath
	if *ggufPath != "" {
		f, err := ggufload.Open(*ggufPath)
		if err != nil {
			return err
		}
		meta = f.ExportMetadata()
	} else {
		model = *metaPath
		b, err := os.ReadFile(*metaPath)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(b, &meta); err != nil {
			return fmt.Errorf("decode %s: %w", *metaPath, err)
		}
	}
	list := buildOracleShapes(meta, model, ns)
	return writeOracleJSON(stdout, *out, list)
}

func parseOracleNs(s string) ([]int64, error) {
	var ns []int64
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.ParseInt(p, 10, 64)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("--n: bad token count %q", p)
		}
		ns = append(ns, n)
	}
	if len(ns) == 0 {
		return nil, fmt.Errorf("--n: empty")
	}
	return ns, nil
}

func buildOracleShapes(meta ggufload.Metadata, model string, ns []int64) oracleShapeList {
	arch := ""
	for _, kv := range meta.Metadata {
		if kv.Key == "general.architecture" {
			arch, _ = kv.Value.(string)
		}
	}
	type wkey struct {
		dtype string
		m, k  int64
	}
	agg := map[wkey]*oracleShape{}
	hasSSM := false
	for _, t := range meta.Tensors {
		if strings.Contains(t.Name, "ssm_") {
			hasSSM = true
		}
		if !oracleIsMatmulWeight(t) {
			continue
		}
		dt := strings.ToLower(t.Type)
		k := wkey{dt, int64(t.Dims[1]), int64(t.Dims[0])}
		s, ok := agg[k]
		if !ok {
			s = &oracleShape{Op: "MUL_MAT", Dtype: dt, M: k.m, K: k.k}
			agg[k] = s
		}
		s.Count++
		role := oracleTensorRole(t.Name)
		if !containsString(s.Tensors, role) {
			s.Tensors = append(s.Tensors, role)
		}
	}
	list := oracleShapeList{Schema: kernelOracleShapesSchema, Model: model, Arch: arch, Ns: ns}
	for _, s := range agg {
		sort.Strings(s.Tensors)
		for _, n := range ns {
			c := *s
			c.N = n
			list.Shapes = append(list.Shapes, c)
		}
	}
	sort.Slice(list.Shapes, func(i, j int) bool { return oracleShapeLess(list.Shapes[i], list.Shapes[j]) })
	list.Ops = []string{"MUL_MAT", "FLASH_ATTN_EXT", "RMS_NORM"}
	if hasSSM {
		list.Ops = append(list.Ops, "GATED_DELTA_NET")
	}
	list.Commands = []string{
		"test-export-graph-ops -m <gguf> -ub 512 -np 1 -o ops-n1.txt   # pp n=512 + tg n=1",
		"test-export-graph-ops -m <gguf> -ub 512 -np 8 -o ops-n8.txt   # tg n=8",
		"test-backend-ops perf -b Vulkan0 --test-file ops-n1.txt > ggml-perf.txt",
		"FAK_RAWIQ_BENCH_SHAPES=<MxK,...> go test -tags vulkan -run '^$' -bench BenchmarkVulkanRawIQMatvec ./internal/compute/ > fak-bench.txt",
		"FAK_VULKAN_TIMESTAMP_PROFILE=1 fak serve ... 2> fak-profile.txt",
		"fak-dev kernel-oracle rank --shapes shapes.json --ggml ggml-perf.txt --fak-bench fak-bench.txt --fak-profile fak-profile.txt",
	}
	return list
}

func oracleIsMatmulWeight(t ggufload.TensorExport) bool {
	if len(t.Dims) != 2 || !strings.HasSuffix(t.Name, ".weight") {
		return false
	}
	if strings.HasPrefix(t.Name, "token_embd") || strings.Contains(t.Name, "conv1d") || oracleIsEngramTable(t.Name) {
		return false
	}
	if strings.EqualFold(t.Type, "F32") {
		return false
	}
	return t.Dims[0] >= 32 && t.Dims[1] >= 1
}

// Engram tables are gathered row-wise, not consumed as dense matmul weights.
// Match ggufload's table names without excluding the Engram projections.
func oracleIsEngramTable(name string) bool {
	if !strings.HasPrefix(name, "blk.") {
		return false
	}
	layer, suffix, ok := strings.Cut(strings.TrimPrefix(name, "blk."), ".")
	if !ok {
		return false
	}
	if _, err := strconv.Atoi(layer); err != nil {
		return false
	}
	switch strings.TrimSuffix(suffix, ".weight") {
	case "engram_table", "engram_embd":
		return true
	}
	return false
}

func oracleTensorRole(name string) string {
	parts := strings.Split(strings.TrimSuffix(name, ".weight"), ".")
	if len(parts) >= 3 && parts[0] == "blk" {
		return strings.Join(parts[2:], ".")
	}
	return strings.Join(parts, ".")
}

func containsString(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

func oracleShapeLess(a, b oracleShape) bool {
	if a.Dtype != b.Dtype {
		return a.Dtype < b.Dtype
	}
	if a.M != b.M {
		return a.M < b.M
	}
	if a.K != b.K {
		return a.K < b.K
	}
	return a.N < b.N
}

func runKernelOracleRank(stdout, stderr io.Writer, argv []string) error {
	fs := flag.NewFlagSet("kernel-oracle rank", flag.ContinueOnError)
	fs.SetOutput(stderr)
	shapesPath := fs.String("shapes", "", "shape list from `kernel-oracle shapes`")
	ggmlPath := fs.String("ggml", "", "test-backend-ops perf console output")
	benchPath := fs.String("fak-bench", "", "go test -bench output (BenchmarkVulkanRawIQMatvec)")
	profilePath := fs.String("fak-profile", "", "stderr carrying FAK_VULKAN_TIMESTAMP_PROFILE ts-stage lines")
	backend := fs.String("ggml-backend", "Vulkan0", "test-backend-ops backend section to read")
	benchMode := fs.String("fak-bench-mode", "native_stream", "benchmark sub-mode to read (native_stream streams from DRAM)")
	evidence := fs.String("evidence-class", "unlabeled", "HW-WITNESSED | SW-VERIFIED | FIXTURE | unlabeled")
	node := fs.String("node", "", "node the inputs were measured on")
	out := fs.String("out", "", "write JSON here instead of stdout")
	if err := fs.Parse(argv); err != nil {
		return err
	}
	if *shapesPath == "" || (*ggmlPath == "" && *benchPath == "" && *profilePath == "") {
		fmt.Fprintln(stderr, "kernel-oracle rank: --shapes and at least one of --ggml, --fak-bench, --fak-profile are required")
		return errKernelOracleUsage
	}
	var shapes oracleShapeList
	inputs := map[string]string{}
	if err := readOracleJSON(*shapesPath, &shapes, inputs, "shapes"); err != nil {
		return err
	}
	in := oracleRankInput{shapes: shapes}
	if *ggmlPath != "" {
		text, err := readOracleInput(*ggmlPath, inputs, "ggml")
		if err != nil {
			return err
		}
		in.ggml = parseGGMLPerf(text, *backend)
	}
	if *benchPath != "" {
		text, err := readOracleInput(*benchPath, inputs, "fak_bench")
		if err != nil {
			return err
		}
		in.fakBench = parseFakBench(text, *benchMode)
	}
	if *profilePath != "" {
		text, err := readOracleInput(*profilePath, inputs, "fak_profile")
		if err != nil {
			return err
		}
		in.stages = parseFakStages(text)
	}
	rec := rankKernelOracle(in)
	rec.EvidenceClass = *evidence
	rec.Node = *node
	rec.Inputs = inputs
	return writeOracleJSON(stdout, *out, rec)
}

type oracleRankInput struct {
	shapes   oracleShapeList
	ggml     ggmlPerf
	fakBench []oracleMeasurement
	stages   []oracleStage
}

// rankKernelOracle is pure: the same inputs always produce the same receipt bytes.
func rankKernelOracle(in oracleRankInput) oracleReceipt {
	rec := oracleReceipt{
		Schema:      kernelOracleReceiptSchema,
		Model:       in.shapes.Model,
		Rule:        "a kernel > 1.25x slower than ggml at the same op/dtype/shape defaults to port/ADAPT; an own design needs a recorded reason",
		GGMLBackend: in.ggml.Backend,
		GGMLDevice:  in.ggml.Device,
	}
	rows := map[string]*oracleRow{}
	var order []string
	add := func(key string, r oracleRow) *oracleRow {
		if p, ok := rows[key]; ok {
			return p
		}
		rows[key] = &r
		order = append(order, key)
		return rows[key]
	}
	for _, s := range in.shapes.Shapes {
		add(oracleMatKey(s.Dtype, s.M, s.K, s.N), oracleRow{
			Op: s.Op, Shape: fmt.Sprintf("%dx%d", s.M, s.K), Dtype: s.Dtype, N: s.N, Count: s.Count, m: s.M, k: s.K,
		})
	}
	for _, g := range in.ggml.Rows {
		r := add(g.key(), oracleRowFor(g))
		us := g.US
		r.GGMLUS = &us
		if g.Op == "MUL_MAT" && g.M > 0 {
			r.GGMLGBps, r.GGMLTFLOPS = oracleMatRates(g.Dtype, g.M, g.K, g.N, us)
		} else if g.GBps > 0 {
			v := g.GBps
			r.GGMLGBps = &v
		}
	}
	for _, f := range in.fakBench {
		r := add(f.key(), oracleRowFor(f))
		us := f.US
		r.FakUS = &us
		r.FakSource = "microbench"
	}
	joinOracleStages(in.stages, rows, in.shapes)
	rec.FakStages = in.stages
	for _, key := range order {
		r := rows[key]
		if r.FakUS == nil && r.FakInsituUS != nil {
			v := *r.FakInsituUS
			r.FakUS = &v
			r.FakSource = "insitu_profile"
		}
		if r.FakUS != nil && r.Op == "MUL_MAT" && r.m > 0 {
			r.FakGBps, r.FakTFLOPS = oracleMatRates(r.Dtype, r.m, r.k, r.N, *r.FakUS)
		}
		switch {
		case r.FakUS != nil && r.GGMLUS != nil:
			r.Status = oracleStatusBoth
			ratio := round3(*r.FakUS / *r.GGMLUS)
			r.Ratio = &ratio
			r.PortOrAdapt = ratio > kernelOraclePortRatio
		case r.GGMLUS != nil:
			r.Status = oracleStatusFakMissing
		case r.FakUS != nil:
			r.Status = oracleStatusGGMLMissing
		default:
			r.Status = oracleStatusBothMissing
		}
	}
	applyOracleShares(rows, order)
	out := make([]oracleRow, 0, len(order))
	for _, key := range order {
		out = append(out, *rows[key])
	}
	sort.SliceStable(out, func(i, j int) bool { return oracleRowLess(out[i], out[j]) })
	rec.Rows = out
	rec.Summary = summarizeOracle(out, in.ggml.Unsupported)
	if in.ggml.Rows == nil {
		rec.Notes = append(rec.Notes, "no ggml rows: every row is ggml_missing (see commands in the shape list)")
	}
	return rec
}

func oracleRowFor(m oracleMeasurement) oracleRow {
	if m.Op == "MUL_MAT" && m.M > 0 {
		return oracleRow{Op: m.Op, Shape: fmt.Sprintf("%dx%d", m.M, m.K), Dtype: m.Dtype, N: m.N, m: m.M, k: m.K}
	}
	return oracleRow{Op: m.Op, Shape: "generic", Dtype: m.Dtype, Params: m.Params}
}

// applyOracleShares weights each comparable row by its share of fak time among
// rows with the same n, counting every model tensor of that shape.
func applyOracleShares(rows map[string]*oracleRow, order []string) {
	total := map[int64]float64{}
	for _, key := range order {
		r := rows[key]
		if r.Status == oracleStatusBoth && r.Count > 0 {
			total[r.N] += float64(r.Count) * *r.FakUS
		}
	}
	for _, key := range order {
		r := rows[key]
		if r.Status != oracleStatusBoth || r.Count == 0 || total[r.N] <= 0 {
			continue
		}
		share := round3(100 * float64(r.Count) * *r.FakUS / total[r.N])
		score := round3(*r.Ratio * share / 100)
		r.SharePct, r.Score = &share, &score
	}
}

func oracleRowLess(a, b oracleRow) bool {
	ab, bb := a.Status == oracleStatusBoth, b.Status == oracleStatusBoth
	if ab != bb {
		return ab
	}
	if ab {
		as, bs := ptrOr(a.Score, -1), ptrOr(b.Score, -1)
		if as != bs {
			return as > bs
		}
		if *a.Ratio != *b.Ratio {
			return *a.Ratio > *b.Ratio
		}
	}
	if a.Status != b.Status {
		return a.Status < b.Status
	}
	if a.Op != b.Op {
		return a.Op < b.Op
	}
	if a.Dtype != b.Dtype {
		return a.Dtype < b.Dtype
	}
	if a.m != b.m {
		return a.m < b.m
	}
	if a.k != b.k {
		return a.k < b.k
	}
	if a.N != b.N {
		return a.N < b.N
	}
	return a.Params < b.Params
}

func summarizeOracle(rows []oracleRow, unsupported int) oracleSummary {
	s := oracleSummary{Rows: len(rows), GGMLUnsupported: unsupported}
	for _, r := range rows {
		switch r.Status {
		case oracleStatusBoth:
			s.Both++
		case oracleStatusFakMissing:
			s.FakMissing++
		case oracleStatusGGMLMissing:
			s.GGMLMissing++
		default:
			s.FakMissing++
			s.GGMLMissing++
		}
		if r.PortOrAdapt {
			s.PortOrAdapt++
		}
	}
	return s
}

func ptrOr(p *float64, d float64) float64 {
	if p == nil {
		return d
	}
	return *p
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

// oracleWeightBytesPerElem is the ggml block size in bytes per weight.
var oracleWeightBytesPerElem = map[string]float64{
	"f32": 4, "f16": 2, "bf16": 2,
	"q8_0": 34.0 / 32, "q4_0": 18.0 / 32, "q4_1": 20.0 / 32, "q5_0": 22.0 / 32, "q5_1": 24.0 / 32, "iq4_nl": 18.0 / 32,
	"q2_k": 84.0 / 256, "q3_k": 110.0 / 256, "q4_k": 144.0 / 256, "q5_k": 176.0 / 256, "q6_k": 210.0 / 256,
	"iq1_s": 50.0 / 256, "iq1_m": 56.0 / 256, "iq2_xxs": 66.0 / 256, "iq2_xs": 74.0 / 256, "iq2_s": 82.0 / 256,
	"iq3_xxs": 98.0 / 256, "iq3_s": 110.0 / 256, "iq4_xs": 136.0 / 256,
}

// oracleMatRates applies one bytes/flops model to both engines so their rates are
// comparable: weight bytes + f32 activations in and out; 2*M*N*K flops.
func oracleMatRates(dtype string, m, k, n int64, us float64) (*float64, *float64) {
	if us <= 0 {
		return nil, nil
	}
	tf := round3(2 * float64(m) * float64(n) * float64(k) / (us * 1e6))
	bpe, ok := oracleWeightBytesPerElem[dtype]
	if !ok {
		return nil, &tf
	}
	bytes := float64(m)*float64(k)*bpe + 4*float64(k*n) + 4*float64(m*n)
	gb := round3(bytes / (us * 1e3))
	return &gb, &tf
}

func readOracleInput(path string, inputs map[string]string, label string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	inputs[label] = "sha256:" + hex.EncodeToString(sum[:])
	return string(b), nil
}

func readOracleJSON(path string, v any, inputs map[string]string, label string) error {
	text, err := readOracleInput(path, inputs, label)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(text), v); err != nil {
		return fmt.Errorf("decode %s: %w", path, err)
	}
	return nil
}

func writeOracleJSON(stdout io.Writer, path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if path == "" {
		_, err = stdout.Write(b)
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
