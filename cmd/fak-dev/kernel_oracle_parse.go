package main

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var (
	oracleANSI        = regexp.MustCompile(`\x1b\[[0-9;]*m`)
	oracleGGMLBackend = regexp.MustCompile(`^\s*Backend \d+/\d+: (\S+)`)
	oracleGGMLDevice  = regexp.MustCompile(`^\s*Device description: (.+?)\s*$`)
	// test-backend-ops console perf row (tests/test-backend-ops.cpp print_perf_console).
	oracleGGMLRow    = regexp.MustCompile(`^\s*([A-Z][A-Z0-9_]*)\((.*)\):\s+(\d+) runs -\s+([0-9.]+) us/run -\s+([0-9.]+) ?(kB|[kMGT]FLOP)/run -\s+([0-9.]+) ?(GB/s|[kMGT]FLOPS)\s*$`)
	oracleGGMLNoSup  = regexp.MustCompile(`^\s*([A-Z][A-Z0-9_]*)\((.*)\):\s+not supported`)
	oracleTensorRef  = regexp.MustCompile(`^([a-z0-9_]+)\[(\d+),(\d+),(\d+),(\d+)\]`)
	oracleBenchRow   = regexp.MustCompile(`^(Benchmark\S+?)(?:-\d+)?\s+(\d+)\s+([0-9.]+) ns/op`)
	oracleBenchShape = regexp.MustCompile(`^out(\d+)_in(\d+)$`)
	oracleStageRow   = regexp.MustCompile(`fak-vulkan ts-stage kernel=(\S+) groups=(\d+)x(\d+) calls/batch=([0-9.]+) ms/batch=[0-9.]+ us/call=([0-9.]+) share=([0-9.]+)%`)
)

type ggmlPerf struct {
	Backend     string
	Device      string
	Rows        []oracleMeasurement
	Unsupported int
}

// parseGGMLPerf reads the rows of one backend section; output without backend
// headers (a single-backend capture) is read whole. Repeated cases keep the
// fastest run, matching min-of-N receipts elsewhere.
func parseGGMLPerf(text, backend string) ggmlPerf {
	var out ggmlPerf
	sawHeader, inSection := false, true
	idx := map[string]int{}
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimRight(oracleANSI.ReplaceAllString(raw, ""), "\r")
		if m := oracleGGMLBackend.FindStringSubmatch(line); m != nil {
			sawHeader = true
			inSection = backend == "" || m[1] == backend
			if inSection {
				out.Backend = m[1]
			}
			continue
		}
		if !inSection {
			continue
		}
		if m := oracleGGMLDevice.FindStringSubmatch(line); m != nil && out.Device == "" {
			out.Device = m[1]
			continue
		}
		if oracleGGMLNoSup.MatchString(line) {
			out.Unsupported++
			continue
		}
		m := oracleGGMLRow.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		meas, ok := ggmlMeasurement(m[1], m[2])
		if !ok {
			continue
		}
		meas.US, _ = strconv.ParseFloat(m[4], 64)
		rate, _ := strconv.ParseFloat(m[7], 64)
		if m[8] == "GB/s" {
			meas.GBps = rate
		} else {
			meas.TFLOPS = rate * oracleFlopScale(m[8]) / 1e12
		}
		meas.Samples = 1
		k := meas.key()
		if i, ok := idx[k]; ok {
			out.Rows[i].Samples++
			if meas.US < out.Rows[i].US {
				meas.Samples = out.Rows[i].Samples
				out.Rows[i] = meas
			}
			continue
		}
		idx[k] = len(out.Rows)
		out.Rows = append(out.Rows, meas)
	}
	if !sawHeader {
		out.Backend = ""
	}
	return out
}

func oracleFlopScale(unit string) float64 {
	switch unit[0] {
	case 'k':
		return 1e3
	case 'M':
		return 1e6
	case 'G':
		return 1e9
	case 'T':
		return 1e12
	}
	return 1
}

// ggmlMeasurement decodes a case's parameter string. Built-in MUL_MAT cases name
// type_a/m/n/k; --test-file (test_generic_op) cases carry ne=[M,N,..] and
// sources=<src0>[K,M,..],<src1>[K,N,..].
func ggmlMeasurement(op, params string) (oracleMeasurement, bool) {
	kv := splitGGMLParams(params)
	meas := oracleMeasurement{Op: op}
	delete(kv, "name")
	if op == "MUL_MAT" {
		if ta, ok := kv["type_a"]; ok {
			meas.Dtype = strings.ToLower(ta)
			meas.M, _ = strconv.ParseInt(kv["m"], 10, 64)
			meas.N, _ = strconv.ParseInt(kv["n"], 10, 64)
			meas.K, _ = strconv.ParseInt(kv["k"], 10, 64)
			if !oracleBatchedBuiltin(kv) {
				return meas, meas.M > 0 && meas.N > 0 && meas.K > 0
			}
			meas.M = 0
		} else if srcs := parseGGMLTensors(kv["sources"]); len(srcs) >= 2 {
			a, b := srcs[0], srcs[1]
			if a.ne[2]*a.ne[3] == 1 && b.ne[2]*b.ne[3] == 1 {
				meas.Dtype, meas.K, meas.M, meas.N = a.typ, a.ne[0], a.ne[1], b.ne[1]
				return meas, true
			}
		}
	}
	if srcs := parseGGMLTensors(kv["sources"]); len(srcs) > 0 {
		meas.Dtype = srcs[0].typ
	} else if t, ok := kv["type"]; ok {
		meas.Dtype = strings.ToLower(t)
	}
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+kv[k])
	}
	meas.Params = strings.Join(parts, ",")
	return meas, true
}

func oracleBatchedBuiltin(kv map[string]string) bool {
	for _, k := range []string{"bs", "nr"} {
		if v, ok := kv[k]; ok && v != "[1,1]" {
			return true
		}
	}
	return false
}

// splitGGMLParams splits on top-level commas; a token without '=' continues the
// previous value (the sources list holds comma-joined tensors).
func splitGGMLParams(s string) map[string]string {
	kv := map[string]string{}
	depth, start, last := 0, 0, ""
	flush := func(tok string) {
		if i := strings.IndexByte(tok, '='); i > 0 && !strings.ContainsAny(tok[:i], "[]") {
			last = tok[:i]
			kv[last] = tok[i+1:]
			return
		}
		if last != "" {
			kv[last] += "," + tok
		}
	}
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '[':
			depth++
		case ']':
			depth--
		case ',':
			if depth == 0 {
				flush(s[start:i])
				start = i + 1
			}
		}
	}
	if start < len(s) {
		flush(s[start:])
	}
	return kv
}

type ggmlTensor struct {
	typ string
	ne  [4]int64
}

func parseGGMLTensors(s string) []ggmlTensor {
	var out []ggmlTensor
	for s != "" {
		m := oracleTensorRef.FindStringSubmatch(s)
		if m == nil {
			return out
		}
		t := ggmlTensor{typ: strings.ToLower(m[1])}
		for i := 0; i < 4; i++ {
			t.ne[i], _ = strconv.ParseInt(m[2+i], 10, 64)
		}
		out = append(out, t)
		s = s[len(m[0]):]
		if strings.HasPrefix(s, "nb[") {
			if j := strings.IndexByte(s, ']'); j >= 0 {
				s = s[j+1:]
			}
		}
		s = strings.TrimPrefix(s, ",")
	}
	return out
}

// parseFakBench reads `go test -bench` rows named <dtype>/out<M>_in<K>/<mode>
// (BenchmarkVulkanRawIQMatvec), keeping the fastest of repeated -count runs.
func parseFakBench(text, mode string) []oracleMeasurement {
	var out []oracleMeasurement
	idx := map[string]int{}
	for _, raw := range strings.Split(text, "\n") {
		m := oracleBenchRow.FindStringSubmatch(strings.TrimSpace(raw))
		if m == nil {
			continue
		}
		parts := strings.Split(m[1], "/")
		if len(parts) < 4 || parts[len(parts)-1] != mode {
			continue
		}
		sh := oracleBenchShape.FindStringSubmatch(parts[len(parts)-2])
		if sh == nil {
			continue
		}
		ns, err := strconv.ParseFloat(m[3], 64)
		if err != nil || ns <= 0 {
			continue
		}
		meas := oracleMeasurement{Op: "MUL_MAT", Dtype: strings.ToLower(parts[len(parts)-3]), N: 1, US: ns / 1e3, Samples: 1}
		meas.M, _ = strconv.ParseInt(sh[1], 10, 64)
		meas.K, _ = strconv.ParseInt(sh[2], 10, 64)
		k := meas.key()
		if i, ok := idx[k]; ok {
			out[i].Samples++
			if meas.US < out[i].US {
				out[i].US = meas.US
			}
			continue
		}
		idx[k] = len(out)
		out = append(out, meas)
	}
	return out
}

// parseFakStages keeps the last report per (kernel, groups): the profiler
// prints cumulative aggregates every FAK_VULKAN_TIMESTAMP_PROFILE_EVERY batches.
func parseFakStages(text string) []oracleStage {
	byKey := map[string]oracleStage{}
	for _, raw := range strings.Split(text, "\n") {
		m := oracleStageRow.FindStringSubmatch(raw)
		if m == nil {
			continue
		}
		st := oracleStage{Kernel: m[1], Groups: m[2] + "x" + m[3]}
		st.gx, _ = strconv.ParseInt(m[2], 10, 64)
		st.gy, _ = strconv.ParseInt(m[3], 10, 64)
		st.CallsBatch, _ = strconv.ParseFloat(m[4], 64)
		st.USPerCall, _ = strconv.ParseFloat(m[5], 64)
		st.SharePct, _ = strconv.ParseFloat(m[6], 64)
		byKey[st.Kernel+"|"+st.Groups] = st
	}
	out := make([]oracleStage, 0, len(byKey))
	for _, st := range byKey {
		out = append(out, st)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].SharePct != out[j].SharePct {
			return out[i].SharePct > out[j].SharePct
		}
		return out[i].Kernel+out[i].Groups < out[j].Kernel+out[j].Groups
	})
	return out
}

// oracleStageDtype maps the decode matvec kernels whose dispatch geometry is
// known (16 output rows per workgroup, TICKET-05j) to the weight dtype they serve.
var oracleStageDtype = map[string]string{
	"iq1s_matvec": "iq1_s", "iq2xxs_matvec": "iq2_xxs", "iq2xs_matvec": "iq2_xs", "iq2s_matvec": "iq2_s",
	"iq3xxs_matvec": "iq3_xxs", "iq3s_matvec": "iq3_s", "iq4xs_matvec": "iq4_xs",
}

const oracleStageRowsPerGroup = 16

// joinOracleStages attaches an in-situ decode (n=1) timing to a shape row only
// when exactly one model shape fits the kernel's dtype and workgroup count.
func joinOracleStages(stages []oracleStage, rows map[string]*oracleRow, shapes oracleShapeList) {
	for i := range stages {
		st := &stages[i]
		dt, ok := oracleStageDtype[st.Kernel]
		if !ok {
			st.Unjoined = "kernel_geometry_unknown"
			continue
		}
		if st.gy != 1 {
			st.Unjoined = "multi_row_grid"
			continue
		}
		var hits []oracleShape
		for _, s := range shapes.Shapes {
			if s.N == 1 && s.Dtype == dt && (s.M+oracleStageRowsPerGroup-1)/oracleStageRowsPerGroup == st.gx {
				hits = append(hits, s)
			}
		}
		switch len(hits) {
		case 0:
			st.Unjoined = "no_model_shape"
		case 1:
			s := hits[0]
			r := rows[oracleMatKey(s.Dtype, s.M, s.K, 1)]
			us := st.USPerCall
			r.FakInsituUS = &us
			st.JoinedShape = fmt.Sprintf("%s %dx%d n=1", s.Dtype, s.M, s.K)
		default:
			st.Unjoined = fmt.Sprintf("ambiguous_%d_shapes", len(hits))
		}
	}
}
