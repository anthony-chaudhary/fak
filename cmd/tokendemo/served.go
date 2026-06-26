package main

// Served HTTP same-read cache proof (#877).
//
// The older -timing proof drives kernel.Syscall in-process. This file crosses the
// served boundary a non-Go client would use: an httptest gateway serving the same
// Handler as `fak serve`, real HTTP POSTs to /v1/fak/syscall and /mcp, and the
// real confined filesystem read engine (fakread) on the miss path. The raw arm is
// direct os.ReadFile repeated once per served call; the fak arm is HTTP syscall
// plus MCP fak_syscall repeated against the same key.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const defaultServedCalls = 4

type servedReadCallProof struct {
	Index          int    `json:"index"`
	Surface        string `json:"surface"`
	Tool           string `json:"tool"`
	Resource       string `json:"resource"`
	ArgsHash       string `json:"args_hash"`
	HTTPStatus     int    `json:"http_status"`
	Verdict        string `json:"verdict"`
	ResultStatus   string `json:"result_status"`
	RawReadTimeNs  int64  `json:"raw_read_time_ns"`
	FakHTTPTimeNs  int64  `json:"fak_http_time_ns"`
	FakSource      string `json:"fak_source"`
	ServedBy       string `json:"served_by,omitempty"`
	Tier           string `json:"tier,omitempty"`
	Engine         string `json:"engine,omitempty"`
	EngineRanFak   bool   `json:"engine_ran_fak"`
	ResultBytes    int    `json:"result_bytes"`
	TraceID        string `json:"trace_id,omitempty"`
	ResponseMetaOK bool   `json:"response_meta_ok"`
}

type servedMetricEvidence struct {
	KernelSubmits             int64             `json:"kernel_submits"`
	KernelVDSOHits            int64             `json:"kernel_vdso_hits"`
	KernelEngineCalls         int64             `json:"kernel_engine_calls"`
	GatewayHTTPPostSyscall200 int64             `json:"gateway_http_post_syscall_200"`
	GatewayMCPPost200         int64             `json:"gateway_mcp_post_200"`
	GatewaySyscallAllowEngine int64             `json:"gateway_syscall_allow_engine"`
	GatewaySyscallAllowVDSO   int64             `json:"gateway_syscall_allow_vdso"`
	GatewayVDSOHitRatio       float64           `json:"gateway_vdso_hit_ratio"`
	Rows                      map[string]string `json:"rows"`
}

type servedReadProof struct {
	Schema                string                `json:"schema"`
	Surface               string                `json:"surface"`
	Endpoints             []string              `json:"endpoints"`
	Tool                  string                `json:"tool"`
	Engine                string                `json:"engine"`
	CallsPerSurface       int                   `json:"calls_per_surface"`
	Calls                 int                   `json:"calls"`
	FileBytes             int                   `json:"file_bytes"`
	RawEngineCalls        int64                 `json:"raw_engine_calls"`
	FakEngineCalls        int64                 `json:"fak_engine_calls"`
	VDSOHits              int64                 `json:"vdso_hits"`
	RoundtripsCollapsed   int64                 `json:"roundtrips_collapsed"`
	EngineCallsAvoided    int64                 `json:"engine_calls_avoided"`
	RawTotalNs            int64                 `json:"raw_total_ns"`
	FakTotalNs            int64                 `json:"fak_total_ns"`
	RawMinusFakNs         int64                 `json:"raw_minus_fak_ns"`
	RawP50Ns              int64                 `json:"raw_p50_ns"`
	RawP95Ns              int64                 `json:"raw_p95_ns"`
	FakP50Ns              int64                 `json:"fak_p50_ns"`
	FakP95Ns              int64                 `json:"fak_p95_ns"`
	HonestyScope          string                `json:"honesty_scope"`
	CallsDetail           []servedReadCallProof `json:"calls_detail"`
	GatewayMetricEvidence servedMetricEvidence  `json:"gateway_metric_evidence"`
}

func runServedReadJSON(calls int) int {
	proof, err := buildServedReadProof(context.Background(), calls)
	if err != nil {
		fmt.Fprintf(os.Stderr, "served proof: %v\n", err)
		return 1
	}
	b, _ := json.MarshalIndent(proof, "", "  ")
	fmt.Println(string(b))
	return 0
}

func runServedReadPrint(calls int) int {
	p := colors()
	proof, err := buildServedReadProof(context.Background(), calls)
	if err != nil {
		fmt.Fprintf(os.Stderr, "served proof: %v\n", err)
		return 1
	}

	fmt.Printf("\n  %s - HTTP + MCP (%d calls/surface, %d served calls)\n",
		p.paint(p.bold, "fak · served read-cache proof"), proof.CallsPerSurface, proof.Calls)
	fmt.Printf("  %s\n\n", p.paint(p.dim, "raw baseline repeats os.ReadFile; fak arm uses /v1/fak/syscall and MCP fak_syscall on one gateway"))
	fmt.Printf("  %-3s  %-5s  %-12s  %9s  %9s  %-12s  %-8s  %-5s  %-7s  %-6s\n",
		"#", "wire", "args", "raw ms", "wire ms", "fak source", "engine", "tier", "verdict", "bytes")
	fmt.Printf("  %s\n", strings.Repeat("-", 92))
	for _, c := range proof.CallsDetail {
		lineColor := p.dim
		if c.ServedBy == "vdso" && c.Tier == "2" {
			lineColor = p.green
		}
		fmt.Printf("  %s\n", p.paint(lineColor, fmt.Sprintf("%-3d  %-5s  %-12s  %9s  %9s  %-12s  %-8s  %-5s  %-7s  %-6d",
			c.Index,
			c.Surface,
			c.ArgsHash,
			formatMs(c.RawReadTimeNs),
			formatMs(c.FakHTTPTimeNs),
			padTrim(c.FakSource, 12),
			padTrim(c.Engine, 8),
			padTrim(c.Tier, 5),
			padTrim(c.Verdict, 7),
			c.ResultBytes,
		)))
	}
	fmt.Printf("  %s\n", strings.Repeat("-", 92))
	fmt.Printf("  raw read executions: %d   served engine calls: %d   served vDSO hits: %d   collapsed: %d\n",
		proof.RawEngineCalls, proof.FakEngineCalls, proof.VDSOHits, proof.RoundtripsCollapsed)
	fmt.Printf("  timing totals: raw %.3fms   served wire %.3fms   raw-minus-served %.3fms\n",
		nsToMs(proof.RawTotalNs), nsToMs(proof.FakTotalNs), nsToMs(proof.RawMinusFakNs))
	fmt.Printf("  gateway metrics: kernel_engine=%d kernel_vdso=%d http_post_syscall_200=%d mcp_post_200=%d op_engine=%d op_vdso=%d\n\n",
		proof.GatewayMetricEvidence.KernelEngineCalls,
		proof.GatewayMetricEvidence.KernelVDSOHits,
		proof.GatewayMetricEvidence.GatewayHTTPPostSyscall200,
		proof.GatewayMetricEvidence.GatewayMCPPost200,
		proof.GatewayMetricEvidence.GatewaySyscallAllowEngine,
		proof.GatewayMetricEvidence.GatewaySyscallAllowVDSO)
	return 0
}

func buildServedReadProof(ctx context.Context, calls int) (servedReadProof, error) {
	if calls < 2 {
		return servedReadProof{}, errors.New("served proof needs at least two calls")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	surfaces := []string{"http", "mcp"}
	totalCalls := calls * len(surfaces)

	dir, err := os.MkdirTemp("", "fak-tokendemo-served-")
	if err != nil {
		return servedReadProof{}, err
	}
	defer os.RemoveAll(dir)

	const relPath = "same-read-cache-proof.txt"
	body := []byte(strings.Repeat("fak served same-read cache proof\n", 256))
	absPath := filepath.Join(dir, relPath)
	if err := os.WriteFile(absPath, body, 0o644); err != nil {
		return servedReadProof{}, err
	}

	rawDurations := make([]int64, 0, totalCalls)
	for i := 0; i < totalCalls; i++ {
		start := time.Now()
		got, err := os.ReadFile(absPath)
		rawDurations = append(rawDurations, elapsedNs(start))
		if err != nil {
			return servedReadProof{}, fmt.Errorf("raw read %d: %w", i+1, err)
		}
		if !bytes.Equal(got, body) {
			return servedReadProof{}, fmt.Errorf("raw read %d returned changed bytes", i+1)
		}
	}

	configureServedReadWorld(dir)
	vdso.Default.SetGranularity(vdso.Resource)
	vdso.Default.BumpWorld()

	srv, err := gateway.New(gateway.Config{
		EngineID:     agent.FakReadEngineID,
		Model:        "tokendemo-served-read",
		VDSO:         true,
		Invalidation: "resource",
		Logf:         func(string, ...any) {},
	})
	if err != nil {
		return servedReadProof{}, err
	}
	defer srv.Close()

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	client := ts.Client()
	client.Timeout = 10 * time.Second

	args, _ := json.Marshal(map[string]string{"file_path": relPath})
	argsHash := argsHash(json.RawMessage(args))
	details := make([]servedReadCallProof, 0, totalCalls)
	fakDurations := make([]int64, 0, totalCalls)
	var firstContent string
	idx := 0
	for _, surface := range surfaces {
		for i := 0; i < calls; i++ {
			start := time.Now()
			status, resp, err := servedReadCall(ctx, client, ts.URL, surface, args, idx+1)
			httpNs := elapsedNs(start)
			fakDurations = append(fakDurations, httpNs)
			if err != nil {
				return servedReadProof{}, fmt.Errorf("served %s read %d: %w", surface, i+1, err)
			}
			if status != http.StatusOK {
				return servedReadProof{}, fmt.Errorf("served %s read %d HTTP status = %d, want 200", surface, i+1, status)
			}
			if resp.Verdict.Kind != "ALLOW" {
				return servedReadProof{}, fmt.Errorf("served %s read %d verdict = %q, want ALLOW", surface, i+1, resp.Verdict.Kind)
			}
			if resp.Result == nil {
				return servedReadProof{}, fmt.Errorf("served %s read %d returned nil result", surface, i+1)
			}
			if firstContent == "" {
				firstContent = resp.Result.Content
			} else if resp.Result.Content != firstContent {
				return servedReadProof{}, fmt.Errorf("served %s read %d content differs from first served read", surface, i+1)
			}

			meta := resp.Result.Meta
			source := servedReadSource(meta)
			row := servedReadCallProof{
				Index:          idx + 1,
				Surface:        surface,
				Tool:           "Read",
				Resource:       relPath,
				ArgsHash:       argsHash,
				HTTPStatus:     status,
				Verdict:        resp.Verdict.Kind,
				ResultStatus:   resp.Result.Status,
				RawReadTimeNs:  rawDurations[idx],
				FakHTTPTimeNs:  httpNs,
				FakSource:      source,
				ServedBy:       meta["served_by"],
				Tier:           meta["tier"],
				Engine:         meta["engine"],
				EngineRanFak:   meta["engine"] == agent.FakReadEngineID,
				ResultBytes:    len(resp.Result.Content),
				TraceID:        resp.TraceID,
				ResponseMetaOK: source == "engine" || source == "vdso_tier2",
			}
			if idx == 0 {
				if row.Engine != agent.FakReadEngineID || row.ServedBy == "vdso" {
					return servedReadProof{}, fmt.Errorf("first served read should reach %s, got engine=%q served_by=%q tier=%q",
						agent.FakReadEngineID, row.Engine, row.ServedBy, row.Tier)
				}
			} else if row.ServedBy != "vdso" || row.Tier != "2" || row.EngineRanFak {
				return servedReadProof{}, fmt.Errorf("served read %d should be tier-2 vDSO hit with no engine, got meta=%v", idx+1, meta)
			}
			details = append(details, row)
			idx++
		}
	}

	metricsText, err := getServedMetrics(ctx, client, ts.URL)
	if err != nil {
		return servedReadProof{}, err
	}
	metricEvidence := servedReadMetricEvidence(metricsText)
	rawCalls := int64(totalCalls)
	fakEngineCalls := metricEvidence.KernelEngineCalls
	vdsoHits := metricEvidence.KernelVDSOHits
	if metricEvidence.KernelSubmits != rawCalls {
		return servedReadProof{}, fmt.Errorf("metrics kernel submits = %d, want %d", metricEvidence.KernelSubmits, rawCalls)
	}
	if fakEngineCalls != 1 {
		return servedReadProof{}, fmt.Errorf("metrics kernel engine calls = %d, want 1", fakEngineCalls)
	}
	if vdsoHits != rawCalls-1 {
		return servedReadProof{}, fmt.Errorf("metrics kernel vdso hits = %d, want %d", vdsoHits, rawCalls-1)
	}
	if metricEvidence.GatewayHTTPPostSyscall200 != int64(calls) {
		return servedReadProof{}, fmt.Errorf("metrics HTTP POST /v1/fak/syscall = %d, want %d",
			metricEvidence.GatewayHTTPPostSyscall200, calls)
	}
	if metricEvidence.GatewayMCPPost200 != int64(calls) {
		return servedReadProof{}, fmt.Errorf("metrics HTTP POST /mcp = %d, want %d",
			metricEvidence.GatewayMCPPost200, calls)
	}
	if metricEvidence.GatewaySyscallAllowEngine != 1 || metricEvidence.GatewaySyscallAllowVDSO != rawCalls-1 {
		return servedReadProof{}, fmt.Errorf("metrics operation rows engine/vdso = %d/%d, want 1/%d",
			metricEvidence.GatewaySyscallAllowEngine, metricEvidence.GatewaySyscallAllowVDSO, rawCalls-1)
	}

	rawTotal := sumNs(rawDurations)
	fakTotal := sumNs(fakDurations)
	return servedReadProof{
		Schema:                "fak.tokendemo.served-read-cache.v1",
		Surface:               "http+mcp",
		Endpoints:             []string{"/v1/fak/syscall", "/mcp"},
		Tool:                  "Read",
		Engine:                agent.FakReadEngineID,
		CallsPerSurface:       calls,
		Calls:                 totalCalls,
		FileBytes:             len(body),
		RawEngineCalls:        rawCalls,
		FakEngineCalls:        fakEngineCalls,
		VDSOHits:              vdsoHits,
		RoundtripsCollapsed:   rawCalls - fakEngineCalls,
		EngineCallsAvoided:    rawCalls - fakEngineCalls,
		RawTotalNs:            rawTotal,
		FakTotalNs:            fakTotal,
		RawMinusFakNs:         rawTotal - fakTotal,
		RawP50Ns:              percentileNs(rawDurations, 50),
		RawP95Ns:              percentileNs(rawDurations, 95),
		FakP50Ns:              percentileNs(fakDurations, 50),
		FakP95Ns:              percentileNs(fakDurations, 95),
		HonestyScope:          "tool-side only: cached bytes are still returned to the caller; this proof does not claim model-context token savings or native Claude Code tool routing",
		CallsDetail:           details,
		GatewayMetricEvidence: metricEvidence,
	}, nil
}

func configureServedReadWorld(root string) {
	agent.RegisterReadEngine(root)
	adjudicator.Default.SetPolicy(adjudicator.Policy{
		Allow: map[string]bool{"Read": true},
	})
}

func servedReadCall(ctx context.Context, client *http.Client, baseURL, surface string, args []byte, index int) (int, gateway.SyscallResponse, error) {
	reqBody, _ := json.Marshal(gateway.SyscallRequest{
		Tool:      "Read",
		Arguments: json.RawMessage(args),
		ReadOnly:  true,
		TraceID:   "tokendemo-served-read",
	})
	if surface == "mcp" {
		resp, err := servedMCPSyscall(ctx, client, baseURL, index, gateway.SyscallRequest{
			Tool:      "Read",
			Arguments: json.RawMessage(args),
			ReadOnly:  true,
			TraceID:   "tokendemo-served-read",
		})
		if err != nil {
			return 0, gateway.SyscallResponse{}, err
		}
		return http.StatusOK, resp, nil
	}
	if surface != "http" {
		return 0, gateway.SyscallResponse{}, fmt.Errorf("unknown served surface %q", surface)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/fak/syscall", bytes.NewReader(reqBody))
	if err != nil {
		return 0, gateway.SyscallResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Tokendemo-Call", strconv.Itoa(index))
	r, err := client.Do(req)
	if err != nil {
		return 0, gateway.SyscallResponse{}, err
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return r.StatusCode, gateway.SyscallResponse{}, err
	}
	var resp gateway.SyscallResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return r.StatusCode, gateway.SyscallResponse{}, fmt.Errorf("decode response: %w; body=%s", err, string(body))
	}
	return r.StatusCode, resp, nil
}

func getServedMetrics(ctx context.Context, client *http.Client, baseURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/metrics", nil)
	if err != nil {
		return "", err
	}
	r, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return "", err
	}
	if r.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET /metrics status = %d, body=%s", r.StatusCode, string(body))
	}
	return string(body), nil
}

func servedReadSource(meta map[string]string) string {
	if meta == nil {
		return "unknown"
	}
	if meta["served_by"] == "vdso" && meta["tier"] != "" {
		return "vdso_tier" + meta["tier"]
	}
	if meta["engine"] != "" {
		return "engine"
	}
	return "unknown"
}

func servedReadMetricEvidence(text string) servedMetricEvidence {
	prefixes := map[string]string{
		"kernel_submits":                "fak_kernel_submits_total ",
		"kernel_vdso_hits":              "fak_kernel_vdso_hits_total ",
		"kernel_engine_calls":           "fak_kernel_engine_calls_total ",
		"gateway_http_post_syscall_200": `fak_gateway_http_requests_total{route="/v1/fak/syscall",method="POST",status="200"} `,
		"gateway_mcp_post_200":          `fak_gateway_http_requests_total{route="/mcp",method="POST",status="200"} `,
		"gateway_syscall_allow_engine":  `fak_gateway_operations_total{operation="syscall",verdict="ALLOW",reason="",disposition="",by="monitor"} `,
		"gateway_syscall_allow_vdso":    `fak_gateway_operations_total{operation="syscall",verdict="ALLOW",reason="",disposition="",by="vdso"} `,
		"gateway_vdso_hit_ratio":        "fak_gateway_vdso_hit_ratio ",
	}
	rows := make(map[string]string, len(prefixes))
	for k, p := range prefixes {
		rows[k] = firstMetricLine(text, p)
	}
	return servedMetricEvidence{
		KernelSubmits:             metricLineInt(rows["kernel_submits"]),
		KernelVDSOHits:            metricLineInt(rows["kernel_vdso_hits"]),
		KernelEngineCalls:         metricLineInt(rows["kernel_engine_calls"]),
		GatewayHTTPPostSyscall200: metricLineInt(rows["gateway_http_post_syscall_200"]),
		GatewayMCPPost200:         metricLineInt(rows["gateway_mcp_post_200"]),
		GatewaySyscallAllowEngine: metricLineInt(rows["gateway_syscall_allow_engine"]),
		GatewaySyscallAllowVDSO:   metricLineInt(rows["gateway_syscall_allow_vdso"]),
		GatewayVDSOHitRatio:       metricLineFloat(rows["gateway_vdso_hit_ratio"]),
		Rows:                      rows,
	}
}

func firstMetricLine(text, prefix string) string {
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, prefix) {
			return line
		}
	}
	return ""
}

func metricLineInt(line string) int64 {
	v := metricLineFloat(line)
	if v < 0 {
		return int64(v - 0.5)
	}
	return int64(v + 0.5)
}

func metricLineFloat(line string) float64 {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return 0
	}
	f, _ := strconv.ParseFloat(fields[len(fields)-1], 64)
	return f
}
