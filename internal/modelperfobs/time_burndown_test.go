package modelperfobs

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownSeparatesCacheArmsAndReconcilesAPIWall(t *testing.T) {
	rows := []Observation{
		burndownObservation("cold-1", "cold", 100, 80),
		burndownObservation("cold-2", "cold", 300, 180),
		burndownObservation("cold-3", "cold", 200, 130),
		burndownObservation("warm-1", "warm", 20, 5),
		burndownObservation("warm-2", "warm", 40, 15),
		burndownObservation("warm-3", "warm", 30, 10),
		burndownObservation("legacy", "", 50, 0),
		burndownObservation("prime", "cold", 10000, 9000),
		burndownObservation("failed", "cold", 700, 600),
	}
	rows[7].Warmup = true
	rows[8].Status = 503
	rows[8].Error = "upstream_failed"
	report, err := ReduceTimeBurndown(rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	if report.Requests != 9 || report.IncludedRequests != 7 || report.WarmupExcluded != 1 || report.FailedRequests != 1 || report.NativeRequests != 0 {
		t.Fatalf("counts requests=%d included=%d warmups=%d failed=%d native=%d", report.Requests, report.IncludedRequests, report.WarmupExcluded, report.FailedRequests, report.NativeRequests)
	}
	if len(report.Groups) != 3 {
		t.Fatalf("cache arms=%d, want cold, warm and unknown", len(report.Groups))
	}
	cases := []struct {
		cache                  string
		requests, samples      int
		warmups, failed, ttfts int
		duration, ttft, wall   float64
		phases                 []TimeBurndownPhase
	}{
		{"cold", 4, 3, 1, 1, 3, 200, 130, 600, []TimeBurndownPhase{{Phase: "before_first_output", WallMS: 390, Percent: 65}, {Phase: "after_first_output", WallMS: 210, Percent: 35}}},
		{"warm", 3, 3, 0, 0, 3, 30, 10, 90, []TimeBurndownPhase{{Phase: "after_first_output", WallMS: 60, Percent: 200.0 / 3}, {Phase: "before_first_output", WallMS: 30, Percent: 100.0 / 3}}},
		{"unknown", 1, 1, 0, 0, 0, 50, 0, 50, []TimeBurndownPhase{{Phase: "unpartitioned_request", WallMS: 50, Percent: 100}}},
	}
	for _, tc := range cases {
		t.Run(tc.cache, func(t *testing.T) {
			g := burndownGroup(t, report, tc.cache)
			if g.Requests != tc.requests || g.Samples != tc.samples || g.WarmupExcluded != tc.warmups || g.FailedRequests != tc.failed || g.TTFTSamples != tc.ttfts {
				t.Fatalf("arm counts requests=%d samples=%d warmups=%d failed=%d ttfts=%d", g.Requests, g.Samples, g.WarmupExcluded, g.FailedRequests, g.TTFTSamples)
			}
			if g.DurationP50MS != tc.duration || g.TTFTP50MS != tc.ttft || g.APIWallMS != tc.wall {
				t.Fatalf("duration p50=%g TTFT p50=%g API wall=%g", g.DurationP50MS, g.TTFTP50MS, g.APIWallMS)
			}
			if g.Attribution != "api_only" || g.AttributionUnavailable != "backend-does-not-expose-dispatch-attribution" || len(g.NativeRank) != 0 {
				t.Fatalf("API-only arm claims attribution=%s unavailable=%s native phases=%d", g.Attribution, g.AttributionUnavailable, len(g.NativeRank))
			}
			if len(g.APIRank) != len(tc.phases) {
				t.Fatalf("API partitions=%d want %d", len(g.APIRank), len(tc.phases))
			}
			var total, percent float64
			for i, want := range tc.phases {
				got := g.APIRank[i]
				if got.Phase != want.Phase || got.WallMS != want.WallMS || math.Abs(got.Percent-want.Percent) > 1e-9 {
					t.Errorf("rank %d phase=%s wall=%g percent=%g", i, got.Phase, got.WallMS, got.Percent)
				}
				if got.ActiveMS != nil || got.WaitMS != nil {
					t.Errorf("API partition %s invents active/wait attribution", got.Phase)
				}
				total += got.WallMS
				percent += got.Percent
			}
			if total != g.APIWallMS || math.Abs(percent-100) > 1e-9 {
				t.Fatalf("partition wall=%g API wall=%g percent=%g", total, g.APIWallMS, percent)
			}
		})
	}
	slices.Reverse(rows)
	reordered, err := ReduceTimeBurndown(rows, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report, reordered) {
		t.Fatal("row order changes cache-arm or cumulative-phase ranking")
	}
	nonstream := burndownObservation("nonstream", "warm", 60, 20)
	nonstream.Streaming = false
	unpartitioned, err := ReduceTimeBurndown([]Observation{nonstream}, nil)
	if err != nil {
		t.Fatal(err)
	}
	ng := burndownGroup(t, unpartitioned, "warm")
	if ng.TTFTSamples != 0 || ng.TTFTP50MS != 0 || len(ng.APIRank) != 1 || ng.APIRank[0].Phase != "unpartitioned_request" || ng.APIRank[0].WallMS != 60 {
		t.Fatal("nonstream response was treated as observed first-output timing")
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownJoinsExclusiveNativePhasesAndRejectsFalseCoverage(t *testing.T) {
	row := burndownObservation("native-1", "warm", 120, 90)
	envelope := burndownNativeReceipt(row)
	report, err := ReduceTimeBurndown([]Observation{row}, []RequestPhaseReceipt{envelope})
	if err != nil {
		t.Fatal(err)
	}
	g := burndownGroup(t, report, "warm")
	if report.NativeRequests != 1 || g.NativeRequests != 1 || g.Attribution != "native" || g.AttributionUnavailable != "" {
		t.Fatalf("native requests=%d group native=%d attribution=%s unavailable=%s", report.NativeRequests, g.NativeRequests, g.Attribution, g.AttributionUnavailable)
	}
	if g.APIWallMS != 120 || g.NativeCoveredAPIWallMS != 120 || g.NativeWallMS != 100 || g.OutsideNativeMS != 20 {
		t.Fatalf("API=%g covered API=%g native=%g outside=%g", g.APIWallMS, g.NativeCoveredAPIWallMS, g.NativeWallMS, g.OutsideNativeMS)
	}
	if len(g.NativeRank) != len(nativeperf.Phases()) {
		t.Fatalf("native phase set=%d want %d", len(g.NativeRank), len(nativeperf.Phases()))
	}
	var wall, active, wait, percent float64
	for _, phase := range g.NativeRank {
		if phase.ActiveMS == nil || phase.WaitMS == nil {
			t.Fatalf("native phase %s lacks active/wait partition", phase.Phase)
		}
		wall += phase.WallMS
		active += *phase.ActiveMS
		wait += *phase.WaitMS
		percent += phase.Percent
	}
	if wall != 100 || active != 80 || wait != 20 || math.Abs(percent-100) > 1e-9 || wall+g.OutsideNativeMS != g.NativeCoveredAPIWallMS {
		t.Fatalf("exclusive wall=%g active=%g wait=%g percent=%g", wall, active, wait, percent)
	}
	if g.NativeRank[0].Phase != "kernel" || g.NativeRank[0].WallMS != 40 || g.NativeRank[1].Phase != "prefill" || g.NativeRank[1].WallMS != 40 {
		t.Fatal("inclusive nesting was double-counted or equal cumulative phases lack stable ordering")
	}
	apiOnly := burndownObservation("api-only", "warm", 80, 60)
	partial, err := ReduceTimeBurndown([]Observation{row, apiOnly}, []RequestPhaseReceipt{envelope})
	if err != nil {
		t.Fatal(err)
	}
	pg := burndownGroup(t, partial, "warm")
	if pg.Attribution != "partial" || pg.AttributionUnavailable == "" || pg.APIWallMS != 200 || pg.NativeCoveredAPIWallMS != 120 || pg.OutsideNativeMS != 20 {
		t.Fatal("partial native coverage was projected onto unmatched API requests")
	}
	cases := []struct {
		name   string
		mutate func([]Observation, []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt)
	}{
		{"duplicate observation", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			return append(rows, rows[0]), receipts
		}},
		{"duplicate receipt", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			return rows, append(receipts, receipts[0])
		}},
		{"unmatched receipt", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			receipts[0].RequestID = "unmatched"
			return rows, receipts
		}},
		{"model mismatch", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			receipts[0].Model = "another-model"
			return rows, receipts
		}},
		{"backend mismatch", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			receipts[0].Backend = "http://another.test"
			return rows, receipts
		}},
		{"native longer than API", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			rows[0].DurationMS = 99
			return rows, receipts
		}},
		{"fallback execution", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			receipts[0].Receipt.FallbackActive = true
			return rows, receipts
		}},
		{"incomplete phase set", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			receipts[0].Receipt.Phases = receipts[0].Receipt.Phases[1:]
			return rows, receipts
		}},
		{"negative API wall", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			rows[0].DurationMS = -1
			return rows, receipts
		}},
		{"nonfinite API wall", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			rows[0].DurationMS = math.Inf(1)
			return rows, receipts
		}},
		{"TTFT outside API wall", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			rows[0].TTFTMS = 121
			return rows, receipts
		}},
		{"known cache state with unknown source", func(rows []Observation, receipts []RequestPhaseReceipt) ([]Observation, []RequestPhaseReceipt) {
			rows[0].CacheStateSource = "unknown"
			return rows, receipts
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows, receipts := tc.mutate([]Observation{row}, []RequestPhaseReceipt{burndownNativeReceipt(row)})
			if _, err := ReduceTimeBurndown(rows, receipts); !errors.Is(err, ErrInvalidTimeBurndown) {
				t.Fatalf("invalid join error=%v, want ErrInvalidTimeBurndown", err)
			}
		})
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownWireReadersAndPrivacyBoundedReports(t *testing.T) {
	row := burndownObservation("private-request-unique", "warm", 120, 90)
	row.Backend = "https://secret-host.example:8443/private-route?token=secret-value"
	row.Model = "private-model-unique"
	envelope := burndownNativeReceipt(row)
	var wire bytes.Buffer
	if err := json.NewEncoder(&wire).Encode(envelope); err != nil {
		t.Fatal(err)
	}
	payload := wire.String()
	decoded, err := ReadRequestPhaseReceipts(&wire)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, []RequestPhaseReceipt{envelope}) {
		t.Fatal("request correlation or exclusive phase receipt changed on JSONL read")
	}
	for name, invalid := range map[string]string{
		"schema":        `{"schema":"not-a-receipt"}`,
		"unknown field": `{"unknown_field":true,` + payload[1:],
		"trailing JSON": strings.TrimSpace(payload) + ` {}`,
		"malformed":     `{`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ReadRequestPhaseReceipts(strings.NewReader(invalid)); !errors.Is(err, ErrInvalidTimeBurndown) {
				t.Fatalf("invalid receipt error=%v", err)
			}
		})
	}
	report, err := ReduceTimeBurndown([]Observation{row}, decoded)
	if err != nil {
		t.Fatal(err)
	}
	var prom, markdown bytes.Buffer
	if err := WriteTimeBurndownPrometheus(&prom, report); err != nil {
		t.Fatal(err)
	}
	if err := WriteTimeBurndownMarkdown(&markdown, report); err != nil {
		t.Fatal(err)
	}
	if prom.Len() == 0 || markdown.Len() == 0 {
		t.Fatal("a report writer omitted its nonempty report")
	}
	for _, secret := range []string{row.RequestID, row.Backend, row.Model, "secret-host", "private-route", "secret-value"} {
		if strings.Contains(prom.String(), secret) {
			t.Errorf("Prometheus exposed unbounded request identity %q", secret)
		}
	}
	for _, want := range []string{`cache_state="warm"`, `phase="kernel"`, `phase="queue_admission"`} {
		if !strings.Contains(prom.String(), want) {
			t.Errorf("bounded metric dimension missing %s", want)
		}
	}
	label := regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"`)
	phaseSeconds := map[string]float64{}
	for _, line := range strings.Split(prom.String(), "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			t.Fatalf("metric line lacks one numeric sample: %q", line)
		}
		labels := map[string]string{}
		for _, match := range label.FindAllStringSubmatch(fields[0], -1) {
			if match[1] != "cache_state" && match[1] != "plane" && match[1] != "phase" && match[1] != "kind" {
				t.Errorf("unbounded metric label key %s", match[1])
			}
			labels[match[1]] = match[2]
		}
		if strings.HasPrefix(fields[0], "fak_time_burndown_seconds{") {
			if labels["kind"] != "wall" && labels["kind"] != "active" && labels["kind"] != "wait" {
				t.Errorf("unbounded phase timing kind %q", labels["kind"])
			}
			if labels["plane"] != "native" && labels["kind"] != "wall" {
				t.Errorf("%s metric invents active/wait attribution", labels["plane"])
			}
			seconds, err := strconv.ParseFloat(fields[1], 64)
			if err != nil {
				t.Fatal(err)
			}
			phaseSeconds[labels["plane"]+"_"+labels["kind"]] += seconds
		}
	}
	for dimension, want := range map[string]float64{"api_wall": .12, "native_wall": .1, "native_active": .08, "native_wait": .02, "outside_native_wall": .02} {
		if math.Abs(phaseSeconds[dimension]-want) > 1e-12 {
			t.Errorf("Prometheus %s seconds=%g want %g", dimension, phaseSeconds[dimension], want)
		}
	}
	for _, raw := range []string{"cold", "warm", "steady", "unknown", ""} {
		got, err := ParseCacheState(raw)
		want := raw
		if want == "" {
			want = "unknown"
		}
		if err != nil || got != want {
			t.Errorf("cache state %q => %q, %v; want %q", raw, got, err, want)
		}
	}
	if _, err := ParseCacheState("unbounded-state"); !errors.Is(err, ErrInvalidTimeBurndown) {
		t.Fatalf("invalid cache state error=%v", err)
	}
}

func burndownObservation(id, cache string, wall, ttft float64) Observation {
	return Observation{Schema: Schema, RequestID: id, Backend: "http://backend.test", Model: "model", Status: 200, Streaming: true, CacheState: cache, DurationMS: wall, TTFTMS: ttft}
}

func burndownGroup(t *testing.T, report TimeBurndown, cache string) TimeBurndownGroup {
	t.Helper()
	for _, group := range report.Groups {
		if group.CacheState == cache {
			return group
		}
	}
	t.Fatalf("missing cache arm %s", cache)
	return TimeBurndownGroup{}
}

func burndownNativeReceipt(row Observation) RequestPhaseReceipt {
	phases := make([]nativeperf.PhaseAccounting, 0, len(nativeperf.Phases()))
	for _, phase := range nativeperf.Phases() {
		account := nativeperf.PhaseAccounting{Phase: phase}
		switch phase {
		case nativeperf.PhaseQueueAdmission:
			account.Inclusive = nativeperf.PhaseTiming{Wall: 20 * time.Millisecond, Wait: 20 * time.Millisecond}
			account.Exclusive = account.Inclusive
		case nativeperf.PhasePrefill:
			account.Inclusive = nativeperf.PhaseTiming{Wall: 80 * time.Millisecond, Active: 80 * time.Millisecond}
			account.Exclusive = nativeperf.PhaseTiming{Wall: 40 * time.Millisecond, Active: 40 * time.Millisecond}
		case nativeperf.PhaseKernel:
			account.Parent = nativeperf.PhasePrefill
			account.Inclusive = nativeperf.PhaseTiming{Wall: 40 * time.Millisecond, Active: 40 * time.Millisecond}
			account.Exclusive = account.Inclusive
		}
		phases = append(phases, account)
	}
	return RequestPhaseReceipt{Schema: "fak-request-phases/1", RequestID: row.RequestID, Model: row.Model, Backend: row.Backend, Receipt: nativeperf.PhaseReceipt{Engine: "inkernel", Backend: "cuda", ForwardPath: "qwen3.8_cuda", Wall: 100 * time.Millisecond, Phases: phases}}
}
