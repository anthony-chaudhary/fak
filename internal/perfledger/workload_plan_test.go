package perfledger

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// wplanGoldenPlan mirrors a plan object hashed with upstream's
// canonical_json_bytes; the frozen bytes and digests below came from Python's
// json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).
func wplanGoldenPlan() WorkloadPlan {
	return WorkloadPlan{
		PlanID:        "smoke",
		ModelID:       "MiniMax/Video",
		ModelRevision: strings.Repeat("a", 40),
		Cases: []WorkloadCase{
			{CaseID: "c1", Prompt: "a cat\n\"jumps\"", Seed: 42},
			{CaseID: "c2", Prompt: "rain\t\x01", Seed: 7},
		},
		Repetitions: 3,
		WarmupRuns:  1,
		Generation: map[string]any{
			"num_inference_steps": 30,
			"flow_shift":          5.0,
			"guidance":            1.5e-05,
			"fps":                 24,
			"negative":            "",
			"tag":                 "café <&>",
		},
		Fairness: map[string]any{
			"conditioning_cache": "disabled_across_requests",
			"schedule": map[string]any{
				"sigma_points":      31,
				"denoising_updates": 30,
				"big":               1e16,
				"neg":               math.Copysign(0, -1),
			},
		},
		Host: "aegis",
		GPU:  "RTX 5080",
		Port: 18102,
	}
}

const (
	wplanGoldenJSON = `{"cases":[{"case_id":"c1","prompt":"a cat\n\"jumps\"","seed":42},{"case_id":"c2","prompt":"rain\t\u0001","seed":7}],` +
		`"fairness":{"conditioning_cache":"disabled_across_requests","schedule":{"big":1e+16,"denoising_updates":30,"neg":-0.0,"sigma_points":31}},` +
		`"generation":{"flow_shift":5.0,"fps":24,"guidance":1.5e-05,"negative":"","num_inference_steps":30,"tag":"café <&>"},` +
		`"model_id":"MiniMax/Video","model_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","plan_id":"smoke","repetitions":3,"schema_version":2,"warmup_runs":1}`
	wplanGoldenSHA        = "5d5caab2432d7454687f7a761409afbe1ce0fd8e0e51d40d861577829ca3c8da"
	wplanGoldenShortSHA   = "a2ecdf6b4afe68a3fd1d8891d6e3c023c1365822622a9d7d215d8e0eb9544936"
	wplanGoldenShortPlan  = "smoke-r1-w0"
	wplanGoldenShortReps  = 1
	wplanGoldenShortWarms = 0
)

func TestWorkloadPlanHashGolden(t *testing.T) {
	t.Parallel()
	p := wplanGoldenPlan()
	got, err := CanonicalPlanJSON(WorkloadPlanIdentity(p))
	if err != nil {
		t.Fatalf("CanonicalPlanJSON: %v", err)
	}
	if string(got) != wplanGoldenJSON {
		t.Fatalf("canonical bytes drifted\n got: %s\nwant: %s", got, wplanGoldenJSON)
	}
	sum, err := WorkloadPlanSHA256(p)
	if err != nil {
		t.Fatalf("WorkloadPlanSHA256: %v", err)
	}
	if sum != wplanGoldenSHA {
		t.Fatalf("plan_sha256 = %s, want %s", sum, wplanGoldenSHA)
	}

	short, err := WithWorkloadCountOverride(p, wplanGoldenShortReps, wplanGoldenShortWarms)
	if err != nil {
		t.Fatalf("WithWorkloadCountOverride: %v", err)
	}
	if short.PlanID != wplanGoldenShortPlan {
		t.Fatalf("override plan id = %q, want %q", short.PlanID, wplanGoldenShortPlan)
	}
	shortSum, err := WorkloadPlanSHA256(short)
	if err != nil {
		t.Fatalf("WorkloadPlanSHA256(short): %v", err)
	}
	if shortSum != wplanGoldenShortSHA {
		t.Fatalf("short plan_sha256 = %s, want %s", shortSum, wplanGoldenShortSHA)
	}
}

func TestWorkloadPlanHashExcludesRuntimeIdentity(t *testing.T) {
	t.Parallel()
	base := wplanGoldenPlan()
	tests := []struct {
		name   string
		mutate func(*WorkloadPlan)
		same   bool
	}{
		{"host", func(p *WorkloadPlan) { p.Host = "raider" }, true},
		{"gpu", func(p *WorkloadPlan) { p.GPU = "H200" }, true},
		{"port", func(p *WorkloadPlan) { p.Port = 8000 }, true},
		{"generation map rebuilt in another order", func(p *WorkloadPlan) {
			g := map[string]any{}
			for _, k := range []string{"tag", "negative", "fps", "guidance", "flow_shift", "num_inference_steps"} {
				g[k] = p.Generation[k]
			}
			p.Generation = g
		}, true},
		{"seed", func(p *WorkloadPlan) {
			p.Cases = append([]WorkloadCase(nil), p.Cases...)
			p.Cases[0].Seed++
		}, false},
		{"prompt", func(p *WorkloadPlan) {
			p.Cases = append([]WorkloadCase(nil), p.Cases...)
			p.Cases[1].Prompt = "snow"
		}, false},
		{"case order", func(p *WorkloadPlan) { p.Cases = []WorkloadCase{p.Cases[1], p.Cases[0]} }, false},
		{"generation param", func(p *WorkloadPlan) {
			p.Generation = map[string]any{"num_inference_steps": 31}
		}, false},
		{"fairness dropped", func(p *WorkloadPlan) { p.Fairness = nil }, false},
		{"model revision", func(p *WorkloadPlan) { p.ModelRevision = strings.Repeat("b", 40) }, false},
		{"repetitions without rename", func(p *WorkloadPlan) { p.Repetitions = 9 }, false},
	}
	want, err := WorkloadPlanSHA256(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			p := wplanGoldenPlan()
			tc.mutate(&p)
			got, err := WorkloadPlanSHA256(p)
			if err != nil {
				t.Fatal(err)
			}
			if (got == want) != tc.same {
				t.Fatalf("hash equal=%v, want equal=%v (got %s)", got == want, tc.same, got)
			}
		})
	}
}

func TestWorkloadPlanHashCountOverride(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		reps     int
		warm     int
		wantID   string
		wantErr  bool
		sameHash bool
	}{
		{name: "own counts keep identity", reps: 3, warm: 1, wantID: "smoke", sameHash: true},
		{name: "fewer repetitions", reps: 1, warm: 1, wantID: "smoke-r1-w1"},
		{name: "only warmup changed", reps: 3, warm: 0, wantID: "smoke-r3-w0"},
		{name: "more repetitions", reps: 10, warm: 2, wantID: "smoke-r10-w2"},
		{name: "zero repetitions refused", reps: 0, warm: 1, wantErr: true},
		{name: "negative warmup refused", reps: 3, warm: -1, wantErr: true},
	}
	base := wplanGoldenPlan()
	baseSum, err := WorkloadPlanSHA256(base)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := WithWorkloadCountOverride(base, tc.reps, tc.warm)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("want error, got plan %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.PlanID != tc.wantID || got.Repetitions != tc.reps || got.WarmupRuns != tc.warm {
				t.Fatalf("got id=%q reps=%d warm=%d", got.PlanID, got.Repetitions, got.WarmupRuns)
			}
			sum, err := WorkloadPlanSHA256(got)
			if err != nil {
				t.Fatal(err)
			}
			if (sum == baseSum) != tc.sameHash {
				t.Fatalf("hash equal=%v, want %v", sum == baseSum, tc.sameHash)
			}
		})
	}
	if base.PlanID != "smoke" || base.Repetitions != 3 {
		t.Fatalf("override mutated the source plan: %+v", base)
	}
}

func TestWorkloadPlanHashCanonicalJSON(t *testing.T) {
	t.Parallel()
	type tagged struct {
		B int     `json:"b"`
		A float64 `json:"a"`
	}
	tests := []struct {
		name    string
		in      any
		want    string
		wantErr bool
	}{
		{name: "nil", in: nil, want: `null`},
		{name: "nested sorted", in: map[string]any{"z": []any{1, true, nil}, "a": map[string]int{"y": 2, "x": 1}}, want: `{"a":{"x":1,"y":2},"z":[1,true,null]}`},
		{name: "no html escaping", in: "<a>&</a>", want: `"<a>&</a>"`},
		{name: "raw utf8 and line separator", in: "é \U0001F600 ", want: "\"é \U0001F600 \""},
		{name: "control escapes", in: "\b\f\x1f\x7f", want: "\"\\b\\f\\u001f\x7f\""},
		{name: "float integral", in: 2.0, want: `2.0`},
		{name: "float small", in: 0.0001, want: `0.0001`},
		{name: "float tiny", in: 0.00001, want: `1e-05`},
		{name: "float large fixed", in: 1234567890123456.0, want: `1234567890123456.0`},
		{name: "float large sci", in: 1.5e300, want: `1.5e+300`},
		{name: "float32", in: float32(0.1), want: `0.1`},
		{name: "uint", in: uint64(math.MaxUint64), want: `18446744073709551615`},
		{name: "int64 min", in: int64(math.MinInt64), want: `-9223372036854775808`},
		{name: "empty containers", in: map[string]any{"m": map[string]any{}, "s": []string{}}, want: `{"m":{},"s":[]}`},
		{name: "struct via json tags", in: tagged{B: 1, A: 0.5}, want: `{"a":0.5,"b":1}`},
		{name: "pointer", in: &tagged{B: 2}, want: `{"a":0,"b":2}`},
		{name: "NaN refused", in: math.NaN(), wantErr: true},
		{name: "Inf nested refused", in: map[string]any{"x": []any{math.Inf(1)}}, wantErr: true},
		{name: "negative Inf refused", in: math.Inf(-1), wantErr: true},
		{name: "invalid utf8 refused", in: "\xff", wantErr: true},
		{name: "invalid utf8 key refused", in: map[string]int{"\xfe": 1}, wantErr: true},
		{name: "non-string key refused", in: map[int]int{1: 1}, wantErr: true},
		{name: "unsupported type refused", in: make(chan int), wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := CanonicalPlanJSON(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("want error, got %s", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tc.want {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
}

func TestWorkloadPlanHashStableAcrossFileRoundTrip(t *testing.T) {
	t.Parallel()
	b, err := CanonicalPlanJSON(WorkloadPlanIdentity(wplanGoldenPlan()))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	read, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(read) != wplanGoldenJSON {
		t.Fatalf("persisted bytes differ from golden")
	}
}
