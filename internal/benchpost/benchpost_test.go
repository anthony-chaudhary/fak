package benchpost

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func f64(v float64) *float64 { return &v }

// --- resolution -------------------------------------------------------------

func TestResolveTokenAndChannelFromBenchEnv(t *testing.T) {
	t.Setenv("FAK_BENCH_TOKEN", "xoxb-bench-token")
	t.Setenv("FAK_BENCH_CHANNEL", "C_BENCH")
	if got := ResolveToken(); got != "xoxb-bench-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-bench-token", got)
	}
	if got := ResolveChannel(); got != "C_BENCH" {
		t.Fatalf("ResolveChannel env = %q, want C_BENCH", got)
	}
}

func TestResolveTokenFallsBackToScoreboardToken(t *testing.T) {
	// The dedicated key is unset; the bench channel commonly shares the scoreboard
	// workspace, so the token must fall back to FAK_SCOREBOARD_TOKEN — never to the lab
	// SLACK_BOT_TOKEN.
	t.Setenv("FAK_BENCH_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-scoreboard-token")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token-must-not-leak")
	chdir(t, t.TempDir()) // no .env.slack.local
	if got := ResolveToken(); got != "xoxb-scoreboard-token" {
		t.Fatalf("ResolveToken fallback = %q, want the scoreboard token", got)
	}
}

func TestResolveTokenNeverLeaksLabToken(t *testing.T) {
	t.Setenv("FAK_BENCH_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir())
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked a token: got %q, want empty", got)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_BENCH_TOKEN", "")
	t.Setenv("FAK_BENCH_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_BENCH_TOKEN=xoxb-file-bench\n" +
		"FAK_BENCH_CHANNEL=C_FILE_BENCH\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-bench" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-bench", got)
	}
	if got := ResolveChannel(); got != "C_FILE_BENCH" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE_BENCH", got)
	}
}

func TestResolveChannelFallsThroughToDefault(t *testing.T) {
	// With no env / .env.slack.local value, ResolveChannel falls through to the public
	// built-in ChannelDefault (#1428) — the channel id is public, only the token is secret —
	// so the bench surface never resolves to NO channel and silently dry-runs.
	t.Setenv("FAK_BENCH_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := ResolveChannel(); got != ChannelDefault {
		t.Fatalf("ResolveChannel unset = %q, want the built-in default %q", got, ChannelDefault)
	}
	if ChannelDefault == "" {
		t.Fatal("bench ChannelDefault must be a real public channel id, not empty")
	}
}

// --- loaders ----------------------------------------------------------------

func TestLoadCatalogReadsTopLevelRuns(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "catalog.json")
	body := `{"$schema":"benchmark/catalog.v1","runs":[
	  {"run_id":"r1","machine_id":"m1","model":"SmolLM2","precision":"q8","peak_tok_per_sec":31.0,"timestamp":"2026-06-20T00:00:00Z","provenance":"measured"},
	  {"run_id":"r2","machine_id":"m2","model":"Qwen","precision":"Q8_0","peak_tok_per_sec":null,"baseline_tok_per_sec":12.0,"timestamp":"2026-06-21T00:00:00Z","provenance":"observed"}
	]}`
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cat, err := LoadCatalog(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Runs) != 2 {
		t.Fatalf("runs = %d, want 2", len(cat.Runs))
	}
	if v, ok := cat.Runs[0].Val(); !ok || v != 31.0 {
		t.Fatalf("run0 Val = (%v,%v), want (31,true)", v, ok)
	}
	// r2 has a null peak but a real baseline -> baseline is the value.
	if v, ok := cat.Runs[1].Val(); !ok || v != 12.0 {
		t.Fatalf("run1 Val = (%v,%v), want (12,true)", v, ok)
	}
}

// --- folds: latest rollup ---------------------------------------------------

func TestRollupFromCatalogOrdersAndLabels(t *testing.T) {
	cat := &Catalog{Runs: []Run{
		{MachineID: "m2", Model: "B", Precision: "q4", PeakTokS: f64(20), Timestamp: "2026-06-03T00:00:00Z", Provenance: "observed"},
		{MachineID: "m1", Model: "A", Precision: "q8", PeakTokS: f64(10), Timestamp: "2026-06-02T00:00:00Z", Provenance: "measured"},
		{MachineID: "m1", Model: "C", Timestamp: "2026-06-01T00:00:00Z", Tags: []string{"llama.cpp"}},
	}}
	p := RollupFromCatalog(cat, 2)
	txt := p.Text()
	if !strings.Contains(txt, "3 runs on record across 2 machines") {
		t.Fatalf("lead wrong:\n%s", txt)
	}
	// Only n=2 lines shown, and both provenance labels appear.
	if got := strings.Count(txt, "•"); got != 2 {
		t.Fatalf("lines = %d, want 2:\n%s", got, txt)
	}
	if !strings.Contains(txt, "WITNESSED") || !strings.Contains(txt, "OBSERVED") {
		t.Fatalf("expected WITNESSED+OBSERVED labels:\n%s", txt)
	}
	// Newest first: m2 (06-03) leads m1/A (06-01); m1/C (06-02) is dropped by n=2.
	lines := strings.Split(txt, "•")
	if !strings.HasPrefix(lines[1], " `m2`") {
		t.Fatalf("newest run (m2) should lead:\n%s", txt)
	}
}

func TestProvenanceLabelFromTagsWhenFieldBlank(t *testing.T) {
	cases := []struct {
		run  Run
		want string
	}{
		{Run{Provenance: "measured"}, "WITNESSED"},
		{Run{Provenance: "observed"}, "OBSERVED"},
		{Run{Tags: []string{"fak-native"}}, "WITNESSED"},
		{Run{Tags: []string{"engine-llama"}}, "OBSERVED"},
		{Run{Provenance: "unknown"}, "UNLABELED"},
	}
	for _, c := range cases {
		if got := provenanceLabel(c.run); got != c.want {
			t.Errorf("provenanceLabel(%+v) = %q, want %q", c.run, got, c.want)
		}
	}
}

func TestFmtTokShowsDashForGap(t *testing.T) {
	if got := fmtTok(Run{}); got != "—" {
		t.Fatalf("gap fmtTok = %q, want —", got)
	}
	if got := fmtTok(Run{PeakTokS: f64(38.07)}); got != "38.07 tok/s" {
		t.Fatalf("fmtTok = %q, want 38.07 tok/s", got)
	}
	if got := fmtTok(Run{PeakTokS: f64(56)}); got != "56 tok/s" {
		t.Fatalf("fmtTok whole = %q, want 56 tok/s", got)
	}
}

// --- folds: regression ------------------------------------------------------

func TestRegressionFlagsDualThresholdDropOnly(t *testing.T) {
	// Baseline key must match fullKey(run) = slug(machine)/benchKey(model-precision).
	cat := &Catalog{Runs: []Run{
		// big drop: 30 -> 20 (33%, 10 tok/s) -> flagged
		{MachineID: "node-a", Model: "SmolLM2-135M-Instruct", Precision: "q8", PeakTokS: f64(20), Timestamp: "2026-06-10T00:00:00Z"},
		// tiny relative wobble: 100 -> 98 (2%) -> NOT flagged
		{MachineID: "node-b", Model: "Qwen", Precision: "Q8_0", PeakTokS: f64(98), Timestamp: "2026-06-10T00:00:00Z"},
		// big % but sub-tok absolute: 2.0 -> 1.5 (25% but 0.5 tok/s) -> NOT flagged
		{MachineID: "node-c", Model: "Micro", Precision: "q4", PeakTokS: f64(1.5), Timestamp: "2026-06-10T00:00:00Z"},
	}}
	bl := &Baseline{Baselines: map[string]float64{
		fullKey(cat.Runs[0]): 30.0,
		fullKey(cat.Runs[1]): 100.0,
		fullKey(cat.Runs[2]): 2.0,
	}}
	p := RegressionFromCatalogVsBaseline(cat, bl, 15.0, 1.0)
	txt := p.Text()
	if p.Emoji != ":red_circle:" {
		t.Fatalf("want red emoji for a drop, got %q\n%s", p.Emoji, txt)
	}
	if !strings.Contains(txt, "1 benchmark(s) dropped") {
		t.Fatalf("want exactly 1 flagged drop:\n%s", txt)
	}
	if !strings.Contains(txt, "30.00 → 20.00") {
		t.Fatalf("want the 30->20 drop line:\n%s", txt)
	}
	if strings.Contains(txt, "node-b") || strings.Contains(txt, "node-c") {
		t.Fatalf("sub-threshold drops must not be flagged:\n%s", txt)
	}
}

func TestRegressionCleanWhenNoDrops(t *testing.T) {
	cat := &Catalog{Runs: []Run{
		{MachineID: "m", Model: "X", Precision: "q8", PeakTokS: f64(50), Timestamp: "2026-06-10T00:00:00Z"},
	}}
	bl := &Baseline{Baselines: map[string]float64{fullKey(cat.Runs[0]): 50.0}}
	p := RegressionFromCatalogVsBaseline(cat, bl, 15.0, 1.0)
	if p.Emoji != ":white_check_mark:" {
		t.Fatalf("clean check should be a green check, got %q", p.Emoji)
	}
	if !strings.Contains(p.Text(), "no tok/s regressions") {
		t.Fatalf("clean lead wrong:\n%s", p.Text())
	}
}

func TestRegressionNilBaselineIsClean(t *testing.T) {
	cat := &Catalog{Runs: []Run{{MachineID: "m", Model: "X", PeakTokS: f64(1)}}}
	p := RegressionFromCatalogVsBaseline(cat, nil, 15.0, 1.0)
	if p.Emoji != ":white_check_mark:" {
		t.Fatalf("nil baseline should be clean, got %q", p.Emoji)
	}
}

// --- folds: request ---------------------------------------------------------

func TestRequestFromPlanFoldsPerMachineAndHonesty(t *testing.T) {
	raw := []byte(`{
	  "schema":"benchmark/plan","ok":true,"now":"20260627T143000Z",
	  "honesty":"PLAN ONLY -- no benchmark was run.",
	  "per_machine_next":{
	    "a100":{"machine_id":"a100","workload_kind":"gpu-benchmark","model":"qwen2.5-3b","precision":"Q8_0","intent":"learn-collect","suggested_command":"on a100 (NVIDIA A100): go run -tags cuda ./cmd/gpucheck  # HINT -- not run"},
	    "mac":{"machine_id":"mac","workload_kind":"fan-benchmark","model":"None","precision":"n/a","intent":"coverage","suggested_command":"on mac (Apple): go run ./cmd/fanbench"}
	  },
	  "ranked":[]
	}`)
	plan, err := ParsePlan(raw)
	if err != nil {
		t.Fatal(err)
	}
	p := RequestFromPlan(plan, 0)
	txt := p.Text()
	if !strings.Contains(txt, "PLAN ONLY") || !strings.Contains(txt, "planned 20260627T143000Z") {
		t.Fatalf("honesty banner missing:\n%s", txt)
	}
	if !strings.Contains(txt, "`a100` → gpu-benchmark · qwen2.5-3b/Q8_0 [learn-collect]") {
		t.Fatalf("a100 row wrong:\n%s", txt)
	}
	// "None"/"n/a" -> agent-workload, command hint stripped of prefix/comment.
	if !strings.Contains(txt, "`mac` → fan-benchmark · (agent workload)") {
		t.Fatalf("mac row wrong:\n%s", txt)
	}
	if !strings.Contains(txt, "go run -tags cuda ./cmd/gpucheck") || strings.Contains(txt, "# HINT") {
		t.Fatalf("command should be bare (no prefix/comment):\n%s", txt)
	}
}

func TestRequestTopCapsMachines(t *testing.T) {
	raw := []byte(`{"per_machine_next":{
	  "a":{"machine_id":"a","workload_kind":"k"},
	  "b":{"machine_id":"b","workload_kind":"k"},
	  "c":{"machine_id":"c","workload_kind":"k"}}}`)
	plan, _ := ParsePlan(raw)
	p := RequestFromPlan(plan, 2)
	if got := strings.Count(p.Text(), "•"); got != 2 {
		t.Fatalf("top=2 should show 2 lines, got %d:\n%s", got, p.Text())
	}
}

// --- render -----------------------------------------------------------------

func TestBlocksCarrySameFacts(t *testing.T) {
	p := Post{Emoji: ":bar_chart:", Title: "T", Lead: "L", Lines: []string{"x", "y"}, Source: "ci"}
	b := p.Blocks()
	if len(b) == 0 {
		t.Fatal("no blocks")
	}
	// Smoke: a non-empty post yields a header + lead + body + source context + S/N context.
	if len(b) != 5 {
		t.Fatalf("blocks = %d, want 5 (header/lead/body/source-context/sn-context)", len(b))
	}
}

// chdir switches to dir for the test and restores the prior cwd after.
func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}
