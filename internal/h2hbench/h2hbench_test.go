package h2hbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeEngine streams an OpenAI SSE answer after a delay proportional to the
// uncached prompt, with a prefix cache keyed on the longest previously seen
// prompt, so the runner's TTFT, prefill, decode and cache readings are checkable.
type fakeEngine struct {
	mu          sync.Mutex
	seen        []string
	perTokenDur time.Duration
	reportCache bool
	timings     bool
}

func (f *fakeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages []message `json:"messages"`
		Max      int       `json:"max_tokens"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	var sb strings.Builder
	for _, m := range req.Messages {
		sb.WriteString(m.Role + ":" + m.Content + "\n")
	}
	prompt := sb.String()
	tokens := len(strings.Fields(prompt))
	f.mu.Lock()
	cachedChars := 0
	for _, s := range f.seen {
		n := 0
		for n < len(s) && n < len(prompt) && s[n] == prompt[n] {
			n++
		}
		if n > cachedChars {
			cachedChars = n
		}
	}
	f.seen = append(f.seen, prompt)
	f.mu.Unlock()
	cached := len(strings.Fields(prompt[:cachedChars]))
	if cached > tokens {
		cached = tokens
	}
	time.Sleep(time.Duration(tokens-cached) * f.perTokenDur)
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	for i := 0; i < req.Max; i++ {
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"w%d \"}}]}\n\n", i)
		fl.Flush()
		time.Sleep(time.Millisecond)
	}
	usage := map[string]any{"prompt_tokens": tokens, "completion_tokens": req.Max}
	if f.reportCache {
		usage["prompt_tokens_details"] = map[string]int{"cached_tokens": cached}
	}
	final := map[string]any{"choices": []any{}, "usage": usage}
	if f.timings {
		final["timings"] = ServerTimings{CacheN: cached, PromptN: tokens - cached, PredictedN: req.Max, PredictedPerSecond: 99}
	}
	b, _ := json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
}

func TestRunMeasuresAllScenariosAndKeepsArmsCacheIsolated(t *testing.T) {
	// Both arms share ONE upstream: the nonce must stop arm B riding arm A's cache.
	eng := &fakeEngine{perTokenDur: 20 * time.Microsecond, reportCache: true, timings: true}
	srv := httptest.NewServer(eng)
	defer srv.Close()
	fakArm, _ := ParseArm("fak-serve=" + srv.URL + "/v1/")
	llamaArm, _ := ParseArm("llama=" + srv.URL + "/v1")
	if fakArm.Role != RoleFak || llamaArm.Role != RoleBaseline || strings.HasSuffix(fakArm.BaseURL, "/") {
		t.Fatalf("ParseArm roles/url: %+v %+v", fakArm, llamaArm)
	}
	var sunk int
	rows, err := Run(context.Background(), Config{
		RunID: "t1", Model: "m", Arms: []Arm{fakArm, llamaArm},
		ColdSizes: []int{200, 800}, Reps: 2, DecodeTokens: 12, Turns: 3, SystemTokens: 400, TurnTokens: 40, AnswerTokens: 4,
		Sink: func(Row) error { sunk++; return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	// cold 2 sizes*2 reps*2 arms + decode 2 reps*2 arms + multiturn 2 convs*3 turns*2 arms
	if want := 8 + 4 + 12; len(rows) != want || sunk != want {
		t.Fatalf("rows=%d sunk=%d want %d", len(rows), sunk, want)
	}
	for _, r := range rows {
		if !r.OK || r.TTFTMS <= 0 || r.E2EMS < r.TTFTMS || r.Schema != Schema {
			t.Fatalf("bad row %+v", r)
		}
		switch r.Scenario {
		case ScenarioCold:
			// The nonce sits in front of the shared body: nothing beyond the chat
			// role prefix may be served from cache, even for the second arm.
			if r.CacheRatio > 0.05 {
				t.Fatalf("cold row rode another arm's cache: %+v", r)
			}
		case ScenarioMultiTurn:
			if r.Size >= 2 && r.CacheRatio < 0.8 {
				t.Fatalf("turn %d should reuse the conversation prefix, cache_ratio=%.2f", r.Size, r.CacheRatio)
			}
		case ScenarioDecode:
			if r.CompletionTokens != 12 || r.DecodeTPS <= 0 {
				t.Fatalf("decode row %+v", r)
			}
		}
		if r.Server == nil || r.Server.PredictedPerSecond != 99 {
			t.Fatalf("server timings not kept: %+v", r.Server)
		}
	}
	rep := Summarize(rows)
	if rep.RunID != "t1" || len(rep.Cells) != 2*(2+1+3+1) {
		t.Fatalf("cells=%d run=%q", len(rep.Cells), rep.RunID)
	}
}

func TestCachedTokensFallBackToServerTimings(t *testing.T) {
	eng := &fakeEngine{timings: true}
	srv := httptest.NewServer(eng)
	defer srv.Close()
	arm, _ := ParseArm("llama=" + srv.URL + "/v1")
	rows, err := Run(context.Background(), Config{RunID: "t", Model: "m", Arms: []Arm{arm}, Turns: 2, SystemTokens: 200, TurnTokens: 10, AnswerTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	if rows[1].CachedTokens == 0 {
		t.Fatalf("turn 2 cached tokens should come from timings.cache_n: %+v", rows[1])
	}
}

func TestCompareNamesLossesAgainstBestRival(t *testing.T) {
	cells := []Cell{
		{Arm: "fak", Role: RoleFak, Scenario: ScenarioCold, Size: 2048, N: 3, TTFTP50: 1200, PrefillP50: 1700},
		{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioCold, Size: 2048, N: 3, TTFTP50: 1000, PrefillP50: 2000},
		{Arm: "vllm", Role: RoleBaseline, Scenario: ScenarioCold, Size: 2048, N: 3, TTFTP50: 1500, PrefillP50: 1400},
		{Arm: "fak", Role: RoleFak, Scenario: ScenarioDecode, Size: 256, N: 3, DecodeP50: 20},
		{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioDecode, Size: 256, N: 3, DecodeP50: 10},
		{Arm: "fak", Role: RoleFak, Scenario: ScenarioMultiTurn, Size: WarmTurns, N: 3, TTFTP50: 100, CacheRatio: 0.98},
		{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioMultiTurn, Size: WarmTurns, N: 3, TTFTP50: 102, CacheRatio: 0},
		// Per-turn cells carry order bias and are never graded.
		{Arm: "fak", Role: RoleFak, Scenario: ScenarioMultiTurn, Size: 3, N: 3, TTFTP50: 900, CacheRatio: 0.98},
		{Arm: "llama", Role: RoleBaseline, Scenario: ScenarioMultiTurn, Size: 3, N: 3, TTFTP50: 100, CacheRatio: 0.98},
	}
	losses, wins := Compare(cells)
	if len(losses) != 2 {
		t.Fatalf("losses=%+v", losses)
	}
	for _, l := range losses {
		if l.Rival != "llama" || l.Scenario != ScenarioCold {
			t.Fatalf("loss must be vs the BEST rival on cold: %+v", l)
		}
	}
	if losses[0].Metric != "ttft_ms_p50" { // 1200 vs 1000 = 16.7% > 1700 vs 2000 = 15%
		t.Fatalf("losses not sorted worst-first: %+v", losses)
	}
	var sawDecode, sawCache bool
	for _, w := range wins {
		sawDecode = sawDecode || w.Metric == "decode_tps_p50"
		sawCache = sawCache || w.Metric == "cache_ratio_mean"
		if w.Metric == "ttft_ms_p50" {
			t.Fatalf("a 2%% TTFT gap is a tie, not a win: %+v", w)
		}
	}
	if !sawDecode || !sawCache {
		t.Fatalf("wins=%+v", wins)
	}
}

func TestLedgerRoundTripSelectsLatestRun(t *testing.T) {
	path := filepath.Join(t.TempDir(), "h2h.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	_ = enc.Encode(Row{Schema: Schema, RunID: "old", UnixMS: 1, Arm: "a", Scenario: ScenarioCold, OK: true})
	_, _ = f.WriteString("not json\n")
	_ = enc.Encode(Row{Schema: Schema, RunID: "new", UnixMS: 5, Arm: "a", Scenario: ScenarioCold, OK: true})
	_ = enc.Encode(Row{Schema: Schema, RunID: "new", UnixMS: 6, Arm: "b", Scenario: ScenarioCold, OK: false, Error: "x"})
	f.Close()
	rows, err := ReadLedger(path)
	if err != nil || len(rows) != 3 {
		t.Fatalf("rows=%d err=%v", len(rows), err)
	}
	if got := SelectRun(rows, ""); len(got) != 2 || got[0].RunID != "new" {
		t.Fatalf("latest run = %+v", got)
	}
	if got := SelectRun(rows, "old"); len(got) != 1 {
		t.Fatalf("explicit run = %+v", got)
	}
	rep := Summarize(SelectRun(rows, ""))
	var sb strings.Builder
	Render(&sb, rep)
	if !strings.Contains(sb.String(), "h2h run new") {
		t.Fatalf("render: %s", sb.String())
	}
}

func TestForeignSlotLoadIsFlaggedAndExcluded(t *testing.T) {
	busy := 0
	mux := http.NewServeMux()
	mux.Handle("/v1/chat/completions", &fakeEngine{})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `[{"id":0,"is_processing":%v},{"id":1,"is_processing":false}]`, busy > 0)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()
	arm, _ := ParseArm("llama=" + srv.URL + "/v1")
	cfg := Config{RunID: "c", Model: "m", Arms: []Arm{arm}, ColdSizes: []int{50}, Reps: 2, AnswerTokens: 2, SlotsURL: srv.URL + "/slots"}
	clean, err := Run(context.Background(), cfg)
	if err != nil || len(clean) != 2 || clean[0].BusySlots == nil || clean[0].Contended() {
		t.Fatalf("idle upstream: rows=%+v err=%v", clean, err)
	}
	busy = 1
	dirty, _ := Run(context.Background(), cfg)
	if !dirty[0].Contended() {
		t.Fatalf("busy upstream not flagged: %+v", dirty[0])
	}
	cells := Summarize(append(clean, dirty...)).Cells
	if len(cells) != 1 || cells[0].N != 2 || cells[0].Contended != 2 {
		t.Fatalf("contended rows must be counted but kept out of the fold: %+v", cells)
	}
}

func TestBusyGPUMarksRowContended(t *testing.T) {
	busyFile := filepath.Join(t.TempDir(), "gpu_busy_percent")
	defer func(d time.Duration) { gpuBusySettle = d }(gpuBusySettle)
	gpuBusySettle = 200 * time.Millisecond
	srv := httptest.NewServer(&fakeEngine{})
	defer srv.Close()
	arm, _ := ParseArm("llama=" + srv.URL + "/v1")
	cfg := Config{RunID: "g", Model: "m", Arms: []Arm{arm}, ColdSizes: []int{20}, AnswerTokens: 2, GPUBusyPath: busyFile}
	for _, tc := range []struct {
		pct       string
		contended bool
	}{{"3 ", false}, {"87 ", true}} {
		if err := os.WriteFile(busyFile, []byte(tc.pct), 0o644); err != nil {
			t.Fatal(err)
		}
		rows, err := Run(context.Background(), cfg)
		if err != nil || rows[0].GPUBusyPct == nil || rows[0].Contended() != tc.contended {
			t.Fatalf("gpu busy %q: row=%+v err=%v", tc.pct, rows[0], err)
		}
	}
}

func TestPositionBalancedMedianIgnoresWhoRanSecond(t *testing.T) {
	// Both arms pay +400 ms when second; fak ran second twice, llama once.
	mk := func(arm, role string, pos int, ttft float64) Row {
		return Row{Schema: Schema, RunID: "p", Arm: arm, Role: role, Scenario: ScenarioCold, Size: 1, Pos: pos, OK: true, TTFTMS: ttft, PromptTokens: 10}
	}
	rows := []Row{
		mk("fak", RoleFak, 0, 1500), mk("fak", RoleFak, 1, 1900), mk("fak", RoleFak, 1, 1910),
		mk("llama", RoleBaseline, 1, 1905), mk("llama", RoleBaseline, 0, 1495), mk("llama", RoleBaseline, 0, 1490),
	}
	rep := Summarize(rows)
	if len(rep.Losses) != 0 || len(rep.Wins) != 0 {
		t.Fatalf("order luck graded as a result: losses=%+v wins=%+v cells=%+v", rep.Losses, rep.Wins, rep.Cells)
	}
}

func TestEveryArmVisitsEveryPosition(t *testing.T) {
	srv := httptest.NewServer(&fakeEngine{})
	defer srv.Close()
	var arms []Arm
	for _, n := range []string{"fak-a", "fak-b", "llama"} {
		a, _ := ParseArm(n + "=" + srv.URL + "/v1")
		arms = append(arms, a)
	}
	rows, err := Run(context.Background(), Config{RunID: "o", Model: "m", Arms: arms, ColdSizes: []int{10}, Reps: 3, AnswerTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]map[int]bool{}
	for _, r := range rows {
		if seen[r.Arm] == nil {
			seen[r.Arm] = map[int]bool{}
		}
		seen[r.Arm][r.Pos] = true
	}
	for arm, pos := range seen {
		if len(pos) != 3 {
			t.Fatalf("%s only ran at positions %v", arm, pos)
		}
	}
}
