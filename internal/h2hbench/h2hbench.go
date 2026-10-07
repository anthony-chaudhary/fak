// Package h2hbench is the engine head-to-head benchmark: the same seeded
// workload replayed against several OpenAI-compatible serving arms (fak serve,
// stock llama-server, vLLM, ollama, ...) on one box, measuring the four numbers
// the inference-engine goal is graded on: TTFT, prefill rate, decode rate, and
// multi-turn prompt-cache reuse.
//
// Every request becomes one Row appended to a JSONL ledger (beside the gateway
// perf ledger) so agents read results back with `fak bench h2h report --json`
// instead of scraping terminal output. Summarize folds the rows per arm and
// scenario; Compare names every cell where a fak arm trails the best rival.
//
// Fairness rules the runner enforces:
//   - Each request opens with a per-(run, arm, conversation) nonce, so one arm
//     can never warm the prompt cache for another when they share an upstream.
//   - Arms are interleaved request by request (order alternates per rep), so
//     thermal and background drift lands on every arm alike.
//   - Rates are computed client-side the same way for every arm; server-reported
//     llama.cpp timings are kept alongside as diagnostics, never mixed in.
package h2hbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// Schema is the per-request ledger row schema.
	Schema = "fak.bench.h2h.v1"
	// ReportSchema is the folded report schema `report --json` emits.
	ReportSchema = "fak.bench.h2h-report.v1"
	// DefaultLedgerRel sits beside perfledger.DefaultLedgerRel.
	DefaultLedgerRel = ".fak/nightrun/bench-h2h.jsonl"

	ScenarioCold      = "cold"      // unique long prompt, short answer: TTFT + prefill
	ScenarioDecode    = "decode"    // short prompt, long answer: decode rate
	ScenarioMultiTurn = "multiturn" // growing conversation: cache reuse + warm TTFT

	RoleFak      = "fak"
	RoleBaseline = "baseline"

	// lossTolerance is the relative gap below which a cell is a tie, not a loss.
	lossTolerance = 0.05

	// WarmTurns is the multiturn cell Size that pools every turn from 2 on.
	// Per-turn cells carry an order bias (whichever arm runs second in a
	// back-to-back pair is measured ~400 ms slower on llama-server, strix3),
	// and arms alternate order per turn, so only the pooled cell is graded.
	WarmTurns = 0
)

// Arm is one serving endpoint under test.
type Arm struct {
	Name    string `json:"name"`
	BaseURL string `json:"base_url"` // OpenAI-compatible root ending in /v1
	Role    string `json:"role"`     // RoleFak or RoleBaseline
	APIKey  string `json:"-"`
}

// ParseArm parses "name=url". An arm whose name starts with "fak" is a fak arm.
func ParseArm(s string) (Arm, error) {
	name, url, ok := strings.Cut(strings.TrimSpace(s), "=")
	name, url = strings.TrimSpace(name), strings.TrimSpace(url)
	if !ok || name == "" || url == "" {
		return Arm{}, fmt.Errorf("arm %q: want name=http://host:port/v1", s)
	}
	role := RoleBaseline
	if strings.HasPrefix(strings.ToLower(name), "fak") {
		role = RoleFak
	}
	return Arm{Name: name, BaseURL: strings.TrimRight(url, "/"), Role: role}, nil
}

// Config is one run's workload.
type Config struct {
	RunID          string
	Host           string
	Model          string
	Arms           []Arm
	ColdSizes      []int // approximate prompt tokens per cold request
	Reps           int
	DecodeTokens   int // max_tokens for the decode scenario; 0 skips it
	Turns          int // multi-turn conversation length; 0 skips it
	SystemTokens   int // multi-turn shared system prefix size
	TurnTokens     int // new user tokens per turn
	AnswerTokens   int // max_tokens for cold and multi-turn answers
	Timeout        time.Duration
	Client         *http.Client
	DisableThink   bool // send chat_template_kwargs.enable_thinking=false
	Progress       io.Writer
	Sink           func(Row) error
	ConversationsN int // multi-turn conversations per arm (default Reps)
	// SlotsURL is a llama-server /slots endpoint on the shared upstream. When
	// set, each row records how many slots OTHER clients held around it, so a
	// reading taken under foreign load is flagged and kept out of the medians.
	SlotsURL string
	// GPUBusyPath is an amdgpu gpu_busy_percent file. When set, the GPU is
	// sampled just before each request: a GPU already busy means another
	// process (a second server, a tuning run) shares the device, which /slots
	// cannot see.
	GPUBusyPath string
}

// gpuBusyContended is the idle-GPU busy percentage above which a reading is
// treated as shared with a foreign process.
const gpuBusyContended = 20

// GPUBusyAuto returns the first amdgpu gpu_busy_percent file on this host, or "".
func GPUBusyAuto() string {
	m, _ := filepath.Glob("/sys/class/drm/card*/device/gpu_busy_percent")
	if len(m) == 0 {
		return ""
	}
	sort.Strings(m)
	return m[0]
}

// gpuBusySettle bounds how long gpuBusy waits for the device to go idle. The
// amdgpu counter decays with a ~1 s half-life after our own previous request
// (measured on strix3: 74% -> under 20% in ~1.3 s), so a busy reading that
// survives this window is foreign load, not our own tail.
var gpuBusySettle = 3 * time.Second

// gpuBusy waits for the device to settle below gpuBusyContended and returns the
// lowest busy percentage seen, or -1 when the counter is unreadable.
func gpuBusy(path string) int {
	if path == "" {
		return -1
	}
	low := -1
	deadline := time.Now().Add(gpuBusySettle)
	for {
		b, err := os.ReadFile(path)
		if err != nil {
			return -1
		}
		v, err := strconv.Atoi(strings.TrimSpace(string(b)))
		if err != nil {
			return -1
		}
		if low < 0 || v < low {
			low = v
		}
		if low < gpuBusyContended || time.Now().After(deadline) {
			return low
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// ServerTimings is llama.cpp's own per-request accounting, when the arm returns it.
type ServerTimings struct {
	CacheN             int     `json:"cache_n"`
	PromptN            int     `json:"prompt_n"`
	PromptMS           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int     `json:"predicted_n"`
	PredictedMS        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
	DraftN             int     `json:"draft_n,omitempty"`
	DraftAccepted      int     `json:"draft_n_accepted,omitempty"`
}

// Row is one measured request.
type Row struct {
	Schema   string `json:"schema"`
	RunID    string `json:"run_id"`
	UnixMS   int64  `json:"unix_ms"`
	Host     string `json:"host,omitempty"`
	Model    string `json:"model"`
	Arm      string `json:"arm"`
	Role     string `json:"role"`
	Scenario string `json:"scenario"`
	Size     int    `json:"size"` // cold: target prompt tokens; decode: max tokens; multiturn: turn number
	Rep      int    `json:"rep"`
	// Pos is this request's place in its back-to-back group of arms. On
	// llama-server the second request of a pair measured ~400 ms slower on warm
	// multi-turn TTFT regardless of arm (strix3), so the fold balances by Pos.
	Pos              int     `json:"pos"`
	OK               bool    `json:"ok"`
	Error            string  `json:"error,omitempty"`
	PromptTokens     int     `json:"prompt_tokens"`
	CachedTokens     int     `json:"cached_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TTFTMS           float64 `json:"ttft_ms"`
	E2EMS            float64 `json:"e2e_ms"`
	PrefillTPS       float64 `json:"prefill_tps"` // uncached prompt tokens / TTFT
	DecodeTPS        float64 `json:"decode_tps"`  // (completion-1) / (e2e - TTFT)
	CacheRatio       float64 `json:"cache_ratio"` // cached / prompt
	// BusySlots is the most upstream slots other clients held just before or
	// just after this request (nil = not probed). >0 marks a contended reading.
	BusySlots *int `json:"busy_slots,omitempty"`
	// GPUBusyPct is the device's busy percentage just before the request was
	// sent (nil = not probed); >= gpuBusyContended marks a contended reading.
	GPUBusyPct *int           `json:"gpu_busy_pct,omitempty"`
	Server     *ServerTimings `json:"server,omitempty"`
}

// Contended reports whether another client shared the upstream with this request.
func (r Row) Contended() bool {
	return (r.BusySlots != nil && *r.BusySlots > 0) || (r.GPUBusyPct != nil && *r.GPUBusyPct >= gpuBusyContended)
}

// busySlots counts processing slots on a llama-server /slots endpoint; -1 = unknown.
func busySlots(ctx context.Context, client *http.Client, url string) int {
	if url == "" {
		return -1
	}
	rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, url, nil)
	if err != nil {
		return -1
	}
	resp, err := client.Do(req)
	if err != nil {
		return -1
	}
	defer resp.Body.Close()
	var slots []struct {
		IsProcessing bool `json:"is_processing"`
	}
	if resp.StatusCode/100 != 2 || json.NewDecoder(resp.Body).Decode(&slots) != nil {
		return -1
	}
	n := 0
	for _, s := range slots {
		if s.IsProcessing {
			n++
		}
	}
	return n
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// words is a fixed vocabulary of common single-token English words, so a
// prompt of N words is close to N tokens on every BPE tokenizer in use.
var words = strings.Fields(`the of and to in is that it for on with as was at by be this from or have an are not
but had his they you were her she which all their there been one would will we can more when if out so up
said what about into than them only other new some could time these two may then first any like now my
such make over our even most made after also did many before must through back years where much your way
well down should because each just those people how too little state good very world still own see men
work long here get both between life being under never day same another know while last might us great
old year off come since against go came right used take three states himself few house use during
without again place around however home small found thought went say part once general high upon school
every point form number end system order level case water give fact group play stand increase early
course change help line city put close need report table value cache token model serve engine decode`)

// filler returns about n words of seeded text.
func filler(r *rand.Rand, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			if i%16 == 0 {
				b.WriteString(".\n")
			} else {
				b.WriteByte(' ')
			}
		}
		b.WriteString(words[r.Intn(len(words))])
	}
	b.WriteByte('.')
	return b.String()
}

// nonce makes each conversation's prefix unique to (run, arm, conversation).
func nonce(runID, arm, tag string) string {
	return fmt.Sprintf("[h2h run=%s arm=%s conv=%s]\n", runID, arm, tag)
}

// Run replays the workload across every arm and hands each row to cfg.Sink.
func Run(ctx context.Context, cfg Config) ([]Row, error) {
	if len(cfg.Arms) == 0 {
		return nil, errors.New("h2hbench: no arms")
	}
	if cfg.Reps <= 0 {
		cfg.Reps = 1
	}
	if cfg.AnswerTokens <= 0 {
		cfg.AnswerTokens = 16
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	if cfg.ConversationsN <= 0 {
		cfg.ConversationsN = cfg.Reps
	}
	var rows []Row
	emit := func(r Row) error {
		rows = append(rows, r)
		if cfg.Progress != nil {
			status := "ok"
			if !r.OK {
				status = "ERR " + r.Error
			}
			fmt.Fprintf(cfg.Progress, "%-9s %-10s size=%-6d rep=%d prompt=%-6d cached=%-6d ttft=%8.1fms prefill=%7.1ft/s decode=%6.2ft/s %s\n",
				r.Scenario, r.Arm, r.Size, r.Rep, r.PromptTokens, r.CachedTokens, r.TTFTMS, r.PrefillTPS, r.DecodeTPS, status)
		}
		if cfg.Sink != nil {
			return cfg.Sink(r)
		}
		return nil
	}
	order := func(rep int) []Arm {
		arms := append([]Arm(nil), cfg.Arms...)
		if rep%2 == 1 {
			for i, j := 0, len(arms)-1; i < j; i, j = i+1, j-1 {
				arms[i], arms[j] = arms[j], arms[i]
			}
		}
		return arms
	}
	base := func(arm Arm, scenario string, size, rep int) Row {
		return Row{Schema: Schema, RunID: cfg.RunID, Host: cfg.Host, Model: cfg.Model,
			Arm: arm.Name, Role: arm.Role, Scenario: scenario, Size: size, Rep: rep}
	}

	// Cold prefill: identical seeded body per (size, rep), unique nonce per arm.
	for _, size := range cfg.ColdSizes {
		for rep := 0; rep < cfg.Reps; rep++ {
			body := filler(rand.New(rand.NewSource(int64(size*1000+rep))), size)
			for pos, arm := range order(rep) {
				if err := ctx.Err(); err != nil {
					return rows, err
				}
				msgs := []message{{Role: "user", Content: nonce(cfg.RunID, arm.Name, fmt.Sprintf("cold-%d-%d", size, rep)) + body +
					"\n\nIn one short sentence, what is the most frequent word above?"}}
				r := base(arm, ScenarioCold, size, rep)
				r.Pos = pos
				if _, err := measure(ctx, cfg, arm, msgs, cfg.AnswerTokens, &r); err != nil {
					r.Error = err.Error()
				}
				if err := emit(r); err != nil {
					return rows, err
				}
			}
		}
	}

	// Decode: short prompt, long answer.
	if cfg.DecodeTokens > 0 {
		for rep := 0; rep < cfg.Reps; rep++ {
			for pos, arm := range order(rep) {
				if err := ctx.Err(); err != nil {
					return rows, err
				}
				msgs := []message{{Role: "user", Content: nonce(cfg.RunID, arm.Name, fmt.Sprintf("decode-%d", rep)) +
					"Write the integers from 1 to 1000 in English words, separated by commas. Do not stop early."}}
				r := base(arm, ScenarioDecode, cfg.DecodeTokens, rep)
				r.Pos = pos
				if _, err := measure(ctx, cfg, arm, msgs, cfg.DecodeTokens, &r); err != nil {
					r.Error = err.Error()
				}
				if err := emit(r); err != nil {
					return rows, err
				}
			}
		}
	}

	// Multi-turn: one conversation per arm per rep, turns interleaved across arms
	// so every arm's turn t runs back to back.
	if cfg.Turns > 0 {
		for conv := 0; conv < cfg.ConversationsN; conv++ {
			r0 := rand.New(rand.NewSource(int64(7_000_000 + conv)))
			system := filler(r0, cfg.SystemTokens)
			turnText := make([]string, cfg.Turns)
			for t := range turnText {
				turnText[t] = filler(r0, cfg.TurnTokens)
			}
			hist := map[string][]message{}
			for _, arm := range cfg.Arms {
				hist[arm.Name] = []message{{Role: "system", Content: nonce(cfg.RunID, arm.Name, fmt.Sprintf("mt-%d", conv)) +
					"You are a terse assistant. Reference material follows.\n" + system}}
			}
			for t := 0; t < cfg.Turns; t++ {
				for pos, arm := range order(conv + t) {
					if err := ctx.Err(); err != nil {
						return rows, err
					}
					h := append(hist[arm.Name], message{Role: "user", Content: fmt.Sprintf("Turn %d notes:\n%s\nReply with one short sentence.", t+1, turnText[t])})
					r := base(arm, ScenarioMultiTurn, t+1, conv)
					r.Pos = pos
					answer, err := measure(ctx, cfg, arm, h, cfg.AnswerTokens, &r)
					if err != nil {
						r.Error = err.Error()
						answer = "ok."
					}
					if strings.TrimSpace(answer) == "" {
						answer = "ok."
					}
					hist[arm.Name] = append(h, message{Role: "assistant", Content: answer})
					if err := emit(r); err != nil {
						return rows, err
					}
				}
			}
		}
	}
	return rows, nil
}

// measure sends one streaming chat request and fills r. It returns the answer text.
func measure(ctx context.Context, cfg Config, arm Arm, msgs []message, maxTokens int, r *Row) (string, error) {
	// GPU settle first, so the previous request's slot has been released too.
	if g := gpuBusy(cfg.GPUBusyPath); g >= 0 {
		r.GPUBusyPct = &g
	}
	before := busySlots(ctx, cfg.Client, cfg.SlotsURL)
	// Mid-stream, our own request holds exactly one slot: anything above that
	// is another client. Probing at the first token avoids the release race an
	// after-the-fact probe has.
	during := make(chan int, 1)
	fired := false
	onFirst := func() {
		fired = true
		go func() { during <- busySlots(ctx, cfg.Client, cfg.SlotsURL) - 1 }()
	}
	answer, err := measureOnce(ctx, cfg, arm, msgs, maxTokens, r, onFirst)
	if before >= 0 {
		busy := before
		if fired {
			select {
			case d := <-during:
				busy = max(busy, d)
			case <-time.After(5 * time.Second):
			}
		}
		r.BusySlots = &busy
	}
	return answer, err
}

func measureOnce(ctx context.Context, cfg Config, arm Arm, msgs []message, maxTokens int, r *Row, onFirst func()) (string, error) {
	r.UnixMS = time.Now().UnixMilli()
	payload := map[string]any{
		"model":          cfg.Model,
		"messages":       msgs,
		"max_tokens":     maxTokens,
		"temperature":    0,
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
	}
	if cfg.DisableThink {
		payload["chat_template_kwargs"] = map[string]bool{"enable_thinking": false}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	rctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, arm.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if arm.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+arm.APIKey)
	}
	start := time.Now()
	resp, err := cfg.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("http %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var first time.Time
	var answer strings.Builder
	var events int
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			break
		}
		var ch struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens        int `json:"prompt_tokens"`
				CompletionTokens    int `json:"completion_tokens"`
				PromptTokensDetails *struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
			Timings *ServerTimings `json:"timings"`
		}
		if json.Unmarshal([]byte(data), &ch) != nil {
			continue
		}
		for _, c := range ch.Choices {
			if c.Delta.Content == "" && c.Delta.ReasoningContent == "" && c.Delta.Reasoning == "" {
				continue
			}
			if first.IsZero() {
				first = time.Now()
				if onFirst != nil {
					onFirst()
				}
			}
			events++
			answer.WriteString(c.Delta.Content)
		}
		if u := ch.Usage; u != nil {
			r.PromptTokens = u.PromptTokens
			r.CompletionTokens = u.CompletionTokens
			if u.PromptTokensDetails != nil {
				r.CachedTokens = u.PromptTokensDetails.CachedTokens
			}
		}
		if ch.Timings != nil {
			r.Server = ch.Timings
		}
	}
	if err := sc.Err(); err != nil {
		return answer.String(), err
	}
	end := time.Now()
	r.E2EMS = ms(end.Sub(start))
	if first.IsZero() {
		return answer.String(), errors.New("no output tokens streamed")
	}
	r.TTFTMS = ms(first.Sub(start))
	if r.CompletionTokens == 0 {
		r.CompletionTokens = events
	}
	if r.Server != nil && r.CachedTokens == 0 && r.Server.CacheN > 0 {
		r.CachedTokens = r.Server.CacheN
	}
	if r.PromptTokens > 0 {
		r.CacheRatio = float64(r.CachedTokens) / float64(r.PromptTokens)
		if uncached := r.PromptTokens - r.CachedTokens; uncached > 0 && r.TTFTMS > 0 {
			r.PrefillTPS = float64(uncached) / (r.TTFTMS / 1000)
		}
	}
	if dt := r.E2EMS - r.TTFTMS; r.CompletionTokens > 1 && dt > 0 {
		r.DecodeTPS = float64(r.CompletionTokens-1) / (dt / 1000)
	}
	r.OK = true
	return answer.String(), nil
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// Cell is one (arm, scenario, size) fold.
type Cell struct {
	Arm          string  `json:"arm"`
	Role         string  `json:"role"`
	Scenario     string  `json:"scenario"`
	Size         int     `json:"size"`
	N            int     `json:"n"`
	Errors       int     `json:"errors"`
	Contended    int     `json:"contended"` // rows dropped: another client shared the upstream
	PromptTokens int     `json:"prompt_tokens_p50"`
	TTFTP50      float64 `json:"ttft_ms_p50"`
	TTFTP90      float64 `json:"ttft_ms_p90"`
	PrefillP50   float64 `json:"prefill_tps_p50"`
	DecodeP50    float64 `json:"decode_tps_p50"`
	CacheRatio   float64 `json:"cache_ratio_mean"`
}

// Loss is one cell where a fak arm trails the best rival by more than lossTolerance.
type Loss struct {
	Scenario string  `json:"scenario"`
	Size     int     `json:"size"`
	Metric   string  `json:"metric"`
	FakArm   string  `json:"fak_arm"`
	FakValue float64 `json:"fak_value"`
	Rival    string  `json:"rival"`
	RivalVal float64 `json:"rival_value"`
	// Gap is how far fak trails, as a fraction: 0.25 = fak is 25% worse.
	Gap float64 `json:"gap"`
}

// Report is the folded, agent-readable view of one run.
type Report struct {
	Schema string `json:"schema"`
	RunID  string `json:"run_id"`
	Host   string `json:"host,omitempty"`
	Model  string `json:"model,omitempty"`
	Rows   int    `json:"rows"`
	Cells  []Cell `json:"cells"`
	Losses []Loss `json:"losses"`
	Wins   []Loss `json:"wins"` // same shape; Gap is how far fak leads
}

func quantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	i := int(math.Ceil(q*float64(len(s)))) - 1
	if i < 0 {
		i = 0
	}
	return s[i]
}

// balancedMedian is the mean of the per-position medians, so an arm that
// happened to run second more often than first is not graded on that luck.
func balancedMedian(byPos map[int][]float64) float64 {
	var sum float64
	n := 0
	for _, xs := range byPos {
		if len(xs) > 0 {
			sum += quantile(xs, 0.5)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// Summarize folds rows (already filtered to one run) into a report.
func Summarize(rows []Row) Report {
	rep := Report{Schema: ReportSchema, Rows: len(rows), Losses: []Loss{}, Wins: []Loss{}}
	type key struct {
		arm, scenario string
		size          int
	}
	type acc struct {
		cell                     Cell
		ttft, prompts            []float64
		ttftP, prefillP, decodeP map[int][]float64 // by Pos
		cache                    float64
	}
	accs := map[key]*acc{}
	var keys []key
	for _, r := range rows {
		if rep.RunID == "" {
			rep.RunID, rep.Host, rep.Model = r.RunID, r.Host, r.Model
		}
		ks := []key{{r.Arm, r.Scenario, r.Size}}
		if r.Scenario == ScenarioMultiTurn && r.Size >= 2 {
			ks = append(ks, key{r.Arm, r.Scenario, WarmTurns})
		}
		for _, k := range ks {
			a := accs[k]
			if a == nil {
				a = &acc{cell: Cell{Arm: r.Arm, Role: r.Role, Scenario: k.scenario, Size: k.size},
					ttftP: map[int][]float64{}, prefillP: map[int][]float64{}, decodeP: map[int][]float64{}}
				accs[k] = a
				keys = append(keys, k)
			}
			switch {
			case !r.OK:
				a.cell.Errors++
			case r.Contended():
				a.cell.Contended++
			default:
				a.cell.N++
				a.ttft = append(a.ttft, r.TTFTMS)
				a.ttftP[r.Pos] = append(a.ttftP[r.Pos], r.TTFTMS)
				a.prompts = append(a.prompts, float64(r.PromptTokens))
				if r.PrefillTPS > 0 {
					a.prefillP[r.Pos] = append(a.prefillP[r.Pos], r.PrefillTPS)
				}
				if r.DecodeTPS > 0 {
					a.decodeP[r.Pos] = append(a.decodeP[r.Pos], r.DecodeTPS)
				}
				a.cache += r.CacheRatio
			}
		}
	}
	sort.SliceStable(keys, func(i, j int) bool {
		if keys[i].scenario != keys[j].scenario {
			return scenarioRank(keys[i].scenario) < scenarioRank(keys[j].scenario)
		}
		if keys[i].size != keys[j].size {
			return keys[i].size < keys[j].size
		}
		return keys[i].arm < keys[j].arm
	})
	for _, k := range keys {
		a := accs[k]
		c := a.cell
		c.TTFTP50, c.TTFTP90 = balancedMedian(a.ttftP), quantile(a.ttft, 0.9)
		c.PrefillP50, c.DecodeP50 = balancedMedian(a.prefillP), balancedMedian(a.decodeP)
		c.PromptTokens = int(quantile(a.prompts, 0.5))
		if c.N > 0 {
			c.CacheRatio = a.cache / float64(c.N)
		}
		rep.Cells = append(rep.Cells, c)
	}
	rep.Losses, rep.Wins = Compare(rep.Cells)
	return rep
}

func scenarioRank(s string) int {
	switch s {
	case ScenarioCold:
		return 0
	case ScenarioDecode:
		return 1
	case ScenarioMultiTurn:
		return 2
	}
	return 3
}

// Compare pits every fak cell against the best baseline cell of the same
// (scenario, size) on the metrics that scenario is built to measure.
func Compare(cells []Cell) (losses, wins []Loss) {
	losses, wins = []Loss{}, []Loss{}
	type sk struct {
		scenario string
		size     int
	}
	groups := map[sk][]Cell{}
	var order []sk
	for _, c := range cells {
		k := sk{c.Scenario, c.Size}
		if _, ok := groups[k]; !ok {
			order = append(order, k)
		}
		groups[k] = append(groups[k], c)
	}
	type metric struct {
		name        string
		get         func(Cell) float64
		lowerBetter bool
	}
	for _, k := range order {
		var ms []metric
		switch k.scenario {
		case ScenarioCold:
			ms = []metric{{"ttft_ms_p50", func(c Cell) float64 { return c.TTFTP50 }, true}, {"prefill_tps_p50", func(c Cell) float64 { return c.PrefillP50 }, false}}
		case ScenarioDecode:
			ms = []metric{{"decode_tps_p50", func(c Cell) float64 { return c.DecodeP50 }, false}}
		case ScenarioMultiTurn:
			if k.size != WarmTurns {
				continue
			}
			ms = []metric{{"ttft_ms_p50", func(c Cell) float64 { return c.TTFTP50 }, true}, {"cache_ratio_mean", func(c Cell) float64 { return c.CacheRatio }, false}}
		}
		for _, m := range ms {
			var best *Cell
			for i := range groups[k] {
				c := groups[k][i]
				// A zero rate means "not observed"; a zero cache ratio is a real reading.
				if c.Role != RoleBaseline || c.N == 0 || (m.get(c) <= 0 && m.name != "cache_ratio_mean") {
					continue
				}
				if best == nil || better(m.get(c), m.get(*best), m.lowerBetter) {
					cc := c
					best = &cc
				}
			}
			if best == nil {
				continue
			}
			for _, c := range groups[k] {
				if c.Role != RoleFak || c.N == 0 {
					continue
				}
				fv, bv := m.get(c), m.get(*best)
				gap := relGap(fv, bv, m.lowerBetter)
				l := Loss{Scenario: k.scenario, Size: k.size, Metric: m.name, FakArm: c.Arm, FakValue: fv, Rival: best.Arm, RivalVal: bv}
				switch {
				case gap > lossTolerance:
					l.Gap = gap
					losses = append(losses, l)
				case gap < -lossTolerance:
					l.Gap = -gap
					wins = append(wins, l)
				}
			}
		}
	}
	sort.SliceStable(losses, func(i, j int) bool { return losses[i].Gap > losses[j].Gap })
	sort.SliceStable(wins, func(i, j int) bool { return wins[i].Gap > wins[j].Gap })
	return losses, wins
}

func better(a, b float64, lowerBetter bool) bool {
	if lowerBetter {
		return a < b
	}
	return a > b
}

// relGap is how much worse fak is than the rival, as a fraction of the better
// value; negative means fak is ahead.
func relGap(fak, rival float64, lowerBetter bool) float64 {
	if lowerBetter {
		if fak <= 0 {
			return 0
		}
		return (fak - rival) / math.Max(fak, rival)
	}
	den := math.Max(fak, rival)
	if den <= 0 {
		return 0
	}
	return (rival - fak) / den
}

// ReadLedger loads every row from a JSONL ledger, skipping malformed lines.
func ReadLedger(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for sc.Scan() {
		var r Row
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.Schema == Schema {
			rows = append(rows, r)
		}
	}
	return rows, sc.Err()
}

// SelectRun returns the rows of runID, or of the most recent run when runID is "".
func SelectRun(rows []Row, runID string) []Row {
	if runID == "" {
		var lastMS int64
		for _, r := range rows {
			if r.UnixMS >= lastMS {
				lastMS, runID = r.UnixMS, r.RunID
			}
		}
	}
	var out []Row
	for _, r := range rows {
		if r.RunID == runID {
			out = append(out, r)
		}
	}
	return out
}

// Render prints the report as a compact table plus the loss list.
func Render(w io.Writer, rep Report) {
	fmt.Fprintf(w, "h2h run %s  host=%s  model=%s  rows=%d\n\n", rep.RunID, rep.Host, rep.Model, rep.Rows)
	fmt.Fprintf(w, "%-9s %6s %-12s %4s %4s %4s %8s %10s %10s %11s %10s %6s\n", "scenario", "size", "arm", "n", "err", "busy", "prompt", "ttft_p50", "ttft_p90", "prefill t/s", "decode t/s", "cache")
	for _, c := range rep.Cells {
		fmt.Fprintf(w, "%-9s %6s %-12s %4d %4d %4d %8d %10.1f %10.1f %11.1f %10.2f %5.0f%%\n",
			c.Scenario, sizeLabel(c.Scenario, c.Size), c.Arm, c.N, c.Errors, c.Contended, c.PromptTokens, c.TTFTP50, c.TTFTP90, c.PrefillP50, c.DecodeP50, 100*c.CacheRatio)
	}
	fmt.Fprintf(w, "\nwhere fak loses (gap > %.0f%%):\n", 100*lossTolerance)
	if len(rep.Losses) == 0 {
		fmt.Fprintln(w, "  none")
	}
	for _, l := range rep.Losses {
		fmt.Fprintf(w, "  %-9s size=%-6s %-16s %s=%.2f vs %s=%.2f  (fak %.0f%% behind)\n", l.Scenario, sizeLabel(l.Scenario, l.Size), l.Metric, l.FakArm, l.FakValue, l.Rival, l.RivalVal, 100*l.Gap)
	}
	if len(rep.Wins) > 0 {
		fmt.Fprintln(w, "where fak leads:")
		for _, l := range rep.Wins {
			fmt.Fprintf(w, "  %-9s size=%-6s %-16s %s=%.2f vs %s=%.2f  (fak %.0f%% ahead)\n", l.Scenario, sizeLabel(l.Scenario, l.Size), l.Metric, l.FakArm, l.FakValue, l.Rival, l.RivalVal, 100*l.Gap)
		}
	}
}

func sizeLabel(scenario string, size int) string {
	if scenario == ScenarioMultiTurn && size == WarmTurns {
		return "warm"
	}
	return strconv.Itoa(size)
}
