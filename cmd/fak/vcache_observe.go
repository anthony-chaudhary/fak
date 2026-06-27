package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// runVCacheObserve is the `fak vcache observe` core: it ingests REAL provider-cache
// telemetry — one or more Claude Code transcripts (.jsonl) and/or a session-telemetry
// JSONL — groups the turns by prefix family, runs the shipped vCache decision leaves
// over the real data, and renders a per-sub-concept observability report.
func runVCacheObserve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache observe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var transcripts multiString
	fs.Var(&transcripts, "transcript", "real Claude Code transcript .jsonl (repeatable)")
	telemetry := fs.String("telemetry", "", "session-telemetry JSONL ('-' for stdin) with session_id + cache counters")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read input-token multiplier")
	write5mMult := fs.Float64("write-5m-mult", vcachegov.WriteMult5Minutes, "5m cache-write input-token multiplier")
	write1hMult := fs.Float64("write-1h-mult", vcachegov.WriteMult1Hour, "1h cache-write input-token multiplier")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}
	if len(transcripts) == 0 && strings.TrimSpace(*telemetry) == "" {
		fmt.Fprintln(stderr, "fak vcache observe: need at least one --transcript or --telemetry")
		return 2
	}

	var turns []vcacheobserve.Turn
	for _, path := range transcripts {
		ts, err := readObserveTranscript(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache observe: transcript %q: %v\n", path, err)
			return 2
		}
		turns = append(turns, ts...)
	}
	if strings.TrimSpace(*telemetry) != "" {
		ts, err := readObserveTelemetry(*telemetry, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache observe: telemetry %q: %v\n", *telemetry, err)
			return 2
		}
		turns = append(turns, ts...)
	}
	if len(turns) == 0 {
		fmt.Fprintln(stderr, "fak vcache observe: no cache-bearing assistant turns found in the supplied sources")
		return 1
	}

	rep := vcacheobserve.Observe(turns, vcacheobserve.Multipliers{
		Read: *readMult, Write5m: *write5mMult, Write1h: *write1hMult,
	})
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	renderObserveReport(stdout, rep)
	return 0
}

// multiString is a repeatable string flag (every --transcript adds one path).
type multiString []string

func (m *multiString) String() string { return strings.Join(*m, ",") }
func (m *multiString) Set(v string) error {
	*m = append(*m, v)
	return nil
}

// observeUsageRec is the subset of a Claude transcript record this verb reads: the
// assistant turn's provider-cache usage and the timestamp. Only the fields needed are
// typed, so a future schema addition is forward-compatible.
type observeUsageRec struct {
	Timestamp string `json:"timestamp"`
	Message   *struct {
		Role  string `json:"role"`
		Model string `json:"model"`
		Usage *struct {
			InputTokens         int64 `json:"input_tokens"`
			CacheCreationTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadTokens     int64 `json:"cache_read_input_tokens"`
			CacheCreation       *struct {
				Ephemeral1h int64 `json:"ephemeral_1h_input_tokens"`
				Ephemeral5m int64 `json:"ephemeral_5m_input_tokens"`
			} `json:"cache_creation"`
		} `json:"usage"`
	} `json:"message"`
}

// readObserveTranscript scans a real Claude Code transcript and returns one Turn per
// cache-bearing assistant turn, tagged with the transcript's session id as the prefix
// family. A malformed line is skipped, never fatal.
func readObserveTranscript(path string) ([]vcacheobserve.Turn, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	family := transcriptFamily(path)

	var out []vcacheobserve.Turn
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20) // a single tool-result line can be large
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec observeUsageRec
		if json.Unmarshal(line, &rec) != nil {
			continue
		}
		if rec.Message == nil || rec.Message.Usage == nil || rec.Message.Role != "assistant" {
			continue
		}
		u := rec.Message.Usage
		if u.CacheReadTokens == 0 && u.CacheCreationTokens == 0 {
			continue // not a cache-bearing turn
		}
		t := vcacheobserve.Turn{
			Family:        family,
			UnixMillis:    parseTranscriptUnix(rec.Timestamp) * 1000,
			InputTokens:   u.InputTokens,
			CacheCreation: u.CacheCreationTokens,
			CacheRead:     u.CacheReadTokens,
		}
		if u.CacheCreation != nil {
			t.Ephemeral1h = u.CacheCreation.Ephemeral1h
			t.Ephemeral5m = u.CacheCreation.Ephemeral5m
		}
		out = append(out, t)
	}
	return out, sc.Err()
}

// transcriptFamily derives a stable prefix-family key from a transcript path: the
// session id is the base filename without its .jsonl suffix.
func transcriptFamily(path string) string {
	base := path
	if i := strings.LastIndexAny(base, "/\\"); i >= 0 {
		base = base[i+1:]
	}
	return strings.TrimSuffix(base, ".jsonl")
}

// observeTelemetryRec is one row of the session-telemetry JSONL (the schema
// tools emit from real transcripts): top-level cache counters plus the session id.
type observeTelemetryRec struct {
	SessionID                string `json:"session_id"`
	CapturedUTC              string `json:"captured_utc"`
	InputTokens              int64  `json:"input_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	Ephemeral1hInputTokens   int64  `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens   int64  `json:"ephemeral_5m_input_tokens"`
}

// readObserveTelemetry reads a session-telemetry JSONL into Turns. Rows are ordered as
// read; the leaf re-sorts each family by timestamp.
func readObserveTelemetry(path string, stdin io.Reader) ([]vcacheobserve.Turn, error) {
	var r io.Reader
	if path == "-" {
		r = stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		r = f
	}
	var out []vcacheobserve.Turn
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	seq := int64(0)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec observeTelemetryRec
		if json.Unmarshal(line, &rec) != nil {
			continue
		}
		if rec.CacheReadInputTokens == 0 && rec.CacheCreationInputTokens == 0 {
			continue
		}
		family := rec.SessionID
		if family == "" {
			family = "telemetry"
		}
		millis := parseTranscriptUnix(rec.CapturedUTC) * 1000
		if millis == 0 {
			// No usable timestamp: preserve file order with a synthetic monotonic clock
			// so the warmth-belief sequencing stays well-defined.
			millis = seq
		}
		seq++
		out = append(out, vcacheobserve.Turn{
			Family:        family,
			UnixMillis:    millis,
			InputTokens:   rec.InputTokens,
			CacheCreation: rec.CacheCreationInputTokens,
			CacheRead:     rec.CacheReadInputTokens,
			Ephemeral1h:   rec.Ephemeral1hInputTokens,
			Ephemeral5m:   rec.Ephemeral5mInputTokens,
		})
	}
	return out, sc.Err()
}

// renderObserveReport prints the per-sub-concept observability as an aligned, scannable
// report: the run header, the per-family economics table, and one panel per sub-concept.
func renderObserveReport(w io.Writer, r vcacheobserve.Report) {
	fmt.Fprintf(w, "vCache sub-concept observability — %d turns across %d prefix families\n",
		r.Turns, r.FamilyCount)
	fmt.Fprintf(w, "aggregate: hit %.1f%%  saved %.1f%% (%.0f token-equiv)  multiplier %.2fx  mean prefix %.0f tok\n",
		100*r.HitRate, r.Aggregate.SavedPct, r.Aggregate.SavedTokenEquiv, r.Multiplier, r.MeanPrefixTokens)
	fmt.Fprintf(w, "grade: MEASURED %s (%d/100)  vs  SYNTHETIC %s (%d/100)   concentration s=%.2f (defeated=%v)\n\n",
		r.GradeMeasured, r.ScoreMeasured, r.GradeSynthetic, r.ScoreSynthetic,
		r.Concentration.ZipfS, r.Concentration.Defeated)

	fmt.Fprintf(w, "%-12s %6s %9s %12s %7s %7s %-14s\n",
		"family", "turns", "hit%", "saved_teq", "saved%", "1stpos", "governor")
	fams := append([]vcacheobserve.Family(nil), r.Families...)
	sort.SliceStable(fams, func(i, j int) bool { return fams[i].CacheReadTokens > fams[j].CacheReadTokens })
	for _, f := range fams {
		fmt.Fprintf(w, "%-12s %6d %8.1f%% %12.0f %6.1f%% %7d %-14s\n",
			truncKey(f.Key, 12), f.Turns, 100*f.HitRate, f.Economics.SavedTokenEquiv,
			f.Economics.SavedPct, f.Economics.FirstPositiveRequest, string(f.GovernorDecision))
	}

	fmt.Fprintln(w, "\nsub-concept panels (OBSERVED = provider's counters · DECISION = fak verdict):")
	for _, p := range r.Panels {
		fmt.Fprintf(w, "\n[%s] %s  (%s · %s)\n", p.Verdict, p.Name, p.Milestone, p.Provenance)
		fmt.Fprintf(w, "  Q: %s\n", p.Question)
		fmt.Fprintf(w, "  %s\n", p.Value)
		if p.Detail != "" {
			fmt.Fprintf(w, "  · %s\n", p.Detail)
		}
		fmt.Fprintf(w, "  witness: %s\n", p.Witness)
	}
	fmt.Fprintln(w, "\ncorrectness depends on cache hit: false  (a hit is a realized rebate, never a trust claim)")
}

func truncKey(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
