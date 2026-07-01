package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// runVCacheContextJoin is the `fak vcache context-join` core (issue #1607): it ingests
// the same provider-cache turn stream `fak vcache observe` reads (--transcript /
// --telemetry) plus a stream of managed-context lifecycle events (--events JSONL —
// resets, compactions, page faults, prefix mutations), and renders the join: for every
// cost-relevant change in the cache telemetry, whether it is explained by a nearby
// lifecycle event (context planning) or has no such explanation (provider cache
// behavior). This is the report the issue's done condition asks for.
func runVCacheContextJoin(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache context-join", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var transcripts multiString
	fs.Var(&transcripts, "transcript", "real Claude Code transcript .jsonl (repeatable)")
	telemetry := fs.String("telemetry", "", "session-telemetry JSONL ('-' for stdin) with session_id + cache counters")
	events := fs.String("events", "", "lifecycle-events JSONL (reset/compaction/page_fault/prefix_mutation), required")
	asJSON := fs.Bool("json", false, "emit the raw JoinReport JSON instead of the human table")
	beforeMillis := fs.Int64("before-millis", 0, "override correlation window: event up to N ms BEFORE a change still explains it (default: 5m TTL)")
	afterMillis := fs.Int64("after-millis", 0, "override correlation window: event up to N ms AFTER a change still explains it (default: 5s)")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}
	if len(transcripts) == 0 && strings.TrimSpace(*telemetry) == "" {
		fmt.Fprintln(stderr, "fak vcache context-join: need at least one --transcript or --telemetry")
		return 2
	}
	if strings.TrimSpace(*events) == "" {
		fmt.Fprintln(stderr, "fak vcache context-join: --events is required (the lifecycle-event stream to join against)")
		return 2
	}

	var turns []vcacheobserve.Turn
	for _, path := range transcripts {
		ts, err := readObserveTranscript(path)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache context-join: transcript %q: %v\n", path, err)
			return 2
		}
		turns = append(turns, ts...)
	}
	if strings.TrimSpace(*telemetry) != "" {
		ts, err := readObserveTelemetry(*telemetry, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak vcache context-join: telemetry %q: %v\n", *telemetry, err)
			return 2
		}
		turns = append(turns, ts...)
	}
	if len(turns) == 0 {
		fmt.Fprintln(stderr, "fak vcache context-join: no cache-bearing assistant turns found in the supplied sources")
		return 1
	}

	evs, err := readLifecycleEvents(*events, os.Stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache context-join: events %q: %v\n", *events, err)
		return 2
	}

	rep := vcacheobserve.JoinContext(vcacheobserve.JoinInput{
		Turns:  turns,
		Events: evs,
		Window: vcacheobserve.CorrelationWindow{BeforeMillis: *beforeMillis, AfterMillis: *afterMillis},
	})
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	renderJoinReport(stdout, rep)
	return 0
}

// lifecycleEventRec is the JSONL row shape for --events: a direct mirror of
// vcacheobserve.LifecycleEvent so a caller can emit one line per real
// resume.Strategy / ctxplan.PageFaultDecision / ctxplan compaction / cachemeta
// TurnDivergence without any translation layer.
type lifecycleEventRec struct {
	Kind       string `json:"kind"`
	Family     string `json:"family"`
	UnixMillis int64  `json:"unix_millis"`
	Outcome    string `json:"outcome"`
	Detail     string `json:"detail"`
}

// readLifecycleEvents reads a lifecycle-events JSONL into LifecycleEvents. An
// unrecognized "kind" is skipped (not fatal) so a future event source can be added
// upstream without breaking older readers; a malformed line is likewise skipped.
func readLifecycleEvents(path string, stdin io.Reader) ([]vcacheobserve.LifecycleEvent, error) {
	r, closeInput, err := openInputOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}
	defer closeInput()
	var out []vcacheobserve.LifecycleEvent
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec lifecycleEventRec
		if json.Unmarshal(line, &rec) != nil {
			continue
		}
		kind := vcacheobserve.LifecycleEventKind(rec.Kind)
		if !vcacheobserve.ValidLifecycleEventKind(kind) {
			continue
		}
		out = append(out, vcacheobserve.LifecycleEvent{
			Kind: kind, Family: rec.Family, UnixMillis: rec.UnixMillis,
			Outcome: rec.Outcome, Detail: rec.Detail,
		})
	}
	return out, sc.Err()
}

// renderJoinReport prints the context-join as a scannable table: the run header, the
// summary headline (the issue's done-condition answer), and one row per attributed
// change.
func renderJoinReport(w io.Writer, r vcacheobserve.JoinReport) {
	fmt.Fprintf(w, "vCache context-join — %d turns, %d lifecycle events, %d cost-relevant changes detected\n",
		r.Turns, r.Events, r.Summary.TotalChanges)
	fmt.Fprintf(w, "attribution: %d/%d context_planning · %d/%d provider_cache_behavior\n\n",
		r.Summary.PlanningAttributed, r.Summary.TotalChanges,
		r.Summary.ProviderAttributed, r.Summary.TotalChanges)

	if len(r.Changes) == 0 {
		fmt.Fprintln(w, "no cost-relevant change detected (steady warm session, or too few turns per family)")
		return
	}

	fmt.Fprintf(w, "%-12s %-20s %-13s %-24s %s\n", "family", "change", "cause", "matched_event", "detail")
	for _, c := range r.Changes {
		matched := "-"
		if c.MatchedEvent != nil {
			matched = fmt.Sprintf("%s(%s)", c.MatchedEvent.Kind, c.MatchedEvent.Outcome)
		}
		fmt.Fprintf(w, "%-12s %-20s %-13s %-24s %s\n",
			truncKey(c.Family, 12), string(c.Change), string(c.Cause), matched, c.Detail)
	}
}
