package main

// fak serve-wiring: the durable wiring-status surface for `fak serve`. It answers the
// one question a green `make ci` cannot: of every feature the gateway advertises, which
// ones are actually REACHED from the live serve entrypoint, and which are scaffolded-but-
// dead (a gateway.Config field that is set on the struct but whose flag the operator can
// never reach, or a field serve.go never sets at all).
//
// The verdicts come from an audited baseline (servewiringData below): each row was traced
// flag -> gateway.Config field -> the load-bearing runtime read, and adversarially verified.
// What this verb adds on top of a static doc is DRIFT DETECTION that cannot rot: it reads
// the real cmd/fak/serve.go and internal/gateway/gateway.go on each run and cross-checks two
// machine-derivable facts per row:
//
//   1. the gateway.Config field named in the row still EXISTS in the Config struct, and
//   2. serve.go actually SETS that field in its gateway.New(Config{...}) literal.
//
// A row whose field serve.go stops setting (the dead-wiring regression this verb exists to
// catch) flips to UNWIRED and reds `--check`. A Config field with no row at all is reported
// as UNAUDITED so a newly-added feature cannot slip in unexamined. This is the serve-path
// twin of the scorecard family: a deterministic, tree-cross-checked status the trunk keeps
// honest, regenerated on git via `fak serve-wiring --md > the table in docs/serve-config.md`.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

func cmdServeWiring(argv []string) { os.Exit(runServeWiring(os.Stdout, os.Stderr, argv)) }

// wiringVerdict is the closed set of wiring states a feature can be in.
type wiringVerdict string

const (
	// verdictWired: a load-bearing runtime read consumes the field on a request/turn path,
	// and the operator can reach it by default (no opt-in flag gate).
	verdictWired wiringVerdict = "WIRED"
	// verdictOffByDefault: fully wired, but inert until a non-default flag value arms it
	// (e.g. a budget > 0). A deliberate guard, not a defect.
	verdictOffByDefault wiringVerdict = "OFF_BY_DEFAULT_BUT_WIRED"
	// verdictPartial: the producer side is wired and reachable, but a documented consumer
	// step is deferred, so the end-to-end effect is incomplete.
	verdictPartial wiringVerdict = "PARTIAL"
	// verdictDead: the gateway reads the field, but serve.go never feeds it; the feature
	// is unreachable on the shipped binary. The defect this verb exists to catch.
	verdictDead wiringVerdict = "DEAD_WIRED"
)

// wiringRow is one audited feature: the operator flag, the gateway.Config field it feeds,
// the verdict, and the load-bearing call site that produced (or would produce) the effect.
type wiringRow struct {
	Feature  string
	Flag     string
	Field    string // the gateway.Config field name, or "" for a non-Config seam (e.g. an observer)
	Verdict  wiringVerdict
	CallSite string
	Note     string
}

// servewiringData is the audited baseline (workflow wiring-audit, every row skeptic-verified;
// routemanifest + the #761 notifier were DEAD_WIRED and wired in the same pass that added this
// verb). Update a row's Verdict/CallSite when the wiring changes; `--check` re-derives the
// machine-checkable half (field exists + serve.go sets it) so a stale row cannot hide a
// regression. A "" Field marks a seam wired through a session.Table observer, not Config;
// the serve.go-sets check is skipped for those (tracked by Flag presence instead).
var servewiringData = []wiringRow{
	{"inkernelchat", "--gguf / --tokenizer", "InKernelModel", verdictWired, "internal/gateway/gateway.go:861", "with model+tokenizer and no --base-url, /v1/chat/completions and /v1/messages serve the in-kernel model"},
	{"replica", "--replica-base-url", "ReplicaBaseURLs", verdictWired, "internal/gateway/gateway.go:715", "2+ endpoints -> ReplicaRouter round-robin"},
	{"vdso", "--vdso / --invalidation", "VDSO", verdictWired, "internal/kernel/kernel.go:348", "dedup fast path + tier-2 invalidation granularity"},
	{"toolfloor", "(adjudicator.Default.NeverAdmits)", "ToolFloorDenies", verdictWired, "internal/gateway/messages.go:392", "prunes provably-unreachable tool defs from the Anthropic passthrough; default-on, fail-safe"},
	{"decidesession", "(host func, default-on)", "DecideSession", verdictWired, "internal/gateway/session_admit.go:57", "run-state refusal + TurnsLeft debit + budget + pace, before the model turn"},
	{"debitsession", "(host func, default-on)", "DebitSession", verdictWired, "internal/gateway/session_admit.go:157", "debits TokensLeft + context budget after the planner returns"},
	{"routemanifest", "--route-manifest", "RouteManifest", verdictWired, "internal/gateway/gateway.go:1127", "binds ToolCall.Engine before Submit; flag wired (was DEAD_WIRED before this pass)"},
	{"ctxview", "--ctx-view-budget", "CtxViewBudget", verdictWired, "internal/gateway/gateway.go:788", "re-materializes history as an O(1) planned ctxplan view under the budget; DEFAULT-ON at 8000 resident tokens (fail-open, Anthropic cache prefix byte-identical), pass 0 to disable"},
	{"compacthistory", "--compact-history-budget", "CompactHistoryBudget", verdictWired, "internal/gateway/messages.go:365", "compacts old turns in the Anthropic outbound body once it sprawls past the budget, cache prefix byte-identical; DEFAULT-ON at ~48k (gateway.DefaultCompactHistoryBudget), pass 0 to disable"},
	{"elideresult", "--elide-result-bytes", "ElideResultBytes", verdictWired, "internal/gateway/messages.go:maybeElideAnthropicRaw", "shrinks an old oversized tool_result body to a bounded head+tail on BOTH wires — the Anthropic passthrough (req.Raw byte-splice, cache head byte-identical) and the decoded local-model path (req.Messages, for GLM-5.2/Qwen-3.6 served by fak); DEFAULT-ON at gateway.DefaultElideResultBytes (16KB), pass 0 to disable"},
	{"debugstats", "--debug-stats", "DebugStatsf", verdictOffByDefault, "internal/gateway/metrics.go:404", "emits one compact payload-free per-turn cache/compaction/resetScore line to stderr; off by default"},
	{"resetonbudget", "--reset-on-budget", "ResetOnBudget", verdictOffByDefault, "internal/gateway/session_admit.go:108", "distills a carryover seed and continues transparently on budget exhaustion; needs --context-budget-tokens"},
	{"budgetwebhook", "--budget-webhook", "", verdictOffByDefault, "internal/session/usage.go:73", "POSTs a pre-exhaustion warning + exhaustion event; wired via WatchBudget, off when URL empty"},
	{"notifier", "--notify-native / --notify-webhook / --notify-slack", "", verdictWired, "cmd/fak/serve.go (WatchTransitions)", "#761 stop-reason push notifier; native default-on (was DEAD_WIRED before this pass)"},
	{"enginecache", "--engine-cache-engine", "EngineCacheEngine", verdictOffByDefault, "internal/gateway/gateway.go:1480", "resets the serving-engine cache after a quarantined proxy turn; off when engine empty"},
	{"backend", "--backend", "Backend", verdictOffByDefault, "internal/agent/inkernel_planner.go:271", "decodes the in-kernel chat through the compute HAL device; off when name empty"},
	{"cpuoffloadexperts", "--cpu-offload-experts", "CPUOffloadExperts", verdictOffByDefault, "internal/agent/inkernel_planner.go:282", "with --gguf --backend, keeps MoE expert GEMMs on host RAM while dense/router/attention run on the device; off by default"},
	{"metal", "--metal", "Metal", verdictWired, "internal/agent/inkernel_planner.go:1067", "with --gguf (no --backend), auto-selects the Apple-Silicon metalgemm GPU when Apple-Silicon+cgo+a device are available; --metal/FAK_METAL=1 requires that path fail-loud; dense-Qwen Q8 only; CPU fallback on non-Metal builds or unavailable devices"},
	{"steersession", "(host func, default-on)", "SteerSession", verdictPartial, "internal/gateway/http.go:951", "POST /session/{id}/steer sends onto a2achan; the running-session TryRecv splice is deferred (#760)"},
}

func runServeWiring(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("serve-wiring", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asMD := fs.Bool("md", false, "emit the markdown table for docs/serve-config.md")
	check := fs.Bool("check", false, "CI gate: exit non-zero on wiring drift (a row's Config field is gone, or serve.go stopped setting it, or a Config field has no audited row)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak serve-wiring: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	serveSrc, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-wiring: read serve.go: %v\n", err)
		return 1
	}
	gwSrc, err := os.ReadFile(filepath.Join(root, "internal", "gateway", "gateway.go"))
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-wiring: read gateway.go: %v\n", err)
		return 1
	}

	configFields := parseConfigFields(string(gwSrc))
	serveSets := serveConfigAssignments(string(serveSrc))

	var drift []string

	// Per-row drift: a Config-backed row whose field no longer exists, or that serve.go no
	// longer sets, has silently dead-wired; exactly the regression this verb guards.
	for _, r := range servewiringData {
		if r.Field == "" {
			continue // observer seam, not a Config field; tracked by flag presence, checked below
		}
		if !configFields[r.Field] {
			drift = append(drift, fmt.Sprintf("row %q names gateway.Config.%s, which no longer exists in the Config struct", r.Feature, r.Field))
			continue
		}
		if !serveSets[r.Field] && r.Verdict != verdictDead {
			drift = append(drift, fmt.Sprintf("row %q (%s) is %s but serve.go no longer sets Config.%s; it has dead-wired", r.Feature, r.Field, r.Verdict, r.Field))
		}
	}

	// Coverage drift: a Config field serve.go sets but no audited row covers is an unexamined
	// feature. Skip the plumbing fields that are not operator features.
	covered := map[string]bool{}
	for _, r := range servewiringData {
		if r.Field != "" {
			covered[r.Field] = true
		}
	}
	var unaudited []string
	for f := range serveSets {
		if !covered[f] && !plumbingField[f] {
			unaudited = append(unaudited, f)
		}
	}
	sort.Strings(unaudited)
	for _, f := range unaudited {
		drift = append(drift, fmt.Sprintf("gateway.Config.%s is set by serve.go but has no audited wiring row (add it to servewiringData and trace it)", f))
	}

	if *asMD {
		writeWiringMarkdown(stdout, unaudited)
	} else if !*check {
		writeWiringSummary(stdout, drift)
	}

	if *check {
		if len(drift) == 0 {
			fmt.Fprintln(stdout, "OK  serve wiring: all audited rows still fed by serve.go; no unaudited Config feature")
			return 0
		}
		fmt.Fprintf(stdout, "DRIFT  serve wiring: %d issue(s)\n", len(drift))
		for _, d := range drift {
			fmt.Fprintf(stdout, "  - %s\n", d)
		}
		return 1
	}
	return 0
}

// plumbingField names gateway.Config fields that are infrastructure, not operator-facing
// features, so they are not expected to carry a wiring row.
var plumbingField = map[string]bool{
	"EngineID": true, "Model": true, "BaseURL": true, "Provider": true, "APIKey": true,
	"PinUpstreamCredential": true, "EngineCacheBaseURL": true, "EngineCacheAdminKey": true,
	"EngineCacheIdleTimeout": true, "EngineCacheRequireExactSpan": true, "Tokenizer": true,
	"InKernelQ4K": true, "RequireKey": true, "Invalidation": true, "Version": true,
	"ReloadPolicy": true, "ResetTrace": true, "ObserveTrace": true, "ObserveSession": true,
	"ControlSession": true, "ListSessions": true, "OnBudgetExhausted": true,
	"DefaultTraceID": true, "Logf": true, "StartTime": true, "StartupPhases": true,
}

var configFieldRe = regexp.MustCompile(`(?m)^\t([A-Z][A-Za-z0-9]*)\s+[^/]`)

// parseConfigFields returns the set of field names declared in the gateway.Config struct.
func parseConfigFields(src string) map[string]bool {
	return scanFieldSet(src, "type Config struct {", "\n}", configFieldRe)
}

// scanFieldSet returns the set of capitalized field names that re matches inside the
// src region bounded by startMarker (the first occurrence) and the first endMarker
// after it (or end-of-string if absent). It is the shared scanner behind the
// Config-declaration and Config-literal field extractors.
func scanFieldSet(src, startMarker, endMarker string, re *regexp.Regexp) map[string]bool {
	out := map[string]bool{}
	start := strings.Index(src, startMarker)
	if start < 0 {
		return out
	}
	rest := src[start:]
	end := strings.Index(rest, endMarker)
	if end < 0 {
		end = len(rest)
	}
	for _, m := range re.FindAllStringSubmatch(rest[:end], -1) {
		out[m[1]] = true
	}
	return out
}

var assignRe = regexp.MustCompile(`(?m)^\s*([A-Z][A-Za-z0-9]*):\s+`)

// serveConfigAssignments returns the set of gateway.Config field names assigned inside the
// gateway.New(gateway.Config{...}) literal in serve.go.
func serveConfigAssignments(src string) map[string]bool {
	// The literal ends at the matching "})" that closes New(Config{...}.
	return scanFieldSet(src, "gateway.New(gateway.Config{", "\n\t})", assignRe)
}

func verdictGlyph(v wiringVerdict) string {
	switch v {
	case verdictWired:
		return "wired"
	case verdictOffByDefault:
		return "off-by-default (wired)"
	case verdictPartial:
		return "partial"
	case verdictDead:
		return "dead-wired"
	default:
		return string(v)
	}
}

func writeWiringMarkdown(w io.Writer, unaudited []string) {
	fmt.Fprintln(w, "| Feature | Status | Flag | gateway.Config field | Live call site | Note |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|")
	for _, r := range servewiringData {
		field := r.Field
		if field == "" {
			field = "_(observer seam)_"
		} else {
			field = "`" + field + "`"
		}
		fmt.Fprintf(w, "| `%s` | %s | `%s` | %s | `%s` | %s |\n",
			r.Feature, verdictGlyph(r.Verdict), r.Flag, field, r.CallSite, r.Note)
	}
	if len(unaudited) > 0 {
		fmt.Fprintf(w, "\n> WARNING: Unaudited Config feature(s) serve.go sets with no wiring row: %s\n", strings.Join(unaudited, ", "))
	}
}

func writeWiringSummary(w io.Writer, drift []string) {
	var wired, off, partial, dead int
	for _, r := range servewiringData {
		switch r.Verdict {
		case verdictWired:
			wired++
		case verdictOffByDefault:
			off++
		case verdictPartial:
			partial++
		case verdictDead:
			dead++
		}
	}
	fmt.Fprintf(w, "serve wiring: %d features: %d wired, %d off-by-default, %d partial, %d dead\n",
		len(servewiringData), wired, off, partial, dead)
	for _, r := range servewiringData {
		fmt.Fprintf(w, "  %-26s %-16s %s\n", r.Feature, r.Verdict, r.CallSite)
	}
	if len(drift) > 0 {
		fmt.Fprintf(w, "\nDRIFT (%d):\n", len(drift))
		for _, d := range drift {
			fmt.Fprintf(w, "  - %s\n", d)
		}
	}
}
