package main

// dispatch_route_health.go — `fak dispatch route-health`, the selected-route smoke gate
// (#3035) that answers "may this provider route spend a live worker slot?" BEFORE an
// issue-worker spawn. The trigger was a DeepSeek/NIM probe that timed out after 20s while
// worker selection still looked possible; the contract is broader than "slow NIM": ANY
// OpenAI-compatible route can fail typed — timeout, auth (401/403), model unavailable
// (404), quota/rate-limited (429 + Retry-After / rate-limit reset), provider 5xx, or an
// unsupported/retired route — and each failure must suppress ONLY that route/model/account
// for a cooldown, never the whole provider family (a healthy sibling like Kimi on the same
// provider keeps dispatching).
//
//	# the cheap smoke: one tiny chat completion against the SELECTED route
//	fak dispatch route-health probe --base-url URL --model M [--provider P] [--account A]
//	# the operator card: last probe age, failure class, cooldown/reset, exact recheck command
//	fak dispatch route-health status [--json]
//	# the pre-spawn gate: exit 0 = route may spend a slot, exit 3 = suppressed (cooldown live)
//	fak dispatch route-health gate --provider P --model M [--account A]
//
// probe appends one fak-route-health/1 row per attempt to .dispatch-runs/route-health.jsonl
// (the same runs-dir family as the skip/timeout ledgers); status and gate are PURE folds
// over that ledger (same rows + clock in, same verdict out), so a dispatcher calls `gate`
// before spawning and a test re-derives every verdict hermetically. The classification is
// a pure function of (HTTP status, headers, body, transport error) and honors provider
// hints — Retry-After (delta-seconds or HTTP-date) and X-RateLimit-Reset (epoch, delta, or
// Go-style duration) — before falling back to per-class default cooldowns. Folding this
// ledger into the tools/dispatch_status.py --fast card is the named follow-on; the fold
// source and every field it needs (probe age, class, cooldown, recheck) live here.
//
// Live validation (#3429, 2026-07-19): probed the real NIM route
// (https://integrate.api.nvidia.com/v1, provider nim, --api-key-env NVIDIA_API_KEY) and
// captured THREE typed classes from one live seat, not just the healthy path —
// meta/llama-3.1-8b-instruct returned class=healthy HTTP 200 exit 0, meta/llama-3.3-70b-instruct
// returned class=timeout exit 0 (the exact 20s trigger failure from #3035, reproduced live and
// then suppressed for a cooldown), and deepseek-ai/deepseek-r1 returned class=model_unavailable
// HTTP 404. status folded all four fak-route-health/1 rows (4 probed, 3 suppressed), gate
// returned pass exit 0 for the healthy route and suppressed exit 3 for the timeout route, and the
// dispatch_status.py --fast "routes:" line rendered "4 probed, 3 suppressed [...] (#3035)" from
// that ledger. This retires the trigger gap — route health assumed rather than checked — with a
// real multi-class witness, but a healthy-at-probe-time row still cannot promise
// healthy-at-spawn-time. Invalidating correction: an earlier #3429 header (commit 31c6538bd4)
// claimed a healthy live probe of model deepseek-ai/deepseek-v4-pro, but that model id returns
// HTTP 404 on this route and no ledger row was ever captured — an unwitnessed claim, replaced
// here by the transcript above.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	routeHealthSchema         = "fak-route-health/1"
	routeHealthStatusSchema   = "fak-route-health-status/1"
	routeHealthGateSchema     = "fak-route-health-gate/1"
	routeHealthLogName        = "route-health.jsonl"
	routeHealthDefaultPrompt  = "Return exactly OK"
	routeHealthDefaultTimeout = 20 * time.Second
)

// routeHealthClass is the closed vocabulary a probe outcome lands in. Every class is a
// typed routing decision, not a log line: rate_limited and provider_5xx are transient
// (short cooldown, provider hint honored), auth and unsupported_route need a human (long
// cooldown), healthy clears the route immediately.
type routeHealthClass string

const (
	routeClassHealthy          routeHealthClass = "healthy"
	routeClassTimeout          routeHealthClass = "timeout"
	routeClassUnreachable      routeHealthClass = "unreachable"
	routeClassAuth             routeHealthClass = "auth"
	routeClassRateLimited      routeHealthClass = "rate_limited"
	routeClassModelUnavailable routeHealthClass = "model_unavailable"
	routeClassProvider5xx      routeHealthClass = "provider_5xx"
	routeClassUnsupported      routeHealthClass = "unsupported_route"
)

// Default cooldowns per class, used only when the provider sent no reset hint. Transient
// classes stay short so a healthy route is re-admitted quickly; auth and retired-model
// classes stay long because they never heal without a human.
const (
	routeCooldownTimeout     = 10 * time.Minute
	routeCooldownUnreachable = 10 * time.Minute
	routeCooldownAuth        = time.Hour
	routeCooldownRateLimited = 15 * time.Minute
	routeCooldownModelGone   = time.Hour
	routeCooldown5xx         = 5 * time.Minute
	routeCooldownUnsupported = 6 * time.Hour
)

// routeHealthRecord is one fak-route-health/1 ledger row: the typed outcome of one probe
// of one route (provider/account/model), with the cooldown window and the exact recheck
// command an operator (or the gate) needs.
type routeHealthRecord struct {
	Schema            string `json:"schema"`
	Route             string `json:"route"`
	Provider          string `json:"provider"`
	Account           string `json:"account,omitempty"`
	Model             string `json:"model"`
	BaseURL           string `json:"base_url,omitempty"`
	Class             string `json:"class"`
	Status            int    `json:"status,omitempty"`
	Detail            string `json:"detail,omitempty"`
	ProbedAtUnix      int64  `json:"probed_at_unix"`
	CooldownUntilUnix int64  `json:"cooldown_until_unix,omitempty"`
	ProviderHint      bool   `json:"provider_hint,omitempty"`
	Recheck           string `json:"recheck"`
}

// routeProbeOutcome is the pure classification of one probe response.
type routeProbeOutcome struct {
	Class        routeHealthClass
	Detail       string
	CooldownSecs int64
	ProviderHint bool
}

// routeHealthKey is the suppression identity: provider/account/model. Suppression at this
// granularity is what keeps a 429 on deepseek-chat from blacklisting kimi on the same
// provider — sibling routes have different keys by construction.
func routeHealthKey(provider, account, model string) string {
	if strings.TrimSpace(account) == "" {
		account = "-"
	}
	return provider + "/" + account + "/" + model
}

// classifyRouteProbe maps one probe result onto the typed vocabulary. Pure: same
// (status, headers, body, err, clock) in, same outcome out. A transport timeout is its own
// class (the trigger failure), not conflated with provider_5xx; 429 and 503 honor the
// provider's reset hints before falling back to defaults.
func classifyRouteProbe(status int, header http.Header, body string, err error, nowUnix int64) routeProbeOutcome {
	if err != nil {
		var nerr net.Error
		if errors.Is(err, context.DeadlineExceeded) || (errors.As(err, &nerr) && nerr.Timeout()) {
			return routeProbeOutcome{
				Class:        routeClassTimeout,
				Detail:       "probe timed out: " + err.Error(),
				CooldownSecs: int64(routeCooldownTimeout / time.Second),
			}
		}
		return routeProbeOutcome{
			Class:        routeClassUnreachable,
			Detail:       "probe transport error: " + err.Error(),
			CooldownSecs: int64(routeCooldownUnreachable / time.Second),
		}
	}
	switch {
	case status >= 200 && status < 300:
		return routeProbeOutcome{Class: routeClassHealthy}
	case status == 401 || status == 403:
		return routeProbeOutcome{
			Class:        routeClassAuth,
			Detail:       fmt.Sprintf("HTTP %d: credential rejected", status),
			CooldownSecs: int64(routeCooldownAuth / time.Second),
		}
	case status == 404:
		return routeProbeOutcome{
			Class:        routeClassModelUnavailable,
			Detail:       "HTTP 404: model or endpoint not found on this route",
			CooldownSecs: int64(routeCooldownModelGone / time.Second),
		}
	case status == 410:
		return routeProbeOutcome{
			Class:        routeClassUnsupported,
			Detail:       "HTTP 410: route retired",
			CooldownSecs: int64(routeCooldownUnsupported / time.Second),
		}
	case status == 429:
		cooldown, hint := routeRetryAfterSecs(header, nowUnix)
		if !hint {
			cooldown = int64(routeCooldownRateLimited / time.Second)
		}
		return routeProbeOutcome{
			Class:        routeClassRateLimited,
			Detail:       "HTTP 429: quota or rate limit",
			CooldownSecs: cooldown,
			ProviderHint: hint,
		}
	case status >= 500:
		cooldown, hint := routeRetryAfterSecs(header, nowUnix)
		if !hint {
			cooldown = int64(routeCooldown5xx / time.Second)
		}
		return routeProbeOutcome{
			Class:        routeClassProvider5xx,
			Detail:       fmt.Sprintf("HTTP %d: provider-side failure", status),
			CooldownSecs: cooldown,
			ProviderHint: hint,
		}
	default:
		// The probe request is fixed and well-formed, so a remaining 4xx is the route
		// refusing it. A body naming a missing model is model_unavailable; retirement /
		// unsupported wording (or anything else) is unsupported_route.
		if routeBodyNamesMissingModel(body) {
			return routeProbeOutcome{
				Class:        routeClassModelUnavailable,
				Detail:       fmt.Sprintf("HTTP %d: model not found on this route", status),
				CooldownSecs: int64(routeCooldownModelGone / time.Second),
			}
		}
		return routeProbeOutcome{
			Class:        routeClassUnsupported,
			Detail:       fmt.Sprintf("HTTP %d: route rejected the probe request", status),
			CooldownSecs: int64(routeCooldownUnsupported / time.Second),
		}
	}
}

// routeBodyNamesMissingModel reports whether an error body says the MODEL is missing
// (vs. retired/unsupported). Checked only on the residual-4xx path.
func routeBodyNamesMissingModel(body string) bool {
	b := strings.ToLower(body)
	if !strings.Contains(b, "model") {
		return false
	}
	for _, hint := range []string{"not found", "does not exist", "unknown model", "no such model"} {
		if strings.Contains(b, hint) {
			return true
		}
	}
	return false
}

// routeRetryAfterSecs extracts the provider's own reset hint: Retry-After as delta-seconds
// or HTTP-date, then the X-RateLimit-Reset family as epoch seconds, delta seconds, or a
// Go-style duration ("6m12s", OpenAI's reset header shape). Returns (seconds-from-now, true)
// when a hint was present and parseable.
func routeRetryAfterSecs(h http.Header, nowUnix int64) (int64, bool) {
	if v := strings.TrimSpace(h.Get("Retry-After")); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			return n, true
		}
		if t, err := http.ParseTime(v); err == nil {
			if d := t.Unix() - nowUnix; d > 0 {
				return d, true
			}
			return 0, true
		}
	}
	for _, key := range []string{"X-RateLimit-Reset", "X-RateLimit-Reset-Requests", "RateLimit-Reset"} {
		v := strings.TrimSpace(h.Get(key))
		if v == "" {
			continue
		}
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n >= 0 {
			if n > 1_000_000_000 { // epoch seconds, not a delta
				if d := n - nowUnix; d > 0 {
					return d, true
				}
				return 0, true
			}
			return n, true
		}
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			return int64(d / time.Second), true
		}
	}
	return 0, false
}

// routeProbeSpec is one probe request: which route to smoke and how.
type routeProbeSpec struct {
	Provider  string
	Account   string
	Model     string
	BaseURL   string
	APIKeyEnv string
	Prompt    string
	Timeout   time.Duration
}

// routeHealthRecheckCommand is the exact command that re-probes this route — surfaced on
// every record so a suppressed route always names its own way back in.
func routeHealthRecheckCommand(spec routeProbeSpec) string {
	parts := []string{
		"fak dispatch route-health probe",
		"--base-url " + spec.BaseURL,
		"--model " + spec.Model,
		"--provider " + spec.Provider,
	}
	if strings.TrimSpace(spec.Account) != "" {
		parts = append(parts, "--account "+spec.Account)
	}
	if strings.TrimSpace(spec.APIKeyEnv) != "" {
		parts = append(parts, "--api-key-env "+spec.APIKeyEnv)
	}
	return strings.Join(parts, " ")
}

// probeProviderRoute runs the cheap smoke — one tiny chat completion against the selected
// route — and classifies the outcome. The only impurity is the HTTP call and the clock;
// classification is delegated to the pure classifyRouteProbe.
func probeProviderRoute(spec routeProbeSpec) (routeHealthRecord, error) {
	endpoint, err := llmdEndpoint(spec.BaseURL, "/chat/completions")
	if err != nil {
		return routeHealthRecord{}, err
	}
	apiKey, err := llmdResolveAPIKey(spec.APIKeyEnv)
	if err != nil {
		return routeHealthRecord{}, err
	}
	if spec.Timeout <= 0 {
		spec.Timeout = routeHealthDefaultTimeout
	}
	if strings.TrimSpace(spec.Prompt) == "" {
		spec.Prompt = routeHealthDefaultPrompt
	}

	payload := map[string]any{
		"model":       spec.Model,
		"messages":    []map[string]string{{"role": "user", "content": spec.Prompt}},
		"max_tokens":  8,
		"temperature": 0,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return routeHealthRecord{}, err
	}

	now := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), spec.Timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return routeHealthRecord{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	llmdApplyAuth(req, apiKey)

	client := &http.Client{Timeout: spec.Timeout}
	var status int
	var header http.Header
	var body string
	resp, doErr := client.Do(req)
	if doErr == nil {
		status = resp.StatusCode
		header = resp.Header
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		body = string(b)
	}
	out := classifyRouteProbe(status, header, body, doErr, now.Unix())

	rec := routeHealthRecord{
		Schema:       routeHealthSchema,
		Route:        routeHealthKey(spec.Provider, spec.Account, spec.Model),
		Provider:     spec.Provider,
		Account:      spec.Account,
		Model:        spec.Model,
		BaseURL:      spec.BaseURL,
		Class:        string(out.Class),
		Status:       status,
		Detail:       out.Detail,
		ProbedAtUnix: now.Unix(),
		ProviderHint: out.ProviderHint,
		Recheck:      routeHealthRecheckCommand(spec),
	}
	if out.CooldownSecs > 0 {
		rec.CooldownUntilUnix = now.Unix() + out.CooldownSecs
	}
	return rec, nil
}

// routeHealthLedgerPath: the route-health rows live beside the skip/timeout ledgers under
// the workspace's .dispatch-runs.
func routeHealthLedgerPath(workspace string) string {
	return filepath.Join(workspace, timeoutLedgerRunsDir, routeHealthLogName)
}

// appendRouteHealthRecord persists one JSONL row, append-only, creating the runs dir if
// needed — the same persistence shape as skip-ledger and timeout-ledger.
func appendRouteHealthRecord(path string, rec routeHealthRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(rec)
}

// loadRouteHealthLatest folds the ledger to the LATEST row per route key. A missing ledger
// is an empty fold, not an error — an unprobed fleet must never stall dispatch (fail open,
// matching the known-bad hold).
func loadRouteHealthLatest(path string) ([]routeHealthRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	latest := map[string]routeHealthRecord{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec routeHealthRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil || rec.Route == "" {
			continue // one corrupt row must not poison the fold
		}
		if prev, ok := latest[rec.Route]; !ok || rec.ProbedAtUnix >= prev.ProbedAtUnix {
			latest[rec.Route] = rec
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	routes := make([]string, 0, len(latest))
	for route := range latest {
		routes = append(routes, route)
	}
	sort.Strings(routes)
	out := make([]routeHealthRecord, 0, len(routes))
	for _, route := range routes {
		out = append(out, latest[route])
	}
	return out, nil
}

// routeHealthGateResult is the fak-route-health-gate/1 verdict for one route.
type routeHealthGateResult struct {
	Schema                string `json:"schema"`
	Route                 string `json:"route"`
	Verdict               string `json:"verdict"` // "pass" or "suppressed"
	Reason                string `json:"reason"`
	Class                 string `json:"class,omitempty"`
	ProbeAgeSecs          int64  `json:"probe_age_secs,omitempty"`
	CooldownUntilUnix     int64  `json:"cooldown_until_unix,omitempty"`
	CooldownRemainingSecs int64  `json:"cooldown_remaining_secs,omitempty"`
	Recheck               string `json:"recheck,omitempty"`
}

// routeHealthGateDecision is the pure pre-spawn verdict for ONE route key: suppressed only
// while that route's own latest probe failed and its cooldown is live. Sibling routes never
// enter the decision — an unprobed or healthy or cooled-down route passes.
func routeHealthGateDecision(records []routeHealthRecord, route string, nowUnix int64) routeHealthGateResult {
	res := routeHealthGateResult{Schema: routeHealthGateSchema, Route: route, Verdict: "pass"}
	for _, rec := range records {
		if rec.Route != route {
			continue
		}
		res.Class = rec.Class
		res.ProbeAgeSecs = nowUnix - rec.ProbedAtUnix
		res.Recheck = rec.Recheck
		if rec.Class == string(routeClassHealthy) {
			res.Reason = fmt.Sprintf("last probe healthy (%ds ago)", res.ProbeAgeSecs)
			return res
		}
		if rec.CooldownUntilUnix > nowUnix {
			res.Verdict = "suppressed"
			res.CooldownUntilUnix = rec.CooldownUntilUnix
			res.CooldownRemainingSecs = rec.CooldownUntilUnix - nowUnix
			res.Reason = fmt.Sprintf("route suppressed: %s (%ds of cooldown left); recheck: %s",
				rec.Class, res.CooldownRemainingSecs, rec.Recheck)
			return res
		}
		res.Reason = fmt.Sprintf("cooldown elapsed after %s; recheck advised: %s", rec.Class, rec.Recheck)
		return res
	}
	res.Reason = "route unprobed — no ledger row; probe advised before first live spawn"
	return res
}

// routeHealthStatusRow is one route in the fak-route-health-status/1 snapshot: the latest
// record plus the derived probe age and live-cooldown view the operator card renders.
type routeHealthStatusRow struct {
	routeHealthRecord
	ProbeAgeSecs          int64 `json:"probe_age_secs"`
	Suppressed            bool  `json:"suppressed"`
	CooldownRemainingSecs int64 `json:"cooldown_remaining_secs,omitempty"`
}

type routeHealthStatusSnapshot struct {
	Schema          string                 `json:"schema"`
	Ledger          string                 `json:"ledger"`
	NowUnix         int64                  `json:"now_unix"`
	RouteCount      int                    `json:"route_count"`
	SuppressedCount int                    `json:"suppressed_count"`
	Routes          []routeHealthStatusRow `json:"routes"`
}

func buildRouteHealthStatus(records []routeHealthRecord, ledger string, nowUnix int64) routeHealthStatusSnapshot {
	snap := routeHealthStatusSnapshot{
		Schema:  routeHealthStatusSchema,
		Ledger:  ledger,
		NowUnix: nowUnix,
	}
	for _, rec := range records {
		row := routeHealthStatusRow{routeHealthRecord: rec, ProbeAgeSecs: nowUnix - rec.ProbedAtUnix}
		if rec.Class != string(routeClassHealthy) && rec.CooldownUntilUnix > nowUnix {
			row.Suppressed = true
			row.CooldownRemainingSecs = rec.CooldownUntilUnix - nowUnix
			snap.SuppressedCount++
		}
		snap.Routes = append(snap.Routes, row)
	}
	snap.RouteCount = len(snap.Routes)
	return snap
}

func renderRouteHealthStatus(snap routeHealthStatusSnapshot) string {
	var b strings.Builder
	fmt.Fprintf(&b, "route health — %d route(s) probed, %d suppressed\n", snap.RouteCount, snap.SuppressedCount)
	fmt.Fprintf(&b, "ledger: %s\n", snap.Ledger)
	if len(snap.Routes) == 0 {
		fmt.Fprint(&b, "no probed routes — run: fak dispatch route-health probe --base-url URL --model M\n")
		return b.String()
	}
	for _, row := range snap.Routes {
		fmt.Fprintf(&b, "  %s  class=%s  probe_age=%ds", row.Route, row.Class, row.ProbeAgeSecs)
		if row.Status != 0 {
			fmt.Fprintf(&b, "  http=%d", row.Status)
		}
		if row.Suppressed {
			fmt.Fprintf(&b, "  cooldown=%ds left (until %s)", row.CooldownRemainingSecs,
				time.Unix(row.CooldownUntilUnix, 0).UTC().Format(time.RFC3339))
			if row.ProviderHint {
				fmt.Fprint(&b, " [provider hint]")
			}
		}
		fmt.Fprintln(&b)
		fmt.Fprintf(&b, "    recheck: %s\n", row.Recheck)
	}
	return b.String()
}

// runDispatchRouteHealth dispatches the probe/status/gate verbs.
func runDispatchRouteHealth(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		routeHealthUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "probe":
		return runDispatchRouteHealthProbe(stdout, stderr, argv[1:])
	case "status":
		return runDispatchRouteHealthStatus(stdout, stderr, argv[1:])
	case "gate":
		return runDispatchRouteHealthGate(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		routeHealthUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak dispatch route-health: unknown subcommand %q (want probe, status, or gate)\n", argv[0])
		routeHealthUsage(stderr)
		return 2
	}
}

func routeHealthUsage(w io.Writer) {
	fmt.Fprint(w, `fak dispatch route-health — selected-route smoke gate before live worker spawn

  fak dispatch route-health probe --base-url URL --model M [--provider P] [--account A] [--api-key-env ENV] [--prompt S] [--timeout DUR] [--workspace DIR] [--json]
  fak dispatch route-health status [--workspace DIR] [--now UNIX] [--json]
  fak dispatch route-health gate (--route KEY | --provider P --model M [--account A]) [--workspace DIR] [--now UNIX] [--json]

probe smokes ONE route (a tiny chat completion), classifies the outcome into the typed
vocabulary (healthy, timeout, unreachable, auth, rate_limited, model_unavailable,
provider_5xx, unsupported_route), honors Retry-After / X-RateLimit-Reset hints for the
cooldown, and appends one fak-route-health/1 row to the ledger. Exit 0 healthy, 1 not.

status folds the ledger to the latest row per route: last probe age, failure class,
cooldown/reset time, and the exact recheck command.

gate is the pre-spawn check for ONE route/model/account key: exit 0 = may spend a worker
slot (healthy, unprobed, or cooldown elapsed), exit 3 = suppressed while its own cooldown
is live. Sibling routes on the same provider are never affected.
`)
}

func runDispatchRouteHealthProbe(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch route-health probe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", "", "OpenAI-compatible base URL of the selected route (required)")
	model := fs.String("model", "", "model id the worker would use (required)")
	provider := fs.String("provider", "", "provider name for the route key (default: base-url host)")
	account := fs.String("account", "", "account/seat identity for the route key")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable holding the API key")
	prompt := fs.String("prompt", routeHealthDefaultPrompt, "smoke prompt")
	timeout := fs.Duration("timeout", routeHealthDefaultTimeout, "probe timeout")
	workspace := fs.String("workspace", ".", "workspace root the ledger is persisted under")
	asJSON := fs.Bool("json", false, "emit the fak-route-health/1 record as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*baseURL) == "" || strings.TrimSpace(*model) == "" {
		fmt.Fprintln(stderr, "fak dispatch route-health probe: --base-url and --model are required")
		return 2
	}
	prov := strings.TrimSpace(*provider)
	if prov == "" {
		if u, err := url.Parse(strings.TrimSpace(*baseURL)); err == nil && u.Host != "" {
			prov = u.Host
		} else {
			prov = "provider"
		}
	}
	spec := routeProbeSpec{
		Provider:  prov,
		Account:   strings.TrimSpace(*account),
		Model:     strings.TrimSpace(*model),
		BaseURL:   strings.TrimSpace(*baseURL),
		APIKeyEnv: strings.TrimSpace(*apiKeyEnv),
		Prompt:    *prompt,
		Timeout:   *timeout,
	}
	rec, err := probeProviderRoute(spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch route-health probe: %v\n", err)
		return 1
	}
	if err := appendRouteHealthRecord(routeHealthLedgerPath(*workspace), rec); err != nil {
		fmt.Fprintf(stderr, "fak dispatch route-health probe: persist ledger: %v\n", err)
		return 1
	}
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, rec, "fak dispatch route-health probe"); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "route-health probe %s: %s", rec.Route, rec.Class)
		if rec.Status != 0 {
			fmt.Fprintf(stdout, " (HTTP %d)", rec.Status)
		}
		fmt.Fprintln(stdout)
		if rec.Detail != "" {
			fmt.Fprintf(stdout, "  detail: %s\n", rec.Detail)
		}
		if rec.CooldownUntilUnix > 0 {
			fmt.Fprintf(stdout, "  cooldown until %s", time.Unix(rec.CooldownUntilUnix, 0).UTC().Format(time.RFC3339))
			if rec.ProviderHint {
				fmt.Fprint(stdout, " [provider hint]")
			}
			fmt.Fprintln(stdout)
		}
		fmt.Fprintf(stdout, "  recheck: %s\n", rec.Recheck)
	}
	if rec.Class == string(routeClassHealthy) {
		return 0
	}
	return 1
}

func runDispatchRouteHealthStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch route-health status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", ".", "workspace root the ledger is read from")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for age/cooldown math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the fak-route-health-status/1 snapshot as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	ledger := routeHealthLedgerPath(*workspace)
	records, err := loadRouteHealthLatest(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch route-health status: read ledger: %v\n", err)
		return 1
	}
	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	snap := buildRouteHealthStatus(records, ledger, now)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, snap, "fak dispatch route-health status")
	}
	fmt.Fprint(stdout, renderRouteHealthStatus(snap))
	return 0
}

func runDispatchRouteHealthGate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch route-health gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	route := fs.String("route", "", "route key provider/account/model (overrides the part flags)")
	provider := fs.String("provider", "", "provider name for the route key")
	account := fs.String("account", "", "account/seat identity for the route key")
	model := fs.String("model", "", "model id for the route key")
	workspace := fs.String("workspace", ".", "workspace root the ledger is read from")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for cooldown math (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the fak-route-health-gate/1 verdict as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	key := strings.TrimSpace(*route)
	if key == "" {
		if strings.TrimSpace(*provider) == "" || strings.TrimSpace(*model) == "" {
			fmt.Fprintln(stderr, "fak dispatch route-health gate: need --route or both --provider and --model")
			return 2
		}
		key = routeHealthKey(strings.TrimSpace(*provider), strings.TrimSpace(*account), strings.TrimSpace(*model))
	}
	records, err := loadRouteHealthLatest(routeHealthLedgerPath(*workspace))
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch route-health gate: read ledger: %v\n", err)
		return 1
	}
	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	res := routeHealthGateDecision(records, key, now)
	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, res, "fak dispatch route-health gate"); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "route-health gate %s: %s — %s\n", res.Route, res.Verdict, res.Reason)
	}
	if res.Verdict == "suppressed" {
		return 3
	}
	return 0
}
