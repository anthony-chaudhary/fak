package main

// `fak slack alert` — the receiver that wires Prometheus/Alertmanager alerts into Slack.
//
// Prometheus evaluates tools/grafana/prometheus-alerts.yml and hands firing alerts to
// Alertmanager, which groups/routes them and POSTs a v4 webhook (a `webhook_config`
// receiver). This verb is the target of that webhook: it folds the payload through
// internal/promalert and ENQUEUES the rendered card into the durable Slack outbox, so an
// alert survives a crash / 429 / token blip and its delivery is witnessable with
// `fak slack outbox status | calls`.
//
//	# one-shot: read an Alertmanager webhook body and deliver it
//	fak slack alert < payload.json
//	fak slack alert --file payload.json
//	fak slack alert --dry-run --file payload.json      # render only, touch nothing
//
//	# the live receiver Alertmanager POSTs to (background drains keep the spool moving)
//	fak slack alert --serve --addr 127.0.0.1:9096      # POST http://127.0.0.1:9096/alerts
//
// The alerts channel resolves from --channel, then FAK_ALERTS_CHANNEL, then the public
// #grafana channel default (alerts are Grafana/Prometheus-adjacent) — the same
// env-then-file-then-default idiom every other surface uses. The bot token is the shared
// scoreboard token; a row never carries a secret.

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/grafanapost"
	"github.com/anthony-chaudhary/fak/internal/promalert"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	// alertsChannelEnv overrides the alerts channel (env or .env.slack.local).
	alertsChannelEnv = "FAK_ALERTS_CHANNEL"
	// alertSource tags outbox rows this receiver produces, for status/calls attribution.
	alertSource = "prom-alert"
	// alertServePath is the HTTP path Alertmanager POSTs its webhook to.
	alertServePath = "/alerts"
	// alertServeMaxBody caps a single webhook body — a defensive read limit, well above any
	// real Alertmanager group (Alertmanager itself truncates large groups).
	alertServeMaxBody = 1 << 20 // 1 MiB
)

// resolveAlertsChannel applies the documented resolution order for the alerts channel:
// FAK_ALERTS_CHANNEL (env then .env.slack.local), then the public #grafana default so the
// surface lands with zero config. Alerts belong next to the Grafana snapshots they page
// on, never silently in #scoreboard.
func resolveAlertsChannel() string {
	if r := slackenv.Lookup(alertsChannelEnv); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	return grafanapost.ChannelDefault
}

// runSlackAlert routes `fak slack alert`: a one-shot fold of a webhook body from stdin/file,
// or (with --serve) the long-running HTTP receiver Alertmanager POSTs to.
func runSlackAlert(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack alert", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "read the Alertmanager webhook JSON from this file (default: stdin)")
	serve := fs.Bool("serve", false, "run the HTTP receiver Alertmanager POSTs to (long-running)")
	addr := fs.String("addr", "127.0.0.1:9096", "listen address for --serve")
	channel := fs.String("channel", "", "target channel id (default: FAK_ALERTS_CHANNEL, then #grafana)")
	token := fs.String("token", "", "bot token (default: $FAK_SCOREBOARD_TOKEN, then .env.slack.local)")
	apiBase := fs.String("api-base", "", "override the Slack API base URL (for testing/proxying)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not enqueue or post")
	maxAlerts := fs.Int("max-alerts", 0, "cap per-alert blocks in one card (default 10)")
	if !parseFlags(fs, argv) {
		return 2
	}

	ch := *channel
	if ch == "" {
		ch = resolveAlertsChannel()
	}
	if ch == "" && !*dryRun {
		fmt.Fprintln(stderr, "fak slack alert: no channel — pass --channel or set "+alertsChannelEnv)
		return 2
	}

	if *serve {
		return runSlackAlertServe(stdout, stderr, *addr, ch, *token, *apiBase, *maxAlerts)
	}

	// One-shot: read the body from --file or stdin.
	var body []byte
	var err error
	if *file != "" {
		body, err = os.ReadFile(*file)
	} else {
		body, err = io.ReadAll(io.LimitReader(os.Stdin, alertServeMaxBody))
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak slack alert: read payload: %v\n", err)
		return 1
	}

	w, err := promalert.ParseBytes(body)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack alert: %v\n", err)
		return 1
	}
	rendered := promalert.Render(w, promalert.RenderOpts{MaxAlerts: *maxAlerts})

	if *dryRun {
		fmt.Fprintf(stdout, "fak slack alert (dry-run): would post to %s\n\n%s\n", chOrDash(ch), rendered)
		return 0
	}

	nonce, err := enqueueAlertCard(ch, rendered)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack alert: enqueue: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak slack alert: enqueued %d alert(s) durably (nonce=%s, ch=%s)\n", len(w.Alerts), nonce, ch)
	drainAlertOutbox(stdout, *token, *apiBase)
	return 0
}

// enqueueAlertCard writes one rendered alert card into the durable outbox and returns its
// nonce. Each webhook POST is its own row: Alertmanager already handles grouping, dedup,
// and repeat cadence, so the receiver posts what it is handed and lets the outbox deliver.
func enqueueAlertCard(channel, text string) (string, error) {
	ob, err := openOutbox()
	if err != nil {
		return "", err
	}
	return ob.Enqueue(slackoutbox.Row{Channel: channel, Text: text, Source: alertSource})
}

// drainAlertOutbox runs one best-effort drain so a one-shot receive delivers now when a
// token is configured; a failure leaves the row durable for `fak slack outbox drain` or the
// watchdog. It prints the drain fold (or the reason delivery was deferred).
func drainAlertOutbox(stdout io.Writer, token, apiBase string) {
	tok := token
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}
	ob, err := openOutbox()
	if err != nil {
		return
	}
	wire, werr := outboxWire(tok, apiBase)
	if werr != nil {
		fmt.Fprintf(stdout, "  delivery deferred: %v — run `fak slack outbox drain` once configured\n", werr)
		return
	}
	rep, derr := ob.Drain(ctx(), wire, stdDrainOpts())
	switch {
	case errors.Is(derr, slackoutbox.ErrDrainBusy):
		fmt.Fprintln(stdout, "  delivery deferred: another drainer holds the lock")
	case derr != nil:
		fmt.Fprintf(stdout, "  delivery deferred: %v\n", derr)
	default:
		fmt.Fprintf(stdout, "  drained: posted %d  refused %d  failed %d  remaining %d\n",
			rep.Posted, rep.Refused, rep.Failed, rep.Remaining)
	}
}

// newAlertServeMux builds the receiver's HTTP routes. It is split out from the listener so a
// test can drive the exact handler Alertmanager hits. onEnqueue, when non-nil, is called
// after a successful enqueue (the live path kicks a background drain; a test can synchronize
// on it).
func newAlertServeMux(stdout io.Writer, channel string, maxAlerts int, onEnqueue func()) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc(alertServePath, func(rw http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(rw, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(io.LimitReader(req.Body, alertServeMaxBody))
		if err != nil {
			http.Error(rw, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
		w, err := promalert.ParseBytes(body)
		if err != nil {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
		rendered := promalert.Render(w, promalert.RenderOpts{MaxAlerts: maxAlerts})
		nonce, err := enqueueAlertCard(channel, rendered)
		if err != nil {
			// A local spool write failed — tell Alertmanager to retry.
			http.Error(rw, "enqueue: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintf(stdout, "fak slack alert: enqueued %d alert(s) (status=%s nonce=%s ch=%s)\n",
			len(w.Alerts), w.Status, nonce, channel)
		if onEnqueue != nil {
			onEnqueue()
		}
		rw.WriteHeader(http.StatusOK)
		fmt.Fprintf(rw, "enqueued nonce=%s alerts=%d\n", nonce, len(w.Alerts))
	})
	// A liveness probe so an operator (or a compose healthcheck) can confirm the receiver.
	mux.HandleFunc("/healthz", func(rw http.ResponseWriter, _ *http.Request) {
		rw.WriteHeader(http.StatusOK)
		fmt.Fprintln(rw, "ok")
	})
	return mux
}

// runSlackAlertServe runs the HTTP receiver. It answers Alertmanager's POST fast (enqueue is
// a local durable write) and drains in the background so a slow Slack post never blocks the
// webhook — Alertmanager retries a non-2xx, so a stalled handler would amplify a storm.
func runSlackAlertServe(stdout, stderr io.Writer, addr, channel, token, apiBase string, maxAlerts int) int {
	mux := newAlertServeMux(stdout, channel, maxAlerts, func() { go backgroundDrain(token, apiBase) })
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	fmt.Fprintf(stdout, "fak slack alert: receiver listening on http://%s%s → channel %s\n", addr, alertServePath, chOrDash(channel))
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fmt.Fprintf(stderr, "fak slack alert: serve: %v\n", err)
		return 1
	}
	return 0
}

// backgroundDrain runs a single bounded drain off the request path.
func backgroundDrain(token, apiBase string) {
	tok := token
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}
	ob, err := openOutbox()
	if err != nil {
		return
	}
	wire, err := outboxWire(tok, apiBase)
	if err != nil {
		return // no token yet — the row stays durable for a later drain
	}
	dctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_, _ = ob.Drain(dctx, wire, stdDrainOpts())
}

// chOrDash renders an empty channel as "-" for a log line.
func chOrDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}
