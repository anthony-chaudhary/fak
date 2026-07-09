// Package promalert parses an Alertmanager webhook payload (the v4 JSON an
// Alertmanager `webhook_config` POSTs) and renders it into compact Slack message
// text. It is the INBOUND translation half of fak's Prometheus-alerts→Slack wiring:
// Prometheus evaluates the rules in tools/grafana/prometheus-alerts.yml, Alertmanager
// groups/routes the firing alerts, and POSTs a webhook here; `fak slack alert` folds
// that payload through Render and enqueues the result into the durable Slack outbox.
//
// Why a fak-native receiver rather than Alertmanager's built-in slack_config: the
// built-in Slack receiver posts through a Slack *incoming-webhook URL*, but every fak
// surface authenticates with a *bot token* (the shared scoreboard token, see
// internal/scoreboard). Routing alerts through this receiver reuses that one token and
// — more importantly — the durable outbox (internal/slackoutbox): an alert survives a
// crash, a 429 storm, or a token blip instead of being fire-and-forgotten, and its
// delivery is witnessable with `fak slack outbox status | calls`.
//
// This package is PURE — parse bytes in, render a string out, no I/O, no time.Now — so
// the fold is unit-tested directly and the CLI/HTTP layer owns the transport.
package promalert

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// Webhook is the Alertmanager v4 webhook payload
// (https://prometheus.io/docs/alerting/latest/configuration/#webhook_config). Only the
// fields the render reads are typed; unknown fields are ignored by encoding/json.
type Webhook struct {
	Version           string            `json:"version"`
	GroupKey          string            `json:"groupKey"`
	TruncatedAlerts   int               `json:"truncatedAlerts"`
	Status            string            `json:"status"` // "firing" | "resolved"
	Receiver          string            `json:"receiver"`
	GroupLabels       map[string]string `json:"groupLabels"`
	CommonLabels      map[string]string `json:"commonLabels"`
	CommonAnnotations map[string]string `json:"commonAnnotations"`
	ExternalURL       string            `json:"externalURL"`
	Alerts            []Alert           `json:"alerts"`
}

// Alert is one firing/resolved alert inside a webhook.
type Alert struct {
	Status       string            `json:"status"` // "firing" | "resolved"
	Labels       map[string]string `json:"labels"`
	Annotations  map[string]string `json:"annotations"`
	StartsAt     string            `json:"startsAt"`
	EndsAt       string            `json:"endsAt"`
	GeneratorURL string            `json:"generatorURL"`
	Fingerprint  string            `json:"fingerprint"`
}

// Parse decodes an Alertmanager webhook payload from r. It rejects a payload with no
// alerts so a malformed or empty POST becomes a caller error rather than an empty,
// meaningless Slack card.
func Parse(r io.Reader) (*Webhook, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("promalert: read payload: %w", err)
	}
	return ParseBytes(b)
}

// ParseBytes decodes an Alertmanager webhook payload from b.
func ParseBytes(b []byte) (*Webhook, error) {
	if len(strings.TrimSpace(string(b))) == 0 {
		return nil, fmt.Errorf("promalert: empty payload")
	}
	var w Webhook
	dec := json.NewDecoder(strings.NewReader(string(b)))
	if err := dec.Decode(&w); err != nil {
		return nil, fmt.Errorf("promalert: decode webhook json: %w", err)
	}
	if len(w.Alerts) == 0 {
		return nil, fmt.Errorf("promalert: webhook carries no alerts (status=%q receiver=%q)", w.Status, w.Receiver)
	}
	return &w, nil
}

// RenderOpts tunes the fold. The zero value is valid (MaxAlerts defaults to
// DefaultMaxAlerts) so callers can pass promalert.RenderOpts{}.
type RenderOpts struct {
	// MaxAlerts caps how many per-alert blocks are rendered; the remainder is noted as a
	// count so one noisy group can never blow past Slack's text ceiling. 0 => default.
	MaxAlerts int
}

// DefaultMaxAlerts bounds the per-alert blocks in one card. A group larger than this is
// almost always the same class fanned across instances; the header already carries the
// total, so the tail is summarized rather than dropped silently.
const DefaultMaxAlerts = 10

// Render folds a webhook into one Slack message body: a header line (status emoji,
// FIRING/RESOLVED, alert count, group), the common labels, then one block per alert
// (severity, alertname, summary, description, distinguishing labels, start time, and the
// generator link). It is deterministic and pure — label order is sorted — so the output
// is stable under test.
func Render(w *Webhook, opts RenderOpts) string {
	if w == nil {
		return ""
	}
	maxAlerts := opts.MaxAlerts
	if maxAlerts <= 0 {
		maxAlerts = DefaultMaxAlerts
	}

	var b strings.Builder

	status := strings.ToLower(strings.TrimSpace(w.Status))
	b.WriteString(statusEmoji(status, commonSeverity(w)))
	b.WriteByte(' ')
	b.WriteString(strings.ToUpper(statusWord(status)))
	fmt.Fprintf(&b, " · %d alert(s)", len(w.Alerts))
	if grp := renderGroup(w.GroupLabels); grp != "" {
		fmt.Fprintf(&b, " · %s", grp)
	}
	b.WriteByte('\n')

	if cl := renderInlineLabels(w.CommonLabels, "alertname"); cl != "" {
		b.WriteString(cl)
		b.WriteByte('\n')
	}

	shown := w.Alerts
	if len(shown) > maxAlerts {
		shown = shown[:maxAlerts]
	}
	for _, a := range shown {
		b.WriteByte('\n')
		b.WriteString(renderAlert(a))
	}
	if hidden := len(w.Alerts) - len(shown); hidden > 0 {
		fmt.Fprintf(&b, "\n… and %d more alert(s) in this group", hidden)
	}
	if w.TruncatedAlerts > 0 {
		fmt.Fprintf(&b, "\n(Alertmanager truncated %d further alert(s))", w.TruncatedAlerts)
	}
	return b.String()
}

// renderAlert folds one alert into its block.
func renderAlert(a Alert) string {
	sev := severity(a.Labels)
	name := field(a.Labels, "alertname")
	if name == "" {
		name = "alert"
	}
	var b strings.Builder
	b.WriteString(statusEmoji(strings.ToLower(a.Status), sev))
	fmt.Fprintf(&b, " %s", name)
	if sev != "" {
		fmt.Fprintf(&b, " (%s)", sev)
	}
	b.WriteByte('\n')

	if s := field(a.Annotations, "summary"); s != "" {
		fmt.Fprintf(&b, "   %s\n", collapse(s))
	}
	if d := field(a.Annotations, "description"); d != "" {
		fmt.Fprintf(&b, "   %s\n", collapse(d))
	}
	// The distinguishing labels: everything except the ones already in the header line.
	if lbls := renderInlineLabels(a.Labels, "alertname", "severity"); lbls != "" {
		fmt.Fprintf(&b, "   %s\n", lbls)
	}
	if ts := shortTime(a.StartsAt); ts != "" {
		if strings.EqualFold(a.Status, "resolved") {
			if end := shortTime(a.EndsAt); end != "" {
				fmt.Fprintf(&b, "   fired %s · resolved %s\n", ts, end)
			} else {
				fmt.Fprintf(&b, "   fired %s\n", ts)
			}
		} else {
			fmt.Fprintf(&b, "   since %s\n", ts)
		}
	}
	if u := strings.TrimSpace(a.GeneratorURL); u != "" {
		fmt.Fprintf(&b, "   ↳ %s\n", u)
	}
	return strings.TrimRight(b.String(), "\n")
}

// commonSeverity returns the group-wide severity when every alert shares one, for the
// header emoji; "" when it is mixed or absent (the status emoji then carries the state).
func commonSeverity(w *Webhook) string {
	if s := field(w.CommonLabels, "severity"); s != "" {
		return s
	}
	return ""
}

func severity(labels map[string]string) string { return field(labels, "severity") }

// statusWord normalizes the alert state word.
func statusWord(status string) string {
	switch status {
	case "resolved":
		return "resolved"
	case "firing", "":
		return "firing"
	default:
		return status
	}
}

// statusEmoji picks the leading glyph: a resolved alert is always the green check; a
// firing alert takes its color from severity (critical red, warning amber, else blue).
func statusEmoji(status, sev string) string {
	if status == "resolved" {
		return "✅"
	}
	switch strings.ToLower(strings.TrimSpace(sev)) {
	case "critical", "page", "fatal":
		return "🔴"
	case "warning", "warn":
		return "⚠️"
	case "info", "information":
		return "🔵"
	default:
		return "🚨"
	}
}

// renderGroup folds the groupLabels into a {k="v"} signature (sorted), matching the shape
// Alertmanager itself uses for a group key.
func renderGroup(labels map[string]string) string {
	pairs := sortedPairs(labels)
	if len(pairs) == 0 {
		return ""
	}
	var parts []string
	for _, kv := range pairs {
		parts = append(parts, fmt.Sprintf("%s=%q", kv[0], kv[1]))
	}
	return "group {" + strings.Join(parts, ", ") + "}"
}

// renderInlineLabels renders labels as a compact `k=v  k=v` line, skipping the excluded
// keys (already shown elsewhere) and the noisy Alertmanager internals.
func renderInlineLabels(labels map[string]string, exclude ...string) string {
	skip := map[string]bool{}
	for _, e := range exclude {
		skip[e] = true
	}
	var parts []string
	for _, kv := range sortedPairs(labels) {
		if skip[kv[0]] {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s=%s", kv[0], collapse(kv[1])))
	}
	return strings.Join(parts, "  ")
}

// sortedPairs returns the map as key-sorted [k,v] pairs so a render is deterministic.
func sortedPairs(m map[string]string) [][2]string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([][2]string, 0, len(keys))
	for _, k := range keys {
		out = append(out, [2]string{k, m[k]})
	}
	return out
}

// field reads a trimmed map value, "" when absent.
func field(m map[string]string, key string) string {
	return strings.TrimSpace(m[key])
}

// collapse flattens whitespace runs so a multi-line annotation stays on one Slack line.
func collapse(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// shortTime parses an RFC3339 timestamp and renders it as a compact UTC clock; the raw
// string is returned if it does not parse, and "" for the RFC3339 zero value (an unset
// endsAt on a firing alert).
func shortTime(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		// Alertmanager uses nanosecond precision; try that shape too.
		if t2, err2 := time.Parse(time.RFC3339Nano, s); err2 == nil {
			t = t2
		} else {
			return s
		}
	}
	if t.IsZero() || t.Year() <= 1 {
		return ""
	}
	return t.UTC().Format("2006-01-02 15:04:05Z")
}
