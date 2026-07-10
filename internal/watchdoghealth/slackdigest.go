package watchdoghealth

import (
	"strconv"
	"strings"
)

// slackdigest.go folds a watchdog health Digest into a channel-ready card — the stack-
// independent sink the epic (#3802) calls for. Today `fak watchdog status --check` exits
// non-zero when a default monitor needs attention but posts NOWHERE: a host without the
// Prometheus → Alertmanager → `fak slack alert` stack running gets no channel signal at all.
// This leaf closes that gap at the pure seam: it renders the Digest (health.go) and its
// triage split (health_triage.go) into a SlackDigest — a title, a body, and the one-bit post
// gate — that the poster shell hands straight to an internal/slackoutbox card.
//
// It reasons about WHAT the channel reads, never HOW it is sent: no clock, no I/O, no
// slackoutbox import. Same digest in, same card out — so the fold is table-testable in
// isolation and the wire (pacing, leak fence, coalescing, idempotency) stays the outbox's
// sole concern, exactly as the run-card seam already draws it.

// SlackSeverity is the closed alert level a watchdog health digest folds to for a channel
// sink. It refines the digest's single NeedsAttention bit through the triage split, so a
// reader can tell a fleet-clears notice from a genuine human page at a glance: a digest whose
// every attention-floor monitor is the fleet's own to restart or re-probe is NOT a page.
type SlackSeverity string

const (
	// SlackOK: no monitor sits at the attention floor. Nothing is owed a channel — the post
	// gate is closed and a healthy layer stays silent.
	SlackOK SlackSeverity = "OK"
	// SlackNotice: a monitor needs attention, but every such monitor is the fleet's to clear
	// (a DOWN the autoheal restarts on its next tick, an UNKNOWN to re-probe). Informational —
	// the layer is recovering itself, no person is blocked.
	SlackNotice SlackSeverity = "NOTICE"
	// SlackAlert: at least one monitor genuinely waits on a person — a GAVE_UP whose restart
	// budget is spent, or a monitor whose heal-reason names an auth wall. This is the
	// irreducible human residual the triage fold keeps the page on.
	SlackAlert SlackSeverity = "ALERT"
)

// SlackDigest is the channel-ready fold of a watchdog health Digest: the closed severity, the
// one-bit post gate, a one-line title, and a coalescible body that leads with the monitors a
// person owns. It carries no free-text verdict — every token is derived from the digest — so a
// caller cannot narrate a health state the monitors do not show.
type SlackDigest struct {
	// Severity is the closed level the digest folds to (OK | NOTICE | ALERT).
	Severity SlackSeverity `json:"severity"`
	// ShouldPost is the post gate: true exactly when the digest sits at or above the attention
	// floor — the SAME condition `fak watchdog status --check` exits non-zero on. A shell posts
	// only when this is set, so the channel speaks when the check fails and is silent otherwise.
	ShouldPost bool `json:"should_post"`
	// Title is the one-line card header: severity, worst-of rollup, and per-status counts.
	Title string `json:"title"`
	// Body is the card detail: the monitors a person owns (with the concrete next move the
	// triage fold named), then the ones the fleet clears itself. "" is never returned — an
	// all-clear digest still renders a healthy one-liner so a recovery edit reads cleanly.
	Body string `json:"body"`
}

// SlackHealthDigest folds a health Digest into its channel card. ShouldPost mirrors the
// digest's NeedsAttention gate (the default `--check` exit condition); Severity refines it
// through PartitionAttention so ALERT is reserved for a genuine human residual and a
// fleet-only recovery reads as NOTICE. Pure and total over any digest — an empty or all-clear
// digest yields a closed, silent SlackOK card.
func SlackHealthDigest(d Digest) SlackDigest {
	needHuman, fleetClears := PartitionAttention(d)
	sd := SlackDigest{ShouldPost: d.NeedsAttention}
	switch {
	case len(needHuman) > 0:
		sd.Severity = SlackAlert
	case len(fleetClears) > 0:
		sd.Severity = SlackNotice
	default:
		sd.Severity = SlackOK
	}
	sd.Title = slackDigestTitle(d, sd.Severity)
	sd.Body = slackDigestBody(d, needHuman, fleetClears)
	return sd
}

// slackDigestTitle renders the card header: "fak watchdog health — <SEVERITY>: <rollup>
// (<counts>)". The rollup is the digest's worst-of status; the counts summary names every
// non-zero status in severity order so the header alone conveys the shape of the fleet.
func slackDigestTitle(d Digest, sev SlackSeverity) string {
	return "fak watchdog health — " + string(sev) + ": " + string(d.Rollup) + " (" + slackCountsSummary(d) + ")"
}

// slackCountsSummary joins the digest's non-zero per-status counts in severity order, e.g.
// "2 HEALTHY, 1 GAVE_UP". An input with no monitors reads "no monitors".
func slackCountsSummary(d Digest) string {
	parts := make([]string, 0, len(d.Counts))
	total := 0
	for _, s := range Statuses() {
		n := d.Counts[s]
		if n == 0 {
			continue
		}
		total += n
		parts = append(parts, strconv.Itoa(n)+" "+string(s))
	}
	if total == 0 {
		return "no monitors"
	}
	return strings.Join(parts, ", ")
}

// slackDigestBody builds the card detail. It leads with the monitors a person owns — each with
// the concrete next move the triage fold named — then lists the ones the fleet clears itself.
// An all-clear digest (both buckets empty) renders a single healthy one-liner rather than "",
// so a card edited from ALERT back to OK still reads as a complete sentence.
func slackDigestBody(d Digest, needHuman, fleetClears []AttentionItem) string {
	var b strings.Builder
	if len(needHuman) > 0 {
		b.WriteString("needs you:\n")
		writeAttentionItems(&b, needHuman)
	}
	if len(fleetClears) > 0 {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString("fleet is clearing:\n")
		writeAttentionItems(&b, fleetClears)
	}
	if b.Len() == 0 {
		return "all monitors healthy (" + slackCountsSummary(d) + ")"
	}
	return strings.TrimRight(b.String(), "\n")
}

// writeAttentionItems appends one "• <id> — <STATUS>: <resolve>" line per item, in the
// digest's monitor order. The resolve clause is dropped when the triage fold named no remedy
// (an UNKNOWN monitor in hand), keeping the line honest rather than inventing a next move.
func writeAttentionItems(b *strings.Builder, items []AttentionItem) {
	for _, it := range items {
		b.WriteString("• " + it.ID + " — " + string(it.Status))
		if r := strings.TrimSpace(it.Resolve); r != "" {
			b.WriteString(": " + r)
		}
		b.WriteString("\n")
	}
}
