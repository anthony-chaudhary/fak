package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

const (
	guardSessionsChannelEnv     = "FAK_GUARD_SESSIONS_CHANNEL"
	guardSessionsTokenEnv       = "FAK_GUARD_SESSIONS_TOKEN"
	guardSessionsChannelDefault = "C0BEZ7513UM"
	guardSessionThreadSource    = "guard-session"

	// guardSessionCardUpdateInterval paces the live root-card edits while a session runs.
	guardSessionCardUpdateInterval = 20 * time.Second
	// guardSessionCardKeepaliveInterval bounds how long an IDLE session's card goes without
	// a refresh. The live line embeds an elapsed clock, so before this gate every 20s tick
	// produced a byte-different chat.update forever — a real API call per tick even when
	// turns/tokens never moved, multiplied across every guarded session on one shared bot
	// token. The tick now spends a chat.update only when the SUBSTANTIVE fold moved, or when
	// this keepalive window has elapsed so an idle card's clock still advances for a reader.
	guardSessionCardKeepaliveInterval = 5 * time.Minute
	// guardSessionFinalDrainTimeout bounds the synchronous flush of the final outcome edit
	// before guard exits (finishGuardChildAndReport calls os.Exit right after).
	guardSessionFinalDrainTimeout = 8 * time.Second
	// guardSessionReplyTextLimit keeps a reply comfortably under Slack's ~4000-char text
	// ceiling; a longer launch banner is chunked across replies.
	guardSessionReplyTextLimit = 3500
)

type guardSessionThreadMeta struct {
	Channel   string
	Nonce     string
	TraceID   string
	Agent     string
	Provider  string
	AuditPath string
	StartedAt time.Time
	PID       int
}

func resolveGuardSessionsChannel() string {
	if r := slackenv.Lookup(guardSessionsChannelEnv); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	return guardSessionsChannelDefault
}

func resolveGuardSessionsToken() string {
	if r := slackenv.Lookup(guardSessionsTokenEnv); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	if r := slackenv.Lookup(scoreboardTokenKey); r.Set() {
		return strings.TrimSpace(r.Value)
	}
	return ""
}

func enqueueGuardSessionThread(traceID, provider string, command []string, auditPath string, startedAt time.Time) (slackoutbox.Row, error) {
	row := guardSessionThreadRow(guardSessionThreadMeta{
		Channel:   resolveGuardSessionsChannel(),
		Nonce:     slackoutbox.NewNonce(),
		TraceID:   traceID,
		Agent:     guardSessionAgentName(command),
		Provider:  provider,
		AuditPath: auditPath,
		StartedAt: startedAt,
		PID:       os.Getpid(),
	})
	ob, err := openOutbox()
	if err != nil {
		return row, err
	}
	if _, err := ob.Enqueue(row); err != nil {
		return row, err
	}
	return row, nil
}

func guardSessionThreadRow(m guardSessionThreadMeta) slackoutbox.Row {
	channel := strings.TrimSpace(m.Channel)
	if channel == "" {
		channel = guardSessionsChannelDefault
	}
	nonce := strings.TrimSpace(m.Nonce)
	if nonce == "" {
		nonce = slackoutbox.NewNonce()
	}
	started := m.StartedAt.UTC()
	if started.IsZero() {
		started = time.Now().UTC()
	}
	text := fmt.Sprintf(
		"fak guard session started\nsession_thread_id: %s\ntrace_id: %s\nagent: %s\nprovider: %s\npid: %d\nstarted_utc: %s\naudit: %s",
		guardSlackField(nonce),
		guardSlackField(m.TraceID),
		guardSlackField(m.Agent),
		guardSlackField(m.Provider),
		m.PID,
		started.Format(time.RFC3339),
		guardSlackAuditRef(m.AuditPath),
	)
	return slackoutbox.Row{
		Nonce:   nonce,
		Channel: channel,
		Text:    text,
		Source:  guardSessionThreadSource,
	}
}

func guardSessionAgentName(command []string) string {
	if len(command) == 0 {
		return "unknown"
	}
	name := strings.TrimSpace(command[0])
	if name == "" {
		return "unknown"
	}
	name = strings.ReplaceAll(name, "\\", "/")
	return path.Base(name)
}

func guardSlackField(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "-"
	}
	return strings.Join(strings.Fields(s), " ")
}

func guardSlackAuditRef(s string) string {
	s = guardSlackField(s)
	if s == "-" {
		return s
	}
	if strings.Contains(strings.ToLower(s), "off") {
		return "off"
	}
	return path.Base(strings.ReplaceAll(s, "\\", "/"))
}

func startGuardSessionThreadDrain() {
	token := resolveGuardSessionsToken()
	if token == "" {
		return
	}
	go func() {
		ob, err := openOutbox()
		if err != nil {
			return
		}
		wire, err := outboxWire(token, "")
		if err != nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = ob.Drain(ctx, wire, stdDrainOpts())
	}()
}

// guardSessionMetricsSource is the slice of the gateway the live root-card line reads —
// just the adjudication fold. Kept as an interface so the render is testable without a
// real *gateway.Server.
type guardSessionMetricsSource interface {
	AdjudicationSummary() gateway.AdjudicationSummary
}

// guardSessionControlContext carries the launch facts rendered into the context reply.
type guardSessionControlContext struct {
	Command    []string
	Cwd        string
	Audit      string // full audit-journal path (not the basename the root card shows)
	Trace      string
	Provider   string
	GatewayURL string
}

// guardSessionCardHandle is the process-wide live session card. A guard process runs
// exactly one session, so a single package-level handle suffices: it is set once in the
// guard main flow right after the session root is queued (newGuardSessionCard) and its
// terminal outcome edit is written from finishGuardChildAndReport (finalizeOutcome). It
// stays nil — and every method a no-op — whenever Slack is unconfigured or the root did not
// queue, so neither the launch path nor the teardown has to branch on Slack being available.
var guardSessionCardHandle *guardSessionCard

// guardSessionCard is the in-memory handle to a guard session's live-updating root card.
// It is nil whenever a root did not queue (Slack unconfigured), and every method is a
// no-op on a nil receiver, so callers never have to branch on Slack being available.
//
// It deliberately holds NO durable state: the guard process lives for the whole session
// (a --restart-on-budget event relaunches the CHILD, not guard), so the root nonce stays
// in memory and the posted ts is resolved from the outbox on demand — there is nothing to
// survive a restart, hence no card-state file.
type guardSessionCard struct {
	channel   string
	rootNonce string
	startedAt time.Time

	srv  guardSessionMetricsSource
	stop chan struct{}
	done chan struct{}

	// Change-gate state, owned solely by the updater goroutine (finalize runs only after
	// stopUpdater joins it), so no lock is needed. lastKey is the substantive fold of the
	// last edit actually enqueued; lastSentAt is when it went out — the keepalive clock.
	lastKey    string
	lastSentAt time.Time
}

// newGuardSessionCard returns a card for a queued root, or nil if the root did not queue.
func newGuardSessionCard(channel, rootNonce string, startedAt time.Time) *guardSessionCard {
	channel = strings.TrimSpace(channel)
	rootNonce = strings.TrimSpace(rootNonce)
	if channel == "" || rootNonce == "" {
		return nil
	}
	return &guardSessionCard{channel: channel, rootNonce: rootNonce, startedAt: startedAt}
}

// startUpdater spawns the goroutine that edits the root card on a fixed cadence with the
// current session progress. It is a no-op if the card or the metrics source is nil.
func (c *guardSessionCard) startUpdater(srv guardSessionMetricsSource) {
	if c == nil || srv == nil {
		return
	}
	c.srv = srv
	c.stop = make(chan struct{})
	c.done = make(chan struct{})
	go func() {
		defer close(c.done)
		t := time.NewTicker(guardSessionCardUpdateInterval)
		defer t.Stop()
		for {
			select {
			case <-c.stop:
				return
			case <-t.C:
				c.tick(time.Now())
			}
		}
	}()
}

// stopUpdater halts the cadence goroutine and waits for it to exit, so no periodic update
// races the final flush. Safe to call once; a no-op if the updater never started.
func (c *guardSessionCard) stopUpdater() {
	if c == nil || c.stop == nil {
		return
	}
	close(c.stop)
	<-c.done
	c.stop = nil
}

// tick is one periodic-update decision. It enqueues an edit (and kicks a drain) ONLY when
// the edit carries new information: the substantive fold (turns/tokens/denied) moved since
// the last edit that actually went out, OR the keepalive window elapsed so an idle card's
// elapsed clock still advances for a reader. An unchanged, not-yet-keepalive tick spends no
// chat.update at all — the fix for the guard-session card being the fleet's dominant Slack
// rate-limit drain. Gate state advances only when an edit was truly spooled, so a tick that
// no-ops because the root has not posted yet (enqueueUpdateSent -> false) is retried, not
// silently swallowed. Returns whether an edit was enqueued (for tests). now is injected so
// the keepalive arithmetic is deterministic under test.
func (c *guardSessionCard) tick(now time.Time) bool {
	if c == nil || c.srv == nil {
		return false
	}
	sum := c.srv.AdjudicationSummary()
	key := guardSessionCardKey(sum)
	changed := key != c.lastKey
	keepaliveDue := !c.lastSentAt.IsZero() && now.Sub(c.lastSentAt) >= guardSessionCardKeepaliveInterval
	if !changed && !keepaliveDue {
		return false // unchanged and keepalive not due — spend no API call
	}
	sent, err := c.enqueueUpdateSent(guardSessionLiveLine(sum, now.Sub(c.startedAt)))
	if err != nil || !sent {
		return false // root not posted yet, or enqueue failed — retry next tick, keep gate state
	}
	c.lastKey = key
	c.lastSentAt = now
	startGuardSessionThreadDrain()
	return true
}

// guardSessionCardKey folds the SUBSTANTIVE fields of the adjudication summary — the ones a
// reader acts on — into the change-gate key. It deliberately excludes the elapsed clock:
// the clock advancing is not new information worth a chat.update, and embedding it is what
// made every idle tick a distinct, non-idempotent edit.
func guardSessionCardKey(sum gateway.AdjudicationSummary) string {
	return fmt.Sprintf("%d|%d|%d|%d|%d",
		sum.Total, sum.InputTokens, sum.OutputTokens, sum.CachedPromptTokens, sum.Denied)
}

// enqueueUpdate queues an edit of the root card, resolving the root's posted ts from the
// outbox. It SKIPS (no error) while the root has not posted yet, so an early tick or a
// token-less run never spams update rows against a nonexistent card.
func (c *guardSessionCard) enqueueUpdate(text string) error {
	_, err := c.enqueueUpdateSent(text)
	return err
}

// enqueueUpdateSent is enqueueUpdate that also reports whether a row was actually spooled.
// It returns false (no error) while the root has not posted yet — there is no card ts to
// edit — so the change-gate can distinguish "skipped, retry me" from "sent, advance my
// state" and never strands the first real update behind the keepalive.
func (c *guardSessionCard) enqueueUpdateSent(text string) (bool, error) {
	if c == nil {
		return false, nil
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return false, nil
	}
	ob, err := openOutbox()
	if err != nil {
		return false, err
	}
	snap, err := ob.Load()
	if err != nil {
		return false, err
	}
	ts := snap.PostedTS(c.rootNonce)
	if ts == "" {
		return false, nil // root not posted yet — nothing to edit
	}
	if _, err := ob.Enqueue(slackoutbox.Row{
		Channel:  c.channel,
		Text:     text,
		UpdateTS: ts,
		Source:   guardSessionThreadSource + ":status",
	}); err != nil {
		return false, err
	}
	return true, nil
}

// finalize writes the final outcome edit and flushes it SYNCHRONOUSLY: guard calls
// os.Exit immediately after, so a background drain would be cut off. Update rows coalesce
// on the card key, so this final line supersedes any pending periodic edit. With no token
// the edit stays durable in the spool for a later drain.
func (c *guardSessionCard) finalize(text string) {
	if c == nil {
		return
	}
	if err := c.enqueueUpdate(text); err != nil {
		return
	}
	token := resolveGuardSessionsToken()
	if token == "" {
		return
	}
	ob, err := openOutbox()
	if err != nil {
		return
	}
	wire, err := outboxWire(token, "")
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), guardSessionFinalDrainTimeout)
	defer cancel()
	// A periodic drain kicked by the last tick may still hold the lock; retry within the
	// bounded window until it releases so the final edit actually ships.
	for {
		if _, derr := ob.Drain(ctx, wire, stdDrainOpts()); !errors.Is(derr, slackoutbox.ErrDrainBusy) {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// finalizeOutcome stops the live updater and writes the terminal outcome edit for the
// session's exit — the single call guard's teardown (finishGuardChildAndReport) makes so it
// never has to know the card's internals or branch on Slack. Nil-safe: a session that never
// queued a root is a no-op. Elapsed is measured from the card's own start so the final line
// shares the live line's clock.
func (c *guardSessionCard) finalizeOutcome(exitCode int, sum gateway.AdjudicationSummary) {
	if c == nil {
		return
	}
	c.stopUpdater()
	c.finalize(guardSessionFinalLine(exitCode, sum, time.Since(c.startedAt)))
}

// enqueueGuardSessionControlPoints queues the launch replies (banner + context) under the
// session root, returning how many queued. Best-effort: a partial failure returns the
// count queued so far plus the error.
func enqueueGuardSessionControlPoints(rootNonce, channel, launchBanner string, cx guardSessionControlContext) (int, error) {
	rows := guardSessionControlPointRows(rootNonce, channel, launchBanner, cx)
	if len(rows) == 0 {
		return 0, nil
	}
	ob, err := openOutbox()
	if err != nil {
		return 0, err
	}
	n := 0
	for _, row := range rows {
		if _, err := ob.Enqueue(row); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// guardSessionControlPointRows builds the deferred-threaded launch replies: the full
// startup banner (chunked under the text ceiling) followed by a launch-context row. Each
// carries ParentNonce=rootNonce so the drainer threads it once the root posts. Pure —
// no I/O — so it is unit-tested directly.
func guardSessionControlPointRows(rootNonce, channel, launchBanner string, cx guardSessionControlContext) []slackoutbox.Row {
	channel = strings.TrimSpace(channel)
	if channel == "" {
		channel = guardSessionsChannelDefault
	}
	rootNonce = strings.TrimSpace(rootNonce)

	var rows []slackoutbox.Row
	chunks := guardSessionChunk(strings.TrimRight(launchBanner, "\n"), guardSessionReplyTextLimit)
	for i, chunk := range chunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}
		label := "launch banner"
		if len(chunks) > 1 {
			label = fmt.Sprintf("launch banner (%d/%d)", i+1, len(chunks))
		}
		rows = append(rows, slackoutbox.Row{
			Channel:     channel,
			Text:        label + ":\n```\n" + chunk + "\n```",
			ParentNonce: rootNonce,
			Source:      guardSessionThreadSource + ":banner",
		})
	}
	rows = append(rows, slackoutbox.Row{
		Channel:     channel,
		Text:        guardSessionContextText(cx),
		ParentNonce: rootNonce,
		Source:      guardSessionThreadSource + ":context",
	})
	return rows
}

// guardSessionContextText renders the launch-context reply with FULL local paths so an
// operator can locate the artifacts on disk.
func guardSessionContextText(cx guardSessionControlContext) string {
	var b strings.Builder
	b.WriteString("launch context:\n")
	fmt.Fprintf(&b, "command: %s\n", guardSlackField(strings.Join(cx.Command, " ")))
	fmt.Fprintf(&b, "cwd: %s\n", guardSlackField(cx.Cwd))
	fmt.Fprintf(&b, "audit: %s\n", guardSlackField(cx.Audit))
	fmt.Fprintf(&b, "trace_id: %s\n", guardSlackField(cx.Trace))
	fmt.Fprintf(&b, "provider: %s\n", guardSlackField(cx.Provider))
	fmt.Fprintf(&b, "gateway: %s", guardSlackField(cx.GatewayURL))
	return b.String()
}

// guardSessionChunk splits s into pieces no longer than max bytes, breaking on line
// boundaries where possible (a single overlong line is hard-split) so a fenced banner
// stays readable across replies.
func guardSessionChunk(s string, max int) []string {
	if max <= 0 || len(s) <= max {
		if s == "" {
			return nil
		}
		return []string{s}
	}
	var chunks []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			chunks = append(chunks, cur.String())
			cur.Reset()
		}
	}
	for _, line := range strings.Split(s, "\n") {
		for len(line) > max {
			flush()
			chunks = append(chunks, line[:max])
			line = line[max:]
		}
		if cur.Len() > 0 && cur.Len()+1+len(line) > max {
			flush()
		}
		if cur.Len() > 0 {
			cur.WriteByte('\n')
		}
		cur.WriteString(line)
	}
	flush()
	return chunks
}

// guardSessionLiveLine renders the running-session status edit.
func guardSessionLiveLine(sum gateway.AdjudicationSummary, elapsed time.Duration) string {
	return fmt.Sprintf(
		"status: running · turns=%d · in=%d out=%d cached=%d · denied=%d · elapsed=%s",
		sum.Total, sum.InputTokens, sum.OutputTokens, sum.CachedPromptTokens, sum.Denied,
		guardSessionElapsed(elapsed),
	)
}

// guardSessionFinalLine renders the terminal outcome edit.
func guardSessionFinalLine(exitCode int, sum gateway.AdjudicationSummary, elapsed time.Duration) string {
	state := "completed"
	if exitCode != 0 {
		state = "failed"
	}
	return fmt.Sprintf(
		"status: %s (exit=%d) · turns=%d · in=%d out=%d cached=%d · denied=%d · elapsed=%s",
		state, exitCode, sum.Total, sum.InputTokens, sum.OutputTokens, sum.CachedPromptTokens, sum.Denied,
		guardSessionElapsed(elapsed),
	)
}

func guardSessionElapsed(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	return d.Round(time.Second).String()
}
