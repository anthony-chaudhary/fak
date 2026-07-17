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
	// Guard sessions are a dedicated delivery surface. A token that can post to the
	// scoreboard is not evidence that its bot belongs to the guard-sessions channel;
	// silently borrowing it produced a durable channel_not_found dead-letter storm.
	// Require an explicitly provisioned token and let `fak slack check` report it absent.
	if r := slackenv.Lookup(guardSessionsTokenEnv); r.Set() {
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
		":large_blue_circle: *guard session · STARTING* — session `%s` · %s/%s\nstarted %s · trace `%s` · audit `%s` · open thread for launch context and outcome",
		guardSessionShortID(nonce),
		guardSlackField(m.Agent),
		guardSlackField(m.Provider),
		started.Format(time.RFC3339),
		guardSlackField(m.TraceID),
		guardSlackAuditRef(m.AuditPath),
	)
	return slackoutbox.Row{
		Nonce:   nonce,
		Channel: channel,
		Text:    text,
		Source:  guardSessionThreadSource,
	}
}

func guardSessionShortID(nonce string) string {
	nonce = guardSlackField(nonce)
	const visible = 8
	if len(nonce) > visible {
		return nonce[:visible]
	}
	return nonce
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

func pluralWord(n int, singular, plural string) string {
	if n == 1 {
		return singular
	}
	return plural
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

// guardSessionWire keeps the production transport fixed while letting the immediate-exit
// witness point the synchronous final drain at an in-process Slack server.
var guardSessionWire = func(token string) (slackoutbox.Wire, error) {
	return outboxWire(token, "")
}

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
	sessionID string
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
	return &guardSessionCard{
		channel:   channel,
		rootNonce: rootNonce,
		sessionID: guardSessionShortID(rootNonce),
		startedAt: startedAt,
	}
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
	sent, err := c.enqueueUpdateSent(guardSessionLiveLine(c.sessionID, sum, now.Sub(c.startedAt)))
	if err != nil || !sent {
		return false // root not posted yet, or enqueue failed — retry next tick, keep gate state
	}
	c.lastKey = key
	c.lastSentAt = now
	startGuardSessionThreadDrain()
	return true
}

// enqueueReply durably appends one lifecycle checkpoint under the session root. ParentNonce
// lets the outbox bind it after Slack assigns the root ts, including when guard exits before
// the asynchronous startup drain wins its first pass.
func (c *guardSessionCard) enqueueReply(text, source string) error {
	if c == nil {
		return nil
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	ob, err := openOutbox()
	if err != nil {
		return err
	}
	_, err = ob.Enqueue(slackoutbox.Row{
		Nonce:       slackoutbox.NewNonce(),
		Channel:     c.channel,
		Text:        text,
		ParentNonce: c.rootNonce,
		Source:      guardSessionThreadSource + ":" + source,
	})
	return err
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

// finalize writes the terminal outcome as BOTH a durable reply and a root-card edit, then
// flushes synchronously: guard calls os.Exit immediately after, so a background drain would
// be cut off. The reply is queued first via ParentNonce, which preserves the final truth even
// when a very fast child failure beats the startup drain and the root has no Slack ts yet.
// After that first drain posts the root, the edit is retried so the channel-level card also
// settles on the terminal state. With no token, the outcome reply remains durable for a later
// drain even though the root edit cannot yet resolve.
func (c *guardSessionCard) finalize(text string) {
	if c == nil {
		return
	}
	if err := c.enqueueReply(text, "outcome"); err != nil {
		return
	}
	sent, updateErr := c.enqueueUpdateSent(text)
	token := resolveGuardSessionsToken()
	if token == "" {
		return
	}
	ob, err := openOutbox()
	if err != nil {
		return
	}
	wire, err := guardSessionWire(token)
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), guardSessionFinalDrainTimeout)
	defer cancel()
	drain := func() error {
		// A periodic drain kicked by the last tick may still hold the lock; retry within the
		// bounded window until it releases so the final reply/edit actually ships.
		for {
			_, derr := ob.Drain(ctx, wire, stdDrainOpts())
			if !errors.Is(derr, slackoutbox.ErrDrainBusy) {
				return derr
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(200 * time.Millisecond):
			}
		}
	}
	if updateErr != nil || drain() != nil {
		return
	}
	if sent {
		return
	}
	// The first pass posted an unposted root and its deferred outcome reply. Resolve the
	// fresh ts now, enqueue the terminal root edit, and flush once more.
	if sent, err = c.enqueueUpdateSent(text); err != nil || !sent {
		return
	}
	_ = drain()
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
	c.finalize(guardSessionFinalLine(c.sessionID, exitCode, sum, time.Since(c.startedAt)))
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

// guardSessionControlPointRows builds the deferred-threaded launch replies: a lifecycle
// checkpoint, the full startup banner (chunked under the text ceiling), and launch context.
// Each carries ParentNonce=rootNonce so the drainer threads it once the root posts. Pure —
// no I/O — so it is unit-tested directly.
func guardSessionControlPointRows(rootNonce, channel, launchBanner string, cx guardSessionControlContext) []slackoutbox.Row {
	channel = strings.TrimSpace(channel)
	if channel == "" {
		channel = guardSessionsChannelDefault
	}
	rootNonce = strings.TrimSpace(rootNonce)

	rows := []slackoutbox.Row{{
		Channel:     channel,
		Text:        "lifecycle: launch prepared · waiting for child process · progress stays on the root card · terminal outcome will reply here",
		ParentNonce: rootNonce,
		Source:      guardSessionThreadSource + ":progress",
	}}
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

// guardSessionLiveLine renders the running-session status edit while preserving the stable
// session identity that lets an operator scan successive cards without opening each thread.
func guardSessionLiveLine(sessionID string, sum gateway.AdjudicationSummary, elapsed time.Duration) string {
	return fmt.Sprintf(
		":large_blue_circle: *guard session · RUNNING* — session `%s` · turns=%d · in=%d out=%d cached=%d · denied=%d · elapsed=%s",
		guardSessionShortID(sessionID), sum.Total, sum.InputTokens, sum.OutputTokens, sum.CachedPromptTokens, sum.Denied,
		guardSessionElapsed(elapsed),
	)
}

// guardSessionFinalLine renders the terminal outcome edit.
func guardSessionFinalLine(sessionID string, exitCode int, sum gateway.AdjudicationSummary, elapsed time.Duration) string {
	state := "completed"
	icon := ":white_check_mark:"
	if exitCode != 0 {
		state = "failed"
		icon = ":red_circle:"
	}
	return fmt.Sprintf(
		"%s *guard session · %s* — session `%s` · exit=%d · turns=%d · in=%d out=%d cached=%d · denied=%d · elapsed=%s",
		icon, strings.ToUpper(state), guardSessionShortID(sessionID), exitCode, sum.Total, sum.InputTokens, sum.OutputTokens, sum.CachedPromptTokens, sum.Denied,
		guardSessionElapsed(elapsed),
	)
}

func guardSessionElapsed(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	return d.Round(time.Second).String()
}
