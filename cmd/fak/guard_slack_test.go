package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestGuardSessionAgentNameNormalizesLauncher(t *testing.T) {
	cases := map[string]string{
		`C:\Program Files\Claude\claude.exe`: "claude.exe",
		"/usr/local/bin/codex":               "codex",
		"opencode":                           "opencode",
		"":                                   "unknown",
	}
	for in, want := range cases {
		if got := guardSessionAgentName([]string{in}); got != want {
			t.Fatalf("guardSessionAgentName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := guardSessionAgentName(nil); got != "unknown" {
		t.Fatalf("guardSessionAgentName(nil) = %q, want unknown", got)
	}
}

func TestGuardSessionThreadRowIsRootPost(t *testing.T) {
	row := guardSessionThreadRow(guardSessionThreadMeta{
		Channel:   "CCHAN",
		Nonce:     "nonce-1",
		TraceID:   "trace one\ntrace two",
		Agent:     "claude",
		Provider:  "anthropic",
		AuditPath: "audit.jsonl",
		StartedAt: time.Unix(1000, 0),
		PID:       123,
	})
	if row.Channel != "CCHAN" || row.Nonce != "nonce-1" || row.Source != guardSessionThreadSource {
		t.Fatalf("row identity = %+v", row)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("guard session row must be a root post, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}
	for _, want := range []string{
		"guard session · STARTING",
		"session `nonce-1`",
		"claude/anthropic",
		"started 1970-01-01T00:16:40Z",
		"trace `trace one trace two`",
		"audit `audit.jsonl`",
		"open thread for launch context and outcome",
	} {
		if !strings.Contains(row.Text, want) {
			t.Fatalf("row text missing %q:\n%s", want, row.Text)
		}
	}
}

func TestEnqueueGuardSessionThreadDefaultsToDedicatedChannel(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)

	row, err := enqueueGuardSessionThread("trace-a", "anthropic", []string{"claude"}, "audit.jsonl", time.Unix(2000, 0))
	if err != nil {
		t.Fatalf("enqueueGuardSessionThread: %v", err)
	}
	if row.Channel != guardSessionsChannelDefault {
		t.Fatalf("channel = %q, want %q", row.Channel, guardSessionsChannelDefault)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("enqueued guard session must create a root thread, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 {
		t.Fatalf("spool rows = %d, want 1", len(snap.Rows))
	}
	got := snap.Rows[0]
	if got.Channel != guardSessionsChannelDefault || got.Source != guardSessionThreadSource || got.ThreadTS != "" {
		t.Fatalf("spooled row = %+v", got)
	}
	if !strings.Contains(got.Text, "guard session · STARTING") ||
		!strings.Contains(got.Text, "claude/anthropic") ||
		!strings.Contains(got.Text, "trace `trace-a`") {
		t.Fatalf("spooled row text missing guard identity:\n%s", got.Text)
	}
}

func TestGuardSessionControlPointRowsThreadUnderRoot(t *testing.T) {
	rows := guardSessionControlPointRows("root-nonce", "CCHAN", "line one\nline two", guardSessionControlContext{
		Command:    []string{"claude", "--flag"},
		Cwd:        `C:\work\fak`,
		Audit:      `C:\work\fak\.dispatch-runs\audit\session.jsonl`,
		Trace:      "trace-x",
		Provider:   "anthropic",
		GatewayURL: "http://127.0.0.1:8080",
	})
	if len(rows) != 3 {
		t.Fatalf("want 1 progress + 1 banner + 1 context reply, got %d rows", len(rows))
	}
	for _, r := range rows {
		if r.ParentNonce != "root-nonce" {
			t.Fatalf("reply not threaded under the root: parent_nonce=%q", r.ParentNonce)
		}
		if r.Channel != "CCHAN" || r.ThreadTS != "" || r.UpdateTS != "" {
			t.Fatalf("reply must be a deferred post: %+v", r)
		}
	}
	progress, banner, context := rows[0], rows[1], rows[2]
	if progress.Source != guardSessionThreadSource+":progress" || !strings.Contains(progress.Text, "launch prepared") {
		t.Fatalf("progress reply wrong: %+v", progress)
	}
	if banner.Source != guardSessionThreadSource+":banner" || !strings.Contains(banner.Text, "line one\nline two") {
		t.Fatalf("banner reply wrong: %+v", banner)
	}
	if context.Source != guardSessionThreadSource+":context" {
		t.Fatalf("context reply source = %q", context.Source)
	}
	// Full paths (not basenames) so an operator can locate artifacts on disk.
	for _, want := range []string{`command: claude --flag`, `cwd: C:\work\fak`, `audit: C:\work\fak\.dispatch-runs\audit\session.jsonl`, "gateway: http://127.0.0.1:8080"} {
		if !strings.Contains(context.Text, want) {
			t.Fatalf("context reply missing %q:\n%s", want, context.Text)
		}
	}
}

func TestGuardSessionControlPointRowsChunkLongBanner(t *testing.T) {
	// A banner well over the reply ceiling splits into multiple labeled banner replies.
	big := strings.Repeat("x", guardSessionReplyTextLimit*2+10)
	rows := guardSessionControlPointRows("root", "CCHAN", big, guardSessionControlContext{})
	banners := 0
	for _, r := range rows {
		if r.Source == guardSessionThreadSource+":banner" {
			banners++
			if !strings.Contains(r.Text, "launch banner (") {
				t.Fatalf("multi-chunk banner not labeled: %q", r.Text[:40])
			}
		}
	}
	if banners < 2 {
		t.Fatalf("long banner not chunked: %d banner replies", banners)
	}
}

func TestGuardSessionControlPointRowsEmptyBannerStillPostsContext(t *testing.T) {
	rows := guardSessionControlPointRows("root", "CCHAN", "", guardSessionControlContext{Trace: "t"})
	if len(rows) != 2 || rows[0].Source != guardSessionThreadSource+":progress" || rows[1].Source != guardSessionThreadSource+":context" {
		t.Fatalf("empty banner should leave progress + context replies, got %d rows", len(rows))
	}
}

type fakeGuardMetrics struct{ sum gateway.AdjudicationSummary }

func (f fakeGuardMetrics) AdjudicationSummary() gateway.AdjudicationSummary { return f.sum }

func TestEnqueueGuardSessionControlPointsSpoolsThreadedReplies(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	n, err := enqueueGuardSessionControlPoints("root-1", "C1", "banner text", guardSessionControlContext{
		Trace:   "t",
		Command: []string{"claude"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("queued %d replies, want 3 (progress + banner + context)", n)
	}
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 3 {
		t.Fatalf("spool rows = %d, want 3", len(snap.Rows))
	}
	for _, r := range snap.Rows {
		if r.ParentNonce != "root-1" {
			t.Fatalf("reply not deferred-threaded under the root: %+v", r)
		}
	}
}

// TestRenderGuardStartupReportWiresTheSessionThread is the regression witness for the
// live operator defect: the render path used to enqueue ONLY the root even though all
// control-point/card primitives already existed. A real guard therefore produced a
// header-only channel. The production startup path must now bind the card handle and
// spool banner/context replies under the root nonce in one transaction-shaped pass.
func TestRenderGuardStartupReportWiresTheSessionThread(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	guardSessionCardHandle = nil
	t.Cleanup(func() {
		guardSessionCardHandle.stopUpdater()
		guardSessionCardHandle = nil
	})

	report := renderGuardStartupReport(guardStartupView{
		up:           "anthropic",
		command:      []string{"claude", "-p", "audit"},
		gwURL:        "http://127.0.0.1:8080",
		floorSource:  "test floor",
		logLabel:     "off",
		auditLabel:   "audit.jsonl",
		guardTraceID: "trace-1",
	})
	if guardSessionCardHandle == nil {
		t.Fatal("startup left the session card handle nil")
	}
	if !strings.Contains(report, "queued 3 control replies") {
		t.Fatalf("startup did not report the banner/context replies:\n%s", report)
	}

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 4 {
		t.Fatalf("startup spooled %d rows, want root + progress + banner + context", len(snap.Rows))
	}
	rootNonce := guardSessionCardHandle.rootNonce
	roots, replies := 0, 0
	for _, row := range snap.Rows {
		switch {
		case row.Nonce == rootNonce && row.ParentNonce == "":
			roots++
		case row.ParentNonce == rootNonce:
			replies++
		}
	}
	if roots != 1 || replies != 3 {
		t.Fatalf("thread shape roots/replies = %d/%d, want 1/3", roots, replies)
	}
}

func TestGuardSessionCardUpdaterStartsAndStopsCleanly(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	card.startUpdater(fakeGuardMetrics{})
	card.stopUpdater()
	card.stopUpdater() // second stop is a safe no-op
	// The 20s tick never fires within the test, and the root is unposted anyway, so no
	// update row may have been enqueued.
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			t.Fatalf("updater enqueued against an unposted root: %+v", r)
		}
	}
}

func TestGuardSessionCardFinalizeWithoutTokenStaysDurable(t *testing.T) {
	clearSlackEnv(t) // no ambient token — finalize enqueues but does not drain
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	card.finalize("status: completed (exit=0) · turns=1")
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range snap.Rows {
		if r.UpdateTS == "1.23" && strings.Contains(r.Text, "completed") {
			found = true
		}
	}
	if !found {
		t.Fatal("final outcome edit was not enqueued durably")
	}
}

// TestGuardSessionCardFinalizeBoundsUndeliveredSpool witnesses the #5354 leak fence: on a box
// with no Slack token the session cards can never drain, so finalize bounds the spool by
// dropping UNDELIVERED cards older than guardOutboxMaxPendingAge — while keeping the fresh
// outcome truth it just recorded. Without the fence the spool grows without limit (12k rows
// on the live tree).
func TestGuardSessionCardFinalizeBoundsUndeliveredSpool(t *testing.T) {
	clearSlackEnv(t) // no ambient token — finalize bounds instead of draining
	outboxTestDir(t)

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	// An ancient undelivered status card (well past the 72h floor) plus the live root.
	oldNonce, err := ob.Enqueue(slackoutbox.Row{
		Channel:    "C1",
		Text:       ":large_blue_circle: *guard session · STARTING* — ancient",
		Source:     guardSessionThreadSource,
		EnqueuedAt: time.Now().Add(-10 * 24 * time.Hour).UTC().Format(time.RFC3339),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-live"}); err != nil {
		t.Fatal(err)
	}

	card := newGuardSessionCard("C1", "root-live", time.Now())
	card.finalize("status: completed (exit=0) · turns=1")

	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if hasRowNonce(snap, oldNonce) {
		t.Fatal("finalize must drop an undelivered card older than the pending-age floor")
	}
	if !hasRowNonce(snap, "root-live") {
		t.Fatal("the fresh root card must survive the bounding pass")
	}
	outcome := false
	for _, r := range snap.Rows {
		if r.ParentNonce == "root-live" && r.Source == guardSessionThreadSource+":outcome" {
			outcome = true
		}
	}
	if !outcome {
		t.Fatal("the fresh outcome reply finalize just enqueued must survive the bounding pass")
	}
}

// hasRowNonce reports whether the snapshot still carries a spool row with the given nonce.
func hasRowNonce(snap *slackoutbox.Snapshot, nonce string) bool {
	for _, r := range snap.Rows {
		if r.Nonce == nonce {
			return true
		}
	}
	return false
}

func TestGuardSessionLiveAndFinalLines(t *testing.T) {
	sum := gateway.AdjudicationSummary{Total: 7, InputTokens: 120, OutputTokens: 40, CachedPromptTokens: 90, Denied: 2}
	live := guardSessionLiveLine("session-123", sum, 65*time.Second)
	for _, want := range []string{"guard session · RUNNING", "session `session-`", "turns=7", "in=120 out=40 cached=90", "denied=2", "elapsed=1m5s"} {
		if !strings.Contains(live, want) {
			t.Fatalf("live line missing %q: %s", want, live)
		}
	}
	if got := guardSessionFinalLine("session-123", 0, sum, time.Second); !strings.Contains(got, "guard session · COMPLETED") || !strings.Contains(got, "exit=0") {
		t.Fatalf("final(0) = %s", got)
	}
	if got := guardSessionFinalLine("session-123", 2, sum, time.Second); !strings.Contains(got, "guard session · FAILED") || !strings.Contains(got, "exit=2") {
		t.Fatalf("final(2) = %s", got)
	}
}

// TestGuardSessionCardFinalizeWinsImmediateExitRace is the captured fast-failure witness:
// finalize runs while the root is still pending, yet the synchronous two-pass drain must
// post the root, thread the terminal outcome, then edit the root to the same final state.
func TestGuardSessionCardFinalizeWinsImmediateExitRace(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	t.Setenv(guardSessionsTokenEnv, "xoxb-test")
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()
	previousWire := guardSessionWire
	guardSessionWire = func(token string) (slackoutbox.Wire, error) {
		return outboxWire(token, srv.URL+"/")
	}
	t.Cleanup(func() { guardSessionWire = previousWire })

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "starting", Nonce: "root-immediate"}); err != nil {
		t.Fatal(err)
	}
	card := newGuardSessionCard("C1", "root-immediate", time.Now())
	card.finalize(guardSessionFinalLine(card.sessionID, 2, gateway.AdjudicationSummary{}, time.Second))

	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got := snap.PostedTS("root-immediate"); got != "1.23" {
		t.Fatalf("root posted ts = %q, want 1.23", got)
	}
	outcome, finalEdit := false, false
	for _, row := range snap.Rows {
		if row.ParentNonce == "root-immediate" && row.Source == guardSessionThreadSource+":outcome" && strings.Contains(row.Text, "FAILED") {
			outcome = true
		}
		if row.UpdateTS == "1.23" && strings.Contains(row.Text, "FAILED") {
			finalEdit = true
		}
	}
	if !outcome || !finalEdit {
		t.Fatalf("immediate exit did not preserve reply/edit: outcome=%v final_edit=%v rows=%+v", outcome, finalEdit, snap.Rows)
	}
	if posts < 3 {
		t.Fatalf("Slack calls = %d, want root + outcome + final edit", posts)
	}
}

// countCardUpdateRows returns how many chat.update rows the card has spooled (rows with an
// UpdateTS set) — i.e. how many real chat.update API calls the drainer will make for it.
func countCardUpdateRows(t *testing.T, ob *slackoutbox.Outbox) int {
	t.Helper()
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			n++
		}
	}
	return n
}

// postedRootCard sets up an outbox with a guard-session root drained to posted (ts "1.23"),
// and returns the outbox plus a card bound to it, ready for tick() exercises.
func postedRootCard(t *testing.T, startedAt time.Time) (*slackoutbox.Outbox, *guardSessionCard) {
	t.Helper()
	clearSlackEnv(t)
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	t.Cleanup(srv.Close)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}
	return ob, newGuardSessionCard("C1", "root-1", startedAt)
}

// TestGuardSessionCardTickGatesIdleUpdates is the before/after proof: an IDLE session (its
// adjudication fold frozen) used to enqueue a byte-different chat.update on every 20s tick
// forever, because the live line embeds an elapsed clock. The change-gate collapses that to
// one edit per keepalive window. Over 30 minutes of 20s ticks (90 ticks) an idle card now
// spends 6 chat.update calls (the keepalives) instead of 90 — a 15x cut against the fleet's
// dominant Slack rate-limit drain.
func TestGuardSessionCardTickGatesIdleUpdates(t *testing.T) {
	t0 := time.Unix(1_000_000, 0)
	ob, card := postedRootCard(t, t0)
	idle := gateway.AdjudicationSummary{Total: 3, InputTokens: 120, OutputTokens: 40, CachedPromptTokens: 90, Denied: 0}
	card.srv = fakeGuardMetrics{sum: idle}

	sends := 0
	const ticks = 90 // 30 minutes at the 20s cadence
	for i := 0; i < ticks; i++ {
		if card.tick(t0.Add(time.Duration(i) * guardSessionCardUpdateInterval)) {
			sends++
		}
	}
	// First tick (new info) + one keepalive per 5m window over 30m = 1 + 5 more = 6.
	wantKeepalives := 1 + int((time.Duration(ticks-1)*guardSessionCardUpdateInterval)/guardSessionCardKeepaliveInterval)
	if sends != wantKeepalives {
		t.Fatalf("idle session enqueued %d edits over %d ticks; want %d (first + keepalives)", sends, ticks, wantKeepalives)
	}
	if got := countCardUpdateRows(t, ob); got != sends {
		t.Fatalf("spooled %d update rows, want %d (one per gated send)", got, sends)
	}
	if sends >= ticks {
		t.Fatalf("gate did not reduce calls: %d sends over %d ticks", sends, ticks)
	}
}

// TestGuardSessionCardTickShipsEveryRealChange proves the gate never costs liveness: when
// the substantive fold moves on every tick, every tick still ships its edit.
func TestGuardSessionCardTickShipsEveryRealChange(t *testing.T) {
	t0 := time.Unix(2_000_000, 0)
	ob, card := postedRootCard(t, t0)

	sends := 0
	const ticks = 10
	for i := 0; i < ticks; i++ {
		// Each tick advances turns/tokens — real progress the reader must see.
		card.srv = fakeGuardMetrics{sum: gateway.AdjudicationSummary{Total: uint64(i + 1), InputTokens: uint64(100 * (i + 1))}}
		if card.tick(t0.Add(time.Duration(i) * guardSessionCardUpdateInterval)) {
			sends++
		}
	}
	if sends != ticks {
		t.Fatalf("changing session shipped %d/%d edits; a real change must never be gated", sends, ticks)
	}
	if got := countCardUpdateRows(t, ob); got != ticks {
		t.Fatalf("spooled %d update rows, want %d", got, ticks)
	}
}

// TestGuardSessionCardKeyIgnoresElapsed pins the gate's contract: the key folds the
// substantive fields and is INDEPENDENT of the elapsed clock, so a clock-only difference
// never counts as new information worth a chat.update.
func TestGuardSessionCardKeyIgnoresElapsed(t *testing.T) {
	sum := gateway.AdjudicationSummary{Total: 5, InputTokens: 10, OutputTokens: 20, CachedPromptTokens: 15, Denied: 1}
	if guardSessionCardKey(sum) != guardSessionCardKey(sum) {
		t.Fatal("key is not stable for an unchanged summary")
	}
	moved := sum
	moved.Total++
	if guardSessionCardKey(sum) == guardSessionCardKey(moved) {
		t.Fatal("key did not change when turns advanced")
	}
}

func TestGuardSessionCardEnqueueUpdateSkipsUnpostedRoot(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	if err := card.enqueueUpdate("progress"); err != nil {
		t.Fatalf("enqueueUpdate: %v", err)
	}
	// The root has not posted, so no update row may be enqueued (no card ts to edit).
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			t.Fatalf("update row enqueued against an unposted root: %+v", r)
		}
	}
	if len(snap.Rows) != 1 {
		t.Fatalf("spool rows = %d, want 1 (root only)", len(snap.Rows))
	}
}

func TestGuardSessionCardEnqueueUpdateEditsPostedRoot(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	// Drain the root so it reaches posted state (ts "1.23" from okSlackServer).
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}

	card := newGuardSessionCard("C1", "root-1", time.Now())
	if err := card.enqueueUpdate("progress a"); err != nil {
		t.Fatal(err)
	}
	if err := card.enqueueUpdate("progress b"); err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	updates := 0
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			updates++
			if r.UpdateTS != "1.23" {
				t.Fatalf("update row not bound to the root ts: %+v", r)
			}
			if r.Source != guardSessionThreadSource+":status" {
				t.Fatalf("update row source = %q", r.Source)
			}
		}
	}
	if updates != 2 {
		t.Fatalf("want 2 update rows (coalesced at drain), got %d", updates)
	}
}
