package slackoutbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// Ephemeral reaping keeps the dgx-bridge Slack channels clean. A fleet's session/run cards
// and status posts pile up — the transcript floods with stale echoes the design note warns
// about (SLACK-CONTROL-FOUNDATION-2026-07-02.md). The reaper deletes a posted message once
// it has been IDLE past its TTL, measured from its last activity (the newest posted
// transition across the rows that share its ts), so a live card being updated every tick
// stays visible and only clears once the run goes quiet.
//
// Scope is opt-in by channel: a message is reaped only when its channel is on the ephemeral
// allowlist (the caller resolves it from FAK_SLACK_EPHEMERAL_CHANNELS) or a producer set an
// explicit per-row DeleteAfterS. Everything else is left forever — the reaper never touches a
// channel nobody opted in. Deletion is done by the ONE transport's chat.delete (a bot may
// delete only its own posts, which every outbox row is), recorded as the terminal `reaped`
// state, and is idempotent: a message already gone (message_not_found) is recorded reaped
// without a re-probe. It runs automatically at the tail of every Drain and on demand via
// `fak slack outbox reap`.

// DefaultReapTTL is the idle window an ephemeral (bridge-channel) message survives before the
// reaper deletes it, measured from its last activity. Thirty minutes keeps a live session
// card visible while it is being updated and clears the channel shortly after a run goes quiet.
const DefaultReapTTL = 30 * time.Minute

// DefaultReapMaxDeletes bounds how many messages one reap pass deletes, so first enabling the
// reaper on a channel with a long backlog does not hold drain.lock for minutes — the rest are
// reaped on the following drains as the fleet keeps ticking.
const DefaultReapMaxDeletes = 50

// ReapOpts configures one reap pass. Zero values take the documented defaults.
type ReapOpts struct {
	// Ephemeral reports whether a channel auto-expires its posted messages by default. nil
	// means no channel default — only rows with an explicit DeleteAfterS are then reaped.
	Ephemeral func(channel string) bool
	// TTL is the idle window an ephemeral-channel message survives; <=0 => DefaultReapTTL. A
	// row's own DeleteAfterS overrides this for that one message.
	TTL time.Duration
	// MaxDeletes bounds deletes per pass; <=0 => DefaultReapMaxDeletes.
	MaxDeletes int
	// Now is the reference clock; zero => o.now().
	Now time.Time
	// Gap paces consecutive chat.delete calls into the SAME channel (default 1s).
	Gap time.Duration
	// Sleep is the pacing wait, injectable so tests witness gaps. It must honor ctx. Default: timer.
	Sleep func(ctx context.Context, d time.Duration) error
	// DryRun computes what would be reaped without deleting or recording anything.
	DryRun bool

	held bool // internal: the caller already holds drain.lock (the in-drain auto path)
}

func (r ReapOpts) norm(o *Outbox) ReapOpts {
	if r.TTL <= 0 {
		r.TTL = DefaultReapTTL
	}
	if r.MaxDeletes <= 0 {
		r.MaxDeletes = DefaultReapMaxDeletes
	}
	if r.Gap <= 0 {
		r.Gap = time.Second
	}
	if r.Sleep == nil {
		r.Sleep = ctxSleep
	}
	if r.Now.IsZero() {
		r.Now = o.now()
	}
	return r
}

// ReapReport is what one reap pass did (or, for a dry run, would do).
type ReapReport struct {
	Scanned  int  `json:"scanned"`  // distinct posted messages (channel+ts) examined
	Eligible int  `json:"eligible"` // messages idle past their TTL (the reap set)
	Deleted  int  `json:"deleted"`  // chat.delete calls that removed a message
	Gone     int  `json:"gone"`     // messages already absent (message_not_found) — recorded reaped anyway
	Failed   int  `json:"failed"`   // transient delete failures — left posted, retried next pass
	DryRun   bool `json:"dry_run,omitempty"`
}

// Reap deletes posted messages that are idle past their ephemeral TTL, holding drain.lock for
// the pass (unless the caller already holds it). It returns ErrDrainBusy when another holder
// owns the lock. A dry run reports the eligible set without deleting or recording anything.
func (o *Outbox) Reap(ctx context.Context, w Wire, opts ReapOpts) (*ReapReport, error) {
	opts = opts.norm(o)
	if !opts.held {
		lock, err := os.OpenFile(filepath.Join(o.dir, lockFile), os.O_CREATE|os.O_RDWR, 0o644)
		if err != nil {
			return nil, err
		}
		defer lock.Close()
		if err := flock.TryLock(lock); err != nil {
			if errors.Is(err, flock.ErrLockBusy) {
				return nil, ErrDrainBusy
			}
			return nil, err
		}
		defer func() { _ = flock.Unlock(lock) }()
	}
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	return o.reapSnap(ctx, w, snap, opts)
}

// reapMsg is one posted message (a channel+ts) folded across every spool row that shares it —
// a plain post is one row; a run card is its post plus each in-place update.
type reapMsg struct {
	channel  string
	ts       string
	newestAt time.Time     // last activity: newest posted-transition timestamp across the rows
	ttl      time.Duration // effective idle window; 0 means no row opted this message in
	nonces   []string      // every spool nonce whose posted ts is this message
}

// reapSnap runs one reap pass over an already-loaded snapshot. The caller owns lock discipline
// (Drain passes held=true and its own pre-pass snapshot; Reap loads fresh under the lock). It
// is best-effort on ctx cancellation — it stops and returns the partial report — and only a
// state-append (disk) fault returns a non-nil error.
func (o *Outbox) reapSnap(ctx context.Context, w Wire, snap *Snapshot, opts ReapOpts) (*ReapReport, error) {
	opts = opts.norm(o)
	rep := &ReapReport{DryRun: opts.DryRun}

	// Fold posted rows into per-message groups keyed by channel+ts. A message is reapable
	// when any of its rows opts in (an explicit DeleteAfterS, or an ephemeral channel); its
	// TTL is the SHORTEST such window (most-eager wins) and it expires TTL after last activity.
	byMsg := map[string]*reapMsg{}
	var order []string
	for _, r := range snap.Rows {
		s := snap.state(r.Nonce)
		if s.State != statePosted || s.TS == "" {
			continue
		}
		key := r.Channel + "\x00" + s.TS
		m := byMsg[key]
		if m == nil {
			m = &reapMsg{channel: r.Channel, ts: s.TS}
			byMsg[key] = m
			order = append(order, key)
		}
		m.nonces = append(m.nonces, r.Nonce)
		if s.At.After(m.newestAt) {
			m.newestAt = s.At
		}
		if ttl, ok := rowTTL(r, opts); ok {
			if m.ttl == 0 || ttl < m.ttl {
				m.ttl = ttl
			}
		}
	}
	sort.Strings(order) // deterministic order: channel then ts (key is channel\x00ts)

	sentInChannel := map[string]bool{}
	for _, key := range order {
		m := byMsg[key]
		rep.Scanned++
		if m.ttl == 0 { // no row opted this message into reaping
			continue
		}
		if m.newestAt.IsZero() || opts.Now.Sub(m.newestAt) < m.ttl {
			continue // still within its idle window (or an unparseable timestamp — never reap on a guess)
		}
		rep.Eligible++
		if opts.DryRun {
			continue
		}
		if rep.Deleted+rep.Gone >= opts.MaxDeletes {
			break // bounded per pass; the backlog clears over the next drains
		}
		if ctx.Err() != nil {
			break // cancelled — stop reaping, best-effort
		}
		// Pace consecutive deletes into the same channel (chat.delete is Tier 3, but the
		// per-channel courtesy keeps a burst from tripping a 429 the wire would then serialize).
		if sentInChannel[m.channel] {
			if err := opts.Sleep(ctx, opts.Gap); err != nil {
				break // ctx cancelled during the pacing wait
			}
		}
		sentInChannel[m.channel] = true

		reason := "reaped: idle past " + m.ttl.String()
		switch derr := w.DeleteMessage(ctx, m.channel, m.ts); {
		case derr == nil:
			rep.Deleted++
		case isGone(derr):
			// Already absent — the goal state holds; record it reaped so we never probe again.
			reason = "reaped: already absent"
			rep.Gone++
		default:
			rep.Failed++
			continue // transient — leave it posted, retry next pass
		}
		for _, nonce := range m.nonces {
			if err := o.appendState(transition{Nonce: nonce, State: stateReaped, TS: m.ts, Reason: reason}); err != nil {
				return rep, err
			}
		}
	}
	return rep, nil
}

// rowTTL returns the effective ephemeral TTL for one row and whether it opts into reaping at
// all: an explicit per-row DeleteAfterS wins; otherwise an ephemeral channel takes opts.TTL; a
// row in no ephemeral channel with no explicit TTL is never reaped.
func rowTTL(r Row, opts ReapOpts) (time.Duration, bool) {
	if r.DeleteAfterS > 0 {
		return time.Duration(r.DeleteAfterS) * time.Second, true
	}
	if opts.Ephemeral != nil && opts.Ephemeral(r.Channel) {
		return opts.TTL, true
	}
	return 0, false
}

// isGone reports whether a delete error is Slack's "the message is already gone" answer —
// message_not_found — which the reaper treats as success (the channel is already clean).
func isGone(err error) bool {
	var apiErr *slackwire.APIError
	if errors.As(err, &apiErr) {
		return apiErr.Code == "message_not_found"
	}
	return false
}

// ReapReportLine renders one reap report as a single human line (the CLI's non-JSON output).
func ReapReportLine(rep *ReapReport) string {
	verb := "reaped"
	if rep.DryRun {
		verb = "would reap"
	}
	return fmt.Sprintf("%s: scanned %d  eligible %d  deleted %d  already-gone %d  failed %d",
		verb, rep.Scanned, rep.Eligible, rep.Deleted, rep.Gone, rep.Failed)
}
