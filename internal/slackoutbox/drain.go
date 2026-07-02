package slackoutbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// Wire is the transport slice the drainer needs; *slackwire.Client satisfies it, a test
// injects an in-memory fake.
type Wire interface {
	PostMessageIdem(ctx context.Context, channel, text string, blocks []any, threadTS, nonce string) (string, error)
	UpdateMessage(ctx context.Context, channel, ts, text string, blocks []any) error
	History(ctx context.Context, channel, oldestTS string, limit int) ([]slackwire.Message, error)
}

var _ Wire = (*slackwire.Client)(nil)

// DrainOpts configures one drain pass. Zero values take the documented defaults.
type DrainOpts struct {
	// Root locates the leak fence's optional private-needle sidecar (the repo root).
	// "" scans with the base needles/shapes only.
	Root string
	// MaxAttempts dead-letters a row after this many failed sends (default 5).
	MaxAttempts int
	// Gap paces consecutive sends into the SAME channel (default 1s — chat.postMessage's
	// ~1 msg/s special tier; the wire already honors Retry-After inside a call).
	Gap time.Duration
	// Sleep is the pacing wait, injectable so tests witness gaps instead of serving
	// them. It must honor ctx cancellation. Defaults to a timer wait.
	Sleep func(ctx context.Context, d time.Duration) error
	// HistoryProbeLimit bounds the ambiguity probe's history read (default 100).
	HistoryProbeLimit int
}

func (d DrainOpts) norm() DrainOpts {
	if d.MaxAttempts <= 0 {
		d.MaxAttempts = 5
	}
	if d.Gap <= 0 {
		d.Gap = time.Second
	}
	if d.Sleep == nil {
		d.Sleep = ctxSleep
	}
	if d.HistoryProbeLimit <= 0 {
		d.HistoryProbeLimit = 100
	}
	return d
}

func ctxSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// ErrDrainBusy means another drainer holds the lock — the caller should simply not
// drain (the holder is doing the work); it is not a failure of the outbox.
var ErrDrainBusy = errors.New("slackoutbox: another drainer holds the lock")

// DrainReport is what one pass did, for the verb's human/JSON output.
type DrainReport struct {
	Posted     int `json:"posted"`
	Updated    int `json:"updated"`
	Recovered  int `json:"recovered"` // resolved from the nonce probe without re-sending
	Refused    int `json:"refused"`
	Superseded int `json:"superseded"`
	Failed     int `json:"failed"` // transient failures recorded this pass (still pending)
	Dead       int `json:"dead"`
	Remaining  int `json:"remaining"` // rows still owed after this pass
}

// PlanItem is one send the next drain pass would perform, in order.
type PlanItem struct {
	Row        Row      `json:"row"`
	Action     string   `json:"action"`                // "post" | "update"
	Supersedes []string `json:"supersedes,omitempty"`  // update nonces this row coalesced away
	Attempts   int      `json:"attempts,omitempty"`    // failed attempts so far
	NeedsProbe bool     `json:"needs_probe,omitempty"` // ambiguous last attempt — history probe first
}

// Plan folds the snapshot into the ordered per-channel send plan WITHOUT side effects —
// the `--dry-run` view, and the first half of Drain. Channels are ordered
// lexicographically for determinism; rows within a channel keep enqueue (FIFO) order.
// Update rows sharing a CardKey coalesce to the newest; the older nonces ride along in
// Supersedes so Drain can record them.
func (o *Outbox) Plan() ([]PlanItem, *Snapshot, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, nil, err
	}

	byChannel := map[string][]Row{}
	var channels []string
	for _, r := range snap.Rows {
		if snap.state(r.Nonce).terminal() {
			continue
		}
		if _, ok := byChannel[r.Channel]; !ok {
			channels = append(channels, r.Channel)
		}
		byChannel[r.Channel] = append(byChannel[r.Channel], r)
	}
	sort.Strings(channels)

	var plan []PlanItem
	for _, ch := range channels {
		rows := byChannel[ch]
		// Coalesce updates: for each card key keep only the LAST pending update row.
		winner := map[string]int{} // card key -> index in rows of the newest update
		for i, r := range rows {
			if r.UpdateTS != "" {
				winner[r.CardKey] = i
			}
		}
		superseded := map[int][]string{} // winner index -> losing nonces, in order
		for i, r := range rows {
			if r.UpdateTS != "" && winner[r.CardKey] != i {
				w := winner[r.CardKey]
				superseded[w] = append(superseded[w], r.Nonce)
			}
		}
		for i, r := range rows {
			if r.UpdateTS != "" && winner[r.CardKey] != i {
				continue // coalesced away; recorded via the winner's Supersedes
			}
			action := "post"
			if r.UpdateTS != "" {
				action = "update"
			}
			rs := snap.state(r.Nonce)
			plan = append(plan, PlanItem{
				Row: r, Action: action, Supersedes: superseded[i],
				Attempts:   rs.Attempts,
				NeedsProbe: r.UpdateTS == "" && (rs.State == stateSending || (rs.State == stateFailed && rs.Ambiguous)),
			})
		}
	}
	return plan, snap, nil
}

// Drain runs ONE serialized pass over the pending queue: coalesce, fence, probe, send,
// record. Transient failures stop the failing CHANNEL (FIFO order is part of the card
// contract — later messages never jump a stuck head) and the pass continues with other
// channels; the retry is simply the next drain. It returns ErrDrainBusy when another
// drainer holds the lock.
func (o *Outbox) Drain(ctx context.Context, w Wire, opts DrainOpts) (*DrainReport, error) {
	opts = opts.norm()

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

	plan, _, err := o.Plan()
	if err != nil {
		return nil, err
	}

	rep := &DrainReport{}
	stopped := map[string]bool{} // channel -> stop sending this pass (transient failure at head)
	sentInChannel := map[string]bool{}
	for _, item := range plan {
		ch := item.Row.Channel
		if stopped[ch] {
			rep.Remaining++
			continue
		}
		if err := ctx.Err(); err != nil {
			rep.Remaining++
			continue
		}

		// Coalesced-away updates are recorded first so a crash mid-pass never
		// resurrects a stale card state.
		for _, n := range item.Supersedes {
			if err := o.appendState(transition{Nonce: n, State: stateSuperseded, Reason: "coalesced into " + item.Row.Nonce}); err != nil {
				return rep, err
			}
			rep.Superseded++
		}

		// Leak fence — refusal is terminal and carries the finding as its reason.
		if findings := hooks.ScanOutboundText(fencePayload(item.Row), opts.Root); len(findings) > 0 {
			f := findings[0]
			reason := fmt.Sprintf("%s: %s (line %d; %d finding(s))", f.Gate, f.Detail, f.Line, len(findings))
			if err := o.appendState(transition{Nonce: item.Row.Nonce, State: stateRefused, Reason: reason}); err != nil {
				return rep, err
			}
			rep.Refused++
			continue
		}

		// Ambiguity probe: a prior attempt may have landed (crash between post and
		// record, or a transport error after the request left). The nonce rode in
		// message metadata, so recent history answers without a re-send.
		if item.NeedsProbe {
			ts, found, perr := o.probe(ctx, w, item.Row, opts.HistoryProbeLimit)
			if perr != nil {
				// The probe itself failing is a transient channel condition.
				if err := o.recordFailure(item, "probe: "+perr.Error(), false, opts, rep); err != nil {
					return rep, err
				}
				stopped[ch] = true
				continue
			}
			if found {
				if err := o.appendState(transition{Nonce: item.Row.Nonce, State: statePosted, TS: ts, Reason: "recovered from nonce probe"}); err != nil {
					return rep, err
				}
				rep.Recovered++
				continue
			}
		}

		// Pace consecutive sends into the same channel.
		if sentInChannel[ch] {
			if err := opts.Sleep(ctx, opts.Gap); err != nil {
				rep.Remaining++
				stopped[ch] = true
				continue
			}
		}

		// Intent marker BEFORE the send: after a crash the row is probed, not
		// re-sent blind — this is what closes the post-then-die double-post window.
		if item.Action == "post" {
			if err := o.appendState(transition{Nonce: item.Row.Nonce, State: stateSending, Attempts: item.Attempts}); err != nil {
				return rep, err
			}
		}
		sentInChannel[ch] = true

		var sendErr error
		var postedTS string
		if item.Action == "update" {
			sendErr = w.UpdateMessage(ctx, ch, item.Row.UpdateTS, item.Row.Text, item.Row.Blocks)
			postedTS = item.Row.UpdateTS
		} else {
			postedTS, sendErr = w.PostMessageIdem(ctx, ch, item.Row.Text, item.Row.Blocks, item.Row.ThreadTS, item.Row.Nonce)
		}

		if sendErr == nil {
			if err := o.appendState(transition{Nonce: item.Row.Nonce, State: statePosted, TS: postedTS}); err != nil {
				return rep, err
			}
			if item.Action == "update" {
				rep.Updated++
			} else {
				rep.Posted++
			}
			continue
		}

		// A transport error on a POST is ambiguous (the request may have landed);
		// an ok:false API answer is not. Updates are never ambiguous — re-updating
		// a card with the same content is harmless.
		var apiErr *slackwire.APIError
		ambiguous := item.Action == "post" && !errors.As(sendErr, &apiErr)
		if err := o.recordFailure(item, sendErr.Error(), ambiguous, opts, rep); err != nil {
			return rep, err
		}
		stopped[ch] = true
	}

	// Failed rows are recorded but still owed, so they count as remaining alongside
	// the rows a stopped channel never reached.
	rep.Remaining += rep.Failed
	if err := o.appendState(transition{State: stateDrainPass}); err != nil {
		return rep, err
	}
	return rep, nil
}

// recordFailure appends failed-or-dead for one item, honoring MaxAttempts.
func (o *Outbox) recordFailure(item PlanItem, reason string, ambiguous bool, opts DrainOpts, rep *DrainReport) error {
	attempts := item.Attempts + 1
	if attempts >= opts.MaxAttempts {
		rep.Dead++
		return o.appendState(transition{Nonce: item.Row.Nonce, State: stateDead, Reason: reason, Attempts: attempts})
	}
	rep.Failed++
	return o.appendState(transition{Nonce: item.Row.Nonce, State: stateFailed, Reason: reason, Attempts: attempts, Ambiguous: ambiguous})
}

// probe scans recent channel history for the row's nonce (ridden in message metadata by
// PostMessageIdem). found=true returns the landed message's ts.
func (o *Outbox) probe(ctx context.Context, w Wire, r Row, limit int) (ts string, found bool, err error) {
	msgs, err := w.History(ctx, r.Channel, "", limit)
	if err != nil {
		return "", false, err
	}
	for _, m := range msgs {
		if m.IdemNonce() == r.Nonce {
			return m.TS, true, nil
		}
	}
	return "", false, nil
}

// fencePayload is the text the leak fence scans: the message body plus the serialized
// blocks (a needle inside a Block Kit field must refuse the row exactly like one in the
// fallback text).
func fencePayload(r Row) string {
	if len(r.Blocks) == 0 {
		return r.Text
	}
	b, err := json.Marshal(r.Blocks)
	if err != nil {
		return r.Text
	}
	return r.Text + "\n" + string(b)
}
