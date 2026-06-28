package main

// notify.go - the SIGCHLD-equivalent push notifier (#761). The field's #1 remote-control
// complaint is "the agent silently waits": a session hits a PAUSED-needs-input / DRAINING /
// budget-exhaustion boundary and nobody is told. fak has the closed stop-reason vocabulary
// (session/decide.go) and two typed observer seams (session.BudgetObserver #743 +
// session.TransitionObserver #761); this fans ONE unified event out to the configured sinks
// in effort order: native (default-on) -> webhook -> Slack (both opt-in).
//
// It OBSERVES; it never gates or re-decides. Each push is idempotent, keyed on the session
// record's Rev, so a re-fire or a re-delivery of the same (trace, rev) notifies at most once.
//
// The package stays the host: the session package delivers typed values, this owns the I/O
// and the fan-out policy - exactly as cmd/fak owns the #743 webhook POST.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// StopEvent is the cmd-side union of a budget event and a transition event - the one shape
// every sink renders. Reason is the closed stop-reason token (PAUSED / DRAINING /
// BUDGET_CONTEXT_EXHAUSTED / ...). To is the lowercase run-state token, or "budget" for a
// budget-origin event. Rev is the idempotency key (monotonic per trace).
type StopEvent struct {
	TraceID        string  `json:"trace_id"`
	Reason         string  `json:"reason"`
	To             string  `json:"to"`
	ContinuationID string  `json:"continuation_id,omitempty"`
	Rev            uint64  `json:"rev"`
	Fraction       float64 `json:"fraction_consumed,omitempty"`
}

// line renders the one-line human form a native/Slack sink shows.
func (e StopEvent) line() string {
	return fmt.Sprintf("fak: session %s -> %s (%s) rev=%d", e.TraceID, e.To, e.Reason, e.Rev)
}

// notifySink is one delivery channel. name is for stderr logging; send does the (possibly
// fire-and-forget) delivery and returns a build/encode error only - transport errors are
// logged inside the sink, never surfaced (a notifier must not fail a served turn).
type notifySink struct {
	name string
	send func(StopEvent) error
}

// Notifier fans one StopEvent to its sinks in effort order, deduped by (TraceID, Rev).
type Notifier struct {
	sinks   []notifySink
	mu      sync.Mutex
	lastRev map[string]uint64
}

// fire delivers ev to every sink, at most once per (TraceID, Rev). A stale lower-or-equal
// Rev re-delivery is dropped (Rev is monotonic per trace via the table's putLocked), so a
// re-entry or a double-fire notifies exactly once.
func (n *Notifier) fire(ev StopEvent) {
	if n == nil {
		return
	}
	n.mu.Lock()
	if prev, ok := n.lastRev[ev.TraceID]; ok && ev.Rev <= prev {
		n.mu.Unlock()
		return
	}
	n.lastRev[ev.TraceID] = ev.Rev
	n.mu.Unlock()
	for _, s := range n.sinks {
		if err := s.send(ev); err != nil {
			fmt.Fprintf(os.Stderr, "fak: notify sink %s failed: %v\n", s.name, err)
		}
	}
}

// budgetObserver adapts the #743 budget seam onto the unified notifier.
func (n *Notifier) budgetObserver() session.BudgetObserver {
	return func(ev session.BudgetEvent) {
		n.fire(StopEvent{
			TraceID:        ev.TraceID,
			Reason:         ev.Reason,
			To:             "budget",
			ContinuationID: ev.ContinuationID,
			Rev:            ev.Rev,
			Fraction:       ev.FractionConsumed,
		})
	}
}

// transitionObserver adapts the #761 transition seam onto the unified notifier.
func (n *Notifier) transitionObserver() session.TransitionObserver {
	return func(ev session.TransitionEvent) {
		n.fire(StopEvent{
			TraceID:        ev.TraceID,
			Reason:         ev.Reason,
			To:             ev.To.String(),
			ContinuationID: ev.ContinuationID,
			Rev:            ev.Rev,
		})
	}
}

// nativeSink writes one structured line to w (default os.Stderr) per event - the always-on,
// dependency-free, portable native notification. It is a best-effort inline write; a future
// --notify-native-cmd could shell to a desktop toast (tools/notify.ps1) but that is heavy,
// platform-specific fleet tooling, not the v1 default.
func nativeSink(w io.Writer) notifySink {
	return notifySink{name: "native", send: func(ev StopEvent) error {
		fmt.Fprintln(w, ev.line())
		return nil
	}}
}

// webhookSink POSTs the StopEvent JSON to rawURL, reusing the shared webhookPOST helper.
func webhookSink(rawURL string) notifySink {
	return notifySink{name: "webhook", send: func(ev StopEvent) error {
		body, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		webhookPOST("webhook", rawURL, body, "")
		return nil
	}}
}

// slackSink POSTs a Slack incoming-webhook payload ({"text": ...}) to rawURL.
func slackSink(rawURL string) notifySink {
	return notifySink{name: "slack", send: func(ev StopEvent) error {
		body, err := json.Marshal(map[string]string{"text": ev.line()})
		if err != nil {
			return err
		}
		webhookPOST("webhook", rawURL, body, "application/json")
		return nil
	}}
}

// newNotifier builds the notifier with sinks in effort order: native (when nativeOn) ->
// webhook (when webhookURL non-empty) -> slack (when slackURL non-empty). nativeW is the
// native sink's writer (os.Stderr in production, a buffer in tests). Returns nil when NO
// sink is configured, so the observer seams stay the byte-identical no-op default.
func newNotifier(nativeOn bool, nativeW io.Writer, webhookURL, slackURL string) *Notifier {
	if nativeW == nil {
		nativeW = os.Stderr
	}
	var sinks []notifySink
	if nativeOn {
		sinks = append(sinks, nativeSink(nativeW))
	}
	if webhookURL != "" {
		sinks = append(sinks, webhookSink(webhookURL))
	}
	if slackURL != "" {
		sinks = append(sinks, slackSink(slackURL))
	}
	if len(sinks) == 0 {
		return nil
	}
	return &Notifier{sinks: sinks, lastRev: map[string]uint64{}}
}

// combineBudgetObservers chains budget observers, dropping nils. Returns nil if all are nil
// (the no-op seam) and the single observer unchanged if only one is non-nil, so #743's lone
// --budget-webhook wiring stays byte-identical when the notifier is not configured.
func combineBudgetObservers(obs ...session.BudgetObserver) session.BudgetObserver {
	var live []session.BudgetObserver
	for _, o := range obs {
		if o != nil {
			live = append(live, o)
		}
	}
	switch len(live) {
	case 0:
		return nil
	case 1:
		return live[0]
	default:
		return func(ev session.BudgetEvent) {
			for _, o := range live {
				o(ev)
			}
		}
	}
}

// webhookPOST fires a fire-and-forget JSON POST: a goroutine under a short timeout, any
// transport error logged to stderr but never blocking or failing the served turn that
// produced the event. label prefixes the stderr lines (e.g. "webhook", "budget webhook").
// contentType defaults to application/json when empty. It is the shared core that #743's
// budget webhook and #761's sinks send through. A blank URL is a no-op.
func webhookPOST(label, rawURL string, body []byte, contentType string) {
	if rawURL == "" {
		return
	}
	if contentType == "" {
		contentType = "application/json"
	}
	ua := "fak/" + appversion.Current()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, rawURL, bytes.NewReader(body))
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak: %s build failed: %v\n", label, err)
			return
		}
		req.Header.Set("Content-Type", contentType)
		req.Header.Set("User-Agent", ua)
		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak: %s POST to %s failed: %v\n", label, rawURL, err)
			return
		}
		_ = resp.Body.Close()
	}()
}
