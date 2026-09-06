package trajctl

// steer.go — issue #2540, spine step 7 of the trajectory-control epic (#2533):
// the re-anchor nudge — the first steering rung through the session steer
// channel, regime-gated. The curve fold (#2538) derives DRIFT/STALL and the
// turn-end runner (#2539) gives the curve a point every turn, but nothing acts
// on a detected drift. This rung closes the spin: on DRIFT or STALL it composes
// a re-anchor packet (objective statement + plan state + curve excerpt,
// checkpoint-and-re-read) and delivers it via the EXISTING operator steer
// channel (POST /v1/fak/session/{id}/steer, #760) — no protocol change.
//
// The REGIME GATE stands in front: a healthy recent curve means NO action —
// mid-trajectory intervention consistently degrades high-success sessions
// (arXiv:2602.03338) — and one delivered nudge holds per episode until the
// curve returns to HEALTHY, so a persistent drift is nudged exactly once, not
// hammered every turn. EVERY decision is ledgered, the no-action ones included,
// so the follow-on calibration child can score the policy from evidence.
//
// The fold stays pure and tier-1: DecideNudge reads only the folded State and
// the derived curve; the single side effect (delivery) enters through the
// SteerFunc seam, and the wire client below speaks the steer route's pinned
// shape with the stdlib client because trajctl(1) cannot import gateway(4).
// DETOUR_OVERRUN triggers the return-to-main loop (#2552), reusing the spine
// steer channel to nudge or warn referencing the paused parent.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// SteerAction is the closed vocabulary of a regime-gate decision.
type SteerAction string

const (
	// ActionNudge: a re-anchor or return-to-main packet was composed for delivery
	// through the session steer channel.
	ActionNudge SteerAction = "nudge"
	// ActionWarn: repeated overrun escalates one rung to warn.
	ActionWarn SteerAction = "warn"
	// ActionNone: the regime gate decided against intervening. The decision is
	// still ledgered — a silent no-action would starve the calibration child.
	ActionNone SteerAction = "none"
)

// SteerDecision is one regime-gate decision for one objective at one turn
// boundary — the ledgered unit of this rung. A nudge row carries the composed
// packet and whether the steer channel accepted it; a none row carries the
// reason the gate held.
type SteerDecision struct {
	ObjectiveID string      `json:"objective_id"`
	Action      SteerAction `json:"action"`
	Signal      Signal      `json:"signal"`
	Reason      string      `json:"reason"`
	Packet      string      `json:"packet,omitempty"`
	Delivered   bool        `json:"delivered,omitempty"`
	DeliverErr  string      `json:"deliver_err,omitempty"`
	UnixMillis  int64       `json:"unix_millis,omitempty"`
	SessionID   string      `json:"session_id,omitempty"`
	RunID       string      `json:"run_id,omitempty"`
}

// SteerRecord builds a ledger row for a SteerDecision.
func SteerRecord(d SteerDecision) Row {
	return Row{Schema: Schema, Kind: KindSteer, Steer: &d}
}

// SteersFor returns the ledgered steer decisions for objectiveID in append
// order — the episode history the regime gate's re-arm scan reads.
func (s State) SteersFor(objectiveID string) []SteerDecision {
	out := make([]SteerDecision, 0)
	for _, d := range s.Steers {
		if d.ObjectiveID == objectiveID {
			out = append(out, d)
		}
	}
	return out
}

// nudgeArmed reports whether the gate may nudge this objective: a DELIVERED
// nudge disarms the episode, and a later decision taken on a HEALTHY curve
// re-arms it (recovery ends the episode). An undelivered nudge — the channel
// refused or errored — does not disarm, so a transient delivery failure is
// retried at the next boundary rather than silently consuming the one nudge.
func nudgeArmed(steers []SteerDecision) bool {
	armed := true
	for _, d := range steers {
		switch {
		case d.Action == ActionNudge && d.Delivered:
			armed = false
		case d.Signal == SignalHealthy:
			armed = true
		}
	}
	return armed
}

// DecideNudge is the regime gate: it derives this rung's decision for one
// objective from its folded curve and the ledgered episode history. Pure —
// composing a nudge does not deliver it; the caller owns the side effect.
func (s State) DecideNudge(oc ObjectiveCurve) SteerDecision {
	d := SteerDecision{ObjectiveID: oc.ObjectiveID, Action: ActionNone, Signal: oc.Signal}
	switch oc.Signal {
	case SignalDrift, SignalStall:
		// the rung's trigger regime — fall through to the episode scan.
	case SignalDetourOverrun:
		obj, ok := s.Objectives[oc.ObjectiveID]
		if !ok {
			d.Reason = "regime gate: objective was never declared — nothing to re-anchor on"
			return d
		}
		parentID := oc.ParentID
		if parentID == "" {
			parentID = obj.ParentID
		}
		parent, okParent := s.Objectives[parentID]
		if parentID == "" || !okParent || parent.Status != StatusPaused {
			d.Reason = "regime gate: DETOUR_OVERRUN belongs to the return-to-main rung, not the re-anchor nudge"
			return d
		}
		return s.DecideReturnToMain(oc)
	default:
		d.Reason = "regime gate: recent curve healthy — mid-trajectory intervention degrades a high-success session"
		return d
	}
	if !nudgeArmed(s.SteersFor(oc.ObjectiveID)) {
		d.Reason = "regime gate: a delivered nudge is outstanding for this episode — re-arms when the curve returns to HEALTHY"
		return d
	}
	obj, ok := s.Objectives[oc.ObjectiveID]
	if !ok {
		d.Reason = "regime gate: objective was never declared — nothing to re-anchor on"
		return d
	}
	d.Action = ActionNudge
	d.Reason = fmt.Sprintf("regime gate: %s — %s", oc.Signal, oc.Detail)
	d.Packet = ComposeReAnchor(obj, oc)
	return d
}

// reAnchorExcerptPoints bounds the curve excerpt a packet carries. The packet
// must stay a bounded re-read, not a transcript re-injection (the no-length-
// gameable-steering fence).
const reAnchorExcerptPoints = 5

// ComposeReAnchor serializes the re-anchor packet: objective statement + plan
// state + a bounded curve excerpt. Checkpoint-and-re-read — the session reads
// its objective back fresh instead of trusting context continuity, the
// goal-drift literature's most robust mitigation (arXiv:2505.02709).
func ComposeReAnchor(obj Objective, oc ObjectiveCurve) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[fak trajctl re-anchor] The witnessed progress curve for objective %q signals %s: %s\n", obj.ID, oc.Signal, oc.Detail)
	fmt.Fprintf(&b, "Objective: %s\n", obj.Statement)
	if len(obj.Plan) > 0 {
		b.WriteString("Plan state:")
		for _, p := range obj.Plan {
			if p.Title != "" {
				fmt.Fprintf(&b, " %s (%s);", p.ID, p.Title)
			} else {
				fmt.Fprintf(&b, " %s;", p.ID)
			}
		}
		b.WriteString("\n")
	}
	if pts := progressPoints(oc.Methods); len(pts) > 0 {
		if len(pts) > reAnchorExcerptPoints {
			pts = pts[len(pts)-reAnchorExcerptPoints:]
		}
		vals := make([]string, 0, len(pts))
		for _, p := range pts {
			vals = append(vals, fmt.Sprintf("%.2f", p.Value))
		}
		fmt.Fprintf(&b, "Curve excerpt (%s, last %d): %s (latest %.2f, delta %+.2f)\n",
			CommitScorerMethod, len(pts), strings.Join(vals, " -> "), oc.Latest, oc.Delta)
	}
	b.WriteString("Re-read the objective and plan above, compare them against what you are doing right now, and re-anchor on the objective before your next action.")
	return b.String()
}

// SteerFunc delivers a composed re-anchor packet to a session's steer channel.
// It is the rung's single impurity seam: production wiring uses GatewaySteer;
// tests substitute a recorder.
type SteerFunc func(sessionID, text string) error

// SteerSweep runs the regime gate over every OPEN objective at a turn boundary,
// delivers each armed nudge through deliver, and returns the decisions to
// ledger — every one, acted-on or not. Fail-open by the turn-cadence contract:
// a refused, failed, or panicking delivery is captured on the decision row
// (Delivered=false, DeliverErr) and never unwinds the caller, so a broken
// channel costs the nudge, not the turn. The caller appends the returned
// decisions (AppendSteerDecisions) as its single ledger side effect.
func (s State) SteerSweep(stamp Stamp, unixMillis int64, deliver SteerFunc) []SteerDecision {
	out := make([]SteerDecision, 0)
	for _, id := range openObjectiveIDs(s.Objectives) {
		oc, ok := s.CurveFor(id)
		if !ok {
			continue
		}
		d := s.DecideNudge(oc)
		d.UnixMillis = unixMillis
		d.SessionID = stamp.SessionID
		d.RunID = stamp.RunID
		if d.Action == ActionNudge || d.Action == ActionWarn {
			switch {
			case deliver == nil:
				d.DeliverErr = "no steer channel configured"
			case stamp.SessionID == "":
				d.DeliverErr = "no session id to steer"
			default:
				if err := deliverGuarded(deliver, stamp.SessionID, d.Packet); err != nil {
					d.DeliverErr = err.Error()
				} else {
					d.Delivered = true
				}
			}
		}
		out = append(out, d)
	}
	return out
}

// AppendSteerDecisions writes each decision to the ledger at path, in order,
// and returns the count appended — the thin I/O wrapper over the pure sweep,
// mirroring AppendSample. A row that fails validation stops the append and
// returns the error with the count written so far.
func AppendSteerDecisions(path string, ds []SteerDecision) (int, error) {
	n := 0
	for _, d := range ds {
		if err := Append(path, SteerRecord(d)); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}

// deliverGuarded calls deliver behind a recover shield so a panicking channel
// yields a captured error on the decision row instead of unwinding the turn.
func deliverGuarded(deliver SteerFunc, sessionID, text string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("steer delivery panicked: %v", r)
		}
	}()
	return deliver(sessionID, text)
}

// GatewaySteer returns a SteerFunc that POSTs the packet to the existing
// session steer channel, POST {base}/v1/fak/session/{id}/steer (#760). The wire
// shape — body {"text": ...}, success 202 — is pinned by the gateway steer
// route and its tests; trajctl is tier-1 and cannot import gateway(4), so this
// speaks the shape directly with the stdlib client (the macbench precedent).
// bearer is optional; hc defaults to a 15s-timeout client, matching the signal
// verb's.
func GatewaySteer(baseURL, bearer string, hc *http.Client) SteerFunc {
	if hc == nil {
		hc = &http.Client{Timeout: 15 * time.Second}
	}
	base := strings.TrimRight(baseURL, "/")
	return func(sessionID, text string) error {
		body, err := json.Marshal(struct {
			Text string `json:"text"`
		}{Text: text})
		if err != nil {
			return err
		}
		req, err := http.NewRequest(http.MethodPost, base+"/v1/fak/session/"+url.PathEscape(sessionID)+"/steer", bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		resp, err := hc.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode > 299 {
			detail, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			return fmt.Errorf("steer channel refused: %s: %s", resp.Status, strings.TrimSpace(string(detail)))
		}
		return nil
	}
}

// validateSteer checks a steer decision row: closed action vocabulary, a valid
// trigger signal, a stated reason, and a packet on (exactly) the nudge rows.
func validateSteer(d SteerDecision) error {
	if d.ObjectiveID == "" {
		return errors.New("trajctl: steer objective id is required")
	}
	switch d.Action {
	case ActionNudge, ActionWarn, ActionNone:
	default:
		return fmt.Errorf("trajctl: invalid steer action %q", d.Action)
	}
	if !validSignal(d.Signal) {
		return fmt.Errorf("trajctl: invalid steer signal %q", d.Signal)
	}
	if d.Reason == "" {
		return errors.New("trajctl: steer reason is required")
	}
	if (d.Action == ActionNudge || d.Action == ActionWarn) && d.Packet == "" {
		return errors.New("trajctl: a nudge or warn row must carry its packet")
	}
	if d.Action == ActionNone && (d.Packet != "" || d.Delivered) {
		return errors.New("trajctl: a none row must not carry a packet or a delivery")
	}
	return nil
}

func validSignal(sig Signal) bool {
	switch sig {
	case SignalHealthy, SignalStall, SignalDrift, SignalDetourOverrun:
		return true
	default:
		return false
	}
}
