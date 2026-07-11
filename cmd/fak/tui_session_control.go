package main

// tui_session_control.go — the OOB operator control SURFACE for the sessions pane
// (#2763, child of the out-of-band operator control epic #2753). `fak console
// sessions` is otherwise render-only: it SHOWS live drive state but mutates nothing.
// This adds interactive control KEYBINDINGS — pause / resume / throttle / drain — that
// map a keypress on the highlighted session to the existing session-control route
// (POST /v1/fak/session/{id}/run), so a human watching the console can steer a running
// session's lifecycle without leaving the TUI.
//
// The design keeps a clean seam between "what request does this keypress emit" and
// "dispatch it", so the witness is captured, not live-dialed:
//
//   planTUISessionControlKey  — PURE: keypress + selected row -> the control request it
//                               would emit (with the destructive-confirmation gate), no I/O.
//   dispatchTUISessionControl — the thin wiring that hands an emit-ready plan to the
//                               existing sessionClient.control route.
//
// Destructive ops (drain) require an explicit confirmation before any request is emitted;
// the redirect key is DECLARED but deferred until its sibling op (a child of #2753) lands,
// so pressing it names the pending dependency instead of dialing a half-built route.

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// tuiSessionControlBinding declares one sessions-pane keybinding: the key an operator
// presses, the human label, and the control op it drives. All the live drive changes
// ride the single "run" verb (matching `fak session pause/resume/throttle`), setting the
// target run-state token. Destructive bindings gate on a confirmation; a Deferred binding
// is declared for discoverability but emits no request until its op lands.
type tuiSessionControlBinding struct {
	Key         rune   // the key the operator presses on the highlighted row
	Label       string // short human label for the legend ("pause", "drain")
	Verb        string // the control route verb — "run" for every live drive change
	Run         string // the target run-state wire token (paused|running|throttled|draining)
	Reason      string // optional reason token recorded on the transition (audit "why")
	Destructive bool   // true ⇒ require an explicit confirmation before emitting
	Deferred    string // non-empty ⇒ the op is not yet wired; the string names the blocker
}

// tuiSessionControlBindings is the ordered sessions-pane keybinding table. Pause/resume/
// throttle are reversible drive flips; drain is the destructive terminal request (a
// confirmation gate stands in front of it). Redirect is reserved but blocked-by the
// redirect child of #2753 — declared so the surface is complete, inert until it lands.
func tuiSessionControlBindings() []tuiSessionControlBinding {
	return []tuiSessionControlBinding{
		{Key: 'p', Label: "pause", Verb: "run", Run: "paused"},
		{Key: 'r', Label: "resume", Verb: "run", Run: "running"},
		{Key: 't', Label: "throttle", Verb: "run", Run: "throttled", Reason: "operator-tui-throttle"},
		{Key: 'd', Label: "drain", Verb: "run", Run: "draining", Reason: "operator-tui-drain", Destructive: true},
		{Key: '>', Label: "redirect", Deferred: "redirect op (child of #2753) not yet landed"},
	}
}

// lookupTUISessionControlBinding finds the binding for a pressed key. The bool is false
// for an unbound key, so the caller fails closed (does nothing) rather than guessing.
func lookupTUISessionControlBinding(key rune) (tuiSessionControlBinding, bool) {
	for _, b := range tuiSessionControlBindings() {
		if b.Key == key {
			return b, true
		}
	}
	return tuiSessionControlBinding{}, false
}

// tuiSessionControlPlan is the captured outcome of a keypress on a selected session: the
// exact control request the keybinding would emit, plus the gate flags that decide whether
// it is emitted. It is the witness surface — a test asserts this without dialing a gateway.
type tuiSessionControlPlan struct {
	TraceID      string                        // the selected session's trace id
	Key          string                        // the pressed key, rendered
	Label        string                        // the binding's human label
	Verb         string                        // the control route verb ("run"), empty when nothing is emitted
	Request      gateway.SessionControlRequest // the request that would be POSTed to the route
	Destructive  bool                          // the op is destructive (drain)
	NeedsConfirm bool                          // destructive and not yet confirmed ⇒ emit withheld until confirmed
	Deferred     string                        // non-empty ⇒ the op's dependency has not landed; nothing emitted
	Emit         bool                          // true ⇒ dispatchTUISessionControl should POST Request
}

// planTUISessionControlKey maps a keypress on the highlighted session to the control
// request it would emit — the PURE core of the keybinding, with no I/O. It enforces the
// two gates the issue names: a destructive op (drain) is NOT emitted until confirmed, and
// a deferred op (redirect) is never emitted. The bool is false for an unbound key. This is
// the seam that makes the witness "captured, not live-dialed": a test feeds a key here and
// asserts plan.Request without a live gateway.
func planTUISessionControlKey(row tuiSessionRow, key rune, confirmed bool) (tuiSessionControlPlan, bool) {
	b, ok := lookupTUISessionControlBinding(key)
	if !ok {
		return tuiSessionControlPlan{}, false
	}
	plan := tuiSessionControlPlan{
		TraceID:     row.TraceID,
		Key:         string(b.Key),
		Label:       b.Label,
		Destructive: b.Destructive,
	}
	if b.Deferred != "" {
		plan.Deferred = b.Deferred
		return plan, true
	}
	if b.Destructive && !confirmed {
		plan.NeedsConfirm = true
		return plan, true // withhold the request until the operator confirms
	}
	plan.Verb = b.Verb
	plan.Request = gateway.SessionControlRequest{Run: b.Run, Reason: b.Reason}
	plan.Emit = true
	return plan, true
}

// dispatchTUISessionControl hands an emit-ready plan to the existing session-control route
// via the shared sessionClient, returning the new drive state. It refuses to dispatch a
// plan that was withheld (needs confirmation, deferred, or nothing emitted) — the gate
// lives in planTUISessionControlKey and dispatch never re-decides it.
func dispatchTUISessionControl(c *sessionClient, plan tuiSessionControlPlan) (gateway.SessionState, error) {
	if !plan.Emit {
		switch {
		case plan.Deferred != "":
			return gateway.SessionState{}, fmt.Errorf("%s: %s", plan.Label, plan.Deferred)
		case plan.NeedsConfirm:
			return gateway.SessionState{}, fmt.Errorf("%s is destructive: pass --confirm to apply it to %s", plan.Label, plan.TraceID)
		default:
			return gateway.SessionState{}, fmt.Errorf("no control op to dispatch")
		}
	}
	if strings.TrimSpace(plan.TraceID) == "" {
		return gateway.SessionState{}, fmt.Errorf("no session selected for %s", plan.Label)
	}
	return c.control(plan.TraceID, plan.Verb, plan.Request)
}

// selectTUISessionRow resolves the row a keypress acts on: the explicitly named session
// when id is non-empty, else the highlighted (top-of-attention, already sorted) row. The
// bool is false when no such session exists, so the caller reports it instead of driving a
// phantom trace.
func selectTUISessionRow(report tuiSessionReport, id string) (tuiSessionRow, bool) {
	id = strings.TrimSpace(id)
	if id == "" {
		if len(report.Rows) == 0 {
			return tuiSessionRow{}, false
		}
		return report.Rows[0], true // the highlighted row (rows are attention-sorted)
	}
	for _, row := range report.Rows {
		if row.TraceID == id {
			return row, true
		}
	}
	return tuiSessionRow{}, false
}

// renderTUISessionControlLegend renders the one-line control-key legend shown under the
// session queue, so the pane advertises the control surface (not just live state). A
// destructive key is tagged "(confirm)" and a deferred key names its pending dependency,
// so an operator reads what each key does and which are gated before pressing one.
func renderTUISessionControlLegend() string {
	parts := make([]string, 0, len(tuiSessionControlBindings()))
	for _, b := range tuiSessionControlBindings() {
		seg := fmt.Sprintf("%c %s", b.Key, b.Label)
		switch {
		case b.Deferred != "":
			seg += " (pending)"
		case b.Destructive:
			seg += " (confirm)"
		}
		parts = append(parts, seg)
	}
	return "Controls  " + strings.Join(parts, "  ")
}

// tuiSessionControlKeys returns the bound keys in a stable order — a small helper for
// tests and callers that need to enumerate the control surface deterministically.
func tuiSessionControlKeys() []string {
	keys := make([]string, 0, len(tuiSessionControlBindings()))
	for _, b := range tuiSessionControlBindings() {
		keys = append(keys, string(b.Key))
	}
	sort.Strings(keys)
	return keys
}
