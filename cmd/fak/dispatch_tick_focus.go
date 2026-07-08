package main

// dispatch_tick_focus.go -- wires the focusscore focus_debt / WIP-breadth signal
// (internal/focusscore, graded on the docs/nightrun/trajctl.jsonl ledger) into the
// dispatch tick as WARN-FIRST breadth backpressure (#3223). The pure fold lives in
// internal/dispatchtick/preflight_focus.go; this file is the impure shell that reads the
// same focusscore fold the scorecard uses (no new source of truth) and grades whether
// THIS tick's spawn OPENS a new concurrent objective while the fleet is at/over its WIP
// cap. It throttles ONLY new-objective spawns; continuing an already-open objective is
// never held.

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/focusscore"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// dispatchFocusHoldPosture resolves the WARN-vs-HOLD posture for the focus WIP
// backpressure term: default WARN (advise + still spawn), flipped to HOLD by the
// --focus-hold flag or FLEET_DISPATCH_FOCUS_HOLD env. Warn keeps the live fleet
// byte-identical until an operator opts in.
func dispatchFocusHoldPosture(opts dispatchTickOptions) bool {
	return opts.FocusHold || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_FOCUS_HOLD"))
}

// dispatchEvaluateFocus reads the focusscore fold over the workspace's trajctl ledger and
// grades whether dispatching issue `target` OPENS a new concurrent objective over the WIP
// cap. It is the seam the tick consults just before it would spawn a worker. A missing
// ledger folds to Present=false, so the term abstains (no advisory) and the tick is
// byte-identical to today.
func dispatchEvaluateFocus(root string, hold bool, target int) dispatchtick.FocusAdmission {
	ev := focusscore.Build(focusscore.Options{Root: root}).Evidence
	return dispatchtick.EvaluateFocusAdmission(dispatchtick.FocusCheck{
		Active:       ev.Active,
		WIPCap:       ev.WIPCap,
		Present:      ev.LedgerPresent,
		NewObjective: dispatchTargetOpensNewObjective(root, target),
		Hold:         hold,
	})
}

// dispatchTargetOpensNewObjective reports whether dispatching issue #target OPENS a new
// concurrent objective (true) or CONTINUES one already open in the trajctl ledger (false).
// A continuation is any ACTIVE or PAUSED objective whose id or statement references the
// issue token `#<target>` -- the fleet already declared that issue as live work, so
// re-dispatching it adds no breadth and is never held by the focus term. An issue no open
// objective references opens a new objective. A missing/empty ledger has no open
// objectives, so every candidate opens a new objective (and the term still abstains via
// Present=false, keeping the tick unchanged).
func dispatchTargetOpensNewObjective(root string, target int) bool {
	if target <= 0 {
		return true
	}
	ledger := filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	// #<target> as a whole token (a trailing \b keeps #4 from matching #42).
	re := regexp.MustCompile(`#` + strconv.Itoa(target) + `\b`)
	for _, o := range st.Objectives {
		if o.Status != trajctl.StatusActive && o.Status != trajctl.StatusPaused {
			continue
		}
		if re.MatchString(o.ID) || re.MatchString(o.Statement) {
			return false
		}
	}
	return true
}
