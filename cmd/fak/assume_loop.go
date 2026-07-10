package main

// assume_loop.go — `fak assume loop`, the LIVE re-witness guard (#3823, epic
// #3818 C5): the impure shell over the pure internal/assumecheck Tick. `fak
// assume check` is a one-shot, so a registered premise silently rots between
// runs; the loop re-witnesses EVERY registered assumption on an interval —
// through the SAME registry -> driver -> kernel path check uses — appends every
// verdict to an append-only ledger, and on a HOLDS->VIOLATED transition queues
// ONE soft, reversible re-anchor event for the SAME adjudicated steer bus the
// doom-loop guard drains onto (doomloop.go). No new transport: the outbox /
// ledger / deliver seams are the doomloop shapes reused verbatim (appendJSONL,
// dlWriteJSON, archiveNudge, sessionClient.steerAs, classifyDeliverErr).
//
//	fak assume loop --once                       # one tick: re-witness, ledger, queue
//	fak assume loop --interval 5m                # the standing guard (Ctrl-C to stop)
//	fak assume loop --once --deliver --steer-session <id>   # enact queued events inline
//
// The correction ladder is SOFT and REVERSIBLE end to end: (1) queue the
// re-anchor packet (droppable before any delivery), (2) on --deliver POST it
// onto the adjudicated steer bus attributed to the assume-guard machine
// principal, failing CLOSED per event — a delivery the bus did not accept is
// reported refused, never claimed, and (3) the terminal rung on a refused
// delivery is an operator-escalation RECORD under <store>/escalations/. There
// is no destructive rung.
//
// Storage layout (mirrors .dos/doomloop):
//
//	<store>/assume-decisions.jsonl        append-only ledger, one row per
//	                                      registered assumption per tick
//	<store>/outbox/<assumption>-<ts>.json a queued re-anchor event
//	<store>/delivered/<...>.json          events the bus accepted
//	<store>/escalations/<...>.json        operator escalations (refused delivery)
//
// Exit codes (--once): 0 tick ran, nothing transitioned; 3 event(s) emitted
// (actionable — the doomloop drain gate); 1 runtime error; 2 usage. The
// standing loop runs until interrupted and keeps ticking past per-tick errors —
// a transient registry read must not kill the guard.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/assumecheck"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// assumeSteerPrincipal is the attribution a delivered re-anchor event carries
// onto the steer bus — a distinct MACHINE principal, exactly like
// doomloopSteerPrincipal: truthful attribution that buys no extra trust (the
// a2achan floor gates on caps+taint+scope, not the from-string).
const assumeSteerPrincipal = "assume-guard"

// assumeLoopFloorMin is the absolute cadence floor: like stallscan --watch's
// 3s clamp, the guard must never become the load it measures — and an assume
// tick is far heavier than a stallscan sample (the kernel-loop witness spawns
// `dos loop --json`), so the operator-tunable --floor cannot go below this.
const assumeLoopFloorMin = 3 * time.Second

func assumeLoopStore(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return filepath.Join(".dos", "assume")
}

// assumeLedgerRow is one append-only ledger row: the adjudicated verdict for one
// registered assumption at one tick, plus the transition read the pure Tick made
// about it. No self-report rides here — Outcome is the kernel's judgment of
// driver-gathered evidence.
type assumeLedgerRow struct {
	UnixMillis   int64  `json:"unix_millis"`
	AssumptionID string `json:"assumption_id"`
	Outcome      string `json:"outcome"`
	PrevOutcome  string `json:"prev_outcome"`
	Transition   bool   `json:"transition"`
	Witness      string `json:"witness"`
	Seat         string `json:"seat,omitempty"`
	Reason       string `json:"reason"`
}

// assumeEventPacket is the queued correction artifact — the doomloop
// nudgePacket shape carried onto the SAME steer bus, extended with the
// assumption identity and the closed outcome-class refusal token so the packet
// is self-describing in the outbox audit trail.
type assumeEventPacket struct {
	UnixMillis    int64  `json:"unix_millis"`
	Session       string `json:"session"`
	Kind          string `json:"kind"`
	AssumptionID  string `json:"assumption_id"`
	RefusalReason string `json:"refusal_reason"`
	Reason        string `json:"reason"`
	Message       string `json:"message"`
	Reversible    bool   `json:"reversible"`
}

// assumeEventReport is the per-event outcome surfaced to the operator: where the
// packet was queued and — on --deliver — what the bus actually said.
type assumeEventReport struct {
	AssumptionID  string `json:"assumption_id"`
	RefusalReason string `json:"refusal_reason"`
	Queued        string `json:"queued"`
	Outcome       string `json:"outcome"`
	Detail        string `json:"detail,omitempty"`
}

// assumeTickReport is one tick's machine-readable summary.
type assumeTickReport struct {
	Schema     string              `json:"schema"`
	UnixMillis int64               `json:"unix_millis"`
	Rows       []assumeLedgerRow   `json:"rows"`
	Events     []assumeEventReport `json:"events"`
}

func runAssumeLoop(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("assume loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	once := fs.Bool("once", false, "run a single tick and exit (exit 3 when events were emitted, mirroring `doomloop drain`)")
	interval := fs.Duration("interval", 5*time.Minute, "re-witness cadence for the standing loop")
	floor := fs.Duration("floor", 30*time.Second, "hard cadence floor --interval is clamped to (itself floored at 3s: never become the load we measure)")
	store := fs.String("store", "", "state root for the ledger, outbox, and escalations (default: .dos/assume)")
	deliver := fs.Bool("deliver", false, "DELIVER: POST each emitted event onto the adjudicated steer bus inline (fails closed per event)")
	steerSession := fs.String("steer-session", "", "session whose owned loop the re-anchor is steered into (required with --deliver)")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL (for --deliver)")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (for --deliver, only if the gateway sets --require-key)")
	defHome, regDefault := assumeRegistryDefaults()
	registryPath := fs.String("registry", regDefault, "path to the config-home registry.json (same default as `fak assume check`)")
	homeDir := fs.String("home", defHome, "home dir to discover ~/.claude* under when no registry exists")
	noHeadroom := fs.Bool("no-headroom", false, "witness against the registry-only rotation plan, without the live runtime headroom/cooldown overlay")
	asJSON := fs.Bool("json", false, "emit one JSON report per tick")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *deliver && strings.TrimSpace(*steerSession) == "" {
		fmt.Fprintln(stderr, "fak assume loop: --deliver requires --steer-session (the loop the re-anchor is injected into) — refusing to deliver nowhere")
		return 2
	}
	if *floor < assumeLoopFloorMin {
		*floor = assumeLoopFloorMin
	}
	iv := *interval
	if iv < *floor {
		iv = *floor // hard floor: the stallscan --watch cheap-by-construction posture
	}

	st := assumeLoopStore(*store)
	params := assumeGatherParams{
		registryPath: pathutil.ExpandTilde(*registryPath),
		homeDir:      pathutil.ExpandTilde(*homeDir),
		useHeadroom:  !*noHeadroom,
	}
	var client *sessionClient
	if *deliver {
		client = &sessionClient{base: strings.TrimRight(*addr, "/"), key: *key, hc: &http.Client{Timeout: 15 * time.Second}}
	}

	prev := map[string]assumecheck.Outcome{}
	if *once {
		rc, _ := assumeLoopTick(stdout, stderr, st, params, prev, client, *steerSession, *asJSON)
		return rc
	}

	fmt.Fprintf(stderr, "fak assume loop: interval=%s store=%s (Ctrl-C to stop)\n", iv, st)
	for {
		if _, next := assumeLoopTick(stdout, stderr, st, params, prev, client, *steerSession, *asJSON); next != nil {
			// A failed tick keeps the old prev: an unrecorded verdict must not
			// swallow the transition it would have witnessed.
			prev = next
		}
		time.Sleep(iv)
	}
}

// assumeRegistryDefaults resolves the same registry/home defaults `fak assume
// check` uses, so the two verbs witness against the same source of truth.
func assumeRegistryDefaults() (home, registry string) {
	home, _ = os.UserHomeDir()
	registry = os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if registry == "" && home != "" {
		registry = filepath.Join(home, ".claude-accounts", "registry.json")
	}
	return home, registry
}

// assumeLoopTick runs ONE tick end to end: re-witness every registered
// assumption through the check path (registry row -> wired gatherer -> C3
// driver -> C1 kernel), fold (prev, now) through the pure assumecheck.Tick,
// append every verdict to the ledger, queue each emitted event to the outbox,
// and — when a client is wired — deliver inline, failing closed per event.
// Returns the exit contribution (0 calm / 3 events / 1 error) and the Next map
// for the following tick (nil when the tick failed before classifying).
func assumeLoopTick(stdout, stderr io.Writer, store string, params assumeGatherParams, prev map[string]assumecheck.Outcome, client *sessionClient, steerSession string, asJSON bool) (int, map[string]assumecheck.Outcome) {
	rows := assumecheck.Registry()
	now := make([]assumecheck.Verdict, 0, len(rows))
	seats := make(map[string]string, len(rows))
	for _, a := range rows {
		ev, seat := gatherAssumptionEvidence(a, params)
		// The SAME adjudication `fak assume check` runs (GuardAssumption folds the
		// per-assumption label into the reason); the guard error is the verdict's
		// refusal signal, not a loop failure — the loop records, Tick decides.
		v, _ := assumecheck.GuardAssumption(a, ev)
		now = append(now, v)
		seats[a.ID] = seat
	}

	res := assumecheck.Tick(prev, now)
	ts := time.Now().UnixMilli()

	report := assumeTickReport{Schema: "fak.assume.loop.v1", UnixMillis: ts}
	for _, r := range res.Rows {
		row := assumeLedgerRow{
			UnixMillis:   ts,
			AssumptionID: r.Verdict.AssumptionID,
			Outcome:      string(r.Verdict.Outcome),
			PrevOutcome:  string(r.PrevOutcome),
			Transition:   r.Transition,
			Witness:      string(r.Verdict.Witness),
			Seat:         seats[r.Verdict.AssumptionID],
			Reason:       r.Verdict.Reason,
		}
		if err := appendJSONL(filepath.Join(store, "assume-decisions.jsonl"), row); err != nil {
			fmt.Fprintf(stderr, "fak assume loop: record ledger row: %v\n", err)
			return 1, nil
		}
		report.Rows = append(report.Rows, row)
	}

	outboxDir := filepath.Join(store, "outbox")
	for _, e := range res.Events {
		pkt := assumeEventPacket{
			UnixMillis:    ts,
			Session:       steerSession,
			Kind:          "reanchor",
			AssumptionID:  e.AssumptionID,
			RefusalReason: e.RefusalReason,
			Reason:        e.Reason,
			Message:       e.Reanchor,
			Reversible:    e.Reversible,
		}
		file := fmt.Sprintf("%s-%d.json", e.AssumptionID, ts)
		if err := dlWriteJSON(filepath.Join(outboxDir, file), pkt); err != nil {
			fmt.Fprintf(stderr, "fak assume loop: queue event: %v\n", err)
			return 1, nil
		}
		rep := assumeEventReport{
			AssumptionID:  e.AssumptionID,
			RefusalReason: e.RefusalReason,
			Queued:        filepath.Join(outboxDir, file),
			Outcome:       "queued",
		}
		if client != nil {
			rep.Outcome, rep.Detail = assumeDeliverEvent(client, store, outboxDir, file, pkt, ts)
		}
		report.Events = append(report.Events, rep)
	}

	if asJSON {
		raw, err := json.Marshal(report)
		if err != nil {
			fmt.Fprintf(stderr, "fak assume loop: encode report: %v\n", err)
			return 1, nil
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		counts := map[assumecheck.Outcome]int{}
		for _, r := range res.Rows {
			counts[r.Verdict.Outcome]++
		}
		fmt.Fprintf(stdout, "fak assume loop: tick assumptions=%d holds=%d violated=%d unverifiable=%d stale=%d events=%d ledger=%s\n",
			len(res.Rows), counts[assumecheck.OutcomeHolds], counts[assumecheck.OutcomeViolated],
			counts[assumecheck.OutcomeUnverifiable], counts[assumecheck.OutcomeStale],
			len(res.Events), filepath.Join(store, "assume-decisions.jsonl"))
		for _, rep := range report.Events {
			fmt.Fprintf(stdout, "  event: %s %s %s (%s)%s\n",
				rep.AssumptionID, rep.RefusalReason, rep.Outcome, rep.Queued, detailTail(rep.Detail))
		}
	}

	if len(res.Events) > 0 {
		return 3, res.Next
	}
	return 0, res.Next
}

// assumeDeliverEvent enacts ONE queued event onto the adjudicated steer bus,
// attributed to the assume-guard machine principal — the doomloop deliver
// contract verbatim: an accepted steer archives the packet to
// <store>/delivered/ (a re-tick never re-injects); a refusal or transport error
// leaves the packet queued, is reported truthfully (never claimed as
// delivered), and writes the ladder's TERMINAL rung — an operator-escalation
// RECORD, not any destructive action.
func assumeDeliverEvent(client *sessionClient, store, outboxDir, file string, pkt assumeEventPacket, ts int64) (outcome, detail string) {
	_, err := client.steerAs(pkt.Session, pkt.Message, assumeSteerPrincipal)
	if err == nil {
		if mvErr := archiveNudge(filepath.Join(store, "delivered"), outboxDir, file); mvErr != nil {
			return "delivered-not-archived", mvErr.Error()
		}
		return "delivered", ""
	}
	outcome, detail = classifyDeliverErr(err)
	esc := map[string]any{
		"unix_millis":    ts,
		"session":        pkt.Session,
		"assumption_id":  pkt.AssumptionID,
		"refusal_reason": pkt.RefusalReason,
		"outcome":        outcome,
		"detail":         detail,
		"note":           "the assume-guard could not enact its soft re-anchor on the steer bus; the packet stays queued and operator attention is required",
	}
	if escErr := dlWriteJSON(filepath.Join(store, "escalations", file), esc); escErr != nil {
		detail += "; escalation record not written: " + escErr.Error()
	}
	return outcome, detail
}
