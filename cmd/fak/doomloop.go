package main

// fak doomloop - the meta-monitor shell over internal/doomloop.
//
// This is the impure half of the doom-loop guard: it gathers the two axes the
// pure classifier folds (effort spent vs verified forward progress), maintains a
// durable per-worker sample history, classifies each worker, records every
// decision to an accountability ledger, and - when asked - QUEUES a soft,
// reversible re-anchor nudge to the worker's steer outbox. Queuing is decoupled
// from delivery: the classifier only ever writes the reversible artifact, and
// `drain --deliver` is the wired drainer that POSTs each queued nudge onto the
// SAME adjudicated steer bus an operator steer uses — attributed to a distinct
// "doomloop-guard" machine principal, and failing closed per nudge when the
// target owns no loop (#3528) or the floor refuses (#3529). The destructive rung
// (kill/replace) is not here by design; the ladder tops out at an operator
// escalation record.
//
// The SAMPLING seam is deliberately explicit. `tick` takes the effort/progress
// counters from its caller rather than reading transcripts or git itself, so the
// evidence source is a swappable adapter (a guard hook that counts transcript
// lines; a git-log reader that counts commits on the worker's region). Keeping
// the counters injected keeps this shell testable and keeps the trust boundary
// legible: progress is whatever VERIFIED counter the adapter feeds, never a
// worker's self-report.
//
// See internal/doomloop for the classifier and the reason-token contract, and
// dos.toml [reasons.DOOM_LOOP] for the closed refusal vocabulary this emits.

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/doomloop"
)

const doomloopUsage = `fak doomloop - two-axis doom-loop guard (effort vs verified progress)

  tick      ingest one observation for a worker, classify its history, record the decision
  scan      fold the store into a fleet verdict table (one row per worker)
  drain     output drainer: report queued re-anchor nudges (observe-only by default);
            --deliver POSTs each onto the adjudicated steer bus (fails closed per nudge)
  classify  one-shot: classify a samples JSONL stream from --file or stdin

Common flags:
  --store PATH   state root for sample histories, the decision ledger, and the
                 nudge outbox (default: .dos/doomloop)
  --json         emit machine-readable JSON instead of a text table/line

tick flags:
  --session ID   worker/session identifier (required)
  --effort N     monotone effort counter (transcript lines / turns / tokens)
  --progress M   monotone VERIFIED forward-progress counter (commits / ledger steps)
  --alive        the worker's process/heartbeat is live at this observation
  --correct      apply the recommended reversible correction (queue a re-anchor
                 nudge / escalation) instead of only recording the decision
  --now MS       injectable clock (unix millis); 0 = wall clock`

func cmdDoomloop(argv []string) { os.Exit(runDoomloop(os.Stdout, os.Stderr, os.Stdin, argv)) }

func runDoomloop(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, doomloopUsage)
		return 2
	}
	switch argv[0] {
	case "tick":
		return runDoomloopTick(stdout, stderr, argv[1:])
	case "scan":
		return runDoomloopScan(stdout, stderr, argv[1:])
	case "drain":
		return runDoomloopDrain(stdout, stderr, argv[1:])
	case "classify":
		return runDoomloopClassify(stdout, stderr, stdin, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, doomloopUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak doomloop: unknown subcommand %q\n%s\n", argv[0], doomloopUsage)
		return 2
	}
}

// ---- storage layout ---------------------------------------------------------
//
//   <store>/samples/<session>.jsonl   append-only per-worker Sample history
//   <store>/decisions.jsonl           append-only accountability ledger
//   <store>/outbox/<session>-<ts>.json  a queued re-anchor nudge (the correction)
//   <store>/escalations/<session>-<ts>.json  a queued operator escalation

func doomloopStore(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return filepath.Join(".dos", "doomloop")
}

func dlSamplesPath(store, session string) string {
	return filepath.Join(store, "samples", session+".jsonl")
}

// decision is one accountability-ledger row: what the guard saw and what it did.
// It carries no worker self-report - only the classifier's verified read.
type dlDecision struct {
	UnixMillis    int64  `json:"unix_millis"`
	Session       string `json:"session"`
	Verdict       string `json:"verdict"`
	Correction    string `json:"correction"`
	Applied       string `json:"applied"`
	Reason        string `json:"reason"`
	Streak        int    `json:"burning_flat_streak"`
	EffortDelta   int64  `json:"effort_delta"`
	ProgressDelta int64  `json:"progress_delta"`
	Samples       int    `json:"samples"`
	Artifact      string `json:"artifact,omitempty"`
}

// nudgePacket is the soft, reversible correction: a re-anchor message queued to
// the worker's steer outbox. No transport drains this outbox yet - the packet is
// a queued artifact awaiting a drainer (the intended one is a GatewaySteer-style
// POST to /v1/fak/session/{id}/steer, itself not yet wired into a live loop).
// Queuing here is reversible - the packet can be dropped before any delivery, and
// delivery only injects text.
type nudgePacket struct {
	UnixMillis int64  `json:"unix_millis"`
	Session    string `json:"session"`
	Kind       string `json:"kind"`
	Reason     string `json:"reason"`
	Message    string `json:"message"`
	Reversible bool   `json:"reversible"`
	Streak     int    `json:"burning_flat_streak"`
}

func dlNow(override int64) int64 {
	if override != 0 {
		return override
	}
	return time.Now().UnixMilli()
}

func appendJSONL(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(raw, '\n')); err != nil {
		return err
	}
	return nil
}

func readSamples(path string) ([]doomloop.Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	return decodeSamples(f)
}

func decodeSamples(r io.Reader) ([]doomloop.Sample, error) {
	var out []doomloop.Sample
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var s doomloop.Sample
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			return nil, fmt.Errorf("bad sample line: %w", err)
		}
		out = append(out, s)
	}
	return out, sc.Err()
}

// reanchorMessage is the soft nudge text delivered to a doom-looping worker.
func reanchorMessage(streak int) string {
	return fmt.Sprintf("doom-loop guard: %d consecutive windows of effort with no verified forward progress. "+
		"Stop expanding. Re-read your objective, then land ONE small VERIFIED step (a commit on your region or a "+
		"witnessed ledger step) before continuing. If you are blocked, say so with a structured reason instead of retrying.", streak)
}

// applyCorrection turns a classifier recommendation into an accountability
// outcome. Without `correct` it is a dry run (the decision is recorded, nothing
// is delivered). With `correct` it queues the reversible artifact the ladder
// calls for. It never performs a destructive action.
func applyCorrection(store, session string, res doomloop.Result, correct bool, now int64) (applied, artifact string, err error) {
	switch res.Correction {
	case doomloop.CorrectNudge:
		if !correct {
			return "recorded (dry-run: would queue NUDGE)", "", nil
		}
		pkt := nudgePacket{
			UnixMillis: now,
			Session:    session,
			Kind:       "reanchor",
			Reason:     res.Reason,
			Message:    reanchorMessage(res.BurningFlatStreak),
			Reversible: true,
			Streak:     res.BurningFlatStreak,
		}
		path := filepath.Join(store, "outbox", fmt.Sprintf("%s-%d.json", session, now))
		if err := dlWriteJSON(path, pkt); err != nil {
			return "", "", err
		}
		return "nudge-queued", path, nil
	case doomloop.CorrectEscalate:
		if !correct {
			return "recorded (dry-run: would ESCALATE)", "", nil
		}
		path := filepath.Join(store, "escalations", fmt.Sprintf("%s-%d.json", session, now))
		esc := map[string]any{
			"unix_millis": now,
			"session":     session,
			"reason":      res.Reason,
			"streak":      res.BurningFlatStreak,
			"note":        "a re-anchor nudge did not recover this worker; operator attention required",
		}
		if err := dlWriteJSON(path, esc); err != nil {
			return "", "", err
		}
		return "escalate-queued", path, nil
	case doomloop.CorrectObserve:
		return "observe", "", nil
	default:
		return "none", "", nil
	}
}

func dlWriteJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// ---- tick -------------------------------------------------------------------

func runDoomloopTick(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doomloop tick", flag.ContinueOnError)
	fs.SetOutput(stderr)
	session := fs.String("session", "", "worker/session identifier (required)")
	effort := fs.Int64("effort", 0, "monotone effort counter (transcript lines / turns / tokens)")
	progress := fs.Int64("progress", 0, "monotone VERIFIED forward-progress counter (commits / ledger steps)")
	alive := fs.Bool("alive", false, "worker process/heartbeat is live at this observation")
	correct := fs.Bool("correct", false, "apply the recommended reversible correction, not just record it")
	store := fs.String("store", "", "state root (default: .dos/doomloop)")
	asJSON := fs.Bool("json", false, "emit JSON")
	now := fs.Int64("now", 0, "injectable clock (unix millis); 0 = wall clock")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *session == "" {
		fmt.Fprintln(stderr, "fak doomloop tick: --session is required")
		return 2
	}
	st := doomloopStore(*store)
	ts := dlNow(*now)

	sample := doomloop.Sample{UnixMillis: ts, Effort: *effort, Progress: *progress, Alive: *alive}
	if err := appendJSONL(dlSamplesPath(st, *session), sample); err != nil {
		fmt.Fprintf(stderr, "fak doomloop tick: record sample: %v\n", err)
		return 1
	}
	samples, err := readSamples(dlSamplesPath(st, *session))
	if err != nil {
		fmt.Fprintf(stderr, "fak doomloop tick: read history: %v\n", err)
		return 1
	}

	res := doomloop.Classify(samples, doomloop.DefaultConfig())
	applied, artifact, err := applyCorrection(st, *session, res, *correct, ts)
	if err != nil {
		fmt.Fprintf(stderr, "fak doomloop tick: apply correction: %v\n", err)
		return 1
	}

	dec := dlDecision{
		UnixMillis:    ts,
		Session:       *session,
		Verdict:       string(res.Verdict),
		Correction:    string(res.Correction),
		Applied:       applied,
		Reason:        res.Reason,
		Streak:        res.BurningFlatStreak,
		EffortDelta:   res.EffortDelta,
		ProgressDelta: res.ProgressDelta,
		Samples:       len(samples),
		Artifact:      artifact,
	}
	if err := appendJSONL(filepath.Join(st, "decisions.jsonl"), dec); err != nil {
		fmt.Fprintf(stderr, "fak doomloop tick: record decision: %v\n", err)
		return 1
	}

	if *asJSON {
		raw, _ := json.MarshalIndent(dec, "", "  ")
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	fmt.Fprintf(stdout, "%s\t%s\t%s\tstreak=%d\tΔeffort=%d\tΔprogress=%d\t%s\n",
		*session, res.Verdict, res.Correction, res.BurningFlatStreak, res.EffortDelta, res.ProgressDelta, applied)
	fmt.Fprintf(stdout, "  %s\n", res.Interpretation())
	return 0
}

// ---- scan -------------------------------------------------------------------

func runDoomloopScan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doomloop scan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "state root (default: .dos/doomloop)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	st := doomloopStore(*store)
	dir := filepath.Join(st, "samples")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			if *asJSON {
				fmt.Fprintln(stdout, "[]")
			} else {
				fmt.Fprintf(stdout, "no worker histories under %s\n", dir)
			}
			return 0
		}
		fmt.Fprintf(stderr, "fak doomloop scan: %v\n", err)
		return 1
	}

	type row struct {
		Session string          `json:"session"`
		Result  doomloop.Result `json:"result"`
	}
	var rows []row
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		session := strings.TrimSuffix(e.Name(), ".jsonl")
		samples, err := readSamples(filepath.Join(dir, e.Name()))
		if err != nil {
			fmt.Fprintf(stderr, "fak doomloop scan: %s: %v\n", session, err)
			return 1
		}
		rows = append(rows, row{Session: session, Result: doomloop.Classify(samples, doomloop.DefaultConfig())})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Session < rows[j].Session })

	if *asJSON {
		raw, _ := json.MarshalIndent(rows, "", "  ")
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	if len(rows) == 0 {
		fmt.Fprintf(stdout, "no worker histories under %s\n", dir)
		return 0
	}
	for _, r := range rows {
		fmt.Fprintf(stdout, "%-24s %-10s %-10s streak=%d\n", r.Session, r.Result.Verdict, r.Result.Correction, r.Result.BurningFlatStreak)
	}
	return 0
}

// ---- drain (the correction output drainer) ----------------------------------
//
// drain is the read side of the correction transport. The classifier queues a
// reversible re-anchor nudge to <store>/outbox/; drain enumerates those queued
// packets and, by default, reports each one alongside the steer endpoint it
// would post to — OBSERVE-only: it delivers nothing and removes nothing, so the
// outbox stays an auditable record of what the guard would inject. The actual
// POST is gated behind --deliver (#3529), which enacts each nudge onto the SAME
// adjudicated steer bus an operator steer uses and fails closed per nudge when
// the target owns no loop (#3528) or the floor refuses — the ladder never
// silently injects text, and never claims a delivery the bus did not accept.
// This mirrors `fak loop reap`: report by default, the outward rung behind an
// explicit opt-in.

// doomloopSteerPrincipal is the attribution a delivered nudge carries onto the steer bus.
// It is deliberately NOT "operator": the nudge is a machine correction, and the a2achan
// floor gates on caps+taint+scope (Shared ⇒ Tainted/ScopeFleet) rather than the from-string,
// so naming a distinct machine principal is truthful attribution that buys no extra trust.
const doomloopSteerPrincipal = "doomloop-guard"

type drainRow struct {
	File        string `json:"file"`
	Session     string `json:"session"`
	Kind        string `json:"kind"`
	Reason      string `json:"reason"`
	Streak      int    `json:"burning_flat_streak"`
	Reversible  bool   `json:"reversible"`
	Destination string `json:"destination"`
	// Delivery outcome — populated only on --deliver. Outcome is one of "delivered",
	// "delivered-not-archived", "refused-no-owned-loop", "refused-floor", "error"; Detail
	// carries the refusal/error text so a refusal is never rendered as a silent success.
	Delivered bool   `json:"delivered,omitempty"`
	Outcome   string `json:"outcome,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

type drainReport struct {
	Schema    string     `json:"schema"`
	Pending   int        `json:"pending"`
	Delivered int        `json:"delivered,omitempty"`
	Refused   int        `json:"refused,omitempty"`
	Nudges    []drainRow `json:"nudges"`
}

// steerDestination is the endpoint a drainer POSTs a queued nudge to. Kept as a pure string
// builder so the observe path can report the intended target without opening a transport.
func steerDestination(session string) string {
	return "POST /v1/fak/session/" + session + "/steer"
}

// nudgeItem is one decoded outbox packet paired with its filename, so the deliver path can
// archive the exact file it enacted.
type nudgeItem struct {
	file string
	pkt  nudgePacket
}

// readOutboxNudges enumerates and decodes every queued nudge packet in dir, sorted by
// filename for stable output. A missing dir returns (nil, os.ErrNotExist) so the caller can
// treat "no outbox yet" as an empty queue.
func readOutboxNudges(dir string) ([]nudgeItem, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var items []nudgeItem
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", e.Name(), err)
		}
		var pkt nudgePacket
		if err := json.Unmarshal(raw, &pkt); err != nil {
			return nil, fmt.Errorf("decode %s: %w", e.Name(), err)
		}
		items = append(items, nudgeItem{file: e.Name(), pkt: pkt})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].file < items[j].file })
	return items, nil
}

func runDoomloopDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doomloop drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "state root (default: .dos/doomloop)")
	asJSON := fs.Bool("json", false, "emit JSON")
	deliver := fs.Bool("deliver", false, "DELIVER: POST each queued nudge onto the adjudicated steer bus (fails closed per nudge if the target owns no loop or the floor refuses)")
	addr := fs.String("addr", defaultSessionAddr(), "gateway base URL (for --deliver)")
	key := fs.String("key", defaultGatewayBearerToken(), "bearer credential (for --deliver, only if the gateway sets --require-key)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	st := doomloopStore(*store)
	dir := filepath.Join(st, "outbox")
	items, err := readOutboxNudges(dir)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(stderr, "fak doomloop drain: %v\n", err)
		return 1
	}

	if *deliver {
		return runDoomloopDeliver(stdout, stderr, st, dir, items, *asJSON, *addr, *key)
	}

	// ---- observe path (report the queue; deliver and remove nothing) ----
	rows := make([]drainRow, 0, len(items))
	for _, it := range items {
		rows = append(rows, drainRow{
			File: it.file, Session: it.pkt.Session, Kind: it.pkt.Kind, Reason: it.pkt.Reason,
			Streak: it.pkt.Streak, Reversible: it.pkt.Reversible, Destination: steerDestination(it.pkt.Session),
		})
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr,
			drainReport{Schema: "fak-doomloop-drain/1", Pending: len(rows), Nudges: rows}, "fak doomloop drain")
	}

	if len(rows) == 0 {
		fmt.Fprintf(stdout, "fak doomloop drain: outbox empty (%s) - nothing queued\n", dir)
		return 0
	}
	fmt.Fprintf(stdout, "fak doomloop drain: %d queued nudge(s) in %s (observe-only; nothing delivered or removed)\n\n", len(rows), dir)
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "SESSION\tKIND\tREASON\tSTREAK\tREVERSIBLE\tWOULD-DELIVER-TO")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%v\t%s\n", r.Session, r.Kind, r.Reason, r.Streak, r.Reversible, r.Destination)
	}
	_ = tw.Flush()
	fmt.Fprintln(stdout, "\nnote: observe-only. Run --deliver to POST these onto the adjudicated steer bus (each fails closed if the target owns no loop or the floor refuses).")
	// A pending queue is actionable-but-not-enacted -> exit 3, so a scheduler can gate on
	// "is the correction outbox drained?" without parsing output. Mirrors `fak loop reap`.
	return 3
}

// runDoomloopDeliver is the wired drainer (#3529): it POSTs each queued nudge onto the SAME
// adjudicated /steer bus an operator steer uses, attributed to the doomloop-guard machine
// principal. Delivery is honest per nudge: a 202 archives the packet to <store>/delivered/
// (so a re-drain never re-injects); a 409 STEER_NO_OWNED_LOOP (target owns no loop, #3528) or
// a 422 floor refusal leaves the packet in the outbox and is reported as refused, never as a
// silent success. Exit 3 if any nudge remains undelivered (residual), else 0.
func runDoomloopDeliver(stdout, stderr io.Writer, store, dir string, items []nudgeItem, asJSON bool, addr, key string) int {
	client := &sessionClient{base: strings.TrimRight(addr, "/"), key: key, hc: &http.Client{Timeout: 15 * time.Second}}
	deliveredDir := filepath.Join(store, "delivered")

	rows := make([]drainRow, 0, len(items))
	delivered, refused := 0, 0
	for _, it := range items {
		row := drainRow{
			File: it.file, Session: it.pkt.Session, Kind: it.pkt.Kind, Reason: it.pkt.Reason,
			Streak: it.pkt.Streak, Reversible: it.pkt.Reversible, Destination: steerDestination(it.pkt.Session),
		}
		_, err := client.steerAs(it.pkt.Session, it.pkt.Message, doomloopSteerPrincipal)
		if err == nil {
			row.Delivered, row.Outcome = true, "delivered"
			if mvErr := archiveNudge(deliveredDir, dir, it.file); mvErr != nil {
				// Delivered, but the packet is still in the outbox: report the exact seam so a
				// re-drain that re-injects is a known risk, not a surprise. Still counts as
				// delivered (the steer DID land), but flagged so the operator can clean up.
				row.Outcome, row.Detail = "delivered-not-archived", mvErr.Error()
			}
			delivered++
			rows = append(rows, row)
			continue
		}
		refused++
		row.Outcome, row.Detail = classifyDeliverErr(err)
		rows = append(rows, row)
	}

	// Residual undelivered corrections -> exit 3 (same gate as the observe path), so a
	// scheduler can tell "outbox fully drained" (0) from "some nudges still queued" (3)
	// without parsing output. This is applied in BOTH output modes.
	exit := 0
	if refused > 0 {
		exit = 3
	}

	if asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, drainReport{
			Schema: "fak-doomloop-drain/1", Pending: refused, Delivered: delivered, Refused: refused, Nudges: rows,
		}, "fak doomloop drain"); rc != 0 {
			return rc // encode failure dominates
		}
		return exit
	}

	if len(items) == 0 {
		fmt.Fprintf(stdout, "fak doomloop drain --deliver: outbox empty (%s) - nothing to deliver\n", dir)
		return 0
	}
	fmt.Fprintf(stdout, "fak doomloop drain --deliver: delivered %d of %d nudge(s) onto the adjudicated steer bus as %q (%d refused)\n\n",
		delivered, len(items), doomloopSteerPrincipal, refused)
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "SESSION\tKIND\tREASON\tSTREAK\tOUTCOME\tDETAIL")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\n", r.Session, r.Kind, r.Reason, r.Streak, r.Outcome, r.Detail)
	}
	_ = tw.Flush()
	if refused > 0 {
		fmt.Fprintf(stdout, "\nnote: %d nudge(s) not delivered remain in %s (a proxy-served target owns no loop, or the floor refused). Re-drain against a --native serve, or drop the packet.\n", refused, dir)
	}
	return exit
}

// classifyDeliverErr maps a steer transport error to a truthful outcome token. A typed
// sessionHTTPError lets us branch on the ADJUDICATION (no owned loop / floor refused / other)
// rather than string-matching the human line; a bare transport error is just "error".
func classifyDeliverErr(err error) (outcome, detail string) {
	var he *sessionHTTPError
	if errors.As(err, &he) {
		switch {
		case he.Code == "steer_no_owned_loop":
			return "refused-no-owned-loop", he.Error()
		case he.Status == http.StatusUnprocessableEntity:
			return "refused-floor", he.Error()
		default:
			return "error", he.Error()
		}
	}
	return "error", err.Error()
}

// archiveNudge moves a delivered packet out of the outbox into <store>/delivered/, so the
// enacted correction leaves an auditable record and a re-drain is idempotent.
func archiveNudge(deliveredDir, outboxDir, file string) error {
	if err := os.MkdirAll(deliveredDir, 0o755); err != nil {
		return err
	}
	return os.Rename(filepath.Join(outboxDir, file), filepath.Join(deliveredDir, file))
}

// ---- classify (one-shot over a samples stream) ------------------------------

func runDoomloopClassify(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("doomloop classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "samples JSONL file (default: stdin)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	var r io.Reader = stdin
	if *file != "" {
		f, err := os.Open(*file)
		if err != nil {
			fmt.Fprintf(stderr, "fak doomloop classify: %v\n", err)
			return 1
		}
		defer f.Close()
		r = f
	}
	samples, err := decodeSamples(r)
	if err != nil {
		fmt.Fprintf(stderr, "fak doomloop classify: %v\n", err)
		return 1
	}
	res := doomloop.Classify(samples, doomloop.DefaultConfig())
	if *asJSON {
		raw, _ := json.MarshalIndent(res, "", "  ")
		fmt.Fprintln(stdout, string(raw))
		return 0
	}
	fmt.Fprintf(stdout, "%s\t%s\tstreak=%d\tΔeffort=%d\tΔprogress=%d\n",
		res.Verdict, res.Correction, res.BurningFlatStreak, res.EffortDelta, res.ProgressDelta)
	fmt.Fprintf(stdout, "  %s\n", res.Interpretation())
	return 0
}
