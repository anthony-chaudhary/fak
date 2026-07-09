package main

// fak doomloop - the meta-monitor shell over internal/doomloop.
//
// This is the impure half of the doom-loop guard: it gathers the two axes the
// pure classifier folds (effort spent vs verified forward progress), maintains a
// durable per-worker sample history, classifies each worker, records every
// decision to an accountability ledger, and - when asked - QUEUES a soft,
// reversible re-anchor nudge to the worker's steer outbox. Queuing, NOT
// delivering: no transport drains that outbox yet, so a correction here is
// recorded-and-queued, not applied, until a drainer is wired (see below). The
// destructive rung (kill/replace) is not here by design; the ladder tops out at
// an operator escalation record.
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
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/doomloop"
)

const doomloopUsage = `fak doomloop - two-axis doom-loop guard (effort vs verified progress)

  tick      ingest one observation for a worker, classify its history, record the decision
  scan      fold the store into a fleet verdict table (one row per worker)
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
