// Command surface: `fak supervisor-seat wake` — the dispatch-side wake entry
// point for the deterministic supervisor seat (issue #13464, build-plan step
// S3 of the Chief-of-Staff runtime note). One invocation drives exactly one
// wake through the bounded cycle — assemble -> decide -> gate -> lower ->
// journal — over PAYLOAD-FREE typed witnesses only: the dos_status liveness
// digest head, the fleetmon census rows (closed class tokens), the open
// escalation packet heads, and the live lease table. There is no transcript,
// log body, or free-text field anywhere in the wire shape, and the seat never
// reads one.
//
// S3 lands the DRY-RUN cycle: the full decision + envelope-gate runs, the
// record is journaled durably, and ZERO admission calls are made — the live
// lower stage (real dos_arbitrate / dispatch-admit / packet-emit bindings
// behind warn mode) is the S4 follow-on, and --apply refuses with a named
// token until it ships. The soak switch is the envelope's own environment
// variable (FAK_SUPERVISOR_ENVELOPE_MODE); unset soaks (warn), a malformed
// token enforces — a typo can only tighten the envelope.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/supervisoragent"
)

// defaultSeatJournalPath is where the seat's durable decision journal lives
// when --journal is not given: one append-only JSONL file under the
// workspace's .fak home, next to the other durable operator receipts.
const defaultSeatJournalPath = ".fak/supervisor-seat/decisions.jsonl"

// seatWakeTopicDefault addresses the record when neither the payload nor the
// --topic flag names a wake address.
const seatWakeTopicDefault = "wake://dispatch/supervisor-seat"

func cmdSupervisorSeat(args []string) {
	os.Exit(runSupervisorSeat(args, os.Stdin, os.Stdout, os.Stderr))
}

// runSupervisorSeat is the testable core of the verb: it drives one dry-run
// wake and returns the process exit code. 0 = the wake completed and its
// record was journaled; 2 = usage error or the live-lower refusal; 1 = a
// runtime failure (bad payload, unreadable journal).
func runSupervisorSeat(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 || args[0] != "wake" {
		fmt.Fprintln(stderr, "usage: fak supervisor-seat wake [--payload FILE|-] [--journal PATH] [--envelope FILE] [--topic ADDR] [--json] [--apply]")
		fmt.Fprintln(stderr, "  one dry-run wake of the deterministic supervisor seat: assemble -> decide -> gate -> journal, zero admission calls (#13464 S3)")
		return 2
	}
	fs := flag.NewFlagSet("supervisor-seat", flag.ContinueOnError)
	fs.SetOutput(stderr)
	payloadPath := fs.String("payload", "-", "typed wake payload: JSON file, or - for stdin")
	journalPath := fs.String("journal", defaultSeatJournalPath, "durable decision journal (append-only JSONL)")
	envelopePath := fs.String("envelope", "", "earned-envelope JSON file (omit: the zero-width fail-closed envelope)")
	topic := fs.String("topic", seatWakeTopicDefault, "the wake address the decision record is written for (a payload topic overrides it)")
	jsonOut := fs.Bool("json", false, "print the journaled decision record as JSON")
	apply := fs.Bool("apply", false, "request live lowering (refused: S3 is dry-run only)")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if *apply {
		fmt.Fprintln(stderr, "SUPERVISOR_SEAT_LIVE_LOWER_NOT_SHIPPED: S3 lands the dry-run cycle only; live lowering behind warn mode is the S4 follow-on under #13464")
		return 2
	}

	var raw []byte
	var err error
	if *payloadPath == "-" {
		raw, err = io.ReadAll(stdin)
	} else {
		raw, err = os.ReadFile(*payloadPath)
	}
	if err != nil {
		fmt.Fprintf(stderr, "SUPERVISOR_SEAT_PAYLOAD_UNREADABLE: %v\n", err)
		return 1
	}
	payload, err := parseSeatWakePayload(raw)
	if err != nil {
		fmt.Fprintf(stderr, "SUPERVISOR_SEAT_PAYLOAD_INVALID: %v\n", err)
		return 1
	}

	env, err := loadSeatEnvelope(*envelopePath)
	if err != nil {
		fmt.Fprintf(stderr, "SUPERVISOR_SEAT_ENVELOPE_INVALID: %v\n", err)
		return 1
	}

	journal, err := supervisoragent.NewFileJournal(*journalPath)
	if err != nil {
		fmt.Fprintf(stderr, "SUPERVISOR_SEAT_JOURNAL_UNOPENABLE: %v\n", err)
		return 1
	}

	wakeTopic := *topic
	if payload.Topic != "" {
		wakeTopic = payload.Topic
	}
	at := time.Now().UTC()
	if !payload.DecidedAt.IsZero() {
		at = payload.DecidedAt
	}

	// S3 is dry-run by construction: the seat binds NO admission verbs, so
	// the zero-side-effect proof is structural — the preview cannot reach a
	// seam that is not there. The envelope still gates the proposal, and the
	// soak mode comes from the envelope's own environment switch.
	seat := &supervisoragent.Seat{
		Journal:  journal,
		Verbs:    nil,
		Envelope: env,
		Mode:     supervisoragent.ModeFromValue(os.Getenv(supervisoragent.EnvelopeModeEnv)),
	}
	rec, err := seat.RunWake(wakeTopic, payload.sources(), at, true)
	if err != nil {
		fmt.Fprintf(stderr, "SUPERVISOR_SEAT_WAKE_FAILED: %v\n", err)
		return 1
	}

	if *jsonOut {
		b, _ := json.Marshal(rec)
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprintf(stdout, "SUPERVISOR_SEAT_WAKE seq=%d topic=%s verb=%s reason=%s verdict=%s(%s) outcome=%s artifact=%s\n",
		rec.Seq, rec.Topic, rec.ProposedVerb, rec.ReasonToken,
		rec.EnvelopeVerdict.Auth, rec.EnvelopeVerdict.Reason, rec.Outcome, orDash(rec.ArtifactRef))
	return 0
}

// ---- closed, payload-free wire shapes -------------------------------------
//
// Every surface is a pointer witness: a missing key OR present:false is an
// ABSENT witness (the fence-#1 escalate marker); present:true carries the
// typed value. Unknown keys anywhere are a hard error — the closed contract
// cannot acquire a payload field by drift.

type seatWitness[T any] struct {
	Present bool `json:"present"`
	Value   T    `json:"value"`
}

type seatLaneRefJSON struct {
	Lane string `json:"lane"`
	Kind string `json:"kind"`
}

type seatLivenessJSON struct {
	RunID     string            `json:"run_id"`
	Class     string            `json:"class"`
	Commits   int               `json:"commits"`
	SilentFor time.Duration     `json:"silent_for"`
	Region    []seatLaneRefJSON `json:"region"`
}

type seatWorkerJSON struct {
	RunID string `json:"run_id"`
	Issue string `json:"issue"`
	Lane  string `json:"lane"`
	Class string `json:"class"` // closed fleetmon class token
}

type seatEscalationJSON struct {
	ID         string `json:"id"`
	RunID      string `json:"run_id"`
	Issue      string `json:"issue"`
	Class      string `json:"class"`
	Severity   string `json:"severity"`
	ReasonCode string `json:"reason_code"`
}

type seatLeaseJSON struct {
	Lane string   `json:"lane"`
	Kind string   `json:"kind"`
	Tree []string `json:"tree"`
}

type seatWakePayload struct {
	Topic       string                             `json:"topic,omitempty"`
	DecidedAt   time.Time                          `json:"decided_at,omitempty"`
	Liveness    *seatWitness[seatLivenessJSON]     `json:"liveness"`
	Workers     *seatWitness[[]seatWorkerJSON]     `json:"workers"`
	Escalations *seatWitness[[]seatEscalationJSON] `json:"escalations"`
	Leases      *seatWitness[[]seatLeaseJSON]      `json:"leases"`
}

// seatFleetmonClasses is the closed census-class token set the wire accepts:
// fleetmon's seven-way classification, verbatim. A token outside the set is a
// payload-level error, not a silently-folded verdict.
var seatFleetmonClasses = map[fleetmon.Classification]bool{
	fleetmon.ClassHealthy:         true,
	fleetmon.ClassCompletedFinal:  true,
	fleetmon.ClassDead:            true,
	fleetmon.ClassStaleTranscript: true,
	fleetmon.ClassAuthRateBlocked: true,
	fleetmon.ClassStaleChild:      true,
	fleetmon.ClassAttention:       true,
}

// seatActionKinds is the closed envelope-width class set: the six action
// vocabulary tokens, verbatim.
var seatActionKinds = map[string]supervisoragent.ActionKind{
	"spawn":    supervisoragent.ActionSpawn,
	"replace":  supervisoragent.ActionReplace,
	"replan":   supervisoragent.ActionRedispatch,
	"widen":    supervisoragent.ActionWiden,
	"escalate": supervisoragent.ActionEscalate,
	"hold":     supervisoragent.ActionHold,
}

func parseSeatWakePayload(raw []byte) (seatWakePayload, error) {
	var p seatWakePayload
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return p, fmt.Errorf("decode wake payload: %w", err)
	}
	if tok, err := dec.Token(); err == nil && tok != nil {
		return p, errors.New("wake payload must be exactly one JSON document")
	}
	for i, w := range p.valueSlice() {
		if !seatFleetmonClasses[fleetmon.Classification(w.Class)] {
			return p, fmt.Errorf("workers[%d].class %q is not a closed fleetmon class token", i, w.Class)
		}
	}
	return p, nil
}

// valueSlice flattens the workers witness to a plain slice for validation
// (an absent or empty witness validates as nothing).
func (p seatWakePayload) valueSlice() []seatWorkerJSON {
	if p.Workers == nil || !p.Workers.Present {
		return nil
	}
	return p.Workers.Value
}

// sources projects the parsed payload onto the package's closed Sources:
// every PRESENT surface maps to its payload-free projection, every ABSENT
// one stays an explicit absent witness (never a silently-empty present).
func (p seatWakePayload) sources() supervisoragent.Sources {
	src := supervisoragent.Sources{
		Liveness:    supervisoragent.Absent[supervisoragent.Liveness](),
		Workers:     supervisoragent.Absent[[]supervisoragent.WorkerCensus](),
		Escalations: supervisoragent.Absent[[]supervisoragent.Escalation](),
		Leases:      supervisoragent.Absent[[]leaseref.ArbiterLease](),
	}
	if p.Liveness != nil && p.Liveness.Present {
		l := p.Liveness.Value
		region := make([]supervisoragent.LaneRef, 0, len(l.Region))
		for _, r := range l.Region {
			region = append(region, supervisoragent.LaneRef{Lane: r.Lane, Kind: r.Kind})
		}
		src.Liveness = supervisoragent.Seen(supervisoragent.Liveness{
			RunID:     l.RunID,
			Class:     l.Class,
			Commits:   l.Commits,
			SilentFor: l.SilentFor,
			Region:    region,
		})
	}
	if p.Workers != nil && p.Workers.Present {
		rows := make([]supervisoragent.WorkerCensus, 0, len(p.Workers.Value))
		for _, w := range p.Workers.Value {
			rows = append(rows, supervisoragent.WorkerCensus{
				RunID: w.RunID,
				Issue: w.Issue,
				Lane:  w.Lane,
				Class: fleetmon.Classification(w.Class),
			})
		}
		src.Workers = supervisoragent.Seen(rows)
	}
	if p.Escalations != nil && p.Escalations.Present {
		packets := make([]supervisoragent.Escalation, 0, len(p.Escalations.Value))
		for _, e := range p.Escalations.Value {
			packets = append(packets, supervisoragent.Escalation{
				ID:         e.ID,
				RunID:      e.RunID,
				Issue:      e.Issue,
				Class:      e.Class,
				Severity:   e.Severity,
				ReasonCode: e.ReasonCode,
			})
		}
		src.Escalations = supervisoragent.Seen(packets)
	}
	if p.Leases != nil && p.Leases.Present {
		leases := make([]leaseref.ArbiterLease, 0, len(p.Leases.Value))
		for _, l := range p.Leases.Value {
			leases = append(leases, leaseref.ArbiterLease{Lane: l.Lane, LaneKind: l.Kind, Tree: l.Tree})
		}
		src.Leases = supervisoragent.Seen(leases)
	}
	return src
}

// seatEnvelopeFile is the earned-envelope wire shape: per-class earned widths
// keyed by the closed action-vocabulary tokens, plus the witness_refused
// collapse bit. Omitted file = the zero-width fail-closed envelope.
type seatEnvelopeFile struct {
	Widths         map[string]int `json:"widths"`
	WitnessRefused bool           `json:"witness_refused"`
}

func loadSeatEnvelope(path string) (supervisoragent.Envelope, error) {
	env := supervisoragent.Envelope{Widths: map[supervisoragent.ActionKind]int{}}
	if path == "" {
		return env, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return env, fmt.Errorf("read envelope file: %w", err)
	}
	var f seatEnvelopeFile
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return env, fmt.Errorf("decode envelope file: %w", err)
	}
	for k, w := range f.Widths {
		kind, ok := seatActionKinds[k]
		if !ok {
			return env, fmt.Errorf("envelope width class %q is not a closed action-vocabulary token", k)
		}
		if w < 0 {
			return env, fmt.Errorf("envelope width for %q is negative (%d); widths earn, they never un-earn", k, w)
		}
		env.Widths[kind] = w
	}
	env.WitnessRefused = f.WitnessRefused
	return env, nil
}
