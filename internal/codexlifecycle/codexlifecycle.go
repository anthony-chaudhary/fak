// Package codexlifecycle folds a native Codex rollout transcript
// (~/.codex/sessions/**/*.jsonl) into an EXACTLY-ONCE task lifecycle keyed by the
// exact turn_id (#4785).
//
// THE DEFECT IT FENCES. A rollout can emit task_started(turn_id), do substantial
// work, and then emit ANOTHER task_started without either task_complete or
// turn_aborted for the earlier turn. A fold that tracks "am I in a turn?" as a
// boolean (tools/codex_turn_health.py's in_turn flag) silently resets on the second
// start, so the abandoned turn's duration, tokens, and tool outcomes leak into
// whichever turn happens to be open next. That is producer/data-integrity debt: a
// budget governor or critical-path analysis cannot know where the abandoned task
// ended. The 2026-07-15 corpus audit found 37 such mid-session gaps that are real
// task bodies (one spans 8,357 records incl. 2,253 function calls), not empty
// bookkeeping.
//
// THE CONTRACT. For every observed task_started(turn_id) this fold persists exactly
// one TYPED terminal state, and never fabricates a success:
//
//   - Complete     — an observed task_complete for that exact turn_id.
//   - Aborted      — an observed turn_aborted for that exact turn_id.
//   - Superseded   — SYNTHESIZED: a later task started while this turn was still
//     active. The boundary is evidence-backed (the succeeding start's own
//     timestamp), and the outcome is explicitly non-success.
//   - ProcessDeath — SYNTHESIZED: the rollout's final start never terminated and the
//     rollout is STALE, so the writer died or the file is truncated.
//   - Live         — SYNTHESIZED: the final start never terminated but the rollout is
//     FRESH, so the task may genuinely still be running. EndedAt stays empty — a live
//     turn has not ended, and inventing an end is the fabrication this fences.
//
// Provenance separates what was READ from what was INFERRED, so a consumer can refuse
// to attribute post-boundary token/tool deltas to a turn whose end fak synthesized.
// Raw events are never rewritten; the fold is a projection beside them.
//
// Tier: foundation (1) — see internal/architest. Pure: events in, reconciled
// lifecycle out. Stdlib-only, imports nothing internal, off the hot path. Freshness
// is an injected caller decision (a bool), never a clock read, so the fold stays
// deterministic and testable.
package codexlifecycle

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// Event kinds, as they appear in a rollout's event_msg payload `type`.
const (
	KindStarted  = "task_started"
	KindComplete = "task_complete"
	KindAborted  = "turn_aborted"
)

// Outcome is a task's typed terminal state. Every reconciled task carries exactly
// one; there is no "unknown" member, because an unclassified start is precisely the
// defect this package exists to remove.
type Outcome string

const (
	Complete     Outcome = "complete"
	Aborted      Outcome = "aborted"
	Superseded   Outcome = "superseded"
	ProcessDeath Outcome = "process_death"
	Live         Outcome = "live"
)

// Success reports whether the outcome is a genuine completion. Superseded,
// ProcessDeath, and Live are NOT successes — a consumer must not count them as one.
func (o Outcome) Success() bool { return o == Complete }

// Provenance records whether a terminal state was observed in the rollout or
// synthesized by this reconciler.
type Provenance string

const (
	Observed    Provenance = "observed"
	Synthesized Provenance = "synthesized"
)

// Event is one lifecycle-relevant record read from a rollout, flattened from
// {"timestamp":…,"type":"event_msg","payload":{"type":…,"turn_id":…}}.
type Event struct {
	Kind       string `json:"kind"`
	TurnID     string `json:"turn_id,omitempty"`
	Timestamp  string `json:"timestamp,omitempty"`
	Reason     string `json:"reason,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
}

// Task is one reconciled lifecycle: a start bound to exactly one typed terminal.
type Task struct {
	TurnID             string     `json:"turn_id"`
	StartedAt          string     `json:"started_at,omitempty"`
	EndedAt            string     `json:"ended_at,omitempty"` // empty for Live — a running turn has no end
	Outcome            Outcome    `json:"outcome"`
	Provenance         Provenance `json:"provenance"`
	Reason             string     `json:"reason,omitempty"`
	DurationMS         int64      `json:"duration_ms,omitempty"`
	TrailingEmptyAbort bool       `json:"trailing_empty_abort,omitempty"`
}

// Report is the reconciled lifecycle of ONE rollout. The three integrity classes are
// reported SEPARATELY (not merged into one "bad" count) because they have different
// causes and different fixes.
type Report struct {
	Tasks []Task `json:"tasks"`

	// SubstantiveCompleted indicates at least one substantive task turn completed.
	SubstantiveCompleted bool `json:"substantive_completed,omitempty"`

	// CompletedWithTrailingAbort indicates substantive completion followed by an empty abort.
	CompletedWithTrailingAbort bool `json:"completed_with_trailing_abort,omitempty"`

	// Orphans are terminals whose turn_id was never started in this rollout — a
	// truncated head, or a terminal for a turn that began in an earlier file.
	Orphans []Event `json:"orphans,omitempty"`

	// Reused are turn_ids started more than once. The exact-id contract makes this
	// detectable; a boolean in_turn fold cannot see it at all.
	Reused []string `json:"reused,omitempty"`

	// MultiplyTerminated are turn_ids that received a second terminal after an
	// already-OBSERVED one.
	MultiplyTerminated []string `json:"multiply_terminated,omitempty"`
}

// Unclassified returns tasks the fold left without a typed terminal state. It is the
// corpus witness's assertion target and must always be empty: the fold assigns an
// outcome to every start, so a non-empty result is a bug in this package, not data.
func (r Report) Unclassified() []Task {
	var out []Task
	for _, t := range r.Tasks {
		if t.Outcome == "" {
			out = append(out, t)
		}
	}
	return out
}

// CountByOutcome tallies reconciled tasks by outcome — the before/after shape a
// provider/version rollup reports.
func (r Report) CountByOutcome() map[Outcome]int {
	out := map[Outcome]int{}
	for _, t := range r.Tasks {
		out[t.Outcome]++
	}
	return out
}

// Fold reconciles events into an exactly-once lifecycle. fresh says whether the
// rollout is still warm (recent mtime / an live session); it decides ONLY how the
// final unterminated start is typed — Live when fresh, ProcessDeath when stale — so
// process death is never confused with a running task.
//
// Events must be in rollout order (append-only files already are).
func Fold(events []Event, fresh bool) Report {
	rep := Report{Tasks: []Task{}}
	idx := map[string]int{}         // turn_id -> index into rep.Tasks
	terminated := map[string]bool{} // turn_id -> already carries a terminal
	active := -1                    // index of the open task, or -1

	for _, ev := range events {
		switch ev.Kind {
		case KindStarted:
			if ev.TurnID == "" {
				// A start with no id cannot be addressed; it would silently alias
				// every other unkeyed turn. Record it as an orphan rather than
				// opening an unreachable task.
				rep.Orphans = append(rep.Orphans, ev)
				continue
			}
			if _, seen := idx[ev.TurnID]; seen {
				rep.Reused = append(rep.Reused, ev.TurnID)
				continue
			}
			// THE REPAIR: a new task while an older one is still active must never
			// leave the old turn open. Close it with a typed NON-success outcome and
			// an evidence-backed boundary — this start's own timestamp — so no later
			// token/tool delta is attributable to the abandoned turn.
			if active >= 0 {
				t := &rep.Tasks[active]
				t.Outcome = Superseded
				t.Provenance = Synthesized
				t.EndedAt = ev.Timestamp
				t.Reason = "superseded_by_turn:" + ev.TurnID
				terminated[t.TurnID] = true
			}
			rep.Tasks = append(rep.Tasks, Task{TurnID: ev.TurnID, StartedAt: ev.Timestamp})
			idx[ev.TurnID] = len(rep.Tasks) - 1
			active = len(rep.Tasks) - 1

		case KindComplete, KindAborted:
			i := -1
			if ev.TurnID != "" {
				if j, seen := idx[ev.TurnID]; seen {
					i = j
				}
			} else if active >= 0 {
				// Legacy rollouts emit turn_aborted with no turn_id. Within one
				// rollout a turn is single-threaded, so the unkeyed terminal can only
				// belong to the open turn. Bind it there; with no open turn it is an
				// orphan.
				i = active
			}
			if i < 0 {
				rep.Orphans = append(rep.Orphans, ev)
				continue
			}
			t := &rep.Tasks[i]
			if terminated[t.TurnID] {
				if t.Provenance == Observed {
					rep.MultiplyTerminated = append(rep.MultiplyTerminated, t.TurnID)
					continue
				}
				// A real terminal arrived for a turn we had SYNTHESIZED closed.
				// Observed evidence outranks an inference: repair the row rather than
				// keep the guess.
			}
			t.Outcome = Complete
			if ev.Kind == KindAborted {
				t.Outcome = Aborted
			}
			t.Provenance = Observed
			t.EndedAt = ev.Timestamp
			t.Reason = ev.Reason
			t.DurationMS = ev.DurationMS
			terminated[t.TurnID] = true
			if active == i {
				active = -1
			}
		}
	}

	// The final start, if it never terminated: freshness — not a guess — separates a
	// dead writer from a live one.
	if active >= 0 {
		t := &rep.Tasks[active]
		t.Provenance = Synthesized
		if fresh {
			t.Outcome = Live
			t.Reason = "rollout_fresh_task_may_still_be_running"
		} else {
			t.Outcome = ProcessDeath
			t.Reason = "rollout_stale_no_terminal_observed"
		}
	}

	hasComplete := false
	for i := range rep.Tasks {
		if rep.Tasks[i].Outcome == Complete {
			hasComplete = true
		}
		if i > 0 && rep.Tasks[i-1].Outcome == Complete && rep.Tasks[i].Outcome == Aborted {
			if rep.Tasks[i].DurationMS <= 2000 {
				rep.Tasks[i].TrailingEmptyAbort = true
			}
		}
	}
	rep.SubstantiveCompleted = hasComplete
	if len(rep.Tasks) > 1 && rep.Tasks[len(rep.Tasks)-1].TrailingEmptyAbort {
		rep.CompletedWithTrailingAbort = true
	}
	return rep
}

// ParseRollout reads a Codex rollout JSONL stream and returns its lifecycle events
// in file order. Non-lifecycle records are skipped. Rollouts are append-only and can
// carry a torn final line from a crashed writer or a stray non-JSON line, so a
// malformed line is skipped rather than failing the whole read — the truncated tail
// is exactly the process-death evidence this package must survive to classify.
func ParseRollout(r io.Reader) ([]Event, error) {
	_, events, err := ReadRollout(r)
	return events, err
}

// ReadRollout reads a rollout stream and returns both its identifying Meta and its
// lifecycle events. Only the FIRST session_meta record identifies the file: a
// subagent rollout carries the PARENT's metadata in its inherited context further
// down, so a last-wins read would relabel every child as its parent.
func ReadRollout(r io.Reader) (Meta, []Event, error) {
	return readRollout(r, nil)
}

// ReadRolloutCensus is ReadRollout plus the #10668 sidecar: every event_msg
// payload type counted (including the ones this reader does not interpret, so
// upstream's future typed error/retry/terminal events are observed, not
// dropped) and the torn-tail shape. The returned events and Meta are
// identical to ReadRollout's.
func ReadRolloutCensus(r io.Reader) (Meta, []Event, RolloutCensus, error) {
	var census RolloutCensus
	meta, events, err := readRollout(r, &census)
	return meta, events, census, err
}

func readRollout(r io.Reader, census *RolloutCensus) (Meta, []Event, error) {
	var meta Meta
	var out []Event
	haveMeta := false
	lastParsed := true

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Timestamp string `json:"timestamp"`
			Type      string `json:"type"`
			Payload   struct {
				Type          string `json:"type"`
				TurnID        string `json:"turn_id"`
				Reason        string `json:"reason"`
				DurationMS    int64  `json:"duration_ms"`
				ID            string `json:"id"`
				AltID         string `json:"session_id"`
				ModelProvider string `json:"model_provider"`
				CLIVersion    string `json:"cli_version"`
				CWD           string `json:"cwd"`
			} `json:"payload"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil {
			lastParsed = false
			continue
		}
		lastParsed = true
		switch rec.Type {
		case "session_meta":
			if haveMeta {
				continue
			}
			haveMeta = true
			meta = Meta{
				RolloutID:  firstNonEmpty(rec.Payload.ID, rec.Payload.AltID),
				Provider:   strings.TrimSpace(rec.Payload.ModelProvider),
				CLIVersion: strings.TrimSpace(rec.Payload.CLIVersion),
				CWD:        strings.TrimSpace(rec.Payload.CWD),
			}
		case "event_msg":
			// THE #10668 CENSUS: every event_msg payload type is counted by
			// name, interpreted or not — an unrecognized type is never an
			// error, and the census is the FULL event-type inventory (the
			// interpreted kinds included), not just its unknown tail. Free
			// text (e.g. a terminal task_complete's last_agent_message) is
			// never retained here; class extraction lives in the analytics
			// reader (errorclass.go's contract).
			census.addPayload(rec.Payload.Type)
			switch rec.Payload.Type {
			case KindStarted, KindComplete, KindAborted:
				out = append(out, Event{
					Kind:       rec.Payload.Type,
					TurnID:     strings.TrimSpace(rec.Payload.TurnID),
					Timestamp:  rec.Timestamp,
					Reason:     rec.Payload.Reason,
					DurationMS: rec.Payload.DurationMS,
				})
			}
		}
	}
	if census != nil {
		census.TornTail = !lastParsed
	}
	if err := sc.Err(); err != nil {
		return meta, out, err
	}
	return meta, out, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return ""
}
