package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// guard_lookahead.go — the P0 slice of the look-ahead reset (#5207, epic #5202,
// docs/notes/CONCEPT-LOOKAHEAD-RESET-2026-07-17.md).
//
// The full protocol is: at the PreCompact reset boundary, gate a speculative rollout
// (fork + turn-capped headless resume under a deny floor), witness where it ends,
// distill a compact lesson, and inject that lesson at the next SessionStart(compact)
// beside the compacted summary — so the resumed agent carries the FORESIGHT of what
// would have happened without the POISON of the drifted path.
//
// What is REAL here (the honest, shippable slice):
//   - the base-SHA-keyed lesson CHANNEL (append + fresh-read + render);
//   - the SessionStart(source=compact) PICKUP that injects a fresh same-base-SHA lesson —
//     the one half WIRED to a live hook (runGuardSessionStartHook, guard_sessionstart.go);
//   - the PreCompact admission GATE via loopmgr.AdmitSpeculation (advisory, fail-open) —
//     BUILT and unit-witnessed, but NOT yet called from the PreCompact hook: gating a
//     rollout that cannot be spawned buys nothing, so its call site lands with the
//     transcript-fork transport below.
//
// What is NOT yet built (reported as such, never faked):
//   - the transcript-fork transport (a `claude --resume <fork>` on a COPY of the live
//     transcript; `--fork-session` has no repo referent) — so resolveLookaheadForkSession
//     returns none and no live rollout is spawned against the real session;
//   - the rollout runner that witnesses (build/affected-tests -> W3, divergence -> W2) and
//     distills the Lesson — i.e. the PRODUCER that fills the channel.
//
// Every path here is FAIL-OPEN: any error, missing store, or missing transport is a silent
// no-op that leaves the hook's exit code untouched. A look-ahead must never wedge compaction.

// lookaheadLesson is one witnessed, base-SHA-scoped lesson minted by a speculative rollout
// and injected at the next compaction boundary. It is deliberately small: the branch outcome
// distilled to a single rendered claim plus the evidence rung it may assert at.
type lookaheadLesson struct {
	TS      string `json:"ts"`
	Session string `json:"session,omitempty"`
	// BaseSHA is the trunk HEAD the rollout ran at. A lesson is only FRESH — and only
	// injected — while trunk still points at this SHA; once trunk moves the lesson is stale
	// (the same ExpiresSHA-ancestry discipline as dos_recall / GitEvidenceResolver).
	BaseSHA string `json:"base_sha"`
	// Rung is the witnessed authority the lesson asserts at: "W3" (diff/test-witnessed FACT)
	// or "W2" (activity-divergence RISK). W1/W0 self-report never reaches this channel.
	Rung string `json:"rung,omitempty"`
	Text string `json:"text"`
}

// lookaheadLessonLedger is the per-workspace lesson store, co-located in the same fleet regDir
// the drive-carry and identity stores already use so no new env has to be injected.
func lookaheadLessonLedger(regDir string) string {
	return filepath.Join(regDir, "lookahead_lessons.jsonl")
}

// appendLookaheadLesson records one lesson to the channel. It is the seam the (not-yet-built)
// rollout runner writes through; exposed now so the consumer half is testable end to end.
func appendLookaheadLesson(regDir string, l lookaheadLesson) error {
	return appendJSONL(lookaheadLessonLedger(regDir), l)
}

// readFreshLookaheadLesson returns the most-recent lesson whose BaseSHA matches baseSHA — the
// only lessons still valid at inject time. A blank baseSHA (git unavailable), a missing store,
// or a parse miss yields (_, false): fail-open, never inject a stale or unverifiable lesson.
func readFreshLookaheadLesson(regDir, baseSHA string) (lookaheadLesson, bool) {
	baseSHA = strings.TrimSpace(baseSHA)
	if baseSHA == "" {
		return lookaheadLesson{}, false
	}
	raw, err := os.ReadFile(lookaheadLessonLedger(regDir))
	if err != nil {
		return lookaheadLesson{}, false
	}
	var fresh lookaheadLesson
	found := false
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var l lookaheadLesson
		if json.Unmarshal([]byte(line), &l) != nil {
			continue // a torn/partial append is skipped, not fatal
		}
		if strings.TrimSpace(l.BaseSHA) != baseSHA {
			continue
		}
		if strings.TrimSpace(l.Text) == "" {
			continue
		}
		fresh = l
		found = true // last matching row wins (most recent)
	}
	return fresh, found
}

// renderLookaheadLesson formats a lesson for injection, leading with its witnessed rung so the
// reader sees the authority (a W3 FACT vs a W2 RISK flag) explicitly — the rung is load-bearing,
// not decoration.
func renderLookaheadLesson(l lookaheadLesson) string {
	rung := strings.TrimSpace(l.Rung)
	label := "Look-ahead lesson"
	switch strings.ToUpper(rung) {
	case "W3":
		label = "Look-ahead lesson (Witnessed W3)"
	case "W2":
		label = "Look-ahead lesson (Risk flag W2)"
	}
	sha := strings.TrimSpace(l.BaseSHA)
	if len(sha) > 12 {
		sha = sha[:12]
	}
	return fmt.Sprintf("%s [base %s]: %s", label, sha, strings.TrimSpace(l.Text))
}

// lookaheadSpeculationCandidate builds the admission input for a PreCompact rollout. A rollout
// is effect-free by construction (it runs in a reaped fork under a deny floor, landing nothing),
// so it proves the read-only + idempotent surface the governor requires. The EV/slack figures
// are conservative P0 placeholders — the measured slack and P(correct) wiring is a follow-on;
// they are documented as constants rather than fabricated per-turn estimates.
func lookaheadSpeculationCandidate() loopmgr.SpeculationCandidate {
	return loopmgr.SpeculationCandidate{
		Tool:           "lookahead-rollout",
		ReadOnlyHint:   true, // reaped fork: no trunk mutation escapes
		IdempotentHint: true,
		Destructive:    false,
		// Placeholder economics (P0): positive EV with headroom. Replaced by measured
		// slack + witnessed P(correct) when the rollout runner lands.
		CorrectProbability:  0.5,
		LatencySavedMillis:  2000,
		CostIfWrongMillis:   500,
		SlackMillis:         5000,
		EstimatedWorkMillis: 1000,
	}
}

// resolveLookaheadForkSession would resolve a hermetic COPY of the live transcript to roll
// forward speculatively. That transcript-fork transport is the unbuilt seam (the flagship
// `fak guard -- claude` prefix lives in the Claude Code transcript JSONL, and `--fork-session`
// has no repo referent), so today it returns ("", false): the admission gate still runs and is
// logged, but no rollout is ever spawned against the real session. This is the single honest
// "not yet" that keeps the P0 hook safe on the shared tree.
func resolveLookaheadForkSession() (string, bool) {
	return "", false
}

// lookaheadRolloutArgv builds the detached rollout command SHAPE: the child resumes the fork
// transcript headless, capped at --max-turns 3, fronted by `fak guard` so it runs under the
// guard policy floor (the rollout-specific tightened floor that also denies push/gh/steer is a
// follow-on). It uses only verified claude flags (--resume/-p/--max-turns/--dangerously-skip-
// permissions are all live in-repo). Pure and side-effect-free so it is unit-testable without
// spawning anything.
func lookaheadRolloutArgv(fakExe, claudeExe, forkSession string) []string {
	child := []string{
		claudeExe, "--resume", forkSession,
		"-p", "Continue the current plan.",
		"--max-turns", "3",
		"--dangerously-skip-permissions",
	}
	if strings.TrimSpace(fakExe) == "" {
		return child
	}
	front := []string{fakExe, "guard", "--"}
	return append(front, child...)
}

// maybeSpawnLookaheadRolloutFailOpen is the PreCompact producer trigger — built and unit-
// witnessed here, but with NO PreCompact call site yet (see the file header). On an allowed
// compaction it (1) gates a speculative rollout through loopmgr.AdmitSpeculation, logging the
// advisory decision, and (2) spawns the detached, turn-capped rollout ONLY when a fork
// transcript is resolvable — which, until the fork transport lands, it never is. It returns
// immediately either way: the hook never waits on a rollout. Fail-open throughout.
func maybeSpawnLookaheadRolloutFailOpen(stderr io.Writer) {
	defer func() { _ = recover() }() // a look-ahead must never wedge compaction
	dec := loopmgr.AdmitSpeculation(lookaheadSpeculationCandidate())
	if !dec.Admit {
		fmt.Fprintf(stderr, "fak guard PreCompact: look-ahead rollout not admitted (%s); continuing\n", dec.Reason)
		return
	}
	fork, ok := resolveLookaheadForkSession()
	if !ok {
		// Admitted, but the transcript-fork transport is not yet wired: honest no-op. We never
		// resume the LIVE session UUID concurrently (that would collide), so nothing is spawned.
		fmt.Fprintln(stderr, "fak guard PreCompact: look-ahead rollout admitted; transcript-fork transport not yet wired, skipping (fail-open)")
		return
	}
	_ = fork // unreachable today; the spawn lands when resolveLookaheadForkSession resolves.
}

// lookaheadNow is the injectable clock for lesson timestamps (kept for the eventual producer
// and its tests); it mirrors the pattern the other guard writers use.
func lookaheadNow() string { return time.Now().UTC().Format(time.RFC3339) }

// parseHookSource best-effort parses the `source` a Claude Code SessionStart hook payload
// carries on stdin (one of startup|resume|clear|compact). An empty body or parse miss returns
// "" — the source is advisory routing, never a gate.
func parseHookSource(payload []byte) string {
	if len(payload) == 0 {
		return ""
	}
	var p struct {
		Source string `json:"source"`
	}
	if json.Unmarshal(payload, &p) != nil {
		return ""
	}
	return strings.TrimSpace(p.Source)
}

// lookaheadLessonForCompact is the SessionStart(compact) pickup: if the hook payload's source is
// "compact" AND a fresh look-ahead lesson exists at the CURRENT trunk base SHA, it returns the
// rendered lesson for injection. Every other case (nil/blank stdin, non-compact source, git
// unavailable, no fresh lesson) returns ("", false) — fail-open, inject nothing.
func lookaheadLessonForCompact(stdin io.Reader) (string, bool) {
	return lookaheadLessonForCompactPayload(readHookStdin(stdin))
}

func lookaheadLessonForCompactPayload(payload []byte) (string, bool) {
	source := parseHookSource(payload)
	cwd, err := os.Getwd()
	if err != nil {
		return "", false
	}
	baseSHA := currentHeadSha(findRepoRoot(cwd))
	return lookaheadLessonForCompactSource(source, resolveSweepRegDir(""), baseSHA)
}

// lookaheadLessonForCompactSource is the pure pickup core: it returns the rendered fresh lesson
// only when the SessionStart source is "compact" AND a lesson exists at baseSHA. Every other
// case is ("", false). Split out from the stdin/git resolution so the source-gating and
// staleness logic is testable without a live hook payload or repo.
func lookaheadLessonForCompactSource(source, regDir, baseSHA string) (string, bool) {
	if strings.TrimSpace(source) != "compact" {
		return "", false
	}
	lesson, ok := readFreshLookaheadLesson(regDir, baseSHA)
	if !ok {
		return "", false
	}
	return renderLookaheadLesson(lesson), true
}
