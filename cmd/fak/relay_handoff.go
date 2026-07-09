// relay_handoff.go — rung C6 of the perpetual-session relay track (#1875, epic #1860):
// `fak relay handoff`, the OFFLINE write half of the baton IO pair whose read half is
// `fak relay resume` (relay_resume.go, C7). A closing relay leg (or an operator staging
// one) states its leg's end-state in plain flags and gets back a canonical
// `fak.relay.baton.v1` file its successor will pick up — the "write a deterministic
// baton from leg state" half of the issue's Done condition made concrete.
//
// Like `fak relay resume` and `fak session envelope`, this is OFFLINE by design and
// deliberately pure: it PROJECTS the stated flags into a relay.Baton and encodes it with
// relay.Marshal (the same byte-stable, no-clock, no-map codec resume reads back), then
// writes those bytes. No gateway dial, no live rotation, no resolver calls, no clock. The
// determinism is the witness: the same flags produce byte-identical baton bytes across
// runs and processes, so `fak relay handoff ... | fak relay resume --baton - --json`
// round-trips exactly (relay.Marshal(relay.Parse(x)) == x for canonical x).
//
//	fak relay handoff --relay-id RID-... --leg 0 --start-sha <sha> \
//	  --objective "…" --done-when "…" --next-action "…" \
//	  --tombstone-reason RELAY_ROTATED --tombstone-at-sha <sha> \
//	  --artifact commit:<sha> --artifact issue:#1875 \
//	  --held-region "cmd/fak/**" --out .fak/relay/leg0.baton.json
//	fak relay handoff --relay-id RID-... --start-sha <sha> --json   # bytes to stdout
//
// What it is NOT: it does not observe the leg (it does not read git for the sha or the
// objective digest) — a baton is a STATED handoff a successor RE-VERIFIES, so the closing
// leg supplies the pointers and this verb just canonicalizes them. Content validation
// beyond "encodes to a well-formed baton" (ArtifactKind vocabulary, ref resolution, the
// schema-tag gate) is the reader's job on the resume side; the one thing enforced here is
// that a relay id and a progress-cursor start sha are present, since a baton with neither
// is not a usable handoff.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/relay"
)

// artifactFlag collects repeated --artifact "kind:ref" values in flag order. It is an
// exported-in-file flag.Value so a test can drive the parse directly.
type artifactFlag []relay.Artifact

func (a *artifactFlag) String() string { return fmt.Sprintf("%d artifact(s)", len(*a)) }

// Set parses one "kind:ref" pair. The split is on the FIRST colon only, so a ref that
// itself contains a colon (a URL-shaped file ref, say) survives intact. A missing colon
// or an empty half is a usage error — a half-formed artifact would decode but point
// nowhere.
func (a *artifactFlag) Set(v string) error {
	kind, ref, ok := strings.Cut(v, ":")
	kind, ref = strings.TrimSpace(kind), strings.TrimSpace(ref)
	if !ok || kind == "" || ref == "" {
		return fmt.Errorf("artifact %q must be kind:ref (for example commit:<sha> or issue:#1875)", v)
	}
	*a = append(*a, relay.Artifact{Kind: kind, Ref: ref})
	return nil
}

// stringsFlag collects repeated string flags (--held-region, --open-question,
// --do-not-rederive) in flag order, so the emitted slice order is the flag order and the
// baton bytes stay deterministic.
type stringsFlag []string

func (s *stringsFlag) String() string { return strings.Join(*s, ",") }
func (s *stringsFlag) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// runRelayHandoff is the testable shell: it returns the process exit code (0 ok, 1 a
// runtime/write error, 2 a usage error) and takes its streams explicitly, mirroring
// runRelayResume and runSessionEnvelope. It projects the stated flags into a relay.Baton,
// encodes it with relay.Marshal, and writes the canonical bytes to --out (or stdout with
// --json / when no --out is given).
func runRelayHandoff(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("relay handoff", flag.ContinueOnError)
	fs.SetOutput(stderr)

	relayID := fs.String("relay-id", "", "stable id for the whole relay (REQUIRED), e.g. RID-2026-07-06-a")
	leg := fs.Int("leg", 0, "the closing leg's number (>= 0, lineage ordering — never progress)")
	parentTrace := fs.String("parent-trace", "", "trace id of the closing leg (recorded by the successor as lineage)")
	objective := fs.String("objective", "", "the active objective text carried verbatim into the successor")
	pinID := fs.String("pin-id", "", "objective pin id (default: derived from --relay-id); the digest is computed, never stated")
	doneWhen := fs.String("done-when", "", "one-line durable-store predicate the successor evaluates before doing more work")
	nextAction := fs.String("next-action", "", "one line naming the next atomic action (not a recap)")
	startSHA := fs.String("start-sha", "", "git commit the closing leg anchored progress on (REQUIRED — the re-verifiable cursor)")
	ledgerRef := fs.String("ledger-ref", "", "optional intent/run/DOS ledger row the successor re-reads for verified progress")
	tombReason := fs.String("tombstone-reason", "", "relay reason token, e.g. RELAY_ROTATED, RELAY_GOAL_DONE, RELAY_PARKED_UNSAFE")
	tombAtSHA := fs.String("tombstone-at-sha", "", "git commit observed when the baton was written (default: --start-sha)")
	tombNote := fs.String("tombstone-note", "", "short display-only operator note (never consumed as progress)")
	out := fs.String("out", "", "write the baton to this path (default: stdout)")
	asJSON := fs.Bool("json", false, "emit the canonical baton wire bytes to stdout (implied when --out is empty)")

	var artifacts artifactFlag
	fs.Var(&artifacts, "artifact", "durable pointer as kind:ref; repeatable (commit:<sha>, issue:#1875, memory:<slug>, ledger:<id>, file:<glob>)")
	var heldRegion stringsFlag
	fs.Var(&heldRegion, "held-region", "lane/path glob the successor must re-acquire before writing; repeatable")
	var openQuestions stringsFlag
	fs.Var(&openQuestions, "open-question", "durable pointer/short label for an unresolved decision; repeatable")
	var doNotRederive stringsFlag
	fs.Var(&doNotRederive, "do-not-rederive", "durable pointer to a closed dead end the successor should not retry; repeatable")

	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	// A baton with no relay id or no progress anchor is not a usable handoff: the
	// successor keys the read on the relay id and re-verifies the start sha. Refuse both
	// as usage errors rather than writing a baton that resume would have to reject.
	if strings.TrimSpace(*relayID) == "" {
		fmt.Fprintln(stderr, "fak relay handoff: --relay-id is required")
		return 2
	}
	if strings.TrimSpace(*startSHA) == "" {
		fmt.Fprintln(stderr, "fak relay handoff: --start-sha is required (the progress cursor's re-verifiable anchor)")
		return 2
	}
	if *leg < 0 {
		fmt.Fprintf(stderr, "fak relay handoff: --leg must be >= 0, got %d\n", *leg)
		return 2
	}

	// The objective pin's digest is COMPUTED by ctxplan.NewObjectivePin over (pin id,
	// text, leg), never hand-set — so the successor's ObjectivePin.Verify runs the same
	// digest and a mismatch means corrupt input, not objective drift. Default the pin id
	// to the relay id so a caller who only states --objective still gets a stable pin.
	resolvedPinID := strings.TrimSpace(*pinID)
	if resolvedPinID == "" {
		resolvedPinID = strings.TrimSpace(*relayID)
	}
	pin := ctxplan.NewObjectivePin(resolvedPinID, *objective, *leg)

	// The tombstone's observed sha defaults to the progress anchor: a leg that does not
	// separately state where it stood when it wrote the baton stood at its start sha.
	atSHA := strings.TrimSpace(*tombAtSHA)
	if atSHA == "" {
		atSHA = strings.TrimSpace(*startSHA)
	}

	baton := relay.Baton{
		Schema:      relay.Schema,
		RelayID:     strings.TrimSpace(*relayID),
		Leg:         *leg,
		ParentTrace: strings.TrimSpace(*parentTrace),
		Objective:   pin,
		DoneWhen:    strings.TrimSpace(*doneWhen),
		ProgressCursor: relay.ProgressCursor{
			StartSHA:   strings.TrimSpace(*startSHA),
			LedgerRef:  strings.TrimSpace(*ledgerRef),
			HeldRegion: heldRegion,
		},
		NextAction:    strings.TrimSpace(*nextAction),
		OpenQuestions: openQuestions,
		Artifacts:     artifacts,
		DoNotRederive: doNotRederive,
		Tombstone: relay.Tombstone{
			Reason: strings.TrimSpace(*tombReason),
			AtSHA:  atSHA,
			Note:   strings.TrimSpace(*tombNote),
		},
	}

	// relay.Marshal projects nil slices to `[]` and encodes in declaration order, so the
	// bytes are byte-stable for a given flag set — the determinism that is the witness.
	wire, err := relay.Marshal(baton)
	if err != nil {
		fmt.Fprintf(stderr, "fak relay handoff: %v\n", err)
		return 1
	}

	// No --out (or explicit --json) prints the canonical bytes to stdout so the verb
	// composes into a pipe (`fak relay handoff ... | fak relay resume --baton -`). A
	// stated --out writes the file and echoes the path, mirroring the offline-write idiom.
	if strings.TrimSpace(*out) == "" || *asJSON {
		if strings.TrimSpace(*out) == "" {
			fmt.Fprintln(stdout, string(wire))
			return 0
		}
	}

	if dir := filepath.Dir(*out); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			fmt.Fprintf(stderr, "fak relay handoff: create baton dir: %v\n", err)
			return 1
		}
	}
	if err := os.WriteFile(*out, append(wire, '\n'), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak relay handoff: write baton: %v\n", err)
		return 1
	}
	if *asJSON {
		fmt.Fprintln(stdout, string(wire))
		return 0
	}
	fmt.Fprintf(stdout, "wrote %s baton for relay %s (leg %d) to %s\n", relay.Schema, baton.RelayID, baton.Leg, *out)
	return 0
}
