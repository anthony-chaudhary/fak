package guardtrace

// bench.go is the FIXTURE -> REPLAYABLE-TRACE bridge for issue #1846 ("fak ablate
// --from-session"). It turns a guardtrace.Fixture into a *bench.Trace (the shape
// ablate.Sweep replays) plus an *engine.Cassette that answers each call with the
// REAL provider usage the fixture recorded, instead of the mock engine's
// payload-size synthesis.
//
// WHY guardtrace.Fixture IS THE SOURCE, NOT THE DECISION JOURNAL. #1846 asked to
// "reuse the decision-journal / trajectory recorder as the source". Both were
// investigated and neither carries what a replay needs:
//   - internal/journal.Row stores Tool + ArgsDigest (a content HASH), by explicit
//     design ("never materializes a blob... leaks no payload into the log") — no
//     args bytes to replay, and no usage/token fields at all.
//   - internal/trajectory.Turn is the same shape for the same reason: ArgsDigest/
//     ResultDigest, a single TokenEstimate int, no 4-way usage split.
//   - The dojo live-episode corpus (.dojo/live-episodes, internal/dojo/live.go) is
//     start-marker-only ({mode, command, started, workspace}) — no turns at all.
//
// guardtrace.Fixture is the one format in the tree that already carries BOTH the
// real call args (Call.Args json.RawMessage) and the real per-turn usage
// (Turn.Usage{InputTokens, OutputTokens, CacheReadInputTokens,
// CacheCreationInputTokens}) — because it is the INPUT `fak guard --replay-trace`
// feeds through the real floor. Reusing it as the ablate source means: (a) no new
// wire format, (b) the same parser (LoadFixture/ParseFixture) both verbs share,
// (c) a session captured this way is provably the same shape guard already
// round-trips end to end.
//
// CAVEAT — the write side. This file is the READ side only. A live-session
// CAPTURE writer (dump a Fixture from a running `fak guard` / `fak serve`
// session) is NOT wired here: the natural hook point is the gateway's per-turn
// request handling (internal/gateway), which has other agents' uncommitted work
// in flight as of this change — touching it was avoided on collision-safety
// grounds, not because it is architecturally hard. Until that writer exists, the
// --from-session input is a Fixture JSON file authored the same way
// testdata/guard-trace-e2e.json is (by hand, or by a future capture writer using
// this exact schema) — see cmd/fak/ablate.go's --from-session flag help and
// docs/cli-reference.md for the operator-facing statement of this gap.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// SessionTracePrefix marks a bench.Trace built FROM A CAPTURED SESSION (as opposed
// to a checked-in testdata suite like "tau2-smoke"). ablate's report provenance
// carries the trace's SliceID verbatim, so a report bound to a captured session is
// visibly distinct from the frozen mock suite at a glance (acceptance #1 of #1846:
// "provenance/workload hash is bound to real captured traffic, not tau2-smoke").
const SessionTracePrefix = "session:"

// ToBenchTrace converts a Fixture into a *bench.Trace: one bench.Call per fixture
// call, in turn order, flattened across all turns. The trace's SliceID is
// SessionTracePrefix + the fixture's SliceID (or "captured" if the fixture left it
// blank), so WorkloadHash — and every report/provenance field derived from it —
// is bound to the captured session, never confusable with a suite trace.
func (f *Fixture) ToBenchTrace() *bench.Trace {
	slice := f.SliceID
	if slice == "" {
		slice = "captured"
	}
	t := &bench.Trace{SliceID: SessionTracePrefix + slice}
	for ti, turn := range f.Turns {
		for ci, c := range turn.Calls {
			t.Calls = append(t.Calls, bench.Call{
				Tool: c.Tool,
				Args: append(json.RawMessage(nil), c.Args...),
				Meta: map[string]string{
					"turn": itoa(ti),
					"call": itoa(ci),
				},
			})
		}
	}
	return t
}

// ToCassette builds an engine.Cassette that answers every call in the fixture with
// the REAL per-turn usage the session recorded — carrying the input / cache_read /
// cache_creation / output split onto whichever CassetteEntry matches that call's
// (tool, args), so an ablate arm replayed with --engine cassette:<id> (see
// RegisterSessionEngine) reports actual billed usage, not a synthesized count
// (acceptance #2 of #1846).
//
// A caveat inherent to the cassette's (tool, args) keying: if the SAME (tool, args)
// pair repeats across turns with DIFFERENT usage (e.g. a cache_read that grows
// turn over turn), the cassette can only bind ONE usage value per key — the LAST
// turn's usage for that pair wins. This does not affect the aggregate token totals
// bench.RunArm sums (each trace call still contributes once), only which turn's
// per-call usage a duplicate (tool,args) reports if inspected individually.
func (f *Fixture) ToCassette() *engine.Cassette {
	var entries []engine.CassetteEntry
	for _, turn := range f.Turns {
		perCall := splitTurnUsage(turn)
		for i, c := range turn.Calls {
			entries = append(entries, engine.CassetteEntry{
				Tool: c.Tool,
				Args: append([]byte(nil), c.Args...),
				Usage: engine.Usage{
					InputTokens:         perCall[i].InputTokens,
					OutputTokens:        perCall[i].OutputTokens,
					CacheReadTokens:     perCall[i].CacheReadInputTokens,
					CacheCreationTokens: perCall[i].CacheCreationInputTokens,
				},
			})
		}
	}
	return engine.NewCassette(entries)
}

// splitTurnUsage divides one turn's aggregate provider usage evenly across its
// calls so each call carries a share of the turn's real billed tokens (the
// provider bills per TURN, not per call — a turn's tool_use blocks share one
// usage object on the wire). Remainders land on the first call so the per-turn
// sum reconstructs exactly. This is an approximation where a turn has more than
// one call — documented, not hidden: the ablate report's TOTAL token columns for
// the session are exact (they sum back to the turn usage), but a per-call
// breakdown within a multi-call turn is an even split, not the provider's true
// per-call attribution (the wire does not report one).
func splitTurnUsage(t Turn) []Usage {
	n := len(t.Calls)
	if n == 0 {
		return nil
	}
	out := make([]Usage, n)
	dist := func(total int) []int {
		base := total / n
		rem := total % n
		vals := make([]int, n)
		for i := range vals {
			vals[i] = base
		}
		for i := 0; i < rem; i++ {
			vals[i]++
		}
		return vals
	}
	in := dist(t.Usage.InputTokens)
	out2 := dist(t.Usage.OutputTokens)
	cr := dist(t.Usage.CacheReadInputTokens)
	cc := dist(t.Usage.CacheCreationInputTokens)
	for i := 0; i < n; i++ {
		out[i] = Usage{
			InputTokens:              in[i],
			OutputTokens:             out2[i],
			CacheReadInputTokens:     cr[i],
			CacheCreationInputTokens: cc[i],
		}
	}
	return out
}

// SessionEngineID derives a stable, collision-resistant engine id for a captured
// session's cassette, so registering it (abi.RegisterEngine) never clashes with the
// built-in "mock"/"cassette"/"inkernel" ids or with another session's registration
// in the same process. Two loads of the SAME fixture bytes get the SAME id
// (content-addressed), so re-running --from-session against an unchanged file is
// idempotent instead of leaking a fresh registry entry each time.
func SessionEngineID(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "session:" + hex.EncodeToString(sum[:])[:16]
}

// LoadSessionTrace loads a Fixture from path and returns the bench.Trace + the
// Cassette to register as its engine — the whole read-side pipeline
// `fak ablate --from-session` drives. engineID is derived from the fixture's own
// bytes (SessionEngineID) so the caller can register it deterministically.
func LoadSessionTrace(path string) (t *bench.Trace, cas *engine.Cassette, engineID string, err error) {
	raw, rerr := os.ReadFile(path)
	if rerr != nil {
		return nil, nil, "", fmt.Errorf("guardtrace: load session %s: %w", path, rerr)
	}
	f, perr := ParseFixture(raw)
	if perr != nil {
		return nil, nil, "", fmt.Errorf("guardtrace: load session %s: %w", path, perr)
	}
	return f.ToBenchTrace(), f.ToCassette(), SessionEngineID(raw), nil
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
