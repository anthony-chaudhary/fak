package rulesynth

// stream.go is rung 1's LIVE wiring: it folds the kernel's running adjudication
// stream into a mineable near-miss corpus — the args/command-bearing dual of
// internal/harvest's LabelRow harvester (#537).
//
// THE GAP IT CLOSES. Detect is the near-miss PREDICATE (is THIS call a near-miss?),
// but nothing turned the kernel's actual LOG into a corpus of them: the package's
// tests hand-built []NearMiss, and harvest.Harvester — the one Emitter that does ride
// the live stream — folds into abi.LabelRow, which carries no command. So "mine the
// refusal/near-miss LOG" (the issue title) had no producer. Harvester is that
// producer: attach it as an abi.Emitter and every ADMITTED shell/exec call whose
// command reached a guarded tree by a verb the floor does not yet recognize becomes a
// row Propose can cluster into the next structural rule.
//
// It changes no verdict and lands nothing: like harvest, it is opt-in (attached by a
// bench / the compiled loop via abi.RegisterEmitter), it only OBSERVES, and its sole
// output is a corpus an operator mines through Propose -> Validate -> ManifestDiff.

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// NearMissCorpus is the thread-safe append log of command-bearing near-misses the
// Harvester mines from the live adjudication stream — the mineable input Propose
// consumes. It is the args/command-bearing corpus abi.LabelRow cannot hold.
type NearMissCorpus struct {
	mu   sync.Mutex
	rows []NearMiss
}

// NewNearMissCorpus returns an empty corpus.
func NewNearMissCorpus() *NearMissCorpus { return &NearMissCorpus{} }

func (c *NearMissCorpus) add(nm NearMiss) {
	c.mu.Lock()
	c.rows = append(c.rows, nm)
	c.mu.Unlock()
}

// Rows returns a snapshot copy of the captured near-misses (safe to hand to Propose).
func (c *NearMissCorpus) Rows() []NearMiss {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]NearMiss(nil), c.rows...)
}

// Len is the number of captured near-misses.
func (c *NearMissCorpus) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.rows)
}

// Harvester is the abi.Emitter that folds the kernel's live adjudication stream into
// a NearMissCorpus (rung 1). It records a near-miss for every call the floor ADMITTED
// whose command reached a guarded tree by a verb the floor does not yet recognize —
// the Detect definition, reused verbatim so the live capture cannot drift from the
// predicate the package tests. A DENY needs no new rule (it is already caught), so
// only Allow verdicts are mined.
//
// Reliability is carried by the downstream honesty gate, not the capture: the corpus
// is deliberately broad (a guarded tree NAMED by an admitted command), so it can hold
// an admitted read of a guarded file as well as a true unrecognized write — exactly as
// Detect does. Validate is what refuses to ship an over-broad synthesized rule (it
// REVERTs any rule that regresses a benign call), so the capture errs toward recall and
// the gate, not the miner, decides what lands. This is the AUTOHARNESS thesis: the
// harness carries reliability, not the proposer.
type Harvester struct {
	corpus *NearMissCorpus
	globs  []string
}

// NewHarvester builds a stream harvester that mines near-misses reaching any of
// guardedGlobs into corpus. guardedGlobs are the protected path fragments (the floor's
// SelfModifyGlobs); DefaultHarnessGlobs is the harness/witness set.
func NewHarvester(corpus *NearMissCorpus, guardedGlobs []string) *Harvester {
	return &Harvester{corpus: corpus, globs: append([]string(nil), guardedGlobs...)}
}

// Emit folds one adjudication event into the near-miss corpus. It keys ONLY on
// EvDecide (the verdict-resolved event, emitted exactly once per decided call) so an
// allowed call's later EvDispatch/EvComplete cannot double-record it, and only on an
// Allow verdict (a deny is already caught). The command is read from the call's args
// exactly as the floor reads it, and Detect is the single source of the near-miss
// definition.
func (h *Harvester) Emit(ev abi.Event) {
	if ev.Kind != abi.EvDecide || ev.Verdict == nil || ev.Verdict.Kind != abi.VerdictAllow {
		return
	}
	cmd, arg := commandArg(ev.Call)
	if cmd == "" {
		return // no command-bearing arg — nothing to mine
	}
	if nm, ok := Detect(Call{Tool: ev.Call.Tool, Arg: arg, Command: cmd}, h.globs); ok {
		h.corpus.add(nm)
	}
}

// commandArg extracts the shell command string and the arg key carrying it from a
// call's args, checking the same keys the adjudicator's commandSelfModify rung reads
// ("command", then "cmd"). It resolves the args Ref the way the floor does (inline
// directly, else the active resolver), so the harvester mines the exact command the
// floor adjudicated. Returns "","" when no command-bearing scalar arg is present.
func commandArg(c *abi.ToolCall) (cmd, arg string) {
	if c == nil {
		return "", ""
	}
	b := refBytes(c.Args)
	if len(b) == 0 {
		return "", ""
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return "", ""
	}
	for _, k := range []string{"command", "cmd"} {
		if s, ok := m[k].(string); ok && s != "" {
			return s, k
		}
	}
	return "", ""
}

// refBytes returns the bytes behind a Ref: the inline payload directly, else the
// active resolver's bytes (the same resolution decide.go's refBytes does). It returns
// nil on any failure — a near-miss that cannot be read is simply not mined (fail to
// not-capture, never panic on the observe path).
func refBytes(r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		return nil
	}
	b, err := res.Resolve(context.Background(), r)
	if err != nil {
		return nil
	}
	return b
}
