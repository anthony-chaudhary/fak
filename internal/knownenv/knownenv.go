// Package knownenv is the pure core of the fleet-wide known-ENVIRONMENT-failure
// registry (#2144, epic #2136): a signature over TOOL OUTPUT — an error-text
// needle and/or an exit code — mapped to a not-your-fault verdict
// {known-env, owner/eta}. Tool output matching a signature is annotated inline
// ("known env failure #<id>, not your diff") so an agent recognizes a failure it
// cannot fix and does not burn turns debugging it.
//
// Distinct from internal/knownbad by MATCH KEY, not purpose. Both say "not your
// fault," but:
//
//   - knownbad matches by TREE (a peer broke a package; does MY tree intersect a
//     known-broken tree?). Structural — the failure is located by which files it
//     covers.
//   - knownenv matches by OUTPUT FINGERPRINT (a WSL rsync exit-23, a STALE_CRED
//     seat, a peer-owned TIER_DECLARED red). Textual — the failure is recognized
//     by the bytes the tool emitted, which no tree query can see.
//
// The two are complementary: an env flake with no owning tree (rsync-23,
// STALE_CRED) is invisible to knownbad but caught here; a peer-WIP red package is
// caught by both (knownbad by its tree, knownenv by its TIER_DECLARED banner).
//
// This spine is a pure fold: Annotate is a stateless projection over a registry +
// the (output, exitCode) a tool just produced; it reads no clock, no filesystem,
// no network. The impure shell (reading a fleet JSONL registry, piping a live
// tool's output through it, wiring the annotation into the agent's tool-result
// path) is a separate epic child — the SAME pure-core-first shape knownbad landed.
//
// Distinct from SECURITY result-quarantine (#1958-1960), which is about poisoned
// CONTENT the model must not act on; this is about benign ENVIRONMENT flakes the
// agent should not waste turns on. A quarantine says "do not trust this"; a
// knownenv banner says "this is real but not yours to fix."
package knownenv

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Schema is the JSONL record tag every known-env signature row carries. A reader
// filters a shared, multi-writer registry to rows bearing exactly this schema so
// foreign or future rows are ignored rather than misread — the same robustness the
// other fak ledgers use.
const Schema = "fak.known-env.v1"

// DefaultRegistryRel is the repo-relative, fleet-visible registry the (future)
// impure shell reads by default, following the docs/nightrun/*.jsonl idiom the
// other fak ledgers use. The pure core never reads it — it is named here so the
// shell and the core agree on one path.
const DefaultRegistryRel = "docs/nightrun/known-env.jsonl"

// VerdictKnownEnv is the primary not-your-fault verdict class: the failure is a
// known ENVIRONMENT flake (a host/seat/peer condition), not a defect in the
// agent's own diff. It is a small closed vocabulary so a consumer can branch on
// the verdict rather than parse prose.
const VerdictKnownEnv = "known-env"

// Signature is one known-environment-failure entry: a matcher over the output a
// tool just produced, mapped to a not-your-fault verdict. A signature matches when
// its declared conditions ALL hold (AND): a non-empty Needle must be a substring
// of the output, and a non-nil ExitCode must equal the tool's exit code. A
// signature declaring NEITHER condition matches nothing — a catch-all "annotate
// everything" is refused by construction, so an empty/blank row can never flag a
// real failure as environmental.
type Signature struct {
	Schema string `json:"schema"`
	// ID is the stable, human-readable id the banner cites ("rsync-23",
	// "architest-tier-drift", "stale-cred"). It is what an agent or operator greps
	// the registry by, so it is required for a usable signature.
	ID string `json:"id"`
	// Needle is a case-sensitive substring that identifies the failure in tool
	// output (an "error text" signature). Empty means "do not match on text" — the
	// signature then relies on ExitCode alone.
	Needle string `json:"needle,omitempty"`
	// ExitCode, when non-nil, is the exit code that co-identifies the failure (an
	// "exit pattern" signature, e.g. rsync's 23). Nil means "do not match on exit
	// code" — the signature then relies on Needle alone. A pointer so that 0 (a
	// real exit code) is distinguishable from "unset".
	ExitCode *int `json:"exit_code,omitempty"`
	// Verdict is the not-your-fault class (VerdictKnownEnv today); a closed
	// vocabulary so a consumer branches on it instead of parsing Note.
	Verdict string `json:"verdict"`
	// Owner names who owns the environment condition (a team/seat/host), when
	// known, so a parked agent knows where it is being handled. Optional.
	Owner string `json:"owner,omitempty"`
	// ETA is a free-text estimate of when the condition clears ("next seat
	// rotation", "peer merge in flight"), when known. Optional.
	ETA string `json:"eta,omitempty"`
	// Note is the human explanation the banner carries — WHY it is not your diff
	// and what to do instead (retry, park behind the peer, rotate the seat).
	Note string `json:"note,omitempty"`
}

// Matchable reports whether a signature declares at least one condition. A row
// with neither a Needle nor an ExitCode can never legitimately identify a failure,
// so Annotate skips it — the guard against a blank row flagging every failure as
// environmental.
func (s Signature) Matchable() bool {
	return strings.TrimSpace(s.Needle) != "" || s.ExitCode != nil
}

// Match reports whether this signature identifies the given tool output + exit
// code. All DECLARED conditions must hold (AND): a non-empty Needle must be a
// substring of output, and a non-nil ExitCode must equal exitCode. A signature
// that declares no condition (see Matchable) never matches.
func (s Signature) Match(output string, exitCode int) bool {
	if !s.Matchable() {
		return false
	}
	if n := s.Needle; n != "" {
		if !strings.Contains(output, n) {
			return false
		}
	}
	if s.ExitCode != nil {
		if *s.ExitCode != exitCode {
			return false
		}
	}
	return true
}

// Banner renders the standard inline, agent-facing not-your-fault annotation for a
// matched signature, echoing the issue's witness phrase verbatim: "known env
// failure #<id>, not your diff". It appends the verdict, the owner/eta when known,
// and the note so a single line tells the agent both WHAT it is and that it is not
// theirs to fix.
func (s Signature) Banner() string {
	var b strings.Builder
	fmt.Fprintf(&b, "known env failure #%s, not your diff", strings.TrimSpace(s.ID))
	verdict := strings.TrimSpace(s.Verdict)
	if verdict == "" {
		verdict = VerdictKnownEnv
	}
	fmt.Fprintf(&b, " (%s", verdict)
	if o := strings.TrimSpace(s.Owner); o != "" {
		fmt.Fprintf(&b, "; owner=%s", o)
	}
	if e := strings.TrimSpace(s.ETA); e != "" {
		fmt.Fprintf(&b, "; eta=%s", e)
	}
	b.WriteString(")")
	if note := strings.TrimSpace(s.Note); note != "" {
		fmt.Fprintf(&b, ": %s", note)
	}
	return b.String()
}

// Annotation is one matched signature rendered as an agent-facing not-your-fault
// banner: the matched id + verdict for a consumer to branch on, and the rendered
// Line to surface inline.
type Annotation struct {
	ID      string `json:"id"`
	Verdict string `json:"verdict"`
	Owner   string `json:"owner,omitempty"`
	ETA     string `json:"eta,omitempty"`
	Line    string `json:"line"`
}

// Invariant: known environment matching is fail-closed and deterministic. An empty
// or unmatchable signature is strictly rejected by Matchable, guaranteeing that
// ordinary failures without an explicit environment fingerprint are never classified
// as environmental flakes.
//
// Guard condition: Annotate returns an empty slice whenever no signature matches,
// ensuring the caller fails closed and never suppresses agent responsibility.
//
// Annotate scans the output + exit code a tool just produced against a registry
// and returns every signature that matches, each rendered as a not-your-fault
// banner, in registry order. An EMPTY result is the important negative signal:
// nothing in the registry recognizes this failure as environmental, so the agent
// should treat it as a normal (possibly its-own) failure and NOT assume it is
// blameless. Pure: no clock, no I/O — the same (output, exitCode, registry) always
// yields the same annotations.
func Annotate(output string, exitCode int, registry []Signature) []Annotation {
	var out []Annotation
	for _, s := range registry {
		if !s.Match(output, exitCode) {
			continue
		}
		verdict := strings.TrimSpace(s.Verdict)
		if verdict == "" {
			verdict = VerdictKnownEnv
		}
		out = append(out, Annotation{
			ID:      strings.TrimSpace(s.ID),
			Verdict: verdict,
			Owner:   strings.TrimSpace(s.Owner),
			ETA:     strings.TrimSpace(s.ETA),
			Line:    s.Banner(),
		})
	}
	return out
}

// intPtr is a small helper so DefaultRegistry can set an ExitCode inline.
func intPtr(v int) *int { return &v }

// DefaultRegistry is the compiled-in seed registry: the environment failures the
// issue (#2144) documents by name, so the capability recognizes them out of the
// box before any fleet JSONL registry is populated. The (future) shell MERGES this
// seed with the fleet registry at DefaultRegistryRel — the seed is the floor, the
// ledger the live extension. Each entry names its grounding in the issue.
func DefaultRegistry() []Signature {
	return []Signature{
		{
			// wsl_go_test_capture_technique: WSL `go test` output-capture rsync
			// aborts with exit 23. Flaky host plumbing, not a code defect — retry.
			Schema:   Schema,
			ID:       "rsync-23",
			Needle:   "code 23",
			ExitCode: intPtr(23),
			Verdict:  VerdictKnownEnv,
			Owner:    "host",
			ETA:      "transient — retry",
			Note:     "WSL go-test output-capture rsync exit-23 flake; re-run the capture, do not debug the tool",
		},
		{
			// architest_tier_drift_wedges_trunk: a peer's undeclared leaf reds the
			// shared trunk for EVERYONE via architest TIER_DECLARED — not your diff.
			Schema:  Schema,
			ID:      "architest-tier-drift",
			Needle:  "TIER_DECLARED",
			Verdict: VerdictKnownEnv,
			Owner:   "peer",
			ETA:     "peer merge in flight",
			Note:    "a peer's new leaf is missing its architest tier and reds the shared trunk for everyone; park behind the peer, do not edit their tier under a lease you do not hold",
		},
		{
			// A seat whose credential aged out surfaces as STALE_CRED. The seat
			// rotates; the failure is not the agent's diff.
			Schema:  Schema,
			ID:      "stale-cred",
			Needle:  "STALE_CRED",
			Verdict: VerdictKnownEnv,
			Owner:   "fleet",
			ETA:     "next seat rotation",
			Note:    "the launch seat's credential aged out; rotate onto a fresh seat, do not debug the request",
		},
	}
}

// ParseRegistry folds JSONL bytes into signatures. Robust for a shared,
// multi-writer append registry (the same discipline knownbad.ParseLedger uses):
// blank lines are skipped, a line that is not valid JSON or does not carry this
// package's Schema is skipped (a torn append from a crashed peer, or a foreign row
// in a co-located ledger, must not blind every reader), and a row that declares no
// match condition (see Matchable) is dropped so a blank row can never flag every
// failure as environmental. It never errors — the worst a bad line can do is not
// be seen.
func ParseRegistry(data []byte) []Signature {
	var out []Signature
	for _, line := range strings.Split(string(data), "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		var sig Signature
		if err := json.Unmarshal([]byte(s), &sig); err != nil {
			continue
		}
		if sig.Schema != Schema {
			continue
		}
		if !sig.Matchable() {
			continue
		}
		out = append(out, sig)
	}
	return out
}

// MarshalLine renders one signature as a single compact JSONL line (no trailing
// newline). The shell appends "\n" when it writes, matching the other fak
// registries' append idiom.
func MarshalLine(sig Signature) (string, error) {
	b, err := json.Marshal(sig)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// EffectiveRegistry merges the compiled-in seed with a fleet ledger's parsed
// signatures: the seed is the floor, the ledger the live extension. Rows are
// deduplicated by ID (after whitespace trimming). A fleet row sharing a seed
// row's id OVERRIDES (refines) it but keeps the seed row's position — the floor
// slot is stable; the fleet just updates its contents. A fleet row with a new id
// is appended after all seed rows, in fleet order; two fleet rows sharing an id
// resolve last-wins. A row that is not Matchable, or whose trimmed id is empty
// (the banner cites the id, so an id-less row is unusable), never enters the
// effective registry — so a blank fleet row can neither fire nor displace a seed
// floor entry. Pure: returns a fresh slice and mutates neither input.
func EffectiveRegistry(seed, fleet []Signature) []Signature {
	out := make([]Signature, 0, len(seed)+len(fleet))
	index := make(map[string]int, len(seed)+len(fleet))
	admit := func(s Signature) {
		id := strings.TrimSpace(s.ID)
		if id == "" || !s.Matchable() {
			return
		}
		if at, ok := index[id]; ok {
			out[at] = s
			return
		}
		index[id] = len(out)
		out = append(out, s)
	}
	for _, s := range seed {
		admit(s)
	}
	for _, s := range fleet {
		admit(s)
	}
	return out
}

// AnnotateFromLedger is the one-call entrypoint the (future) impure shell uses so
// the dirty-lane wiring stays thin: it folds raw fleet-registry JSONL bytes
// through ParseRegistry, merges them over the compiled-in seed via
// EffectiveRegistry (seed floor ⊕ fleet extension), and annotates the given tool
// output + exit code against the result. The shell reads the file (see
// DefaultRegistryRel); this function never touches disk — nil or empty
// fleetJSONL degrades cleanly to annotating against the seed alone. Pure: the
// same (output, exitCode, fleetJSONL) always yields the same annotations.
func AnnotateFromLedger(output string, exitCode int, fleetJSONL []byte) []Annotation {
	effective := EffectiveRegistry(DefaultRegistry(), ParseRegistry(fleetJSONL))
	return Annotate(output, exitCode, effective)
}
