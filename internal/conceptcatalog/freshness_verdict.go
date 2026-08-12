package conceptcatalog

// The freshness probe's UNKNOWN verdict, and the single surface that renders it (#5962).
//
// FreshnessResult answers a probe that RAN: fresh, or not-fresh plus the artifacts that
// drifted. It has no way to say "I could not check", and CheckFresh fills the result
// with Fresh:true BEFORE it reads the artifacts, then hands that same value back
// alongside a read error. Every CLI surface encoded the result and only then tested the
// error, so `fak concept freshness --check --json` printed `{"fresh":true,...}` on
// stdout while exiting 1. Stdout is what --json is for: a consumer parsing it read the
// reassuring answer for a check that had just failed.
//
// FreshnessProbe closes that by construction. It is built from the (result, error) pair
// TOGETHER, so a probe that errored cannot produce a fresh envelope no matter what flag
// the result carries, and RenderFreshness is the only place the CLI turns either into
// bytes and an exit code.

import (
	"encoding/json"
	"fmt"
	"io"
)

// FreshnessVerdict is a freshness answer in one word. Three values, not two: "I compared
// the bytes and they match" and "I never got to compare" are both non-stale, and only
// the first may be reported as fresh.
type FreshnessVerdict string

const (
	// VerdictFresh: the check ran and every tracked artifact matched a regeneration.
	VerdictFresh FreshnessVerdict = "fresh"
	// VerdictStale: the check ran and named the artifacts that drifted. The only
	// verdict that may drive a refusal or a regeneration.
	VerdictStale FreshnessVerdict = "stale"
	// VerdictUnknown: the check could not run, so it proved neither freshness nor
	// drift. Non-stale, and emphatically not fresh.
	VerdictUnknown FreshnessVerdict = "unknown"
)

// FreshnessProbe is the JSON envelope every freshness surface emits. Fresh and
// StalePaths are kept for the consumers that already read them, and Fresh is false
// under VerdictUnknown, so a reader that only greps `"fresh":true` still cannot be
// told a check succeeded when it did not.
type FreshnessProbe struct {
	Verdict    FreshnessVerdict `json:"verdict"`
	Fresh      bool             `json:"fresh"`
	StalePaths []string         `json:"stale_paths,omitempty"`
	Regenerate string           `json:"regenerate"`
	// Unchecked carries the reason the probe could not run, in the words the error
	// used, so a --json consumer sees the cause rather than only the absence.
	Unchecked string `json:"unchecked,omitempty"`
}

// ProbeFreshness folds a check's (result, error) pair into a verdict. An error always
// outranks the result's own Fresh flag: a check that failed reports what it knows,
// which is nothing. This is a constructor rather than a struct literal at each call
// site precisely because the defect it fixes was call sites reading the two halves of
// the pair in the wrong order.
func ProbeFreshness(res FreshnessResult, err error) FreshnessProbe {
	if err != nil {
		// Keep whatever cure the check managed to name; fall back to the worktree
		// command so an operator reading the unknown verdict still has a next step.
		regenerate := res.Regenerate
		if regenerate == "" {
			regenerate = RegenerateCommand
		}
		return FreshnessProbe{Verdict: VerdictUnknown, Regenerate: regenerate, Unchecked: err.Error()}
	}
	p := FreshnessProbe{Verdict: VerdictStale, Fresh: res.Fresh, StalePaths: res.StalePaths, Regenerate: res.Regenerate}
	if res.Fresh {
		p.Verdict = VerdictFresh
	}
	return p
}

// JSON encodes the probe envelope.
func (p FreshnessProbe) JSON() []byte { b, _ := json.Marshal(p); return b }

// RenderFreshness writes one freshness answer and returns the process exit code.
//
// label names the command for its diagnostics ("fak concept freshness"); treeNote is
// the noun phrase for the tree that was checked (" in the staged tree", or empty for
// the worktree). The JSON envelope is written from the PROBE, never from the raw
// result, so the unknown verdict reaches stdout instead of a stale `"fresh":true`.
// Unknown exits 1 like stale does — a check that could not run is not a pass — but it
// says so in different words, because the cure for "regenerate the artifacts" and the
// cure for "the checker itself is broken" are not the same cure.
func RenderFreshness(stdout, stderr io.Writer, label, treeNote string, res FreshnessResult, err error, asJSON bool) int {
	p := ProbeFreshness(res, err)
	if asJSON {
		_ = json.NewEncoder(stdout).Encode(p)
	}
	switch p.Verdict {
	case VerdictUnknown:
		// Always on stderr, even under --json: the JSON already carries the verdict,
		// and the operator needs the cause on the stream diagnostics live on.
		fmt.Fprintf(stderr, "%s: freshness UNKNOWN%s -- the check could not run, so it proved neither fresh nor stale: %s\n", label, treeNote, p.Unchecked)
		return 1
	case VerdictStale:
		if !asJSON {
			fmt.Fprintln(stderr, "stale generated concept artifacts"+treeNote+":")
			for _, path := range p.StalePaths {
				fmt.Fprintln(stderr, " -", path)
			}
			fmt.Fprintln(stderr, "regenerate:", p.Regenerate)
		}
		return 1
	default:
		if !asJSON {
			fmt.Fprintln(stdout, "concept generated artifacts fresh"+treeNote)
		}
		return 0
	}
}
