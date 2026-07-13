package sessionreplay

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// Replay re-runs the fixture's captured turn under its active regime and returns
// the regime-conditioned verdict. It is the assertable core: a pure, world-safe
// function of `(fixture.Turn, fixture.ActiveRegime)`.
//
//   - It resolves the active regime to a REAL policy floor (regimeManifest) and
//     re-adjudicates the captured tool call through the genuine
//     internal/adjudicator — so the frozen verdict guards the actual harness
//     decision, not a stand-in.
//   - It never calls a model, never touches the network or filesystem, and never
//     mutates the world (the captured args are handed to the adjudicator INLINE;
//     nothing is executed). This is a test-time decision replay, not #4107-R3
//     runtime idempotent replay.
//
// Determinism: the adjudicator's Adjudicate is a pure decision over the policy
// floor and the inline args, so identical `(Turn, ActiveRegime)` always replays
// to the identical verdict.
func Replay(f Fixture) (Verdict, error) {
	if f.Schema != SchemaV1 {
		return Verdict{}, fmt.Errorf("sessionreplay: unsupported schema %q (want %q)", f.Schema, SchemaV1)
	}
	if strings.TrimSpace(f.Turn.Tool) == "" {
		return Verdict{}, fmt.Errorf("sessionreplay: fixture has no captured tool call")
	}

	manifest, err := regimeManifest(f.ActiveRegime)
	if err != nil {
		return Verdict{}, err
	}
	pol, err := policy.Parse(manifest)
	if err != nil {
		return Verdict{}, fmt.Errorf("sessionreplay: regime %q floor did not parse: %w", f.ActiveRegime, err)
	}

	args := []byte(f.Turn.Args)
	if len(args) == 0 {
		args = []byte("{}")
	}
	call := &abi.ToolCall{
		Tool: f.Turn.Tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: args},
	}
	v := adjudicator.New(pol).Adjudicate(context.Background(), call)
	return projectVerdict(v), nil
}

// regimeToPreset maps a harness-native regime NAME to the internal/policy preset
// whose reviewed capability floor realizes it. A regime in fak's harness-native
// program (#2405/#2409/#2759) is a named capability floor, and the preset ladder
// is that floor's concrete realization:
//
//   - "plan": the plan/draft regime (#2409) — may read and draft, but a write or
//     side effect is refused. Realized by the propose-only rung, whose manifest
//     is literally "may draft (diff/plan); apply, push, and deploy are refused".
//   - "autonomous": the broad loop — writes permitted (still floored by the
//     self-modify rung). Realized by the autonomous rung.
//
// The four preset names are also accepted verbatim (see regimeManifest), so any
// reviewed regime can be a fixture without a new alias.
var regimeToPreset = map[string]string{
	"plan":       policy.PresetProposeOnly,
	"autonomous": policy.PresetAutonomous,
}

// regimeManifest resolves a regime name to its policy-floor manifest bytes. A
// harness-native alias (regimeToPreset) resolves to its preset; a bare preset
// name resolves directly. An empty or unknown regime is refused, naming the
// valid regimes — a refusal is a redirect, not a dead end.
func regimeManifest(regime string) ([]byte, error) {
	name := strings.TrimSpace(regime)
	if name == "" {
		return nil, fmt.Errorf("sessionreplay: empty active_regime; name a regime, e.g. one of %s", strings.Join(policy.PresetNames(), ", "))
	}
	if preset, ok := regimeToPreset[name]; ok {
		name = preset
	}
	b, err := policy.PresetManifest(name)
	if err != nil {
		return nil, fmt.Errorf("sessionreplay: active_regime %q does not resolve to a policy floor: %w", regime, err)
	}
	return b, nil
}

// projectVerdict renders an adjudicated abi.Verdict onto the stable, named
// fixture Verdict (the WireVerdict kind/reason vocabulary).
func projectVerdict(v abi.Verdict) Verdict {
	out := Verdict{Kind: verdictKindName(v.Kind)}
	if v.Reason != abi.ReasonNone {
		out.Reason = abi.ReasonName(v.Reason)
	}
	return out
}

// verdictKindName maps a VerdictKind to its stable wire name (the same closed
// vocabulary the gateway's renderVerdict uses). An unknown registered kind
// renders as KIND_<n> rather than leaking an integer.
func verdictKindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return "KIND_" + strconv.FormatUint(uint64(k), 10)
}
