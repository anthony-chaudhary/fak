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

// Replay re-adjudicates a captured turn under its active regime policy floor and returns the verdict.
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

// regimeToPreset maps harness regime aliases to policy preset names.
var regimeToPreset = map[string]string{
	"plan":       policy.PresetProposeOnly,
	"autonomous": policy.PresetAutonomous,
}

// regimeManifest resolves a regime name or preset alias to its policy manifest bytes.
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

// projectVerdict renders an adjudicated abi.Verdict onto a fixture Verdict.
func projectVerdict(v abi.Verdict) Verdict {
	out := Verdict{Kind: verdictKindName(v.Kind)}
	if v.Reason != abi.ReasonNone {
		out.Reason = abi.ReasonName(v.Reason)
	}
	return out
}

// verdictKindName maps a VerdictKind to its stable wire name.
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
