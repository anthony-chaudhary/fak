package adjudicator

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ReasonMissionWriteSetViolation is emitted when a tool call attempts to write outside
// the active mission write-set or violates asymmetric directory extraction rules (#12276).
const ReasonMissionWriteSetViolation abi.ReasonCode = 1120

// ReasonMissionWriteSetViolationName is the stable name registered for ReasonMissionWriteSetViolation.
const ReasonMissionWriteSetViolationName = "MISSION_WRITE_SET_VIOLATION"

// MissionContract defines dynamic file write-sets and asymmetric directory extraction boundaries (#12276).
type MissionContract struct {
	WriteSet    []string `json:"write_set,omitempty"`
	ExtractInto []string `json:"extract_into,omitempty"`
}

// SetMissionContract updates the active mission contract on the adjudicator (#12276).
func (a *Adjudicator) SetMissionContract(mc *MissionContract) {
	if mc == nil {
		a.missionContract.Store(nil)
		return
	}
	cp := *mc
	a.missionContract.Store(&cp)
}

// GetMissionContract returns the active mission contract on the adjudicator, or nil if unset (#12276).
func (a *Adjudicator) GetMissionContract() *MissionContract {
	return a.missionContract.Load()
}

// NewWithMission builds an adjudicator with the given policy and mission contract (#12276).
func NewWithMission(p Policy, mc MissionContract) *Adjudicator {
	a := New(p)
	a.SetMissionContract(&mc)
	return a
}

type missionContractContextKey struct{}

// ContextWithMissionContract returns a derived context carrying the designated mission contract (#12276).
func ContextWithMissionContract(ctx context.Context, mc *MissionContract) context.Context {
	return context.WithValue(ctx, missionContractContextKey{}, mc)
}

// MissionContractFromContext retrieves the mission contract from the context, or nil if unset (#12276).
func MissionContractFromContext(ctx context.Context) *MissionContract {
	if ctx == nil {
		return nil
	}
	if mc, ok := ctx.Value(missionContractContextKey{}).(*MissionContract); ok {
		return mc
	}
	if mc, ok := ctx.Value(missionContractContextKey{}).(MissionContract); ok {
		return &mc
	}
	return nil
}

// resolveMissionContract discovers the active mission contract for a tool call (#12276).
// It checks in precedence order:
// 1. ToolCall.Meta ("mission_contract" JSON, or "write_set" / "extract_into" lists)
// 2. Request context (via MissionContractFromContext)
// 3. Adjudicator instance default (via GetMissionContract)
func (a *Adjudicator) resolveMissionContract(ctx context.Context, c *abi.ToolCall) *MissionContract {
	if c != nil && c.Meta != nil {
		if raw, ok := c.Meta["mission_contract"]; ok && raw != "" {
			var mc MissionContract
			if err := json.Unmarshal([]byte(raw), &mc); err == nil {
				return &mc
			}
		}
		wsRaw, hasWS := c.Meta["write_set"]
		if !hasWS {
			wsRaw, hasWS = c.Meta["mission_write_set"]
		}
		eiRaw, hasEI := c.Meta["extract_into"]
		if !hasEI {
			eiRaw, hasEI = c.Meta["mission_extract_into"]
		}
		if hasWS || hasEI {
			parseList := func(val string) []string {
				val = strings.TrimSpace(val)
				if val == "" {
					return nil
				}
				if strings.HasPrefix(val, "[") && strings.HasSuffix(val, "]") {
					var list []string
					if err := json.Unmarshal([]byte(val), &list); err == nil {
						return list
					}
				}
				var list []string
				for _, item := range strings.Split(val, ",") {
					item = strings.TrimSpace(item)
					if item != "" {
						list = append(list, item)
					}
				}
				return list
			}
			return &MissionContract{
				WriteSet:    parseList(wsRaw),
				ExtractInto: parseList(eiRaw),
			}
		}
	}
	if ctx != nil {
		if mc := MissionContractFromContext(ctx); mc != nil {
			return mc
		}
	}
	return a.GetMissionContract()
}

// missionWriteSetVerdict enforces dynamic mission write-set and asymmetric directory extraction boundaries (#12276).
func (a *Adjudicator) missionWriteSetVerdict(ctx context.Context, p Policy, c *abi.ToolCall, lowerTool string, args map[string]any) (abi.Verdict, bool) {
	mc := a.resolveMissionContract(ctx, c)
	if mc == nil {
		return abi.Verdict{}, false
	}
	targets := a.extractMissionWriteTargets(lowerTool, args)
	if len(targets) == 0 {
		return abi.Verdict{}, false
	}
	for _, target := range targets {
		if err := CheckMissionWriteTarget(target, *mc); err != nil {
			return p.soften(abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  ReasonMissionWriteSetViolation,
				By:      "monitor/mission-writeset",
				Payload: abi.WitnessPayload{Claim: target},
				Meta: map[string]string{
					"rule":        "mission_write_set_violation",
					"target":      target,
					"description": err.Error(),
				},
			}, nil), true
		}
	}
	return abi.Verdict{}, false
}

func (a *Adjudicator) extractMissionWriteTargets(lowerTool string, args map[string]any) []string {
	if len(args) == 0 {
		return nil
	}
	var targets []string
	seen := make(map[string]bool)
	add := func(raw string) {
		trimmed := cleanShellOperand(raw)
		trimmed = strings.Trim(trimmed, `"'`+"`")
		if trimmed == "" || isNullSink(trimmed) {
			return
		}
		if !seen[trimmed] {
			seen[trimmed] = true
			targets = append(targets, trimmed)
		}
	}

	if isWriteOrEditTool(lowerTool) {
		for _, k := range []string{
			"filePath", "file_path", "filepath",
			"path", "file", "target", "filename",
			"destination", "dest", "newPath", "new_path",
		} {
			if v, ok := args[k]; ok {
				if s, ok := v.(string); ok && s != "" {
					add(s)
				}
			}
		}
		for _, k := range []string{"paths", "files", "targets"} {
			if v, ok := args[k]; ok {
				switch sl := v.(type) {
				case []string:
					for _, s := range sl {
						add(s)
					}
				case []any:
					for _, item := range sl {
						if s, ok := item.(string); ok {
							add(s)
						}
					}
				}
			}
		}
	}

	if cmd, ok := commandArg(args); ok && cmd != "" {
		for _, t := range ExtractCommandWriteTargets(cmd) {
			add(t)
		}
		for _, t := range powerShellWriteTargets(cmd) {
			add(t)
		}
	}

	return targets
}
