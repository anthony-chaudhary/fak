package bench

// turnuse.go — the pinned turn-use parity baseline for the agent loop (#2001 M1).
//
// Issue #2001 promotes the `fak agent` loop into a reusable internal/microagent
// runtime with the acceptance criterion "the bench still produces IDENTICAL
// turn-use numbers". "Identical" needs a checked-in baseline to diff against — a
// property test ("turns > 0") can witness liveness but never identity. This file
// freezes the loop-scoped turn-use projection of ONE deterministic agent.RunArm
// session (the same loop fanrun drives, through the real kernel) into a golden
// artifact (testdata/turnuse_baseline.json), so the extraction has a fixed point
// to hold still against.
//
// Field discipline (shared live trunk): the baseline pins ONLY what the LOOP owns
// — turn count, per-turn tool-call sequence, per-turn context growth as the
// planner saw it, and the terminal state. vDSO hit counts, engine calls, and
// token totals are deliberately NOT pinned: those belong to the vdso-registry /
// mock-tool lanes, and a legitimate change there must not break this pin.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// TurnUseSchema tags the baseline artifact (bumped on a breaking field change).
const TurnUseSchema = "turnuse-baseline/1"

// TurnRecord is one model round-trip as the LOOP presented it to the planner:
// how many messages the planner was handed (the context-threading pin) and which
// tool calls it emitted back.
type TurnRecord struct {
	MessagesSeen int      `json:"messages_seen"`
	Tools        []string `json:"tools,omitempty"` // emitted tool names, in order; empty on the final turn
	Final        bool     `json:"final"`           // true when the turn ended with a final answer
}

// TurnUseBaseline is the loop-scoped turn-use projection of one deterministic
// RunArm session — the #2001 acceptance witness. Everything here is a pure
// function of (loop threading × the deterministic research planner × subTurns).
type TurnUseBaseline struct {
	Schema             string       `json:"schema"`
	Task               string       `json:"task"`
	SubTurns           int          `json:"sub_turns"`
	Turns              int          `json:"turns"`
	ToolCalls          int          `json:"tool_calls"`
	ToolErrors         int          `json:"tool_errors"`
	HitTurnCap         bool         `json:"hit_turn_cap"`
	FinalAnswerReached bool         `json:"final_answer_reached"`
	TurnLog            []TurnRecord `json:"turn_log"`
}

// turnRecorder wraps a Planner and records the loop-side view of every turn —
// the seam #2001's Microagent extraction must preserve verbatim.
type turnRecorder struct {
	inner agent.Planner
	log   []TurnRecord
}

func (r *turnRecorder) Model() string { return r.inner.Model() }

func (r *turnRecorder) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	c, err := r.inner.Complete(ctx, messages, tools, opts...)
	if err != nil || c == nil {
		return c, err
	}
	rec := TurnRecord{MessagesSeen: len(messages), Final: len(c.Message.ToolCalls) == 0}
	for _, tc := range c.Message.ToolCalls {
		rec.Tools = append(rec.Tools, tc.Function.Name)
	}
	r.log = append(r.log, rec)
	return c, nil
}

// MeasureTurnUse runs ONE deterministic research session through the real kernel
// (fak arm, vDSO on) and returns its loop-scoped turn-use projection plus the raw
// ArmMetrics (for non-pinned sanity checks). Deterministic in subTurns: the
// planner is offline and stateful on context only, and the session runs in a
// fresh world epoch so no cross-test cache state leaks in.
func MeasureTurnUse(ctx context.Context, subTurns int) (TurnUseBaseline, agent.ArmMetrics, error) {
	if subTurns <= 0 {
		subTurns = 8
	}
	agent.Configure()
	vdso.Default.BumpWorld() // fresh epoch: intra-session dedup only (the liveWave discipline)

	rec := &turnRecorder{inner: plannerFor(true /*shared*/, 0)}
	m, err := agent.RunArm(ctx, rec, agent.DefaultTask, true /*fak*/, subTurns, nil)
	if err != nil {
		return TurnUseBaseline{}, m, err
	}
	return TurnUseBaseline{
		Schema:             TurnUseSchema,
		Task:               agent.DefaultTask,
		SubTurns:           subTurns,
		Turns:              m.Turns,
		ToolCalls:          m.ToolCalls,
		ToolErrors:         m.ToolErrors,
		HitTurnCap:         m.HitTurnCap,
		FinalAnswerReached: m.FinalAnswer != "",
		TurnLog:            rec.log,
	}, m, nil
}
