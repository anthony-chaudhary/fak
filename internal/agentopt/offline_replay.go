package agentopt

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"sync"
	"time"
)

// Family 16: Reliability & evaluation as optimization.
//
// Offline trajectory replay evaluates agent behavior deterministically by
// replaying historical trajectory traces against updated tools or prompts in a
// hermetic mock environment and reporting behavioral divergence.

// EnvironmentObservation represents an environmental signal or output observed during a turn.
type EnvironmentObservation struct {
	Source    string         `json:"source,omitempty"`
	Key       string         `json:"key,omitempty"`
	Content   string         `json:"content"`
	Timestamp int64          `json:"timestamp,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

// ToolResult records the output or error returned by an executed tool call.
type ToolResult struct {
	CallID string `json:"call_id"`
	Output string `json:"output"`
	Error  string `json:"error,omitempty"`
}

// TrajectoryTurn represents one turn in an agent trajectory, containing the prompt,
// tool calls invoked, observations received, and resulting outputs.
type TrajectoryTurn struct {
	TurnIndex    int                      `json:"turn_index"`
	Prompt       string                   `json:"prompt,omitempty"`
	ToolCalls    []ToolCall               `json:"tool_calls,omitempty"`
	Observations []EnvironmentObservation `json:"observations,omitempty"`
	Results      []ToolResult             `json:"results,omitempty"`
	Output       string                   `json:"output,omitempty"`
}

// Trajectory represents an ordered sequence of agent turns with tool calls,
// environment observations, and tool results.
type Trajectory struct {
	ID          string           `json:"id"`
	Description string           `json:"description,omitempty"`
	Prompt      string           `json:"prompt,omitempty"`
	Turns       []TrajectoryTurn `json:"turns"`
	Metadata    map[string]any   `json:"metadata,omitempty"`
}

// ToolSelectionChange describes a divergence in which tools were selected during a turn.
type ToolSelectionChange struct {
	TurnIndex     int      `json:"turn_index"`
	BaselineTools []string `json:"baseline_tools"`
	ReplayedTools []string `json:"replayed_tools"`
	Description   string   `json:"description"`
}

// ArgumentMutation describes a mutated, added, or removed argument for a tool call.
type ArgumentMutation struct {
	TurnIndex    int    `json:"turn_index"`
	CallID       string `json:"call_id"`
	ToolName     string `json:"tool_name"`
	ArgKey       string `json:"arg_key"`
	BaselineVal  any    `json:"baseline_val,omitempty"`
	ReplayedVal  any    `json:"replayed_val,omitempty"`
	MutationType string `json:"mutation_type"` // "modified", "added", "removed"
	Description  string `json:"description"`
}

// TurnDivergence groups all divergences observed at a specific turn.
type TurnDivergence struct {
	TurnIndex           int                  `json:"turn_index"`
	ToolSelectionChange *ToolSelectionChange `json:"tool_selection_change,omitempty"`
	ArgumentMutations   []ArgumentMutation   `json:"argument_mutations,omitempty"`
	ObservationDelta    string               `json:"observation_delta,omitempty"`
	ResultDelta         string               `json:"result_delta,omitempty"`
}

// DivergenceReport summarizes the behavioral differences between a baseline trajectory
// and a replayed trajectory.
type DivergenceReport struct {
	BaselineID           string                `json:"baseline_id"`
	ReplayedID           string                `json:"replayed_id"`
	Diverged             bool                  `json:"diverged"`
	TurnCountDelta       int                   `json:"turn_count_delta"`
	BaselineTurnCount    int                   `json:"baseline_turn_count"`
	ReplayedTurnCount    int                   `json:"replayed_turn_count"`
	ToolSelectionChanges []ToolSelectionChange `json:"tool_selection_changes,omitempty"`
	ArgumentMutations    []ArgumentMutation    `json:"argument_mutations,omitempty"`
	TurnDivergences      []TurnDivergence      `json:"turn_divergences,omitempty"`
	RegressionDetected   bool                  `json:"regression_detected"`
	RegressionScore      float64               `json:"regression_score"` // 0.0 (identical) to 1.0 (completely diverged)
	Summary              string                `json:"summary"`
}

// ToolMockHandler executes or mocks a tool call in the hermetic environment.
type ToolMockHandler func(ctx context.Context, call ToolCall) (string, error)

// HermeticEnvironment provides isolated, deterministic execution of tools and
// observation capturing for offline replay.
type HermeticEnvironment struct {
	mu              sync.RWMutex
	tools           map[string]ToolMockHandler
	recordedResults map[string]ToolResult
	observations    []EnvironmentObservation
	dispatchedCalls []ToolCall
	state           map[string]any
}

// NewHermeticEnvironment constructs a new isolated, deterministic environment.
func NewHermeticEnvironment() *HermeticEnvironment {
	return &HermeticEnvironment{
		tools:           make(map[string]ToolMockHandler),
		recordedResults: make(map[string]ToolResult),
		state:           make(map[string]any),
	}
}

// RegisterTool registers a mock execution handler for a specific tool name.
func (env *HermeticEnvironment) RegisterTool(name string, handler ToolMockHandler) {
	env.mu.Lock()
	defer env.mu.Unlock()
	env.tools[name] = handler
}

// RegisterResult registers a deterministic tool result by call ID.
func (env *HermeticEnvironment) RegisterResult(callID string, res ToolResult) {
	env.mu.Lock()
	defer env.mu.Unlock()
	env.recordedResults[callID] = res
}

// RegisterDigestResult registers a deterministic tool result by tool name and arguments digest.
func (env *HermeticEnvironment) RegisterDigestResult(toolName string, args map[string]any, res ToolResult) {
	env.mu.Lock()
	defer env.mu.Unlock()
	digest := DigestCall(toolName, args)
	env.recordedResults[digest] = res
}

// LoadTrajectory populates recorded results and observations from historical trajectory turns.
func (env *HermeticEnvironment) LoadTrajectory(traj Trajectory) {
	env.mu.Lock()
	defer env.mu.Unlock()
	for _, turn := range traj.Turns {
		for i, call := range turn.ToolCalls {
			var res ToolResult
			if i < len(turn.Results) {
				res = turn.Results[i]
			} else {
				res = ToolResult{CallID: call.ID, Output: "ok"}
			}
			if call.ID != "" {
				env.recordedResults[call.ID] = res
			}
			digest := DigestCall(call.Name, call.Args)
			env.recordedResults[digest] = res
		}
		for _, obs := range turn.Observations {
			env.observations = append(env.observations, obs)
		}
	}
}

// ExecuteTool executes a tool call deterministically within the hermetic environment.
func (env *HermeticEnvironment) ExecuteTool(ctx context.Context, call ToolCall) ToolResult {
	env.mu.Lock()
	defer env.mu.Unlock()

	env.dispatchedCalls = append(env.dispatchedCalls, call)

	if err := ctx.Err(); err != nil {
		return ToolResult{CallID: call.ID, Error: err.Error()}
	}

	// 1. Check if an explicit tool handler is registered.
	if handler, ok := env.tools[call.Name]; ok {
		out, err := handler(ctx, call)
		res := ToolResult{CallID: call.ID, Output: out}
		if err != nil {
			res.Error = err.Error()
		}
		env.observations = append(env.observations, EnvironmentObservation{
			Source:    call.Name,
			Key:       call.ID,
			Content:   out,
			Timestamp: time.Now().UnixNano(),
		})
		return res
	}

	// 2. Check recorded results by call ID.
	if call.ID != "" {
		if res, ok := env.recordedResults[call.ID]; ok {
			return res
		}
	}

	// 3. Check recorded results by call digest.
	digest := DigestCall(call.Name, call.Args)
	if res, ok := env.recordedResults[digest]; ok {
		return res
	}

	// 4. Default hermetic fallback.
	return ToolResult{
		CallID: call.ID,
		Error:  fmt.Sprintf("tool %q not mocked in hermetic environment", call.Name),
	}
}

// AddObservation appends an observation to the environment.
func (env *HermeticEnvironment) AddObservation(obs EnvironmentObservation) {
	env.mu.Lock()
	defer env.mu.Unlock()
	env.observations = append(env.observations, obs)
}

// GetObservations returns a copy of all observations recorded in the environment.
func (env *HermeticEnvironment) GetObservations() []EnvironmentObservation {
	env.mu.RLock()
	defer env.mu.RUnlock()
	out := make([]EnvironmentObservation, len(env.observations))
	copy(out, env.observations)
	return out
}

// DrainObservations retrieves and clears pending observations accumulated in the environment.
func (env *HermeticEnvironment) DrainObservations() []EnvironmentObservation {
	env.mu.Lock()
	defer env.mu.Unlock()
	out := env.observations
	env.observations = nil
	return out
}

// GetDispatchedCalls returns a copy of all tool calls dispatched in this environment.
func (env *HermeticEnvironment) GetDispatchedCalls() []ToolCall {
	env.mu.RLock()
	defer env.mu.RUnlock()
	out := make([]ToolCall, len(env.dispatchedCalls))
	copy(out, env.dispatchedCalls)
	return out
}

// SetState stores a key-value pair in the hermetic environment state.
func (env *HermeticEnvironment) SetState(key string, val any) {
	env.mu.Lock()
	defer env.mu.Unlock()
	env.state[key] = val
}

// GetState retrieves a key-value pair from the hermetic environment state.
func (env *HermeticEnvironment) GetState(key string) (any, bool) {
	env.mu.RLock()
	defer env.mu.RUnlock()
	val, ok := env.state[key]
	return val, ok
}

// Reset clears dispatched calls and observations, retaining tool handlers and recorded results.
func (env *HermeticEnvironment) Reset() {
	env.mu.Lock()
	defer env.mu.Unlock()
	env.dispatchedCalls = nil
	env.observations = nil
}

// AgentReplayFunc models an agent turn generator during offline replay.
// It receives the current turn index, historical turns completed so far,
// and the hermetic environment. It returns the generated turn, a boolean
// indicating whether the agent is done, and any error.
type AgentReplayFunc func(ctx context.Context, turnIndex int, history []TrajectoryTurn, env *HermeticEnvironment) (TrajectoryTurn, bool, error)

// ReplayConfig configures replay execution.
type ReplayConfig struct {
	MaxTurns            int
	UpdatedPrompt       string
	AgentFunc           AgentReplayFunc
	RegressionThreshold float64
}

// OfflineTrajectoryEvaluator coordinates replaying historical trajectory traces
// and evaluating behavioral divergence.
type OfflineTrajectoryEvaluator struct {
	Env *HermeticEnvironment
}

// NewOfflineTrajectoryEvaluator constructs an offline trajectory evaluator with the provided environment.
func NewOfflineTrajectoryEvaluator(env *HermeticEnvironment) *OfflineTrajectoryEvaluator {
	if env == nil {
		env = NewHermeticEnvironment()
	}
	return &OfflineTrajectoryEvaluator{Env: env}
}

// Replay replays a baseline trajectory against updated tools or prompts in the hermetic environment.
func (e *OfflineTrajectoryEvaluator) Replay(ctx context.Context, baseline Trajectory, cfg ReplayConfig) (*Trajectory, *DivergenceReport, error) {
	replayed := &Trajectory{
		ID:          baseline.ID + "-replayed",
		Description: "Replayed trajectory for " + baseline.ID,
		Prompt:      baseline.Prompt,
		Metadata: map[string]any{
			"baseline_id": baseline.ID,
			"replayed_at": time.Now().UTC().Format(time.RFC3339),
		},
	}
	if cfg.UpdatedPrompt != "" {
		replayed.Prompt = cfg.UpdatedPrompt
	}

	maxTurns := cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = len(baseline.Turns) + 10
		if maxTurns < 20 {
			maxTurns = 20
		}
	}

	if cfg.AgentFunc != nil {
		for turnIdx := 0; turnIdx < maxTurns; turnIdx++ {
			if err := ctx.Err(); err != nil {
				return replayed, nil, err
			}
			turn, done, err := cfg.AgentFunc(ctx, turnIdx, replayed.Turns, e.Env)
			if err != nil {
				return replayed, nil, err
			}
			turn.TurnIndex = turnIdx

			// Execute tool calls against the hermetic environment if results not already populated.
			if len(turn.Results) == 0 && len(turn.ToolCalls) > 0 {
				for _, call := range turn.ToolCalls {
					res := e.Env.ExecuteTool(ctx, call)
					turn.Results = append(turn.Results, res)
				}
			}

			// Gather observations emitted during this turn.
			turn.Observations = append(turn.Observations, e.Env.DrainObservations()...)
			replayed.Turns = append(replayed.Turns, turn)

			if done {
				break
			}
		}
	} else {
		// Default replay: re-execute baseline turns against the hermetic environment.
		for turnIdx, baseTurn := range baseline.Turns {
			if err := ctx.Err(); err != nil {
				return replayed, nil, err
			}
			turn := TrajectoryTurn{
				TurnIndex: turnIdx,
				Prompt:    baseTurn.Prompt,
				ToolCalls: baseTurn.ToolCalls,
				Output:    baseTurn.Output,
			}
			if cfg.UpdatedPrompt != "" && turnIdx == 0 {
				turn.Prompt = cfg.UpdatedPrompt
			}
			for _, call := range baseTurn.ToolCalls {
				res := e.Env.ExecuteTool(ctx, call)
				turn.Results = append(turn.Results, res)
			}
			turn.Observations = append(turn.Observations, e.Env.DrainObservations()...)
			replayed.Turns = append(replayed.Turns, turn)
		}
	}

	report := EvaluateDivergence(baseline, *replayed)
	if cfg.RegressionThreshold > 0 {
		report.RegressionDetected = report.RegressionScore > cfg.RegressionThreshold
	}

	return replayed, report, nil
}

// EvaluateDivergence compares a baseline trajectory against a replayed trajectory
// and produces a detailed DivergenceReport.
func EvaluateDivergence(baseline, replayed Trajectory) *DivergenceReport {
	report := &DivergenceReport{
		BaselineID:        baseline.ID,
		ReplayedID:        replayed.ID,
		BaselineTurnCount: len(baseline.Turns),
		ReplayedTurnCount: len(replayed.Turns),
		TurnCountDelta:    len(replayed.Turns) - len(baseline.Turns),
	}

	maxTurns := len(baseline.Turns)
	if len(replayed.Turns) > maxTurns {
		maxTurns = len(replayed.Turns)
	}

	var turnDivergences []TurnDivergence

	for i := 0; i < maxTurns; i++ {
		var td TurnDivergence
		td.TurnIndex = i
		hasTurnDivergence := false

		// 1. Check early termination or extra turns.
		if i >= len(baseline.Turns) {
			repTools := extractToolNames(replayed.Turns[i].ToolCalls)
			tsc := ToolSelectionChange{
				TurnIndex:     i,
				BaselineTools: []string{},
				ReplayedTools: repTools,
				Description:   fmt.Sprintf("turn %d: unexpected extra turn in replayed with tools %v", i, repTools),
			}
			report.ToolSelectionChanges = append(report.ToolSelectionChanges, tsc)
			td.ToolSelectionChange = &tsc
			hasTurnDivergence = true
			turnDivergences = append(turnDivergences, td)
			continue
		}

		if i >= len(replayed.Turns) {
			baseTools := extractToolNames(baseline.Turns[i].ToolCalls)
			tsc := ToolSelectionChange{
				TurnIndex:     i,
				BaselineTools: baseTools,
				ReplayedTools: []string{},
				Description:   fmt.Sprintf("turn %d: replayed terminated early, missing baseline tools %v", i, baseTools),
			}
			report.ToolSelectionChanges = append(report.ToolSelectionChanges, tsc)
			td.ToolSelectionChange = &tsc
			hasTurnDivergence = true
			turnDivergences = append(turnDivergences, td)
			continue
		}

		// 2. Both turns exist: compare tool selection.
		baseTurn := baseline.Turns[i]
		repTurn := replayed.Turns[i]

		baseTools := extractToolNames(baseTurn.ToolCalls)
		repTools := extractToolNames(repTurn.ToolCalls)

		toolsDiffer := len(baseTools) != len(repTools)
		if !toolsDiffer {
			for idx := range baseTools {
				if baseTools[idx] != repTools[idx] {
					toolsDiffer = true
					break
				}
			}
		}

		if toolsDiffer {
			tsc := ToolSelectionChange{
				TurnIndex:     i,
				BaselineTools: baseTools,
				ReplayedTools: repTools,
				Description:   fmt.Sprintf("turn %d: tool selection diverged: baseline %v vs replayed %v", i, baseTools, repTools),
			}
			report.ToolSelectionChanges = append(report.ToolSelectionChanges, tsc)
			td.ToolSelectionChange = &tsc
			hasTurnDivergence = true
		}

		// 3. Compare arguments for matching tool calls.
		minCalls := len(baseTurn.ToolCalls)
		if len(repTurn.ToolCalls) < minCalls {
			minCalls = len(repTurn.ToolCalls)
		}

		var turnMutations []ArgumentMutation
		for j := 0; j < minCalls; j++ {
			bCall := baseTurn.ToolCalls[j]
			cCall := repTurn.ToolCalls[j]
			if bCall.Name == cCall.Name {
				muts := compareArguments(i, bCall, cCall)
				if len(muts) > 0 {
					turnMutations = append(turnMutations, muts...)
					report.ArgumentMutations = append(report.ArgumentMutations, muts...)
				}
			}
		}

		if len(turnMutations) > 0 {
			td.ArgumentMutations = turnMutations
			hasTurnDivergence = true
		}

		// 4. Compare tool results or outputs.
		if baseTurn.Output != repTurn.Output && (baseTurn.Output != "" || repTurn.Output != "") {
			td.ResultDelta = fmt.Sprintf("turn %d: output differs: %q vs %q", i, baseTurn.Output, repTurn.Output)
		}

		if hasTurnDivergence {
			turnDivergences = append(turnDivergences, td)
		}
	}

	report.TurnDivergences = turnDivergences
	report.Diverged = len(report.ToolSelectionChanges) > 0 || len(report.ArgumentMutations) > 0 || report.TurnCountDelta != 0

	if !report.Diverged {
		report.RegressionDetected = false
		report.RegressionScore = 0.0
		report.Summary = fmt.Sprintf("deterministic match: %d turns replayed with zero divergence", report.BaselineTurnCount)
	} else {
		report.RegressionDetected = true
		total := maxTurns
		if total <= 0 {
			total = 1
		}
		// Calculate normalized regression score [0.0, 1.0]
		turnDivergenceRatio := float64(len(turnDivergences)) / float64(total)
		toolChangeRatio := math.Min(1.0, float64(len(report.ToolSelectionChanges))/float64(total))
		argMutationRatio := math.Min(1.0, float64(len(report.ArgumentMutations))/float64(total*2))
		turnDeltaRatio := math.Min(1.0, math.Abs(float64(report.TurnCountDelta))/float64(total))

		score := 0.35*turnDivergenceRatio + 0.30*toolChangeRatio + 0.20*argMutationRatio + 0.15*turnDeltaRatio
		if score > 1.0 {
			score = 1.0
		}
		if score == 0.0 {
			score = 0.01 // non-zero if diverged
		}
		report.RegressionScore = score

		report.Summary = fmt.Sprintf("divergence detected: turn delta %+d (baseline=%d, replayed=%d), %d tool selection changes, %d argument mutations across %d divergent turns (regression score: %.2f)",
			report.TurnCountDelta, report.BaselineTurnCount, report.ReplayedTurnCount,
			len(report.ToolSelectionChanges), len(report.ArgumentMutations), len(report.TurnDivergences), report.RegressionScore)
	}

	return report
}

func extractToolNames(calls []ToolCall) []string {
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names
}

func compareArguments(turnIndex int, baseCall, candCall ToolCall) []ArgumentMutation {
	var mutations []ArgumentMutation
	baseArgs := baseCall.Args
	if baseArgs == nil {
		baseArgs = map[string]any{}
	}
	candArgs := candCall.Args
	if candArgs == nil {
		candArgs = map[string]any{}
	}

	allKeysMap := make(map[string]bool)
	for k := range baseArgs {
		allKeysMap[k] = true
	}
	for k := range candArgs {
		allKeysMap[k] = true
	}

	allKeys := make([]string, 0, len(allKeysMap))
	for k := range allKeysMap {
		allKeys = append(allKeys, k)
	}
	sort.Strings(allKeys)

	callID := candCall.ID
	if callID == "" {
		callID = baseCall.ID
	}

	for _, k := range allKeys {
		bVal, bOk := baseArgs[k]
		cVal, cOk := candArgs[k]

		if bOk && !cOk {
			mutations = append(mutations, ArgumentMutation{
				TurnIndex:    turnIndex,
				CallID:       callID,
				ToolName:     candCall.Name,
				ArgKey:       k,
				BaselineVal:  bVal,
				ReplayedVal:  nil,
				MutationType: "removed",
				Description:  fmt.Sprintf("turn %d %s: argument %q removed (was %v)", turnIndex, candCall.Name, k, bVal),
			})
		} else if !bOk && cOk {
			mutations = append(mutations, ArgumentMutation{
				TurnIndex:    turnIndex,
				CallID:       callID,
				ToolName:     candCall.Name,
				ArgKey:       k,
				BaselineVal:  nil,
				ReplayedVal:  cVal,
				MutationType: "added",
				Description:  fmt.Sprintf("turn %d %s: argument %q added with value %v", turnIndex, candCall.Name, k, cVal),
			})
		} else if !valuesEqual(bVal, cVal) {
			mutations = append(mutations, ArgumentMutation{
				TurnIndex:    turnIndex,
				CallID:       callID,
				ToolName:     candCall.Name,
				ArgKey:       k,
				BaselineVal:  bVal,
				ReplayedVal:  cVal,
				MutationType: "modified",
				Description:  fmt.Sprintf("turn %d %s: argument %q mutated from %v to %v", turnIndex, candCall.Name, k, bVal, cVal),
			})
		}
	}
	return mutations
}

func valuesEqual(a, b any) bool {
	if reflect.DeepEqual(a, b) {
		return true
	}
	ja, errA := json.Marshal(a)
	jb, errB := json.Marshal(b)
	if errA == nil && errB == nil {
		return string(ja) == string(jb)
	}
	return false
}

// ToJSON serializes the trajectory to formatted JSON bytes.
func (t Trajectory) ToJSON() ([]byte, error) {
	return json.MarshalIndent(t, "", "  ")
}

// TrajectoryFromJSON deserializes a trajectory from JSON bytes.
func TrajectoryFromJSON(data []byte) (Trajectory, error) {
	var t Trajectory
	err := json.Unmarshal(data, &t)
	return t, err
}

// ToJSON serializes the divergence report to formatted JSON bytes.
func (r DivergenceReport) ToJSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
