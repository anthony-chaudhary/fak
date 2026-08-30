package main

// Native micro-scout binding for model routing (#2207, epic #595; survey in
// docs/notes/MICRO-SCOUT-NATIVE-ROUTING-2026-07-01.md).
//
// The routing spine (internal/modelroute) reserves the classify-first shape behind
// an injected seam — modelroute.Classifier / ClassifierFunc — so the leaf owns the
// wiring and the classifier owns the model call, with no engine in the spine's
// import graph. The only live binding so far is REMOTE (commit_review.go binds a
// ClassifierFunc to agent.NewHTTPPlanner). This file adds the NATIVE binding:
// bindNativeScout runs a small model IN-PROCESS through the exact loaders `fak run`
// uses (the in-kernel engine), so classify-first sees the raw subject without any
// egress. This is acceptance item 1 of the follow-on — text-decode + parse; the
// logit-scored constrained-decode form (design §4) is a later slice.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// scoutCompleter is the narrow chat-completion capability the native scout needs —
// exactly the method *agent.InKernelPlanner provides. Binding to this interface (not
// the concrete planner) lets a test drive the classify+parse wiring with a fixed
// stub, so the parse contract is provable without any weights on disk.
type scoutCompleter interface {
	Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error)
}

const scoutNativeSystemPrompt = "You are a fast routing scout. Judge how hard the described work is and answer with EXACTLY one word: low, medium, or high. No punctuation, no explanation."

// scoutClassifyPrompt renders a routing Subject into the fixed classification prompt.
// The system prompt + this template are a stable prefix, so after the first call the
// in-kernel KV-prefix cache re-prefills only the subject tail (the cost argument in
// the survey note). Only the closed-vocabulary complexity is requested.
func scoutClassifyPrompt(s modelroute.Subject) string {
	var b strings.Builder
	b.WriteString("Classify the complexity of this work as low, medium, or high.\n")
	if s.Aspect != "" {
		b.WriteString("Aspect: ")
		b.WriteString(string(s.Aspect))
		b.WriteByte('\n')
	}
	if strings.TrimSpace(s.Tool) != "" {
		b.WriteString("Tool: ")
		b.WriteString(strings.TrimSpace(s.Tool))
		b.WriteByte('\n')
	}
	if task := strings.TrimSpace(s.Labels["task"]); task != "" {
		b.WriteString("Task: ")
		b.WriteString(task)
		b.WriteByte('\n')
	}
	b.WriteString("Answer with one word: low, medium, or high.")
	return b.String()
}

// parseScoutLabel folds a scout model's textual answer into a modelroute.ScoutLabel,
// enforcing the closed complexity vocabulary the spine requires (low/medium/high).
// It accepts either an explicit JSON answer ({"complexity":"low"}) or a bare word
// somewhere in the reply, lowercasing before matching. An answer that names no
// in-vocabulary complexity — or a JSON complexity outside the closed set — is a
// fail-loud error, never a silent guess, mirroring scout.go's fail-closed discipline.
func parseScoutLabel(text string) (modelroute.ScoutLabel, error) {
	stripped := stripJSONFence(text)

	// Fast path: an explicit JSON answer {"complexity":"low","labels":{...}}.
	var raw struct {
		Complexity string            `json:"complexity"`
		Labels     map[string]string `json:"labels"`
	}
	if err := json.Unmarshal([]byte(stripped), &raw); err == nil && strings.TrimSpace(raw.Complexity) != "" {
		c := modelroute.Complexity(strings.ToLower(strings.TrimSpace(raw.Complexity)))
		label := modelroute.ScoutLabel{Complexity: c, Labels: raw.Labels}
		if !label.Valid() {
			return modelroute.ScoutLabel{}, fmt.Errorf("scout: out-of-vocabulary complexity %q", c)
		}
		return label, nil
	}

	// Fallback: a bare-word answer somewhere in the reply ("the task is high").
	for _, tok := range strings.FieldsFunc(strings.ToLower(stripped), func(r rune) bool { return !unicode.IsLetter(r) }) {
		switch c := modelroute.Complexity(tok); c {
		case modelroute.ComplexityLow, modelroute.ComplexityMedium, modelroute.ComplexityHigh:
			return modelroute.ScoutLabel{Complexity: c}, nil
		}
	}
	return modelroute.ScoutLabel{}, fmt.Errorf("scout: no complexity label in answer %q", strings.TrimSpace(text))
}

// classifyWithNativeScout runs one classification turn against an already-loaded
// completer and parses the answer. It is the pure wiring between planner.Complete
// and parseScoutLabel, split out so the fixed-stub parity test can exercise it with
// no model on disk. Temperature 0 keeps the tiny decode deterministic.
func classifyWithNativeScout(ctx context.Context, c scoutCompleter, s modelroute.Subject) (modelroute.ScoutLabel, error) {
	temp := 0.0
	comp, err := c.Complete(ctx, []agent.Message{
		{Role: agent.RoleSystem, Content: scoutNativeSystemPrompt},
		{Role: agent.RoleUser, Content: scoutClassifyPrompt(s)},
	}, nil, agent.WithMaxTokens(16), agent.WithTemperature(&temp))
	if err != nil {
		return modelroute.ScoutLabel{}, err
	}
	if comp == nil {
		return modelroute.ScoutLabel{}, fmt.Errorf("scout: model returned nil completion")
	}
	return parseScoutLabel(comp.Message.Content)
}

// loadNativeScoutPlanner resolves modelRef and loads it into the in-kernel engine
// through the SAME loaders `fak run` uses (buildRunPlanner) — modelreg.Resolve,
// hfhub pull-on-demand, loadServeInKernelModel + resolveServeTokenizer, then
// agent.NewInKernelPlanner. It returns an error instead of exiting the process
// (buildRunPlanner's os.Exit is wrong inside a reusable Classifier), so a caller
// can surface a load failure as a classify error.
func loadNativeScoutPlanner(ctx context.Context, modelRef string, nativeConfig nativeControlConfig) (*agent.InKernelPlanner, error) {
	ref, _ := modelreg.Resolve(modelRef)
	ref = pathutil.ExpandTilde(ref)
	if hfhub.IsURI(ref) {
		resolved, err := hfhub.FetchURI(ctx, ref, nil)
		if err != nil {
			return nil, fmt.Errorf("scout: fetch %s: %w", ref, err)
		}
		ref = resolved
	}
	if _, err := os.Stat(ref); err != nil {
		return nil, fmt.Errorf("scout: model %q is not a known alias, an hf:// URI, or an existing .gguf path", modelRef)
	}
	backend, err := resolveServeChatBackend("")
	if err != nil {
		return nil, fmt.Errorf("scout: backend: %w", err)
	}
	if err := applyNativeControls(backend, nativeConfig); err != nil {
		return nil, fmt.Errorf("scout: native controls: %w", err)
	}
	model, q4k, _, _ := loadServeInKernelModel(ref, backend, false, 0, nil, 1)
	if model == nil {
		return nil, fmt.Errorf("scout: failed to load %q into the in-kernel engine", ref)
	}
	tok, ok := resolveServeTokenizer("", ref)
	if !ok || tok == nil {
		return nil, fmt.Errorf("scout: %q has no usable tokenizer; pass a GGUF with an embedded tokenizer", ref)
	}
	// metal=false: this first cut targets the CPU reference path (the preferred device
	// at this size class per the survey) and the cuda HAL, exactly like `fak run`.
	return newNativeScoutInKernelPlanner(model, tok, modelRef, q4k, backend, nativeConfig), nil
}

func newNativeScoutInKernelPlanner(model *model.Model, tok *tokenizer.Tokenizer, modelRef string, q4k bool, backend compute.Backend, nativeConfig nativeControlConfig) *agent.InKernelPlanner {
	return agent.NewInKernelPlannerWithConfig(model, tok, modelRef, q4k, backend, false, nativeConfig.Planner)
}

// bindNativeScout returns a modelroute.ClassifierFunc backed by an in-process model.
// The weights are loaded ONCE (lazily, on the first classify) and reused for every
// subsequent classification — the "load once, classify many" contract the survey
// note calls for. A load failure is returned from every Classify call rather than
// panicking, so the spine's fail-closed ScoutRoute surfaces it cleanly.
func bindNativeScout(modelRef string, nativeConfig nativeControlConfig) modelroute.ClassifierFunc {
	var (
		once    sync.Once
		planner *agent.InKernelPlanner
		loadErr error
	)
	return modelroute.ClassifierFunc(func(ctx context.Context, s modelroute.Subject) (modelroute.ScoutLabel, error) {
		once.Do(func() { planner, loadErr = loadNativeScoutPlanner(ctx, modelRef, nativeConfig) })
		if loadErr != nil {
			return modelroute.ScoutLabel{}, loadErr
		}
		return classifyWithNativeScout(ctx, planner, s)
	})
}
