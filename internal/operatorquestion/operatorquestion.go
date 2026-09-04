package operatorquestion

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// Kind is the closed, harness-independent reason an agent would interrupt an operator.
type Kind string

const (
	// Clarify indicates an interrupt requesting clarification when an instruction is ambiguous or underspecified.
	Clarify Kind = "CLARIFY"
	// ChooseApproach indicates an interrupt requesting a choice between multiple architectural or execution approaches.
	ChooseApproach Kind = "CHOOSE_APPROACH"
	// PlanApproval indicates an interrupt requesting approval of a concrete plan before exiting plan mode.
	PlanApproval Kind = "PLAN_APPROVAL"
	// Permission indicates an interrupt requesting explicit authorization before performing a guarded action.
	Permission Kind = "PERMISSION"
	// ConfirmIrreversible indicates an interrupt requesting confirmation before performing an irreversible action.
	ConfirmIrreversible Kind = "CONFIRM_IRREVERSIBLE"
)

// Valid reports whether k is one of the recognized operator question kinds.
func (k Kind) Valid() bool {
	switch k {
	case Clarify, ChooseApproach, PlanApproval, Permission, ConfirmIrreversible:
		return true
	default:
		return false
	}
}

// Option is one normalized choice and the harness-authored rationale for it.
type Option struct {
	Label     string `json:"label"`
	Rationale string `json:"rationale,omitempty"`
}

// Provenance preserves which native gate produced a normalized question.
type Provenance struct {
	Harness    string `json:"harness"`
	NativeTool string `json:"native_tool"`
}

// OperatorQuestion is the shared question seam consumed by evidence resolvers.
type OperatorQuestion struct {
	Kind       Kind       `json:"kind"`
	Harness    string     `json:"harness"`
	Question   string     `json:"question"`
	Detail     string     `json:"detail,omitempty"`
	Options    []Option   `json:"options,omitempty"`
	Plan       *Plan      `json:"plan,omitempty"`
	Provenance Provenance `json:"provenance"`
}

// Plan is the normalized content adjudicated before plan-mode exit.
type Plan struct {
	FileTree      []string   `json:"file_tree,omitempty"`
	Steps         []PlanStep `json:"steps"`
	DoneCriterion string     `json:"done_criterion,omitempty"`
}

// PlanStep carries a structural tool/args effect plus its independent witness.
type PlanStep struct {
	Text    string         `json:"text"`
	Tool    string         `json:"tool,omitempty"`
	Args    map[string]any `json:"args,omitempty"`
	Witness string         `json:"witness,omitempty"`
}

// NativeGate carries one harness-native operator gate payload.
type NativeGate struct {
	HarnessCommand string
	Tool           string
	Payload        []byte
}

type projection func(profile harnessprofile.HarnessProfile, tool string, payload []byte) (OperatorQuestion, error)

var projections = map[string]projection{
	"claude": projectClaude,
	"codex":  projectCodex,
}

// Normalize dispatches through the harness profile registry, then through one projection
// entry. Adding a harness is one map entry rather than a switch replicated by callers.
func Normalize(g NativeGate) (OperatorQuestion, error) {
	profile, ok := harnessprofile.Lookup(g.HarnessCommand)
	if !ok {
		return OperatorQuestion{}, fmt.Errorf("unknown harness %q", g.HarnessCommand)
	}
	project := projections[profile.Name]
	if project == nil {
		return OperatorQuestion{}, fmt.Errorf("harness %q has no operator-question projection", profile.Name)
	}
	q, err := project(profile, strings.TrimSpace(g.Tool), g.Payload)
	if err != nil {
		return OperatorQuestion{}, err
	}
	q.Harness = profile.Name
	q.Provenance = Provenance{Harness: profile.Name, NativeTool: strings.TrimSpace(g.Tool)}
	if !q.Kind.Valid() || strings.TrimSpace(q.Question) == "" {
		return OperatorQuestion{}, fmt.Errorf("projection produced an invalid question")
	}
	return q, nil
}

// ToSignal projects the normalized seam into the existing choice taxonomy unchanged.
func (q OperatorQuestion) ToSignal() choicetriage.Signal {
	return choicetriage.Signal{
		Severity:    severity(q.Kind),
		Source:      "operator-question:" + q.Harness,
		Question:    q.Question,
		Detail:      q.Detail,
		OptionCount: len(q.Options),
	}
}

func severity(k Kind) string {
	switch k {
	case Permission, ConfirmIrreversible, PlanApproval:
		return "decision"
	default:
		return "action"
	}
}

func projectClaude(_ harnessprofile.HarnessProfile, tool string, payload []byte) (OperatorQuestion, error) {
	switch tool {
	case "AskUserQuestion":
		var wire struct {
			Questions []struct {
				Question    string `json:"question"`
				Header      string `json:"header,omitempty"`
				MultiSelect bool   `json:"multiSelect,omitempty"`
				Options     []struct {
					Label       string `json:"label"`
					Description string `json:"description"`
				} `json:"options"`
			} `json:"questions"`
		}
		if err := strictDecode(payload, &wire); err != nil {
			return OperatorQuestion{}, fmt.Errorf("Claude AskUserQuestion: %w", err)
		}
		if len(wire.Questions) != 1 {
			return OperatorQuestion{}, fmt.Errorf("Claude AskUserQuestion: want exactly one question, got %d", len(wire.Questions))
		}
		q := OperatorQuestion{Kind: kindForOptions(len(wire.Questions[0].Options)), Question: wire.Questions[0].Question}
		for _, option := range wire.Questions[0].Options {
			q.Options = append(q.Options, Option{Label: option.Label, Rationale: option.Description})
		}
		return q, nil
	case "ExitPlanMode":
		var wire struct {
			Plan          string     `json:"plan"`
			FileTree      []string   `json:"file_tree,omitempty"`
			Steps         []PlanStep `json:"steps,omitempty"`
			DoneCriterion string     `json:"done_criterion,omitempty"`
		}
		if err := strictDecode(payload, &wire); err != nil {
			return OperatorQuestion{}, fmt.Errorf("Claude ExitPlanMode: %w", err)
		}
		steps := wire.Steps
		if len(steps) == 0 && strings.TrimSpace(wire.Plan) != "" {
			steps = []PlanStep{{Text: wire.Plan}}
		}
		return OperatorQuestion{Kind: PlanApproval, Question: "Approve this plan?", Detail: wire.Plan, Plan: &Plan{FileTree: wire.FileTree, Steps: steps, DoneCriterion: wire.DoneCriterion}}, nil
	default:
		return OperatorQuestion{}, fmt.Errorf("Claude tool %q is not an operator-question gate", tool)
	}
}

func projectCodex(_ harnessprofile.HarnessProfile, tool string, payload []byte) (OperatorQuestion, error) {
	switch tool {
	case "request_user_input", "functions.request_user_input":
		var wire struct {
			Questions []struct {
				ID       string `json:"id"`
				Header   string `json:"header"`
				Question string `json:"question"`
				Options  []struct {
					Label       string `json:"label"`
					Description string `json:"description"`
				} `json:"options"`
			} `json:"questions"`
		}
		if err := strictDecode(payload, &wire); err != nil {
			return OperatorQuestion{}, fmt.Errorf("Codex request_user_input: %w", err)
		}
		if len(wire.Questions) != 1 {
			return OperatorQuestion{}, fmt.Errorf("Codex request_user_input: want exactly one question, got %d", len(wire.Questions))
		}
		q := OperatorQuestion{Kind: kindForOptions(len(wire.Questions[0].Options)), Question: wire.Questions[0].Question}
		for _, option := range wire.Questions[0].Options {
			q.Options = append(q.Options, Option{Label: option.Label, Rationale: option.Description})
		}
		return q, nil
	case "update_plan", "functions.update_plan":
		var wire struct {
			Explanation string `json:"explanation"`
			Plan        []struct {
				Step    string         `json:"step"`
				Status  string         `json:"status"`
				Tool    string         `json:"tool,omitempty"`
				Args    map[string]any `json:"args,omitempty"`
				Witness string         `json:"witness,omitempty"`
			} `json:"plan"`
			FileTree      []string `json:"file_tree,omitempty"`
			DoneCriterion string   `json:"done_criterion,omitempty"`
		}
		if err := strictDecode(payload, &wire); err != nil {
			return OperatorQuestion{}, fmt.Errorf("Codex update_plan: %w", err)
		}
		stepText := make([]string, 0, len(wire.Plan))
		steps := make([]PlanStep, 0, len(wire.Plan))
		for _, step := range wire.Plan {
			if text := strings.TrimSpace(step.Step); text != "" {
				stepText = append(stepText, text)
				steps = append(steps, PlanStep{Text: text, Tool: step.Tool, Args: step.Args, Witness: step.Witness})
			}
		}
		return OperatorQuestion{Kind: PlanApproval, Question: "Approve this plan?", Detail: strings.Join(stepText, "\n"), Plan: &Plan{FileTree: wire.FileTree, Steps: steps, DoneCriterion: wire.DoneCriterion}}, nil
	default:
		return OperatorQuestion{}, fmt.Errorf("Codex tool %q is not an operator-question gate", tool)
	}
}

func kindForOptions(n int) Kind {
	if n > 1 {
		return ChooseApproach
	}
	return Clarify
}

func strictDecode(payload []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}
