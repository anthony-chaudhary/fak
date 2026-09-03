package agentopt

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"
)

// Family 15: Training & adaptation for agent optimization.
//
// DPO preference dataset builder from agent execution journals:
// Mines recorded agent execution trajectories, identifies witnessed successful
// runs (chosen) and failed attempts (rejected) sharing the same prompt,
// scrubs sensitive tokens/receipts, and exports standardized JSONL preference datasets.

// PairingStrategy dictates how chosen and rejected trajectories on the same prompt are paired.
type PairingStrategy int

const (
	// PairAll pairs every chosen trajectory with every rejected trajectory for the prompt.
	PairAll PairingStrategy = iota
	// PairOneToOne pairs chosen[i] with rejected[i] up to min(len(chosen), len(rejected)).
	PairOneToOne
	// PairBestWorst pairs the first chosen with the first rejected attempt.
	PairBestWorst
)

// ToolReceipt captures the observation and status of a tool invocation.
type ToolReceipt struct {
	ToolName string `json:"tool_name"`
	Output   string `json:"output"`
	Error    string `json:"error,omitempty"`
}

// ExecutionStep represents a discrete reasoning, action, or observation turn in an execution journal.
type ExecutionStep struct {
	StepIndex  int            `json:"step_index,omitempty"`
	Thought    string         `json:"thought,omitempty"`
	ToolName   string         `json:"tool_name,omitempty"`
	ToolArgs   map[string]any `json:"tool_args,omitempty"`
	ToolCall   *ToolCall      `json:"tool_call,omitempty"`
	ToolOutput string         `json:"tool_output,omitempty"`
	Receipt    *ToolReceipt   `json:"receipt,omitempty"`
	Response   string         `json:"response,omitempty"`
	DurationMs int64          `json:"duration_ms,omitempty"`
}

// AgentTrajectory represents a recorded agent execution trace from an execution journal.
type AgentTrajectory struct {
	ID          string          `json:"id"`
	Prompt      string          `json:"prompt"`
	Steps       []ExecutionStep `json:"steps,omitempty"`
	Completion  string          `json:"completion,omitempty"`
	FinalAnswer string          `json:"final_answer,omitempty"`
	Success     bool            `json:"success"`
	Witnessed   bool            `json:"witnessed"`
	Status      string          `json:"status,omitempty"` // "success", "witnessed", "failed", "rejected", "error"
	Error       string          `json:"error,omitempty"`
	Metadata    map[string]any  `json:"metadata,omitempty"`
}

// ExecutionJournalEntry is an alias for AgentTrajectory representing a recorded journal entry.
type ExecutionJournalEntry = AgentTrajectory

// IsVerifiedSuccess reports whether the trajectory represents a successful verified attempt.
func (t AgentTrajectory) IsVerifiedSuccess(requireVerification bool) bool {
	if t.Error != "" || t.Status == "failed" || t.Status == "rejected" || t.Status == "error" {
		return false
	}
	if requireVerification {
		return (t.Success || t.Status == "success" || t.Status == "chosen") &&
			(t.Witnessed || t.Status == "witnessed")
	}
	return t.Success || t.Witnessed || t.Status == "success" || t.Status == "witnessed" || t.Status == "chosen"
}

// IsFailedAttempt reports whether the trajectory represents a failed attempt.
func (t AgentTrajectory) IsFailedAttempt() bool {
	if t.Error != "" || t.Status == "failed" || t.Status == "rejected" || t.Status == "error" {
		return true
	}
	if !t.Success && !t.Witnessed && t.Status != "success" && t.Status != "witnessed" && t.Status != "chosen" {
		return true
	}
	return false
}

// DPOPair represents an aligned Direct Preference Optimization training record
// containing a scrubbed prompt, a witnessed successful completion (chosen),
// and a failed attempt (rejected).
type DPOPair struct {
	Prompt   string         `json:"prompt"`
	Chosen   string         `json:"chosen"`
	Rejected string         `json:"rejected"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ExportJSONL exports the DPOPair as a single newline-delimited JSON line.
func (p DPOPair) ExportJSONL() ([]byte, error) {
	b, err := json.Marshal(p)
	if err != nil {
		return nil, fmt.Errorf("dpo: marshal pair: %w", err)
	}
	return append(b, '\n'), nil
}

// WriteJSONL writes the DPOPair as a single newline-delimited JSON line to w.
func (p DPOPair) WriteJSONL(w io.Writer) error {
	b, err := p.ExportJSONL()
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

// DPODataset represents a collection of DPO preference pairs ready for export or training.
type DPODataset []DPOPair

// ExportJSONL writes all pairs in the dataset as newline-delimited JSON to w.
func (d DPODataset) ExportJSONL(w io.Writer) error {
	for _, pair := range d {
		line, err := pair.ExportJSONL()
		if err != nil {
			return err
		}
		if _, err := w.Write(line); err != nil {
			return fmt.Errorf("dpo: write jsonl: %w", err)
		}
	}
	return nil
}

// ExportJSONLBytes returns the entire dataset serialized to JSONL bytes.
func (d DPODataset) ExportJSONLBytes() ([]byte, error) {
	var buf bytes.Buffer
	if err := d.ExportJSONL(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Scrubbing patterns for sensitive tokens, credentials, private networks, and authorization.
var (
	reBearer       = regexp.MustCompile(`(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.\~+/=]{8,}`)
	reAnthropicKey = regexp.MustCompile(`\bsk-ant-[a-zA-Z0-9_-]{16,}\b`)
	reOpenAIKey    = regexp.MustCompile(`\bsk-[a-zA-Z0-9_-]{16,}\b`)
	reGitHubToken  = regexp.MustCompile(`\b(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{16,}\b`)
	reFakToken     = regexp.MustCompile(`\bfak-[a-zA-Z0-9_-]{12,}\b`)
	reSecretKV     = regexp.MustCompile(`(?i)((?:--?|/)?["']?[\w.-]*(?:token|secret|passw(?:or)?d|api[-_]?key|credential|\bauth\b)[\w.-]*["']?\s*[=:]\s*["']?)[^"',\s;]+`)
	reAuthHeader   = regexp.MustCompile(`(?i)\b(authorization\s*:\s*(?:basic|token)\s+)\S+`)
	rePrivIPv4     = regexp.MustCompile(`\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b`)
	rePrivHost     = regexp.MustCompile(`\b[a-zA-Z0-9][a-zA-Z0-9-]*(?:\.[a-zA-Z0-9-]+)*\.(?:internal|corp|lan|local|intranet)\b`)
)

// ScrubText redacts credentials, secrets, private IPs, and private hostnames from free text.
func ScrubText(s string) string {
	if s == "" {
		return s
	}
	s = reBearer.ReplaceAllString(s, "${1}[REDACTED]")
	s = reAnthropicKey.ReplaceAllString(s, "[REDACTED_API_KEY]")
	s = reOpenAIKey.ReplaceAllString(s, "[REDACTED_API_KEY]")
	s = reGitHubToken.ReplaceAllString(s, "[REDACTED_TOKEN]")
	s = reFakToken.ReplaceAllString(s, "[REDACTED_TOKEN]")
	s = reSecretKV.ReplaceAllString(s, "${1}[REDACTED]")
	s = reAuthHeader.ReplaceAllString(s, "${1}[REDACTED]")
	s = rePrivHost.ReplaceAllString(s, "[REDACTED_HOST]")
	s = rePrivIPv4.ReplaceAllString(s, "[REDACTED_IP]")
	return s
}

// DefaultTrajectoryFormatter formats an AgentTrajectory into a standard conversational/ReAct agent completion.
func DefaultTrajectoryFormatter(t AgentTrajectory, scrub func(string) string) string {
	if scrub == nil {
		scrub = ScrubText
	}
	if strings.TrimSpace(t.Completion) != "" {
		return strings.TrimSpace(scrub(t.Completion))
	}

	var sb strings.Builder
	for i, step := range t.Steps {
		if i > 0 && sb.Len() > 0 && !strings.HasSuffix(sb.String(), "\n") {
			sb.WriteString("\n")
		}
		if step.Thought != "" {
			sb.WriteString("Thought: ")
			sb.WriteString(scrub(strings.TrimSpace(step.Thought)))
			sb.WriteString("\n")
		}

		toolName := step.ToolName
		var toolArgs map[string]any
		if step.ToolCall != nil {
			if toolName == "" {
				toolName = step.ToolCall.Name
			}
			if toolArgs == nil {
				toolArgs = step.ToolCall.Args
			}
		}
		if toolArgs == nil && step.ToolArgs != nil {
			toolArgs = step.ToolArgs
		}

		if toolName != "" {
			sb.WriteString("Action: ")
			sb.WriteString(toolName)
			if len(toolArgs) > 0 {
				argBytes, _ := json.Marshal(toolArgs)
				sb.WriteString("(")
				sb.WriteString(scrub(string(argBytes)))
				sb.WriteString(")")
			} else {
				sb.WriteString("()")
			}
			sb.WriteString("\n")
		}

		receiptOut := step.ToolOutput
		receiptErr := ""
		if step.Receipt != nil {
			if receiptOut == "" {
				receiptOut = step.Receipt.Output
			}
			receiptErr = step.Receipt.Error
		}

		if receiptOut != "" || receiptErr != "" {
			sb.WriteString("Observation: ")
			if receiptOut != "" {
				sb.WriteString(scrub(strings.TrimSpace(receiptOut)))
			}
			if receiptErr != "" {
				if receiptOut != "" {
					sb.WriteString(" [error: ")
				} else {
					sb.WriteString("[error: ")
				}
				sb.WriteString(scrub(strings.TrimSpace(receiptErr)))
				sb.WriteString("]")
			}
			sb.WriteString("\n")
		}

		if step.Response != "" {
			sb.WriteString("Response: ")
			sb.WriteString(scrub(strings.TrimSpace(step.Response)))
			sb.WriteString("\n")
		}
	}

	if t.FinalAnswer != "" {
		if sb.Len() > 0 && !strings.HasSuffix(sb.String(), "\n") {
			sb.WriteString("\n")
		}
		sb.WriteString("Final Answer: ")
		sb.WriteString(scrub(strings.TrimSpace(t.FinalAnswer)))
	} else if t.Error != "" && sb.Len() == 0 {
		sb.WriteString("Error: ")
		sb.WriteString(scrub(strings.TrimSpace(t.Error)))
	}

	res := strings.TrimSpace(sb.String())
	if res == "" && t.Error != "" {
		res = "Error: " + scrub(strings.TrimSpace(t.Error))
	}
	return res
}

// DPOPreferenceDatasetBuilder mines recorded agent trajectories into DPO training pairs.
type DPOPreferenceDatasetBuilder struct {
	RequireVerification bool
	PairingStrategy     PairingStrategy
	MaxPairsPerPrompt   int
	Family              string
	Scrubber            func(string) string
	Formatter           func(AgentTrajectory, func(string) string) string
	trajectories        []AgentTrajectory
}

// DPOBuilder is a convenience alias for DPOPreferenceDatasetBuilder.
type DPOBuilder = DPOPreferenceDatasetBuilder

// BuilderOption configures a DPOPreferenceDatasetBuilder.
type BuilderOption func(*DPOPreferenceDatasetBuilder)

// WithRequireVerification sets whether chosen trajectories must have Witnessed == true.
func WithRequireVerification(require bool) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.RequireVerification = require
	}
}

// WithPairingStrategy sets the pairing strategy for matching chosen against rejected trajectories.
func WithPairingStrategy(strategy PairingStrategy) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.PairingStrategy = strategy
	}
}

// WithMaxPairsPerPrompt caps the maximum number of pairs emitted for a single prompt.
func WithMaxPairsPerPrompt(max int) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.MaxPairsPerPrompt = max
	}
}

// WithFamily configures the optimization family metadata tag.
func WithFamily(family string) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.Family = family
	}
}

// WithScrubber configures a custom scrubber function.
func WithScrubber(scrubber func(string) string) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.Scrubber = scrubber
	}
}

// WithFormatter configures a custom trajectory completion formatter.
func WithFormatter(formatter func(AgentTrajectory, func(string) string) string) BuilderOption {
	return func(b *DPOPreferenceDatasetBuilder) {
		b.Formatter = formatter
	}
}

// NewDPOPreferenceDatasetBuilder constructs a new dataset builder with production defaults.
func NewDPOPreferenceDatasetBuilder(opts ...BuilderOption) *DPOPreferenceDatasetBuilder {
	b := &DPOPreferenceDatasetBuilder{
		PairingStrategy: PairAll,
		Family:          "Family 15: Training & adaptation for agent optimization",
		Scrubber:        ScrubText,
		Formatter:       DefaultTrajectoryFormatter,
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// NewDPOBuilder constructs a new dataset builder.
func NewDPOBuilder(opts ...BuilderOption) *DPOPreferenceDatasetBuilder {
	return NewDPOPreferenceDatasetBuilder(opts...)
}

// AddTrajectory appends one or more agent trajectories to the mining set.
func (b *DPOPreferenceDatasetBuilder) AddTrajectory(trajs ...AgentTrajectory) {
	b.trajectories = append(b.trajectories, trajs...)
}

// AddExecutionJournals appends execution journal entries to the mining set.
func (b *DPOPreferenceDatasetBuilder) AddExecutionJournals(entries ...ExecutionJournalEntry) {
	b.AddTrajectory(entries...)
}

// IngestJournalJSONL parses newline-delimited JSON trajectories from r.
func (b *DPOPreferenceDatasetBuilder) IngestJournalJSONL(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var traj AgentTrajectory
		if err := json.Unmarshal(line, &traj); err != nil {
			return fmt.Errorf("dpo: unmarshal trajectory line: %w", err)
		}
		b.AddTrajectory(traj)
	}
	return scanner.Err()
}

// IngestJournalJSON parses a JSON array or single trajectory from r.
func (b *DPOPreferenceDatasetBuilder) IngestJournalJSON(r io.Reader) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil
	}
	if data[0] == '[' {
		var trajs []AgentTrajectory
		if err := json.Unmarshal(data, &trajs); err != nil {
			return fmt.Errorf("dpo: unmarshal trajectory array: %w", err)
		}
		b.AddTrajectory(trajs...)
		return nil
	}
	var traj AgentTrajectory
	if err := json.Unmarshal(data, &traj); err != nil {
		return fmt.Errorf("dpo: unmarshal trajectory: %w", err)
	}
	b.AddTrajectory(traj)
	return nil
}

// getScrubber returns the configured or default text scrubber.
func (b *DPOPreferenceDatasetBuilder) getScrubber() func(string) string {
	if b.Scrubber != nil {
		return b.Scrubber
	}
	return ScrubText
}

// getFormatter returns the configured or default completion formatter.
func (b *DPOPreferenceDatasetBuilder) getFormatter() func(AgentTrajectory, func(string) string) string {
	if b.Formatter != nil {
		return b.Formatter
	}
	return DefaultTrajectoryFormatter
}

// getFamily returns the configured optimization family.
func (b *DPOPreferenceDatasetBuilder) getFamily() string {
	if b.Family != "" {
		return b.Family
	}
	return "Family 15: Training & adaptation for agent optimization"
}

// ExtractPairs extracts DPO preference pairs from the provided slice of trajectories.
func (b *DPOPreferenceDatasetBuilder) ExtractPairs(trajectories []AgentTrajectory) []DPOPair {
	if len(trajectories) == 0 {
		return nil
	}

	scrub := b.getScrubber()
	format := b.getFormatter()
	family := b.getFamily()

	// Group trajectories by normalized prompt, preserving order of first arrival.
	type promptGroup struct {
		rawPrompt string
		chosen    []AgentTrajectory
		rejected  []AgentTrajectory
	}

	groupMap := make(map[string]*promptGroup)
	var groupKeys []string

	for _, t := range trajectories {
		cleanPrompt := strings.TrimSpace(t.Prompt)
		if cleanPrompt == "" {
			continue
		}
		g, exists := groupMap[cleanPrompt]
		if !exists {
			g = &promptGroup{rawPrompt: t.Prompt}
			groupMap[cleanPrompt] = g
			groupKeys = append(groupKeys, cleanPrompt)
		}

		if t.IsVerifiedSuccess(b.RequireVerification) {
			g.chosen = append(g.chosen, t)
		} else if t.IsFailedAttempt() {
			g.rejected = append(g.rejected, t)
		}
	}

	var pairs []DPOPair

	for _, k := range groupKeys {
		g := groupMap[k]
		if len(g.chosen) == 0 || len(g.rejected) == 0 {
			continue
		}

		scrubbedPrompt := scrub(strings.TrimSpace(g.rawPrompt))
		promptPairsCount := 0

		switch b.PairingStrategy {
		case PairOneToOne:
			limit := len(g.chosen)
			if len(g.rejected) < limit {
				limit = len(g.rejected)
			}
			for i := 0; i < limit; i++ {
				c := g.chosen[i]
				r := g.rejected[i]
				if c.ID != "" && c.ID == r.ID {
					continue
				}

				chosenText := format(c, scrub)
				rejectedText := format(r, scrub)
				if chosenText == rejectedText || chosenText == "" || rejectedText == "" {
					continue
				}

				pair := b.createPair(scrubbedPrompt, chosenText, rejectedText, c, r, family)
				pairs = append(pairs, pair)
				promptPairsCount++
				if b.MaxPairsPerPrompt > 0 && promptPairsCount >= b.MaxPairsPerPrompt {
					break
				}
			}

		case PairBestWorst:
			c := g.chosen[0]
			r := g.rejected[0]
			if c.ID != "" && c.ID == r.ID {
				continue
			}

			chosenText := format(c, scrub)
			rejectedText := format(r, scrub)
			if chosenText != rejectedText && chosenText != "" && rejectedText != "" {
				pairs = append(pairs, b.createPair(scrubbedPrompt, chosenText, rejectedText, c, r, family))
			}

		case PairAll:
			fallthrough
		default:
			for _, c := range g.chosen {
				for _, r := range g.rejected {
					if c.ID != "" && c.ID == r.ID {
						continue
					}

					chosenText := format(c, scrub)
					rejectedText := format(r, scrub)
					if chosenText == rejectedText || chosenText == "" || rejectedText == "" {
						continue
					}

					pair := b.createPair(scrubbedPrompt, chosenText, rejectedText, c, r, family)
					pairs = append(pairs, pair)
					promptPairsCount++
					if b.MaxPairsPerPrompt > 0 && promptPairsCount >= b.MaxPairsPerPrompt {
						break
					}
				}
				if b.MaxPairsPerPrompt > 0 && promptPairsCount >= b.MaxPairsPerPrompt {
					break
				}
			}
		}
	}

	return pairs
}

// createPair constructs a DPOPair with provenance metadata.
func (b *DPOPreferenceDatasetBuilder) createPair(
	prompt, chosen, rejected string,
	c, r AgentTrajectory,
	family string,
) DPOPair {
	meta := map[string]any{
		"family":       family,
		"chosen_id":    c.ID,
		"rejected_id":  r.ID,
		"witnessed":    c.Witnessed,
		"extracted_at": time.Now().UTC().Format(time.RFC3339),
	}
	if len(c.Metadata) > 0 {
		meta["chosen_meta"] = c.Metadata
	}
	if len(r.Metadata) > 0 {
		meta["rejected_meta"] = r.Metadata
	}

	return DPOPair{
		Prompt:   prompt,
		Chosen:   chosen,
		Rejected: rejected,
		Metadata: meta,
	}
}

// Build mines preference pairs from all trajectories added to the builder.
func (b *DPOPreferenceDatasetBuilder) Build() (DPODataset, error) {
	pairs := b.ExtractPairs(b.trajectories)
	return DPODataset(pairs), nil
}

// ExportJSONL writes the provided pairs as newline-delimited JSON to w.
func (b *DPOPreferenceDatasetBuilder) ExportJSONL(w io.Writer, pairs []DPOPair) error {
	return DPODataset(pairs).ExportJSONL(w)
}

// AdaptFrontierTrajectory converts a FrontierTrajectory from tool distillation into an AgentTrajectory.
func AdaptFrontierTrajectory(ft FrontierTrajectory) AgentTrajectory {
	prompt := ft.UserQuery
	if ft.SystemPrompt != "" {
		prompt = fmt.Sprintf("System: %s\nUser: %s", ft.SystemPrompt, ft.UserQuery)
	}

	var steps []ExecutionStep
	for _, s := range ft.Steps {
		step := ExecutionStep{
			StepIndex:  s.StepIndex,
			Thought:    s.Thought,
			DurationMs: s.DurationMs,
		}
		if s.ToolCall.Name != "" {
			tc := s.ToolCall
			step.ToolCall = &tc
			step.ToolName = s.ToolCall.Name
			step.ToolArgs = s.ToolCall.Args
		}
		if s.ToolResult != "" || s.Error != "" {
			step.Receipt = &ToolReceipt{
				ToolName: s.ToolCall.Name,
				Output:   s.ToolResult,
				Error:    s.Error,
			}
		}
		steps = append(steps, step)
	}

	return AgentTrajectory{
		ID:          ft.ID,
		Prompt:      prompt,
		Steps:       steps,
		FinalAnswer: ft.FinalResponse,
		Success:     ft.Success,
		Witnessed:   ft.Success,
		Metadata:    ft.Metadata,
	}
}

// AdaptReplayTrajectory converts an offline replay Trajectory into an AgentTrajectory.
func AdaptReplayTrajectory(t Trajectory) AgentTrajectory {
	var steps []ExecutionStep
	hasError := false

	for _, turn := range t.Turns {
		step := ExecutionStep{
			StepIndex: turn.TurnIndex,
			Response:  turn.Output,
		}
		if len(turn.ToolCalls) > 0 {
			tc := turn.ToolCalls[0]
			step.ToolCall = &tc
			step.ToolName = tc.Name
			step.ToolArgs = tc.Args
		}
		if len(turn.Results) > 0 {
			res := turn.Results[0]
			step.Receipt = &ToolReceipt{
				ToolName: step.ToolName,
				Output:   res.Output,
				Error:    res.Error,
			}
			if res.Error != "" {
				hasError = true
			}
		}
		steps = append(steps, step)
	}

	success := !hasError && len(steps) > 0
	return AgentTrajectory{
		ID:        t.ID,
		Prompt:    t.Prompt,
		Steps:     steps,
		Success:   success,
		Witnessed: success,
		Metadata:  t.Metadata,
	}
}
