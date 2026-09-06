package sessionaudit

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ErrInvalidGeminiSession is returned when JSON content is not a recognized Gemini CLI chat transcript.
var ErrInvalidGeminiSession = errors.New("sessionaudit: invalid gemini session transcript")

// KindGemini identifies Gemini CLI session transcripts.
const KindGemini = "gemini"

// geminiUsageMeta holds token usage metrics from Gemini generateContent responses.
type geminiUsageMeta struct {
	PromptTokens          int64 `json:"promptTokenCount"`
	PromptTokensSnake     int64 `json:"prompt_token_count"`
	PromptTokensShort     int64 `json:"promptTokens"`
	CandidatesTokens      int64 `json:"candidatesTokenCount"`
	CandidatesTokensSnake int64 `json:"candidates_token_count"`
	CandidatesTokensShort int64 `json:"candidatesTokens"`
	OutputTokensSnake     int64 `json:"output_token_count"`
	OutputTokens          int64 `json:"outputTokens"`
	TotalTokens           int64 `json:"totalTokenCount"`
	TotalTokensSnake      int64 `json:"total_token_count"`
	TotalTokensShort      int64 `json:"totalTokens"`
	CachedTokens          int64 `json:"cachedContentTokenCount"`
	CachedTokensSnake     int64 `json:"cached_content_token_count"`
}

func (u *geminiUsageMeta) Prompt() int64 {
	if u == nil {
		return 0
	}
	if u.PromptTokens > 0 {
		return u.PromptTokens
	}
	if u.PromptTokensSnake > 0 {
		return u.PromptTokensSnake
	}
	return u.PromptTokensShort
}

func (u *geminiUsageMeta) Candidates() int64 {
	if u == nil {
		return 0
	}
	if u.CandidatesTokens > 0 {
		return u.CandidatesTokens
	}
	if u.CandidatesTokensSnake > 0 {
		return u.CandidatesTokensSnake
	}
	if u.CandidatesTokensShort > 0 {
		return u.CandidatesTokensShort
	}
	if u.OutputTokensSnake > 0 {
		return u.OutputTokensSnake
	}
	return u.OutputTokens
}

func (u *geminiUsageMeta) Total() int64 {
	if u == nil {
		return 0
	}
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	if u.TotalTokensSnake > 0 {
		return u.TotalTokensSnake
	}
	if u.TotalTokensShort > 0 {
		return u.TotalTokensShort
	}
	return u.Prompt() + u.Candidates()
}

func (u *geminiUsageMeta) Cached() int64 {
	if u == nil {
		return 0
	}
	if u.CachedTokens > 0 {
		return u.CachedTokens
	}
	return u.CachedTokensSnake
}

type geminiFunctionCall struct {
	Name      string          `json:"name"`
	Args      json.RawMessage `json:"args"`
	Arguments json.RawMessage `json:"arguments"`
	ID        string          `json:"id"`
}

func (c *geminiFunctionCall) GetArgs() json.RawMessage {
	if len(c.Args) > 0 {
		return c.Args
	}
	return c.Arguments
}

type geminiFunctionResp struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
	ID       string          `json:"id"`
}

type geminiPart struct {
	Text              string              `json:"text"`
	Thought           bool                `json:"thought"`
	FunctionCall      *geminiFunctionCall `json:"functionCall"`
	FunctionCallSnake *geminiFunctionCall `json:"function_call"`
	FunctionResponse  *geminiFunctionResp `json:"functionResponse"`
	FunctionRespSnake *geminiFunctionResp `json:"function_response"`
}

func (p *geminiPart) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		p.Text = s
		return nil
	}
	type alias geminiPart
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*p = geminiPart(a)
	if p.FunctionCall == nil && p.FunctionCallSnake != nil {
		p.FunctionCall = p.FunctionCallSnake
	}
	if p.FunctionResponse == nil && p.FunctionRespSnake != nil {
		p.FunctionResponse = p.FunctionRespSnake
	}
	return nil
}

type geminiToolCall struct {
	Name     string          `json:"name"`
	Args     json.RawMessage `json:"args"`
	Function *struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type geminiTurn struct {
	Role           string           `json:"role"`
	Model          string           `json:"model"`
	ModelName      string           `json:"modelName"`
	ModelNameSnake string           `json:"model_name"`
	Timestamp      string           `json:"timestamp"`
	Time           string           `json:"time"`
	CreatedAt      string           `json:"created_at"`
	Parts          []geminiPart     `json:"parts"`
	Content        json.RawMessage  `json:"content"`
	Text           string           `json:"text"`
	UsageMetadata  *geminiUsageMeta `json:"usageMetadata"`
	Usage          *geminiUsageMeta `json:"usage"`
	ToolCalls      []geminiToolCall `json:"toolCalls"`
	ToolCallsSnake []geminiToolCall `json:"tool_calls"`
}

func (t *geminiTurn) getRole() string {
	r := strings.ToLower(strings.TrimSpace(t.Role))
	if r != "" {
		return r
	}
	for _, p := range t.getParts() {
		if p.FunctionCall != nil || p.FunctionCallSnake != nil {
			return "model"
		}
		if p.FunctionResponse != nil || p.FunctionRespSnake != nil {
			return "user"
		}
	}
	if len(t.ToolCalls) > 0 || len(t.ToolCallsSnake) > 0 {
		return "model"
	}
	if t.UsageMetadata != nil || t.Usage != nil {
		return "model"
	}
	return "user"
}

func (t *geminiTurn) getModel(defaultModel string) string {
	for _, m := range []string{t.Model, t.ModelName, t.ModelNameSnake, defaultModel} {
		m = strings.TrimSpace(m)
		if m != "" {
			return m
		}
	}
	return "?"
}

func (t *geminiTurn) getTimestamp() string {
	for _, ts := range []string{t.Timestamp, t.Time, t.CreatedAt} {
		ts = strings.TrimSpace(ts)
		if ts != "" {
			return ts
		}
	}
	return ""
}

func (t *geminiTurn) getUsage() *geminiUsageMeta {
	if t.UsageMetadata != nil {
		return t.UsageMetadata
	}
	return t.Usage
}

func (t *geminiTurn) getParts() []geminiPart {
	if len(t.Parts) > 0 {
		return t.Parts
	}
	if len(t.Content) > 0 {
		var str string
		if err := json.Unmarshal(t.Content, &str); err == nil {
			return []geminiPart{{Text: str}}
		}
		var parts []geminiPart
		if err := json.Unmarshal(t.Content, &parts); err == nil {
			return parts
		}
		var obj struct {
			Parts []geminiPart `json:"parts"`
		}
		if err := json.Unmarshal(t.Content, &obj); err == nil && len(obj.Parts) > 0 {
			return obj.Parts
		}
	}
	if t.Text != "" {
		return []geminiPart{{Text: t.Text}}
	}
	return nil
}

type geminiChatFile struct {
	SessionID      string           `json:"sessionId"`
	SessionIDSnake string           `json:"session_id"`
	ID             string           `json:"id"`
	UUID           string           `json:"uuid"`
	StartTime      string           `json:"startTime"`
	Timestamp      string           `json:"timestamp"`
	CreatedAt      string           `json:"created_at"`
	LastUpdateTime string           `json:"lastUpdateTime"`
	UpdatedAt      string           `json:"updated_at"`
	Model          string           `json:"model"`
	ModelName      string           `json:"modelName"`
	ModelNameSnake string           `json:"model_name"`
	Turns          []geminiTurn     `json:"turns"`
	Messages       []geminiTurn     `json:"messages"`
	History        []geminiTurn     `json:"history"`
	UsageMetadata  *geminiUsageMeta `json:"usageMetadata"`
	Usage          *geminiUsageMeta `json:"usage"`
}

func (f *geminiChatFile) getSessionID() string {
	for _, id := range []string{f.SessionID, f.SessionIDSnake, f.ID, f.UUID} {
		id = strings.TrimSpace(id)
		if id != "" {
			return id
		}
	}
	return ""
}

func (f *geminiChatFile) getModel() string {
	for _, m := range []string{f.Model, f.ModelName, f.ModelNameSnake} {
		m = strings.TrimSpace(m)
		if m != "" {
			return m
		}
	}
	return ""
}

func (f *geminiChatFile) getTurns() []geminiTurn {
	if len(f.Turns) > 0 {
		return f.Turns
	}
	if len(f.Messages) > 0 {
		return f.Messages
	}
	return f.History
}

func (f *geminiChatFile) getUsage() *geminiUsageMeta {
	if f.UsageMetadata != nil {
		return f.UsageMetadata
	}
	return f.Usage
}

func updateTimestamp(minStr, maxStr *string, newTS string) {
	if newTS == "" {
		return
	}
	tNew, errNew := parseTimestamp(newTS)
	if *minStr == "" {
		*minStr = newTS
	} else {
		if tMin, errMin := parseTimestamp(*minStr); errNew == nil && errMin == nil {
			if tNew.Before(tMin) {
				*minStr = newTS
			}
		} else if newTS < *minStr {
			*minStr = newTS
		}
	}
	if *maxStr == "" {
		*maxStr = newTS
	} else {
		if tMax, errMax := parseTimestamp(*maxStr); errNew == nil && errMax == nil {
			if tNew.After(tMax) {
				*maxStr = newTS
			}
		} else if newTS > *maxStr {
			*maxStr = newTS
		}
	}
}

// ParseGeminiSession parses a Gemini CLI chat session transcript from an io.Reader.
func ParseGeminiSession(r io.Reader, path string) (Session, error) {
	s := Session{
		Path:        path,
		Session:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		RecordTypes: map[string]int64{},
		Models:      map[string]int64{},
		PerModel:    map[string]ModelCounts{},
		PerTrack:    map[string]ModelCounts{},
		Tools:       map[string]int64{},
	}

	data, err := io.ReadAll(r)
	if err != nil {
		s.Error = err.Error()
		return s, err
	}
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		err := errors.New("sessionaudit: empty gemini session data")
		s.Error = err.Error()
		return s, err
	}

	var root geminiChatFile
	var turns []geminiTurn

	if trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &turns); err != nil {
			s.Error = err.Error()
			return s, err
		}
	} else if trimmed[0] == '{' {
		if err := json.Unmarshal(trimmed, &root); err != nil {
			s.Error = err.Error()
			return s, err
		}
		turns = root.getTurns()
	} else {
		err := ErrInvalidGeminiSession
		s.Error = err.Error()
		return s, err
	}

	if len(turns) == 0 && root.getUsage() == nil && root.getModel() == "" && root.getSessionID() == "" {
		err := ErrInvalidGeminiSession
		s.Error = err.Error()
		return s, err
	}

	if sid := root.getSessionID(); sid != "" {
		s.Session = sid
	}
	rootModel := root.getModel()

	for _, ts := range []string{root.StartTime, root.Timestamp, root.CreatedAt, root.LastUpdateTime, root.UpdatedAt} {
		updateTimestamp(&s.TSMin, &s.TSMax, ts)
	}

	for _, turn := range turns {
		s.NRecords++
		role := turn.getRole()
		s.RecordTypes[role]++

		ts := turn.getTimestamp()
		updateTimestamp(&s.TSMin, &s.TSMax, ts)

		parts := turn.getParts()
		switch role {
		case "user":
			hasPromptText := false
			for _, p := range parts {
				if p.FunctionResponse != nil {
					s.NToolResult++
					s.ToolResultChars += int64(len(p.FunctionResponse.Response))
				}
				if txt := strings.TrimSpace(p.Text); txt != "" {
					hasPromptText = true
					if len(txt) > 400 {
						txt = txt[:400]
					}
					s.Prompts = append(s.Prompts, Prompt{Timestamp: ts, Text: txt})
					s.NText++
				}
			}
			if !hasPromptText && turn.Text != "" {
				txt := strings.TrimSpace(turn.Text)
				if len(txt) > 400 {
					txt = txt[:400]
				}
				s.Prompts = append(s.Prompts, Prompt{Timestamp: ts, Text: txt})
				s.NText++
			}

		case "model", "assistant":
			s.AssistantTurns++
			model := turn.getModel(rootModel)
			s.Models[model]++

			for _, p := range parts {
				if p.Thought {
					s.NThinking++
				}
				if txt := strings.TrimSpace(p.Text); txt != "" {
					s.NText++
				}
				if p.FunctionCall != nil {
					name := p.FunctionCall.Name
					if name == "" {
						name = "?"
					}
					s.NToolUse++
					s.Tools[name]++
					s.ToolInputChars += int64(len(p.FunctionCall.GetArgs()))
				}
			}

			// Also capture any direct toolCalls on the turn object
			allCalls := append(turn.ToolCalls, turn.ToolCallsSnake...)
			for _, tc := range allCalls {
				name := tc.Name
				if name == "" && tc.Function != nil {
					name = tc.Function.Name
				}
				if name == "" {
					name = "?"
				}
				// Only increment if not already counted in parts
				if len(parts) == 0 {
					s.NToolUse++
					s.Tools[name]++
				}
			}

			u := turn.getUsage()
			if u != nil {
				in := u.Prompt()
				out := u.Candidates()
				cached := u.Cached()

				s.Tokens.Input += in
				s.Tokens.Output += out
				s.Tokens.CacheRead += cached

				pm := s.PerModel[model]
				pm.Turns++
				pm.Input += in
				pm.Output += out
				pm.CacheRead += cached
				s.PerModel[model] = pm

				pt := s.PerTrack[TrackMain]
				pt.Turns++
				pt.Input += in
				pt.Output += out
				pt.CacheRead += cached
				s.PerTrack[TrackMain] = pt

				if cost, err := StrictModelCostUSD(model, in, 0, cached, out); err == nil {
					s.CostUSD += cost
				} else if r, ok := PriceFor(model); ok {
					s.CostUSD += rawCostUSD(r, in, 0, cached, out)
				}
			} else {
				pm := s.PerModel[model]
				pm.Turns++
				s.PerModel[model] = pm

				pt := s.PerTrack[TrackMain]
				pt.Turns++
				s.PerTrack[TrackMain] = pt
			}

		case "tool":
			for _, p := range parts {
				if p.FunctionResponse != nil {
					s.NToolResult++
					s.ToolResultChars += int64(len(p.FunctionResponse.Response))
				}
			}
		}
	}

	// Fallback to top-level summary usageMetadata if individual turns carried no usage
	if s.Tokens.Input == 0 && s.Tokens.Output == 0 {
		if u := root.getUsage(); u != nil {
			in := u.Prompt()
			out := u.Candidates()
			cached := u.Cached()

			s.Tokens.Input = in
			s.Tokens.Output = out
			s.Tokens.CacheRead += cached

			if len(s.Models) == 1 {
				for m := range s.Models {
					pm := s.PerModel[m]
					pm.Input = in
					pm.Output = out
					pm.CacheRead = cached
					s.PerModel[m] = pm

					if cost, err := StrictModelCostUSD(m, in, 0, cached, out); err == nil {
						s.CostUSD += cost
					} else if r, ok := PriceFor(m); ok {
						s.CostUSD += rawCostUSD(r, in, 0, cached, out)
					}
				}
			}
			pt := s.PerTrack[TrackMain]
			pt.Input = in
			pt.Output = out
			pt.CacheRead = cached
			s.PerTrack[TrackMain] = pt
		}
	}

	finalizeSession(&s)
	return s, nil
}

// isGeminiChatFile reports whether path points to a Gemini CLI chat session transcript.
func isGeminiChatFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".json" {
		return true
	}
	if ext == ".jsonl" {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 4096)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		return false
	}
	content := string(buf[:n])
	return strings.Contains(content, "usageMetadata") ||
		strings.Contains(content, "candidatesTokenCount") ||
		strings.Contains(content, "promptTokenCount")
}

// ParseGeminiChatFile parses a Gemini CLI chat session transcript from a file path.

func ParseGeminiChatFile(path string) (Session, error) {
	f, err := os.Open(path)
	if err != nil {
		s := Session{
			Path:        path,
			Session:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
			Error:       err.Error(),
			RecordTypes: map[string]int64{},
			Models:      map[string]int64{},
			PerModel:    map[string]ModelCounts{},
			PerTrack:    map[string]ModelCounts{},
			Tools:       map[string]int64{},
		}
		return s, err
	}
	defer f.Close()
	return ParseGeminiSession(f, path)
}

// DefaultGeminiRoots returns the standard Gemini CLI chat session transcript directories.
func DefaultGeminiRoots() []string {
	base := os.Getenv("GEMINI_CLI_HOME")
	if base == "" {
		base = os.Getenv("GEMINI_HOME")
	}
	if base == "" {
		if home, err := os.UserHomeDir(); err == nil {
			base = filepath.Join(home, ".gemini")
		} else {
			base = ".gemini"
		}
	}
	return []string{filepath.Join(base, "tmp")}
}

// DiscoverGemini scans candidate roots for Gemini CLI chat session JSON transcripts (~/.gemini/tmp/**/chats/*.json).
func DiscoverGemini(opts DiscoverOptions) ([]Transcript, error) {
	roots := opts.Roots
	if len(roots) == 0 {
		roots = DefaultGeminiRoots()
	}
	var cutoff time.Time
	if opts.SinceDays != nil {
		cutoff = time.Now().Add(-time.Duration(*opts.SinceDays * float64(24*time.Hour)))
	}
	var out []Transcript
	for _, root := range roots {
		if _, err := os.Stat(root); os.IsNotExist(err) {
			continue
		}
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if !strings.EqualFold(filepath.Ext(path), ".json") {
				return nil
			}
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return nil
			}
			parts := strings.Split(filepath.ToSlash(rel), "/")
			ns := "gemini"
			if len(parts) > 1 {
				ns = parts[0]
			}
			if opts.NamespacePrefix != "" && !strings.HasPrefix(ns, opts.NamespacePrefix) {
				return nil
			} else if excludedNamespace(ns) {
				return nil
			}
			if rec, ok := statTranscript(root, ns, path, KindTop, cutoff); ok {
				out = append(out, rec)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].MTime == out[j].MTime {
			return out[i].Path < out[j].Path
		}
		return out[i].MTime > out[j].MTime
	})
	return out, nil
}
