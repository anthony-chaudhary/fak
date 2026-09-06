package sessionaudit

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
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
	ThoughtsTokens        int64 `json:"thoughtsTokenCount"`
	ThoughtsTokensSnake   int64 `json:"thoughts_token_count"`
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
	cand := u.CandidatesTokens
	if cand == 0 {
		cand = u.CandidatesTokensSnake
	}
	if cand == 0 {
		cand = u.CandidatesTokensShort
	}
	if cand == 0 {
		cand = u.OutputTokensSnake
	}
	if cand == 0 {
		cand = u.OutputTokens
	}
	thoughts := u.ThoughtsTokens
	if thoughts == 0 {
		thoughts = u.ThoughtsTokensSnake
	}
	return cand + thoughts
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

type geminiThought struct {
	Subject     string `json:"subject"`
	Description string `json:"description"`
	Timestamp   string `json:"timestamp"`
}

type geminiTokens struct {
	Input    int64 `json:"input"`
	Output   int64 `json:"output"`
	Cached   int64 `json:"cached"`
	Thoughts int64 `json:"thoughts"`
	Tool     int64 `json:"tool"`
	Total    int64 `json:"total"`
}

type geminiToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Args      json.RawMessage `json:"args"`
	Result    []any           `json:"result"`
	Status    string          `json:"status"`
	Timestamp string          `json:"timestamp"`
	Function  *struct {
		Name string          `json:"name"`
		Args json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type geminiTurn struct {
	ID             string           `json:"id"`
	Role           string           `json:"role"`
	Type           string           `json:"type"`
	Model          string           `json:"model"`
	ModelVersion   string           `json:"modelVersion"`
	ModelName      string           `json:"modelName"`
	ModelNameSnake string           `json:"model_name"`
	Timestamp      string           `json:"timestamp"`
	Time           string           `json:"time"`
	CreatedAt      string           `json:"created_at"`
	Parts          []geminiPart     `json:"parts"`
	Content        json.RawMessage  `json:"content"`
	Text           string           `json:"text"`
	Thoughts       []geminiThought  `json:"thoughts"`
	UsageMetadata  *geminiUsageMeta `json:"usageMetadata"`
	Usage          *geminiUsageMeta `json:"usage"`
	Tokens         *geminiTokens    `json:"tokens"`
	ToolCalls      []geminiToolCall `json:"toolCalls"`
	ToolCallsSnake []geminiToolCall `json:"tool_calls"`
}

func (t *geminiTurn) getRole() string {
	for _, r := range []string{t.Role, t.Type} {
		r = strings.ToLower(strings.TrimSpace(r))
		if r == "gemini" || r == "assistant" {
			return "model"
		}
		if r != "" {
			return r
		}
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
	if t.UsageMetadata != nil || t.Usage != nil || t.Tokens != nil {
		return "model"
	}
	return "user"
}

func (t *geminiTurn) getModel(defaultModel string) string {
	for _, m := range []string{t.Model, t.ModelVersion, t.ModelName, t.ModelNameSnake, defaultModel} {
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
	if t.Usage != nil {
		return t.Usage
	}
	if t.Tokens != nil {
		return &geminiUsageMeta{
			PromptTokens:     t.Tokens.Input,
			CandidatesTokens: t.Tokens.Output,
			ThoughtsTokens:   t.Tokens.Thoughts,
			CachedTokens:     t.Tokens.Cached,
			TotalTokens:      t.Tokens.Total,
		}
	}
	return nil
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
	ProjectHash    string           `json:"projectHash"`
	StartTime      string           `json:"startTime"`
	Timestamp      string           `json:"timestamp"`
	CreatedAt      string           `json:"created_at"`
	LastUpdateTime string           `json:"lastUpdateTime"`
	LastUpdated    string           `json:"lastUpdated"`
	UpdatedAt      string           `json:"updated_at"`
	Model          string           `json:"model"`
	ModelVersion   string           `json:"modelVersion"`
	ModelName      string           `json:"modelName"`
	ModelNameSnake string           `json:"model_name"`
	Kind           string           `json:"kind"`
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
	for _, m := range []string{f.Model, f.ModelVersion, f.ModelName, f.ModelNameSnake} {
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
		Kind:        KindGemini,
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
		// First try unmarshaling as a single geminiChatFile object
		if err := json.Unmarshal(trimmed, &root); err == nil && (len(root.Turns) > 0 || len(root.Messages) > 0 || len(root.History) > 0 || root.getUsage() != nil) {
			turns = root.getTurns()
		} else {
			// If not a standard single object, scan line by line for JSONL formats
			scanner := bufio.NewScanner(bytes.NewReader(trimmed))
			var lineNum int
			for scanner.Scan() {
				lineNum++
				line := bytes.TrimSpace(scanner.Bytes())
				if len(line) == 0 {
					continue
				}
				if lineNum == 1 {
					_ = json.Unmarshal(line, &root)
				}
				// Check for $set wrapper
				var setWrapper struct {
					Set struct {
						Messages []geminiTurn `json:"messages"`
						Turns    []geminiTurn `json:"turns"`
					} `json:"$set"`
				}
				if err := json.Unmarshal(line, &setWrapper); err == nil && (len(setWrapper.Set.Messages) > 0 || len(setWrapper.Set.Turns) > 0) {
					if len(setWrapper.Set.Messages) > 0 {
						turns = append(turns, setWrapper.Set.Messages...)
					} else {
						turns = append(turns, setWrapper.Set.Turns...)
					}
					continue
				}
				// Check if line is a single turn
				var t geminiTurn
				if err := json.Unmarshal(line, &t); err == nil && (t.Role != "" || t.Type != "" || len(t.Parts) > 0 || len(t.Content) > 0 || t.Text != "" || t.Tokens != nil || t.UsageMetadata != nil) {
					turns = append(turns, t)
				}
			}
			if len(turns) == 0 && root.getTurns() != nil {
				turns = root.getTurns()
			}
		}
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

	for _, ts := range []string{root.StartTime, root.Timestamp, root.CreatedAt, root.LastUpdateTime, root.LastUpdated, root.UpdatedAt} {
		updateTimestamp(&s.TSMin, &s.TSMax, ts)
	}

	lens := newBehaviorLens()
	clens := newConfusionLens()

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
					lens.noteToolResult(p.FunctionResponse.ID, false, string(p.FunctionResponse.Response))
				}
				if txt := strings.TrimSpace(p.Text); txt != "" {
					hasPromptText = true
					if looksLikeTypedPrompt(txt) {
						if len(txt) > 400 {
							txt = txt[:400]
						}
						s.Prompts = append(s.Prompts, Prompt{Timestamp: ts, Text: txt})
					}
					s.NText++
				}
			}
			if !hasPromptText && turn.Text != "" {
				txt := strings.TrimSpace(turn.Text)
				if looksLikeTypedPrompt(txt) {
					if len(txt) > 400 {
						txt = txt[:400]
					}
					s.Prompts = append(s.Prompts, Prompt{Timestamp: ts, Text: txt})
				}
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
					clens.noteText(json.RawMessage(strconv.Quote(txt)))
				}
				if p.FunctionCall != nil {
					name := p.FunctionCall.Name
					if name == "" {
						name = "?"
					}
					s.NToolUse++
					s.Tools[name]++
					args := p.FunctionCall.GetArgs()
					s.ToolInputChars += int64(len(args))
					id := p.FunctionCall.ID
					if id == "" {
						id = fmt.Sprintf("call-%d", s.NToolUse)
					}
					lens.noteToolUse(id, name, args, canonicalArgs(args))
				}
			}

			for _, th := range turn.Thoughts {
				s.NThinking++
				if desc := strings.TrimSpace(th.Description); desc != "" {
					clens.noteText(json.RawMessage(strconv.Quote(desc)))
				}
			}

			// Capture any direct toolCalls on the turn object
			hasPartFuncCall := false
			for _, p := range parts {
				if p.FunctionCall != nil {
					hasPartFuncCall = true
					break
				}
			}
			if !hasPartFuncCall {
				allCalls := append(turn.ToolCalls, turn.ToolCallsSnake...)
				for _, tc := range allCalls {
					name := tc.Name
					if name == "" && tc.Function != nil {
						name = tc.Function.Name
					}
					if name == "" {
						name = "?"
					}
					s.NToolUse++
					s.Tools[name]++
					args := tc.Args
					if len(args) == 0 && tc.Function != nil {
						args = tc.Function.Args
					}
					s.ToolInputChars += int64(len(args))
					id := tc.ID
					if id == "" {
						id = fmt.Sprintf("call-%d", s.NToolUse)
					}
					lens.noteToolUse(id, name, args, canonicalArgs(args))
					if len(tc.Result) > 0 {
						s.NToolResult++
						resBytes, _ := json.Marshal(tc.Result)
						s.ToolResultChars += int64(len(resBytes))
						isErr := strings.EqualFold(tc.Status, "error") || strings.EqualFold(tc.Status, "failed")
						lens.noteToolResult(id, isErr, string(resBytes))
					}
				}
			}

			u := turn.getUsage()
			if u != nil {
				in := u.Prompt()
				out := u.Candidates()
				cached := u.Cached()

				fresh := in
				if cached > 0 && in >= cached {
					fresh = in - cached
				}

				s.Tokens.Input += fresh
				s.Tokens.Output += out
				s.Tokens.CacheRead += cached

				pm := s.PerModel[model]
				pm.Turns++
				pm.Input += fresh
				pm.Output += out
				pm.CacheRead += cached
				s.PerModel[model] = pm

				pt := s.PerTrack[TrackMain]
				pt.Turns++
				pt.Input += fresh
				pt.Output += out
				pt.CacheRead += cached
				s.PerTrack[TrackMain] = pt

				if cost, err := StrictModelCostUSD(model, fresh, 0, cached, out); err == nil {
					s.CostUSD += cost
				} else if r, ok := PriceFor(model); ok {
					s.CostUSD += rawCostUSD(r, fresh, 0, cached, out)
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
					lens.noteToolResult(p.FunctionResponse.ID, false, string(p.FunctionResponse.Response))
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

			fresh := in
			if cached > 0 && in >= cached {
				fresh = in - cached
			}

			s.Tokens.Input = fresh
			s.Tokens.Output = out
			s.Tokens.CacheRead += cached

			if len(s.Models) == 1 {
				for m := range s.Models {
					pm := s.PerModel[m]
					pm.Input = fresh
					pm.Output = out
					pm.CacheRead = cached
					s.PerModel[m] = pm

					if cost, err := StrictModelCostUSD(m, fresh, 0, cached, out); err == nil {
						s.CostUSD += cost
					} else if r, ok := PriceFor(m); ok {
						s.CostUSD += rawCostUSD(r, fresh, 0, cached, out)
					}
				}
			}
			pt := s.PerTrack[TrackMain]
			pt.Input = fresh
			pt.Output = out
			pt.CacheRead = cached
			s.PerTrack[TrackMain] = pt
		}
	}

	finalizeSession(&s)
	s.Behavior = lens.summary()
	s.Confusion = clens.summary()
	return s, nil
}

// isGeminiChatFile reports whether path points to a Gemini CLI chat session transcript.
func isGeminiChatFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".json" {
		return true
	}
	if ext == ".jsonl" {
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
			strings.Contains(content, "promptTokenCount") ||
			strings.Contains(content, "sessionId") ||
			strings.Contains(content, "session_id") ||
			strings.Contains(content, "modelVersion")
	}
	return false
}

// ParseGeminiChatFile parses a Gemini CLI chat session transcript from a file path.
func ParseGeminiChatFile(path string) (Session, error) {
	f, err := os.Open(path)
	if err != nil {
		s := Session{
			Path:        path,
			Session:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
			Kind:        KindGemini,
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
	if tmp := os.Getenv("GEMINI_TMP_DIR"); tmp != "" {
		return []string{tmp}
	}
	base := os.Getenv("GEMINI_CLI_HOME")
	if base == "" {
		base = os.Getenv("GEMINI_HOME")
	}
	if base == "" {
		base = os.Getenv("GEMINI_CONFIG_DIR")
	}
	if base != "" {
		return []string{filepath.Join(base, "tmp")}
	}
	// If CLAUDE_CONFIG_DIR was explicitly set to isolate tests, do not touch real user home directory.
	if claudeDir := os.Getenv("CLAUDE_CONFIG_DIR"); claudeDir != "" {
		candidates := []string{
			filepath.Join(claudeDir, ".gemini", "tmp"),
			filepath.Join(claudeDir, "gemini", "tmp"),
			filepath.Join(filepath.Dir(claudeDir), ".gemini", "tmp"),
		}
		for _, c := range candidates {
			if st, err := os.Stat(c); err == nil && st.IsDir() {
				return []string{c}
			}
		}
		return nil
	}
	if home, err := os.UserHomeDir(); err == nil {
		return []string{filepath.Join(home, ".gemini", "tmp")}
	}
	return nil
}

// DiscoverGemini scans candidate roots for Gemini CLI chat session JSON transcripts (~/.gemini/tmp/**/chats/*.json).
func DiscoverGemini(opts DiscoverOptions) ([]Transcript, error) {
	roots := opts.Roots
	if len(roots) == 0 {
		roots = opts.GeminiRoots
	}
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
			ext := strings.ToLower(filepath.Ext(path))
			if ext != ".json" && ext != ".jsonl" {
				return nil
			}
			if ext == ".jsonl" && !isGeminiChatFile(path) {
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
			// Check if parent directory of chats has a .project_root file
			dir := filepath.Dir(path)
			baseDir := filepath.Base(dir)
			if baseDir == "chats" {
				parent := filepath.Dir(dir)
				if prData, err := os.ReadFile(filepath.Join(parent, ".project_root")); err == nil {
					prClean := strings.TrimSpace(string(prData))
					if prClean != "" {
						ns = ProjectNamespace(prClean)
					}
				}
			}
			if opts.NamespacePrefix != "" && !strings.HasPrefix(ns, opts.NamespacePrefix) {
				return nil
			} else if excludedNamespace(ns) {
				return nil
			}
			kind := KindGemini
			if len(parts) > 3 {
				// e.g. proj/chats/sub1/session.json is a subagent chat
				if !opts.IncludeSubagents {
					return nil
				}
				kind = KindSpawned
			}
			if rec, ok := statTranscript(root, ns, path, kind, cutoff); ok {
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
