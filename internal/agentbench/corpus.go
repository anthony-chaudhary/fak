package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const normalCorpusSchema = "fak.agentbench.normal-corpus.v1"

const normalHistorySchedule = "fak.agentbench.history-growth.v1"

// Cumulative exact reference-token growth after the initial 14k envelope.
// Recorded result increments step from 0.5k to 1k and then 2k late.
var normalHistoryTargets = [...]int{0, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384}

type referenceEncoder interface {
	Encode(context.Context, referenceRequest) (referenceEncoding, error)
}

type corpusProvenance struct {
	Kind      string `json:"kind"`
	Path      string `json:"path"`
	Revision  string `json:"revision"`
	SHA256    string `json:"sha256"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

type corpusLayer struct {
	Name            string             `json:"name"`
	SHA256          string             `json:"sha256"`
	Bytes           int                `json:"bytes"`
	ReferenceTokens int                `json:"reference_tokens"`
	Provenance      []corpusProvenance `json:"provenance"`
	content         string
}

type normalTurn struct {
	Ordinal       int               `json:"ordinal"`
	OutputTokens  int               `json:"output_tokens"`
	PromptTokens  int               `json:"prompt_tokens"`
	HistoryTokens int               `json:"history_tokens"`
	HistoryTarget int               `json:"history_target_tokens"`
	Request       referenceRequest  `json:"request"`
	Encoding      referenceEncoding `json:"encoding"`
	LayerDigests  map[string]string `json:"layer_digests"`
}

type normalSession struct {
	ID    string       `json:"id"`
	Area  string       `json:"area"`
	Turns []normalTurn `json:"turns"`
}

type normalPrecondition struct {
	Concurrency      int               `json:"concurrency"`
	PromptTokens     int               `json:"prompt_tokens"`
	NonceTokenOffset int               `json:"nonce_token_offset"`
	Nonce            string            `json:"nonce"`
	EncodingSHA256   string            `json:"encoding_sha256"`
	Request          referenceRequest  `json:"request"`
	Encoding         referenceEncoding `json:"encoding"`
}

type normalCorpus struct {
	Schema          string                     `json:"schema"`
	HistorySchedule string                     `json:"history_schedule"`
	Revision        string                     `json:"revision"`
	SystemTools     corpusLayer                `json:"system_tools"`
	RepositoryMap   corpusLayer                `json:"repository_map"`
	Areas           map[string]corpusLayer     `json:"areas"`
	Sessions        []normalSession            `json:"sessions"`
	Preconditions   map[int]normalPrecondition `json:"preconditions"`
	frozen          frozenInputs
}

func (c *normalCorpus) FrozenInputs() frozenInputs {
	if c == nil {
		return frozenInputs{}
	}
	areas := make(map[string]string, len(c.frozen.area))
	for name, value := range c.frozen.area {
		areas[name] = value
	}
	return frozenInputs{system: c.frozen.system, repo: c.frozen.repo, area: areas}
}

type corpusChunk struct {
	kind, path string
	start, end int
	text       string
}

func buildNormalCorpus(ctx context.Context, encoder referenceEncoder, repo string) (*normalCorpus, error) {
	if encoder == nil {
		return nil, errors.New("normal corpus requires exact reference encoding")
	}
	revision, err := gitText(repo, "rev-parse", "HEAD")
	if err != nil || revision == "" {
		return nil, errors.New("normal corpus requires a committed Git revision")
	}
	listed, err := gitText(repo, "ls-tree", "-r", "--name-only", revision)
	if err != nil {
		return nil, fmt.Errorf("list normal corpus source: %w", err)
	}
	paths := strings.Split(strings.TrimSpace(listed), "\n")
	sort.Strings(paths)
	var sourcePaths []string
	for _, path := range paths {
		if safeSourcePath(path) && strings.HasSuffix(strings.ToLower(path), ".go") {
			sourcePaths = append(sourcePaths, path)
		}
	}
	if len(sourcePaths) < 8 {
		return nil, fmt.Errorf("normal corpus has %d committed source files; need at least 8 distinct files", len(sourcePaths))
	}

	tools := normalToolContracts()
	systemChunks := []corpusChunk{{kind: "tool_contract", path: "agentbench/tool-contract-v1", start: 1, end: 1, text: normalSystemInstructions}}
	if agents, readErr := gitBlob(repo, revision+":AGENTS.md", 1<<20); readErr == nil {
		systemChunks = append(systemChunks, splitCorpusChunks("committed_source", "AGENTS.md", string(agents), 1, 20)...)
	}
	repoChunks := splitCorpusChunks("search_record", "git ls-tree -r --name-only "+revision, "repository\n"+strings.Join(paths, "\n"), 1, 40)
	areaAChunks, areaBChunks, historyChunks, err := normalSourceChunks(repo, revision, sourcePaths)
	if err != nil {
		return nil, err
	}
	system, err := fitCorpusLayer(ctx, encoder, revision, "system_tools", systemChunks, tools, 3500, 4500)
	if err != nil {
		return nil, err
	}
	repository, err := fitCorpusLayer(ctx, encoder, revision, "repository_map", repoChunks, nil, 1500, 2500)
	if err != nil {
		return nil, err
	}
	areaA, err := fitCorpusLayer(ctx, encoder, revision, "A", areaAChunks, nil, 7000, 9000)
	if err != nil {
		return nil, err
	}
	areaB, err := fitCorpusLayer(ctx, encoder, revision, "B", areaBChunks, nil, 7000, 9000)
	if err != nil {
		return nil, err
	}

	corpus := &normalCorpus{
		Schema: normalCorpusSchema, HistorySchedule: normalHistorySchedule, Revision: revision,
		SystemTools: system, RepositoryMap: repository,
		Areas:         map[string]corpusLayer{"A": areaA, "B": areaB},
		Preconditions: make(map[int]normalPrecondition, 4),
		frozen:        frozenInputs{system: system.content, repo: repository.content, area: map[string]string{"A": areaA.content, "B": areaB.content}},
	}
	for session := 1; session <= 8; session++ {
		area := "A"
		if session > 6 {
			area = "B"
		}
		built, err := buildNormalSession(ctx, encoder, fmt.Sprintf("steady-%02d", session), area, corpus, tools, historyChunks)
		if err != nil {
			return nil, err
		}
		corpus.Sessions = append(corpus.Sessions, built)
	}
	if err := buildNormalPreconditions(ctx, encoder, corpus, tools); err != nil {
		return nil, err
	}
	return corpus, nil
}

const normalSystemInstructions = `AgentBench normal replay uses a frozen public repository at a committed revision. Inspect source, search symbols, explain diffs, and interpret recorded test results using only supplied material. Preserve tool schemas and byte ordering. Never claim a live edit, test, hardware property, cache hit, or accepted task from replay evidence.`

func normalToolContracts() []json.RawMessage {
	return []json.RawMessage{
		json.RawMessage(`{"type":"function","function":{"name":"Read","description":"Read a bounded committed source range","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["file_path"]}}}`),
		json.RawMessage(`{"type":"function","function":{"name":"Grep","description":"Search frozen committed source","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}}`),
		json.RawMessage(`{"type":"function","function":{"name":"Edit","description":"Describe a candidate patch without applying it","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["file_path","old_string","new_string"]}}}`),
		json.RawMessage(`{"type":"function","function":{"name":"Bash","description":"Interpret a recorded bounded test command and result","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}`),
	}
}

func normalSourceChunks(repo, revision string, paths []string) ([]corpusChunk, []corpusChunk, []corpusChunk, error) {
	var areaA, areaB []corpusChunk
	history := normalCommandRecords(repo, revision)
	for index, path := range paths {
		blob, err := gitBlob(repo, revision+":"+path, 1<<20)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("read committed %s: %w", path, err)
		}
		body := string(blob)
		lines := strings.Split(body, "\n")
		mid := max(1, len(lines)/2)
		first := strings.Join(lines[:mid], "\n")
		areaChunks := splitCorpusChunks("committed_source", path, first, 1, 20)
		if index%2 == 0 {
			areaA = append(areaA, areaChunks...)
		} else {
			areaB = append(areaB, areaChunks...)
		}
		if mid < len(lines) {
			history = append(history, splitCorpusChunks("committed_source", path, strings.Join(lines[mid:], "\n"), mid+1, 20)...)
		}
	}
	return areaA, areaB, history, nil
}

func normalCommandRecords(repo, revision string) []corpusChunk {
	var records []corpusChunk
	if search, err := gitText(repo, "grep", "-n", "-e", "func ", revision, "--", "*.go"); err == nil && search != "" {
		records = append(records, splitCorpusChunks("search_record", "git grep -n -e func <revision> -- *.go", search, 1, 20)...)
	}
	if diff, err := gitText(repo, "show", "--format=", "--no-ext-diff", "--unified=3", revision); err == nil && diff != "" {
		records = append(records, splitCorpusChunks("diff_record", "git show --format= --no-ext-diff --unified=3 "+revision, diff, 1, 20)...)
	}
	return records
}

func fitCorpusLayer(ctx context.Context, encoder referenceEncoder, revision, name string, chunks []corpusChunk, tools []json.RawMessage, low, high int) (corpusLayer, error) {
	var selected []corpusChunk
	for _, chunk := range chunks {
		selected = append(selected, chunk)
		layer := makeCorpusLayer(revision, name, selected)
		encoded, err := encoder.Encode(ctx, referenceRequest{Messages: []chatMessage{{Role: "system", Content: layer.content}}, Tools: tools})
		if err != nil {
			return corpusLayer{}, err
		}
		layer.ReferenceTokens = encoded.PromptTokens
		if layer.ReferenceTokens >= low && layer.ReferenceTokens <= high {
			return layer, nil
		}
		if layer.ReferenceTokens > high {
			selected = selected[:len(selected)-1]
			for _, lineChunk := range splitCorpusChunks(chunk.kind, chunk.path, chunk.text, chunk.start, 1) {
				trial := append(append([]corpusChunk(nil), selected...), lineChunk)
				candidate := makeCorpusLayer(revision, name, trial)
				measured, measureErr := encoder.Encode(ctx, referenceRequest{Messages: []chatMessage{{Role: "system", Content: candidate.content}}, Tools: tools})
				if measureErr != nil {
					return corpusLayer{}, measureErr
				}
				candidate.ReferenceTokens = measured.PromptTokens
				if candidate.ReferenceTokens >= low && candidate.ReferenceTokens <= high {
					return candidate, nil
				}
				if candidate.ReferenceTokens > high {
					break
				}
				selected = trial
			}
			return corpusLayer{}, fmt.Errorf("normal corpus layer %s cannot enter exact token band without truncating source bytes", name)
		}
	}
	return corpusLayer{}, fmt.Errorf("normal corpus layer %s has insufficient distinct material for %d reference tokens", name, low)
}

func splitCorpusChunks(kind, path, text string, startLine, linesPerChunk int) []corpusChunk {
	lines := strings.Split(text, "\n")
	chunks := make([]corpusChunk, 0, (len(lines)+linesPerChunk-1)/linesPerChunk)
	for start := 0; start < len(lines); start += linesPerChunk {
		end := min(len(lines), start+linesPerChunk)
		chunks = append(chunks, corpusChunk{kind: kind, path: path, start: startLine + start, end: startLine + end - 1, text: strings.Join(lines[start:end], "\n")})
	}
	return chunks
}

func makeCorpusLayer(revision, name string, chunks []corpusChunk) corpusLayer {
	parts := make([]string, 0, len(chunks))
	provenance := make([]corpusProvenance, 0, len(chunks))
	for _, chunk := range chunks {
		parts = append(parts, chunk.text)
		digest := sha256.Sum256([]byte(chunk.text))
		provenance = append(provenance, corpusProvenance{Kind: chunk.kind, Path: chunk.path, Revision: revision, SHA256: hex.EncodeToString(digest[:]), StartLine: chunk.start, EndLine: chunk.end})
	}
	content := strings.Join(parts, "\n\n")
	digest := sha256.Sum256([]byte(content))
	return corpusLayer{Name: name, SHA256: hex.EncodeToString(digest[:]), Bytes: len(content), Provenance: provenance, content: content}
}

func buildNormalSession(ctx context.Context, encoder referenceEncoder, id, area string, corpus *normalCorpus, tools []json.RawMessage, history []corpusChunk) (normalSession, error) {
	base := strings.Join([]string{corpus.SystemTools.content, corpus.RepositoryMap.content, corpus.Areas[area].content}, "\n\n")
	messages := []chatMessage{{Role: "system", Content: base}}
	caps := []int{128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 1024}
	result := normalSession{ID: id, Area: area}
	firstTokens := 0
	historyCursor := 0
	for turn, cap := range caps {
		if turn > 0 {
			if historyCursor >= len(history) {
				return normalSession{}, errors.New("normal corpus has insufficient distinct recorded history")
			}
			callID := fmt.Sprintf("recorded-%02d", turn)
			messages = append(messages,
				chatMessage{Role: "assistant", ToolCalls: []toolCall{{ID: callID, Type: "function", Function: toolFunction{Name: recordedToolName(history[historyCursor].kind), Arguments: `{"source":"frozen committed record"}`}}}},
				chatMessage{Role: "tool", Content: history[historyCursor].text, ToolCallID: callID},
			)
			historyCursor++
		}
		var request referenceRequest
		var encoded referenceEncoding
		for {
			request = referenceRequest{Messages: append([]chatMessage(nil), messages...), Tools: append([]json.RawMessage(nil), tools...), OutputTokens: cap}
			var err error
			encoded, err = encoder.Encode(ctx, request)
			if err != nil {
				return normalSession{}, fmt.Errorf("encode %s turn %d: %w", id, turn+1, err)
			}
			target := firstTokens + normalHistoryTargets[turn]
			if turn == 0 || encoded.PromptTokens >= target {
				break
			}
			if historyCursor >= len(history) {
				return normalSession{}, errors.New("normal corpus has insufficient distinct recorded history")
			}
			if messages[len(messages)-1].Content != "" {
				messages[len(messages)-1].Content += "\n\n"
			}
			messages[len(messages)-1].Content += history[historyCursor].text
			historyCursor++
		}
		if turn == 0 {
			firstTokens = encoded.PromptTokens
		}
		result.Turns = append(result.Turns, normalTurn{Ordinal: turn + 1, OutputTokens: cap, PromptTokens: encoded.PromptTokens, HistoryTokens: encoded.PromptTokens - firstTokens, HistoryTarget: normalHistoryTargets[turn], Request: request, Encoding: encoded, LayerDigests: map[string]string{"system_tools": corpus.SystemTools.SHA256, "repository_map": corpus.RepositoryMap.SHA256, "area": corpus.Areas[area].SHA256}})
	}
	if firstTokens < 13000 || firstTokens > 15000 || result.Turns[11].PromptTokens < 28000 || result.Turns[11].PromptTokens > 31744 {
		return normalSession{}, fmt.Errorf("normal session %s exact context progression is outside 14k-to-32k envelope", id)
	}
	return result, nil
}

func recordedToolName(kind string) string {
	switch kind {
	case "search_record":
		return "Grep"
	case "diff_record", "test_record":
		return "Bash"
	default:
		return "Read"
	}
}

func buildNormalPreconditions(ctx context.Context, encoder referenceEncoder, corpus *normalCorpus, tools []json.RawMessage) error {
	base := corpus.SystemTools.content + "\n\n" + corpus.RepositoryMap.content + "\n\nPreconditioning marker: "
	baseRequest := referenceRequest{Messages: []chatMessage{{Role: "system", Content: base}}, Tools: append([]json.RawMessage(nil), tools...), OutputTokens: 128}
	baseEncoding, err := encoder.Encode(ctx, baseRequest)
	if err != nil {
		return err
	}
	commonDigest := baseEncoding.RenderedSHA256
	var commonTokens int
	commonOffset := -1
	for _, concurrency := range []int{1, 2, 4, 8} {
		nonce := fmt.Sprintf("normal-rung-%02d", concurrency)
		request := referenceRequest{Messages: []chatMessage{{Role: "system", Content: base + nonce}}, Tools: append([]json.RawMessage(nil), tools...), OutputTokens: 128}
		encoded, err := encoder.Encode(ctx, request)
		if err != nil {
			return err
		}
		if commonTokens == 0 {
			commonTokens = encoded.PromptTokens
		} else if encoded.PromptTokens != commonTokens {
			return errors.New("normal rung nonces do not preserve exact reference token count")
		}
		offset := commonTokenPrefix(baseEncoding.TokenIDs, encoded.TokenIDs)
		if commonOffset < 0 {
			commonOffset = offset
		} else if offset != commonOffset {
			return errors.New("normal rung nonces do not occupy the same verified token position")
		}
		corpus.Preconditions[concurrency] = normalPrecondition{Concurrency: concurrency, PromptTokens: encoded.PromptTokens, NonceTokenOffset: offset, Nonce: nonce, EncodingSHA256: commonDigest, Request: request, Encoding: encoded}
	}
	return nil
}

func commonTokenPrefix(left, right []int) int {
	limit := min(len(left), len(right))
	for index := 0; index < limit; index++ {
		if left[index] != right[index] {
			return index
		}
	}
	return limit
}
