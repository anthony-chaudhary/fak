package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const burnInCorpusSchema = "fak.agentbench.burn-in-corpus.v1"

type burnInCorpus struct {
	Schema   string                 `json:"schema"`
	Revision string                 `json:"revision"`
	Areas    map[string]corpusLayer `json:"areas"`
	Sessions []burnInSession        `json:"sessions"`
}

type burnInSession struct {
	ID               string                  `json:"id"`
	Area             string                  `json:"area"`
	Turns            []normalTurn            `json:"turns"`
	AreaTransitions  []burnInAreaTransition  `json:"area_transitions"`
	HistoryRotations []burnInHistoryRotation `json:"history_rotations"`
}

type burnInHistoryRotation struct {
	Turn                  int    `json:"turn"`
	RemovedSHA256         string `json:"removed_sha256"`
	AddedSHA256           string `json:"added_sha256"`
	BeforeSHA256          string `json:"before_sha256"`
	AfterSHA256           string `json:"after_sha256"`
	CandidatePromptTokens int    `json:"candidate_prompt_tokens"`
	ReservedOutputTokens  int    `json:"reserved_output_tokens"`
}

type burnInAreaTransition struct {
	Area         string `json:"area"`
	BeforeSHA256 string `json:"before_sha256"`
	AfterSHA256  string `json:"after_sha256"`
}

func buildBurnInCorpus(ctx context.Context, encoder referenceEncoder, normal *normalCorpus, repo string) (*burnInCorpus, error) {
	if encoder == nil || normal == nil || len(normal.Sessions) != 8 {
		return nil, fmt.Errorf("burn-in corpus requires a complete exact normal corpus")
	}
	areas, foundation, err := loadBurnAreas(ctx, encoder, normal, repo, normal.Revision)
	if err != nil {
		return nil, err
	}
	result := &burnInCorpus{Schema: burnInCorpusSchema, Revision: normal.Revision, Areas: areas}
	tools := normalToolContracts()
	for sessionIndex := 0; sessionIndex < 8; sessionIndex++ {
		sessionCaps := make([]int, 32)
		capPattern := []int{128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 1024}
		for index := range sessionCaps {
			sessionCaps[index] = capPattern[index%len(capPattern)]
		}
		home := []string{"A", "B", "C", "D"}[sessionIndex%4]
		built := burnInSession{ID: fmt.Sprintf("burn-%02d", sessionIndex+1), Area: home}
		messages := []chatMessage{{Role: "system", Content: normal.SystemTools.content + "\n\nPinned shared repository foundation:\n" + foundation}}
		firstPrompt := 0
		for turn := 1; turn <= 32; turn++ {
			areaName := []string{"A", "B", "C", "D"}[(turn-1)/8]
			area := areas[areaName]
			areaTurn := (turn - 1) % 8
			before := burnMessagesDigest(messages)
			cap := sessionCaps[turn-1]
			candidatePrompt := 0
			var removed []chatMessage
			if areaTurn == 0 && turn > 1 {
				currentRequest := referenceRequest{Messages: append([]chatMessage(nil), messages...), Tools: append([]json.RawMessage(nil), tools...)}
				current, measureErr := encoder.Encode(ctx, currentRequest)
				if measureErr != nil {
					return nil, fmt.Errorf("measure burn session %d pre-transition: %w", sessionIndex+1, measureErr)
				}
				candidatePrompt = current.PromptTokens + area.ReferenceTokens
				targetPrompt := 32768 - cap
				if candidatePrompt+cap > 32768 && len(built.Turns) > 0 {
					targetPrompt = min(targetPrompt, built.Turns[len(built.Turns)-1].PromptTokens)
				}
				for current.PromptTokens+area.ReferenceTokens > targetPrompt && len(messages) > 3 {
					removed = append(removed, messages[1:3]...)
					messages = append(messages[:1], messages[3:]...)
					currentRequest.Messages = append([]chatMessage(nil), messages...)
					current, measureErr = encoder.Encode(ctx, currentRequest)
					if measureErr != nil {
						return nil, fmt.Errorf("measure burn session %d compacted transition: %w", sessionIndex+1, measureErr)
					}
				}
			}
			callID := fmt.Sprintf("burn-%02d-area-%s-%02d", sessionIndex+1, areaName, turn)
			content := area.content
			if areaTurn != 0 {
				historyIndex := turn - 2
				start := historyIndex * len(foundation) / 1000
				end := (historyIndex + 1) * len(foundation) / 1000
				content = foundation[start:end]
			}
			added := []chatMessage{
				{Role: "assistant", ToolCalls: []toolCall{{ID: callID, Type: "function", Function: toolFunction{Name: "Read", Arguments: fmt.Sprintf(`{"area":%q,"revision":%q}`, areaName, normal.Revision)}}}},
				{Role: "tool", ToolCallID: callID, Content: content},
			}
			messages = append(messages, added...)
			previewRequest := referenceRequest{Messages: append([]chatMessage(nil), messages...), Tools: append([]json.RawMessage(nil), tools...)}
			preview, previewErr := encoder.Encode(ctx, previewRequest)
			if previewErr != nil {
				return nil, fmt.Errorf("measure burn session %d turn %d: %w", sessionIndex+1, turn, previewErr)
			}
			if preview.PromptTokens > 31744 && cap != 1024 {
				for future := turn; future < len(sessionCaps); future++ {
					if sessionCaps[future] == 1024 {
						sessionCaps[future], sessionCaps[turn-1] = cap, 1024
						cap = 1024
						break
					}
				}
			}
			if candidatePrompt == 0 {
				candidatePrompt = preview.PromptTokens
			}
			postTarget := 32768 - cap
			if preview.PromptTokens+cap > 32768 && len(built.Turns) > 0 {
				postTarget = min(postTarget, built.Turns[len(built.Turns)-1].PromptTokens-64)
			}
			for preview.PromptTokens > postTarget && len(messages) > 3 {
				removed = append(removed, messages[1:3]...)
				messages = append(messages[:1], messages[3:]...)
				previewRequest.Messages = append([]chatMessage(nil), messages...)
				preview, previewErr = encoder.Encode(ctx, previewRequest)
				if previewErr != nil {
					return nil, fmt.Errorf("measure compacted burn session %d turn %d: %w", sessionIndex+1, turn, previewErr)
				}
			}
			if len(removed) > 0 {
				built.HistoryRotations = append(built.HistoryRotations, burnInHistoryRotation{Turn: turn, RemovedSHA256: burnMessagesDigest(removed), AddedSHA256: burnMessagesDigest(added), BeforeSHA256: before, AfterSHA256: burnMessagesDigest(messages), CandidatePromptTokens: candidatePrompt, ReservedOutputTokens: cap})
			}
			request := referenceRequest{Messages: append([]chatMessage(nil), messages...), Tools: append([]json.RawMessage(nil), tools...), OutputTokens: cap}
			encoded, err := encoder.Encode(ctx, request)
			if err != nil || encoded.PromptTokens+request.OutputTokens > 32768 {
				return nil, fmt.Errorf("encode burn session %d turn %d within 32K: %w", sessionIndex+1, turn, err)
			}
			if turn == 1 {
				firstPrompt = encoded.PromptTokens
			}
			built.Turns = append(built.Turns, normalTurn{Ordinal: turn, OutputTokens: request.OutputTokens, PromptTokens: encoded.PromptTokens, HistoryTokens: encoded.PromptTokens - firstPrompt, Request: request, Encoding: encoded})
			if turn == 1 || turn == 9 || turn == 17 || turn == 25 {
				built.AreaTransitions = append(built.AreaTransitions, burnInAreaTransition{Area: areaName, BeforeSHA256: before, AfterSHA256: burnMessagesDigest(messages)})
			}
		}
		result.Sessions = append(result.Sessions, built)
	}
	return result, nil
}

func loadBurnAreas(ctx context.Context, encoder referenceEncoder, normal *normalCorpus, repo, revision string) (map[string]corpusLayer, string, error) {
	listed, err := gitText(repo, "ls-tree", "-r", "--name-only", revision)
	if err != nil {
		return nil, "", err
	}
	paths := strings.Fields(listed)
	sort.Strings(paths)
	type sourceFile struct {
		path string
		body []byte
	}
	var sources []sourceFile
	for _, path := range paths {
		if safeSourcePath(path) && strings.HasSuffix(strings.ToLower(path), ".go") {
			if body, readErr := gitBlob(repo, revision+":"+path, 4<<20); readErr == nil && len(body) > 0 {
				sources = append(sources, sourceFile{path: path, body: body})
			}
		}
	}
	sort.Slice(sources, func(i, j int) bool { return len(sources[i].body) > len(sources[j].body) })
	if len(sources) < 5 {
		return nil, "", fmt.Errorf("burn-in corpus requires four area files plus distinct committed foundation source")
	}
	areas := make(map[string]corpusLayer, 4)
	var foundation []string
	baseText := normal.SystemTools.content + "\n\n" + normal.RepositoryMap.content
	for index := len(sources) - 1; index >= 4; index-- {
		candidate := append(append([]string(nil), foundation...), string(sources[index].body))
		trial := referenceRequest{Messages: []chatMessage{{Role: "system", Content: baseText + "\n\n" + strings.Join(candidate, "\n\n")}}}
		measured, measureErr := encoder.Encode(ctx, trial)
		if measureErr != nil {
			return nil, "", measureErr
		}
		if measured.PromptTokens < 5420 {
			foundation = candidate
			continue
		}
		// Fit the last committed source prefix instead of accepting a whole-file
		// overshoot. This preserves pinned bytes while leaving deterministic
		// headroom for the 32-turn history and its 1K output reservations.
		lines := strings.Split(string(sources[index].body), "\n")
		low, high := 1, len(lines)
		for low < high {
			middle := low + (high-low)/2
			prefix := strings.Join(lines[:middle], "\n")
			parts := append(append([]string(nil), foundation...), prefix)
			probe := referenceRequest{Messages: []chatMessage{{Role: "system", Content: baseText + "\n\n" + strings.Join(parts, "\n\n")}}}
			encoded, encodeErr := encoder.Encode(ctx, probe)
			if encodeErr != nil {
				return nil, "", encodeErr
			}
			if encoded.PromptTokens >= 5420 {
				high = middle
			} else {
				low = middle + 1
			}
		}
		foundation = append(foundation, strings.Join(lines[:low], "\n"))
		if measured.PromptTokens <= 7000 {
			break
		}
		break
	}
	stable := referenceRequest{Messages: []chatMessage{{Role: "system", Content: baseText + "\n\n" + strings.Join(foundation, "\n\n")}}}
	baseEncoding, err := encoder.Encode(ctx, stable)
	if err != nil || baseEncoding.PromptTokens < 5420 || baseEncoding.PromptTokens > 7000 {
		return nil, "", fmt.Errorf("burn-in stable system/repository template cannot enter exact 5.42K-7K band")
	}
	for index := 0; index < 4; index++ {
		source := sources[index]
		name := []string{"A", "B", "C", "D"}[index]
		lines := strings.Split(string(source.body), "\n")
		var fitted corpusLayer
		for end := 20; end <= len(lines); end += 20 {
			if end > len(lines) {
				end = len(lines)
			}
			content := strings.Join(lines[:end], "\n")
			trial := stable
			trial.Messages = append(append([]chatMessage(nil), stable.Messages...), chatMessage{Role: "system", Content: content})
			measured, measureErr := encoder.Encode(ctx, trial)
			if measureErr != nil {
				return nil, "", measureErr
			}
			areaTokens := measured.PromptTokens - baseEncoding.PromptTokens
			if areaTokens >= 7000 && areaTokens <= 9000 {
				sum := sha256.Sum256([]byte(content))
				fitted = corpusLayer{Name: name, SHA256: hex.EncodeToString(sum[:]), Bytes: len(content), ReferenceTokens: areaTokens, Provenance: []corpusProvenance{{Kind: "committed_source", Path: source.path, Revision: revision, SHA256: hex.EncodeToString(sum[:]), StartLine: 1, EndLine: end}}, content: content}
				break
			}
		}
		if fitted.ReferenceTokens == 0 {
			return nil, "", fmt.Errorf("burn-in area %s cannot enter exact 7K-9K reference band", name)
		}
		areas[name] = fitted
	}
	return areas, strings.Join(foundation, "\n\n"), nil
}

func burnMessagesDigest(messages []chatMessage) string {
	body, _ := json.Marshal(messages)
	return burnDigest(string(body))
}

func burnDigest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func burnRequestDigest(request referenceRequest) string {
	body, _ := json.Marshal(request)
	return burnDigest(string(body))
}
