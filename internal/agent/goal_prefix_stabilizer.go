package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// goal_prefix_stabilizer.go — byte-stable prompt prefix restoration for goal-continuation sessions (#10671).
//
// In goal-continuation sessions (e.g. autonomous agent loops in Codex, Claude, or fak guard),
// continuation prompts frequently inject volatile turn-specific progress, step counters, and
// token budget metadata alongside the standing goal statement, or repeatedly inject massive
// redundant world_state snapshots.
//
// When volatile metadata leads a message or is repeated in intermediate turns, the byte prefix
// diverges from token 0, invalidating prompt cache entries (Anthropic prompt cache, OpenAI
// prefix cache, or local vLLM/SGLang RadixAttention) across turns.
//
// This module restores byte-stability by:
//  1. Separating the static goal statement from volatile turn-specific progress/budget fields.
//     Leading goal definitions are canonicalized so they remain byte-identical to earlier turns,
//     and volatile progress deltas are positioned at the end/tail of the turn.
//  2. Rate-limiting and deduplicating world_state injections: the first world_state snapshot
//     in the prefix is preserved verbatim, while subsequent redundant full snapshots in
//     intermediate turns are filtered or converted to compact diffs.

var (
	volatileLinePrefixes = []string{
		"turn:",
		"current turn:",
		"turn number:",
		"turn count:",
		"step:",
		"current step:",
		"iteration:",
		"cycle:",
		"budget:",
		"token budget:",
		"tokens remaining:",
		"remaining budget:",
		"budget remaining:",
		"budget left:",
		"tokens left:",
		"turns left:",
		"turn left:",
		"turns remaining:",
		"steps left:",
		"step left:",
		"steps remaining:",
		"iterations remaining:",
		"iteration left:",
		"tokens used:",
		"cost:",
		"spend:",
		"cost limit:",
		"time remaining:",
		"elapsed:",
		"elapsed time:",
		"timestamp:",
		"progress:",
		"progress delta:",
		"status:",
		"current status:",
		"completion:",
	}

	volatileSections = []string{
		"progress",
		"budget",
		"turn info",
		"status",
		"volatile",
		"telemetry",
	}

	volatileTagNames = []string{
		"turn",
		"current_turn",
		"step",
		"iteration",
		"cycle",
		"budget",
		"token_budget",
		"remaining_budget",
		"tokens_remaining",
		"tokens_used",
		"cost",
		"progress",
		"progress_delta",
		"status",
		"elapsed",
		"timestamp",
		"volatile",
		"delta",
		"turn_context",
	}

	volatileJSONKeys = map[string]bool{
		"turn":             true,
		"current_turn":     true,
		"turn_number":      true,
		"turn_count":       true,
		"step":             true,
		"current_step":     true,
		"iteration":        true,
		"cycle":            true,
		"budget":           true,
		"token_budget":     true,
		"tokens_remaining": true,
		"remaining_budget": true,
		"budget_remaining": true,
		"budget_left":      true,
		"tokens_left":      true,
		"turns_left":       true,
		"turn_left":        true,
		"turns_remaining":  true,
		"steps_left":       true,
		"step_left":        true,
		"steps_remaining":  true,
		"tokens_used":      true,
		"tokens":           true,
		"cost":             true,
		"spend":            true,
		"cost_limit":       true,
		"time_remaining":   true,
		"elapsed":          true,
		"elapsed_time":     true,
		"timestamp":        true,
		"progress":         true,
		"progress_delta":   true,
		"status":           true,
		"current_status":   true,
		"completion":       true,
	}

	goalLineHeaders = []string{
		"goal continuation:",
		"goal:",
		"active goal:",
		"standing goal:",
		"session goal:",
		"current goal:",
		"objective:",
		"# goal",
		"## goal",
		"# goal continuation",
		"## goal continuation",
	}

	volatileTagRegexps = func() []*regexp.Regexp {
		res := make([]*regexp.Regexp, len(volatileTagNames))
		for i, tag := range volatileTagNames {
			res[i] = regexp.MustCompile(`(?is)<` + tag + `(\s+[^>]*)?>.*?</` + tag + `>`)
		}
		return res
	}()

	goalTagStripRe = regexp.MustCompile(`(?is)<[^>]+>`)

	worldStateTagRe     = regexp.MustCompile(`(?is)(.*?)(<world_state(?:\s+[^>]*)?>)(.*?)(</world_state>)(.*)`)
	worldStateAltTagRe  = regexp.MustCompile(`(?is)(.*?)(<(?:system_state|environment_state)(?:\s+[^>]*)?>)(.*?)(</(?:system_state|environment_state)>)(.*)`)
	worldStateBracketRe = regexp.MustCompile(`(?is)(.*?)(\[world_state\])(.*?)(\[/world_state\])(.*)`)
	worldStateHeaderRe  = regexp.MustCompile(`(?is)(.*?)(?:^|\n)(#+\s*world state\b|world state:)(.*)`)
)

// IsGoalContinuationMessage reports whether content is a goal continuation message.
func IsGoalContinuationMessage(content string) bool {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return false
	}

	// 1. Check explicit marker sentinels
	if strings.Contains(content, compactGoalMarker) ||
		strings.Contains(content, "[goal]") ||
		strings.Contains(content, "[goal_continuation]") ||
		strings.Contains(content, "[goal-continuation]") ||
		strings.Contains(content, "[goal:continuation]") {
		return true
	}

	// 2. Check XML / harness context tags
	lower := strings.ToLower(content)
	if strings.Contains(lower, "<codex_internal_context") && strings.Contains(lower, `source="goal"`) {
		return true
	}
	if strings.Contains(lower, "<goal_continuation") ||
		strings.Contains(lower, "<task_goal") ||
		strings.Contains(lower, "<objective_prompt") {
		return true
	}
	if strings.Contains(lower, "<goal>") || strings.Contains(lower, "<goal ") {
		if strings.Contains(lower, "</goal>") {
			return true
		}
	}

	// 3. Check JSON structure
	if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
		var m map[string]any
		if err := json.Unmarshal([]byte(trimmed), &m); err == nil {
			if t, ok := m["type"].(string); ok && (t == "goal_continuation" || t == "goal") {
				return true
			}
			if s, ok := m["source"].(string); ok && s == "goal" {
				return true
			}
			if _, ok := m["goal"]; ok {
				return true
			}
		}
	}

	// 4. Check line headers (leading or line-start to avoid mid-sentence false positives)
	lines := strings.Split(content, "\n")
	hasGoalHeader := false
	hasVolatileHeader := false

	for _, l := range lines {
		tl := strings.TrimSpace(l)
		ll := strings.ToLower(tl)
		// strip markdown bold
		if strings.HasPrefix(ll, "**") {
			ll = strings.TrimPrefix(ll, "**")
			if idx := strings.Index(ll, "**"); idx != -1 {
				ll = strings.TrimSpace(ll[:idx]) + ":" + strings.TrimSpace(ll[idx+2:])
			}
		}

		for _, gh := range goalLineHeaders {
			if strings.HasPrefix(ll, gh) {
				hasGoalHeader = true
				break
			}
		}

		for _, vh := range volatileLinePrefixes {
			if strings.HasPrefix(ll, vh) {
				hasVolatileHeader = true
				break
			}
		}
	}

	if hasGoalHeader {
		return true
	}

	// Goal mentioned + volatile header present in structured form
	if (strings.Contains(lower, "goal:") || strings.Contains(lower, "objective:")) && hasVolatileHeader {
		return true
	}

	return false
}

// StripVolatileContinuationFields separates the static goal statement from volatile
// turn-specific progress/budget fields. When volatile fields are present, it returns
// the canonicalized stablePrefix and the extracted volatileDelta. If no volatile fields
// exist, stablePrefix is content and volatileDelta is empty.
func StripVolatileContinuationFields(content string) (stablePrefix string, volatileDelta string) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return "", ""
	}

	// 1. JSON handling
	if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
		var m map[string]any
		if err := json.Unmarshal([]byte(trimmed), &m); err == nil && len(m) > 0 {
			stableMap := make(map[string]any)
			volatileMap := make(map[string]any)

			for k, v := range m {
				lk := strings.ToLower(strings.TrimSpace(k))
				if volatileJSONKeys[lk] {
					volatileMap[k] = v
				} else {
					stableMap[k] = v
				}
			}

			if len(volatileMap) > 0 {
				var sBytes, vBytes []byte
				if len(stableMap) > 0 {
					sBytes, _ = json.Marshal(stableMap)
				}
				vBytes, _ = json.Marshal(volatileMap)
				return string(sBytes), string(vBytes)
			}
			return content, ""
		}
	}

	// 2. Tag wrapper handling (e.g. <codex_internal_context source="goal"> ... </codex_internal_context>)
	openTag, closeTag, inner, hasWrapper := extractTagWrapper(content)
	if hasWrapper {
		stableInner, vDelta := stripVolatileFromText(inner)
		if vDelta != "" {
			var stableWrapped string
			if strings.TrimSpace(stableInner) != "" {
				stableWrapped = openTag + "\n" + strings.TrimSpace(stableInner) + "\n" + closeTag
			}
			return canonicalizeText(stableWrapped), canonicalizeText(vDelta)
		}
		return canonicalizeText(content), ""
	}

	// 3. General text handling
	stable, volatile := stripVolatileFromText(content)
	if volatile != "" {
		return canonicalizeText(stable), canonicalizeText(volatile)
	}
	return canonicalizeText(content), ""
}

func extractTagWrapper(content string) (openTag, closeTag, inner string, found bool) {
	lower := strings.ToLower(content)

	type wrapperDef struct {
		openPrefix string
		closeExact string
	}

	wrappers := []wrapperDef{
		{openPrefix: `<codex_internal_context source="goal">`, closeExact: `</codex_internal_context>`},
		{openPrefix: `<codex_internal_context source=\"goal\">`, closeExact: `</codex_internal_context>`},
		{openPrefix: `<goal_continuation>`, closeExact: `</goal_continuation>`},
		{openPrefix: `<objective_prompt>`, closeExact: `</objective_prompt>`},
		{openPrefix: `<task_goal>`, closeExact: `</task_goal>`},
		{openPrefix: `<goal>`, closeExact: `</goal>`},
	}

	for _, w := range wrappers {
		startIdx := strings.Index(lower, w.openPrefix)
		if startIdx != -1 {
			endIdx := strings.Index(lower[startIdx:], w.closeExact)
			if endIdx != -1 {
				openEnd := startIdx + len(w.openPrefix)
				closeStart := startIdx + endIdx
				open := content[startIdx:openEnd]
				closeT := content[closeStart : closeStart+len(w.closeExact)]
				in := content[openEnd:closeStart]
				return open, closeT, in, true
			}
		}
	}
	return "", "", "", false
}

func stripVolatileFromText(text string) (string, string) {
	// 1. Extract volatile XML-like tags (e.g. <turn>3</turn>, <budget>...</budget>)
	workingText := text
	var volatilePieces []string

	for _, re := range volatileTagRegexps {
		matches := re.FindAllString(workingText, -1)
		for _, m := range matches {
			volatilePieces = append(volatilePieces, strings.TrimSpace(m))
		}
		workingText = re.ReplaceAllString(workingText, "")
	}

	// 2. Parse remaining text line-by-line
	lines := strings.Split(workingText, "\n")
	var stableLines []string
	inVolatileSection := false
	inVolatileField := false

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			if !inVolatileSection && !inVolatileField {
				stableLines = append(stableLines, "")
			}
			continue
		}

		// Check for section header
		if strings.HasPrefix(trimmed, "#") {
			lower := strings.ToLower(trimmed)
			secName := strings.TrimSpace(strings.TrimLeft(lower, "#"))
			isVolSec := false
			for _, vs := range volatileSections {
				if strings.HasPrefix(secName, vs) {
					isVolSec = true
					break
				}
			}
			if isVolSec {
				inVolatileSection = true
				inVolatileField = false
				volatilePieces = append(volatilePieces, trimmed)
				continue
			} else {
				inVolatileSection = false
			}
		}

		if inVolatileSection {
			volatilePieces = append(volatilePieces, trimmed)
			continue
		}

		// Check for inline pipe-delimited fields (e.g. Goal: Ship #10671 | Budget left: 50 | Turn: 1)
		if strings.Contains(trimmed, "|") {
			parts := strings.Split(trimmed, "|")
			hasVol := false
			for _, p := range parts {
				if isVolatileLineOrSegment(p) {
					hasVol = true
					break
				}
			}
			if hasVol {
				var stableParts []string
				var volParts []string
				for _, p := range parts {
					tp := strings.TrimSpace(p)
					if tp == "" {
						continue
					}
					if isVolatileLineOrSegment(tp) {
						volParts = append(volParts, tp)
					} else {
						stableParts = append(stableParts, tp)
					}
				}
				if len(stableParts) > 0 {
					stableLines = append(stableLines, strings.Join(stableParts, " | "))
				}
				if len(volParts) > 0 {
					volatilePieces = append(volatilePieces, strings.Join(volParts, " | "))
				}
				inVolatileField = false
				continue
			}
		}

		// Check for line header
		if isVolatileLineOrSegment(trimmed) {
			inVolatileField = true
			volatilePieces = append(volatilePieces, trimmed)
			continue
		}

		// Indented lines or bullets under a volatile field belong to it
		if inVolatileField && (strings.HasPrefix(line, "  ") || strings.HasPrefix(line, "\t") || strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ")) {
			volatilePieces = append(volatilePieces, trimmed)
			continue
		}

		inVolatileField = false
		stableLines = append(stableLines, line)
	}

	stableStr := strings.TrimSpace(strings.Join(stableLines, "\n"))
	volatileStr := strings.TrimSpace(strings.Join(volatilePieces, "\n"))

	return stableStr, volatileStr
}

func isVolatileLineOrSegment(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	if strings.HasPrefix(lower, "**") {
		lower = strings.TrimPrefix(lower, "**")
		if idx := strings.Index(lower, "**"); idx != -1 {
			lower = strings.TrimSpace(lower[:idx]) + ":" + strings.TrimSpace(lower[idx+2:])
		}
	}
	for _, vh := range volatileLinePrefixes {
		if strings.HasPrefix(lower, vh) {
			return true
		}
	}
	return false
}

func canonicalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	var cleaned []string
	blankCount := 0

	for _, l := range lines {
		tl := strings.TrimRight(l, " \t")
		if strings.TrimSpace(tl) == "" {
			blankCount++
			if blankCount <= 1 && len(cleaned) > 0 {
				cleaned = append(cleaned, "")
			}
		} else {
			blankCount = 0
			cleaned = append(cleaned, tl)
		}
	}

	return strings.TrimSpace(strings.Join(cleaned, "\n"))
}

func isSameGoal(a, b string) bool {
	norm := func(s string) string {
		s = goalTagStripRe.ReplaceAllString(s, "")
		s = strings.ToLower(s)
		s = strings.TrimPrefix(s, strings.ToLower(compactGoalMarker))
		for _, h := range goalLineHeaders {
			s = strings.TrimPrefix(s, h)
		}
		var b strings.Builder
		for _, r := range s {
			if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
				b.WriteRune(r)
			}
		}
		return b.String()
	}
	na, nb := norm(a), norm(b)
	return na != "" && na == nb
}

// StabilizeGoalContinuationMessages inspects messages in conversation order, canonicalizing
// goal continuation messages so their leading goal definition remains byte-identical across turns,
// and moving any volatile progress deltas to the tail of the turn.
// Returns a copy of the slice if any message was modified, and a boolean indicating if changes occurred.
func StabilizeGoalContinuationMessages(messages []Message) ([]Message, bool) {
	if len(messages) == 0 {
		return messages, false
	}

	var firstGoalDef string
	var tailDeltas []string
	modifiedIndices := make(map[int]string)

	n := len(messages)
	for i, m := range messages {
		if !IsGoalContinuationMessage(m.Content) {
			continue
		}

		stable, volatile := StripVolatileContinuationFields(m.Content)

		// Canonicalize goal definition across turns
		if stable != "" {
			if firstGoalDef == "" {
				firstGoalDef = stable
			} else if isSameGoal(firstGoalDef, stable) {
				stable = firstGoalDef
			}
		}

		if i < n-1 {
			// Intermediate / prefix turn: strip volatile fields completely
			if volatile != "" {
				tailDeltas = append(tailDeltas, volatile)
			}
			if stable != m.Content {
				modifiedIndices[i] = stable
			}
		} else {
			// Last message (the current turn): position volatile delta at tail
			if volatile != "" {
				tailDeltas = append(tailDeltas, volatile)
			}
			modifiedIndices[i] = stable
		}
	}

	// If tail deltas were collected, append them to the last message of the turn
	lastIdx := n - 1
	currentLastContent := messages[lastIdx].Content
	if repl, ok := modifiedIndices[lastIdx]; ok {
		currentLastContent = repl
	}

	if len(tailDeltas) > 0 {
		deltaStr := strings.Join(tailDeltas, "\n\n")
		var targetContent string
		if strings.TrimSpace(currentLastContent) == "" {
			targetContent = deltaStr
		} else {
			targetContent = strings.TrimSpace(currentLastContent) + "\n\n" + deltaStr
		}
		if targetContent != messages[lastIdx].Content {
			modifiedIndices[lastIdx] = targetContent
		} else {
			delete(modifiedIndices, lastIdx)
		}
	} else if repl, ok := modifiedIndices[lastIdx]; ok && repl == messages[lastIdx].Content {
		delete(modifiedIndices, lastIdx)
	}

	if len(modifiedIndices) == 0 {
		return messages, false
	}

	out := make([]Message, n)
	copy(out, messages)
	for idx, newContent := range modifiedIndices {
		out[idx].Content = newContent
	}

	return out, true
}

// --- world_state deduplication and rate-limiting --------------------------------

type worldStateExtraction struct {
	raw      string
	body     string
	isTag    bool
	isJSON   bool
	before   string
	after    string
	jsonData map[string]any
}

func extractWorldState(content string) (worldStateExtraction, bool) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return worldStateExtraction{}, false
	}

	// 1. Tag match <world_state ...> ... </world_state>
	if m := worldStateTagRe.FindStringSubmatch(content); len(m) == 6 {
		raw := m[2] + m[3] + m[4]
		body := strings.TrimSpace(m[3])
		return worldStateExtraction{
			raw:    raw,
			body:   body,
			isTag:  true,
			before: m[1],
			after:  m[5],
		}, true
	}

	// Also check <system_state> or <environment_state>
	if m := worldStateAltTagRe.FindStringSubmatch(content); len(m) == 6 {
		raw := m[2] + m[3] + m[4]
		body := strings.TrimSpace(m[3])
		return worldStateExtraction{
			raw:    raw,
			body:   body,
			isTag:  true,
			before: m[1],
			after:  m[5],
		}, true
	}

	// 2. Bracket marker [world_state] ... [/world_state]
	if m := worldStateBracketRe.FindStringSubmatch(content); len(m) == 6 {
		raw := m[2] + m[3] + m[4]
		body := strings.TrimSpace(m[3])
		return worldStateExtraction{
			raw:    raw,
			body:   body,
			before: m[1],
			after:  m[5],
		}, true
	}

	// 3. JSON check
	if strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}") {
		var jm map[string]any
		if err := json.Unmarshal([]byte(trimmed), &jm); err == nil {
			t, _ := jm["type"].(string)
			_, hasWS := jm["world_state"]
			if t == "world_state" || hasWS {
				return worldStateExtraction{
					raw:      trimmed,
					body:     trimmed,
					isJSON:   true,
					jsonData: jm,
				}, true
			}
		}
	}

	// 4. Line header: World State: or # World State
	if m := worldStateHeaderRe.FindStringSubmatch(content); len(m) == 4 {
		raw := strings.TrimSpace(m[2] + m[3])
		return worldStateExtraction{
			raw:    raw,
			body:   strings.TrimSpace(m[3]),
			before: m[1],
			after:  "",
		}, true
	}

	return worldStateExtraction{}, false
}

func isWorldStateIdentical(a, b worldStateExtraction) bool {
	if a.isJSON && b.isJSON && a.jsonData != nil && b.jsonData != nil {
		ja, _ := json.Marshal(a.jsonData)
		jb, _ := json.Marshal(b.jsonData)
		return string(ja) == string(jb)
	}
	return canonicalizeText(a.body) == canonicalizeText(b.body)
}

func computeWorldStateDiff(prev, curr worldStateExtraction) string {
	if curr.isJSON && prev.isJSON && curr.jsonData != nil && prev.jsonData != nil {
		diff := make(map[string]any)
		for k, v := range curr.jsonData {
			if k == "type" {
				continue
			}
			prevV, exists := prev.jsonData[k]
			if !exists {
				diff[k] = v
			} else {
				pvB, _ := json.Marshal(prevV)
				cvB, _ := json.Marshal(v)
				if string(pvB) != string(cvB) {
					diff[k] = v
				}
			}
		}
		if len(diff) == 0 {
			return `{"type":"world_state_diff","status":"unchanged"}`
		}
		diffBytes, _ := json.Marshal(diff)
		return fmt.Sprintf(`{"type":"world_state_diff","diff":%s}`, string(diffBytes))
	}

	// Line-based diff
	prevLines := strings.Split(canonicalizeText(prev.body), "\n")
	currLines := strings.Split(canonicalizeText(curr.body), "\n")

	prevSet := make(map[string]bool, len(prevLines))
	for _, l := range prevLines {
		if tl := strings.TrimSpace(l); tl != "" {
			prevSet[tl] = true
		}
	}

	var changedLines []string
	for _, l := range currLines {
		tl := strings.TrimSpace(l)
		if tl != "" && !prevSet[tl] {
			changedLines = append(changedLines, tl)
		}
	}

	if len(changedLines) == 0 {
		if curr.isTag {
			return `<world_state_diff status="unchanged"/>`
		}
		return `[world_state: unchanged]`
	}

	sort.Strings(changedLines)
	if curr.isTag {
		return "<world_state_diff>\n" + strings.Join(changedLines, "\n") + "\n</world_state_diff>"
	}
	return "[world_state diff:\n" + strings.Join(changedLines, "\n") + "]"
}

// FilterRedundantWorldState deduplicates and rate-limits world_state injections across turns.
// The earliest full world_state snapshot is preserved verbatim in the prefix, while subsequent
// redundant injections in intermediate turns are filtered or converted to compact diffs.
// Returns the updated message slice and the count of filtered/converted world states.
func FilterRedundantWorldState(messages []Message) ([]Message, int) {
	if len(messages) == 0 {
		return messages, 0
	}

	var (
		out           []Message
		filteredCount int
		hasFirstWS    bool
		firstWS       worldStateExtraction
		lastWS        worldStateExtraction
	)

	for i, m := range messages {
		wsExt, ok := extractWorldState(m.Content)
		if !ok && m.Name != "world_state" {
			if out != nil {
				out = append(out, m)
			}
			continue
		}

		if !ok && m.Name == "world_state" {
			wsExt = worldStateExtraction{
				raw:  m.Content,
				body: m.Content,
			}
		}

		if !hasFirstWS {
			// First full world_state in the conversation: preserve verbatim
			hasFirstWS = true
			firstWS = wsExt
			lastWS = wsExt
			if out != nil {
				out = append(out, m)
			}
			continue
		}

		// Subsequent world_state injection
		if out == nil {
			out = make([]Message, 0, len(messages))
			out = append(out, messages[:i]...)
		}

		if isWorldStateIdentical(wsExt, lastWS) || isWorldStateIdentical(wsExt, firstWS) {
			var replacement string
			if wsExt.isTag {
				replacement = `<world_state_diff status="unchanged"/>`
			} else if wsExt.isJSON {
				replacement = `{"type":"world_state_diff","status":"unchanged"}`
			} else {
				replacement = `[world_state: unchanged]`
			}

			newContent := strings.TrimSpace(wsExt.before + replacement + wsExt.after)
			msgCopy := m
			msgCopy.Content = newContent
			out = append(out, msgCopy)
			filteredCount++
			continue
		}

		// Convert to compact diff
		diff := computeWorldStateDiff(lastWS, wsExt)
		if diff != "" && len(diff) < len(wsExt.raw) {
			newContent := strings.TrimSpace(wsExt.before + diff + wsExt.after)
			msgCopy := m
			msgCopy.Content = newContent
			out = append(out, msgCopy)
			filteredCount++
			lastWS = wsExt
		} else {
			out = append(out, m)
			lastWS = wsExt
		}
	}

	if filteredCount == 0 {
		return messages, 0
	}
	return out, filteredCount
}

// StabilizePromptPrefix restores byte-stable prompt prefixes across goal-continuation sessions
// by filtering redundant world_state injections and canonicalizing goal continuation messages.
func StabilizePromptPrefix(messages []Message) ([]Message, bool) {
	if len(messages) == 0 {
		return messages, false
	}

	filtered, filteredCount := FilterRedundantWorldState(messages)
	stabilized, goalChanged := StabilizeGoalContinuationMessages(filtered)

	if filteredCount == 0 && !goalChanged {
		return messages, false
	}
	return stabilized, true
}
