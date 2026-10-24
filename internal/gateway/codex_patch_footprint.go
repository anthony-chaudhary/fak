package gateway

import (
	"encoding/json"
	"io"
	"strings"
)

// adjudicateCodexPatchTool names only the structured patch tools whose payload
// grammar is understood below. Substring matches would let unrelated "apply" tools
// inherit file-write semantics they did not declare.
func adjudicateCodexPatchTool(tool string) bool {
	switch strings.ToLower(adjudicateToolBase(tool)) {
	case "apply_patch", "patch":
		return true
	default:
		return false
	}
}

// adjudicateCodexPatchFootprint decodes the Codex input spelling and the legacy
// patch spelling. Extra fields and disagreeing aliases are ambiguous effects and
// therefore remain unknown rather than being masked by one apparently safe path.
func adjudicateCodexPatchFootprint(rawArgs string) ([]string, bool) {
	decoder := json.NewDecoder(strings.NewReader(rawArgs))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, false
	}
	values := make(map[string]string, 2)
	for decoder.More() {
		keyToken, err := decoder.Token()
		key, keyOK := keyToken.(string)
		if err != nil || !keyOK || (key != "input" && key != "patch") {
			return nil, false
		}
		if _, duplicate := values[key]; duplicate {
			return nil, false
		}
		var value string
		if err := decoder.Decode(&value); err != nil || strings.TrimSpace(value) == "" {
			return nil, false
		}
		values[key] = value
	}
	if token, err = decoder.Token(); err != nil || token != json.Delim('}') {
		return nil, false
	}
	if _, err = decoder.Token(); err != io.EOF {
		return nil, false
	}
	input, hasInput := values["input"]
	legacy, hasLegacy := values["patch"]
	if !hasInput && !hasLegacy {
		return nil, false
	}
	if hasInput && hasLegacy && input != legacy {
		return nil, false
	}
	if !hasInput {
		input = legacy
	}
	return parseCodexPatchFootprint(input)
}

// parseCodexPatchFootprint recognizes the complete effect-bearing header grammar.
// It deliberately validates the surrounding patch shape as well: a valid-looking
// first path cannot conceal a malformed or second, unaccounted mutation.
func parseCodexPatchFootprint(patch string) ([]string, bool) {
	patch = strings.ReplaceAll(patch, "\r\n", "\n")
	if strings.ContainsRune(patch, '\r') {
		return nil, false
	}
	lines := strings.Split(strings.TrimRight(patch, "\n"), "\n")
	if len(lines) < 3 || lines[0] != "*** Begin Patch" || lines[len(lines)-1] != "*** End Patch" {
		return nil, false
	}

	const (
		patchNone = iota
		patchAdd
		patchUpdate
		patchDelete
	)
	operation := patchNone
	moved := false
	updateHasHunk := false
	updateBodyStarted := false
	updateEOF := false
	var targets []string
	addTarget := func(line, prefix string) bool {
		separator := prefix + " "
		if !strings.HasPrefix(line, separator) || strings.Count(line, ": ") != 1 {
			return false
		}
		path := strings.TrimPrefix(line, separator)
		if path == "" || strings.TrimSpace(path) != path || strings.ContainsRune(path, 0) {
			return false
		}
		targets = append(targets, path)
		return true
	}
	finishOperation := func() bool {
		return operation != patchUpdate || moved || updateHasHunk
	}

	for _, line := range lines[1 : len(lines)-1] {
		switch {
		case strings.HasPrefix(line, "*** Add File:"):
			if !finishOperation() || !addTarget(line, "*** Add File:") {
				return nil, false
			}
			operation, moved = patchAdd, false
		case strings.HasPrefix(line, "*** Update File:"):
			if !finishOperation() || !addTarget(line, "*** Update File:") {
				return nil, false
			}
			operation, moved = patchUpdate, false
			updateHasHunk, updateBodyStarted, updateEOF = false, false, false
		case strings.HasPrefix(line, "*** Delete File:"):
			if !finishOperation() || !addTarget(line, "*** Delete File:") {
				return nil, false
			}
			operation, moved = patchDelete, false
		case strings.HasPrefix(line, "*** Move to:"):
			if operation != patchUpdate || moved || updateBodyStarted || !addTarget(line, "*** Move to:") {
				return nil, false
			}
			moved = true
		case line == "*** End of File":
			if operation != patchUpdate || !updateHasHunk || updateEOF {
				return nil, false
			}
			updateEOF = true
		case strings.HasPrefix(line, "*** "):
			return nil, false
		case operation == patchNone:
			return nil, false
		case operation == patchAdd:
			if !strings.HasPrefix(line, "+") {
				return nil, false
			}
		case operation == patchDelete:
			return nil, false
		case operation == patchUpdate:
			if updateEOF {
				return nil, false
			}
			if strings.HasPrefix(line, "@@") {
				updateHasHunk, updateBodyStarted = true, true
				continue
			}
			if line == "" || !strings.Contains(" +-", line[:1]) {
				return nil, false
			}
			// Codex's executor accepts a first hunk without an @@ locator when its
			// body starts directly with context/add/delete lines. That body is the
			// structural evidence an Update is non-empty; preserve it exactly.
			updateHasHunk, updateBodyStarted = true, true
		}
	}
	if len(targets) == 0 || !finishOperation() {
		return nil, false
	}
	return targets, true
}
