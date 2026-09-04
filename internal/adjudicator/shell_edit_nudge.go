package adjudicator

import (
	"regexp"
	"strings"
)

// ShellEditNudgeAdvisory is the advisory recommendation to adopt structured tools.
const ShellEditNudgeAdvisory = "Prefer structured tool 'Edit' or 'Write' over shell mutation for atomic, diff-witnessed changes."

// ShellEditNudgeResult records the outcome of checking a shell command for in-place edits.
type ShellEditNudgeResult struct {
	IsShellEdit  bool   `json:"is_shell_edit"`
	Suggestion   string `json:"suggestion,omitempty"`
	Blocked      bool   `json:"blocked"`
	DetectedTool string `json:"detected_tool,omitempty"`
	TargetPath   string `json:"target_path,omitempty"`
}

var (
	pythonOpenWritePattern = regexp.MustCompile(
		`(?s)\bopen\s*\(\s*(?:file\s*=\s*)?[rRfF]?['"]([^'"]+)['"]\s*,\s*(?:mode\s*=\s*)?[rRfF]?['"]([waxWAX][^'"]*|[rR]\+[^'"]*)['"]`,
	)
	pythonOpenWriteKeywordPattern = regexp.MustCompile(
		`(?s)\bopen\s*\(\s*mode\s*=\s*[rRfF]?['"]([waxWAX][^'"]*|[rR]\+[^'"]*)['"]\s*,\s*(?:file\s*=\s*)?[rRfF]?['"]([^'"]+)['"]`,
	)
	pythonOpenMethodWritePattern = regexp.MustCompile(
		`(?s)\bopen\s*\(\s*(?:file\s*=\s*)?[rRfF]?['"]([^'"]+)['"]\s*\)\s*\.\s*write`,
	)
	pythonPathWritePattern = regexp.MustCompile(
		`(?s)\b(?:pathlib\.)?Path\s*\(\s*[rRfF]?['"]([^'"]+)['"]\s*\)\s*\.\s*write_(?:text|bytes)`,
	)
)

// CheckShellEditNudge inspects a shell command for in-place file mutations
// (sed -i, sed --in-place, awk/gawk -i inplace, perl -pi/-i, python -c with file write)
// and returns a structured nudge result.
func CheckShellEditNudge(cmd string, strictMode bool) ShellEditNudgeResult {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return ShellEditNudgeResult{}
	}

	// First, try segment-by-segment structured analysis.
	segments := shellSegments(cmd)
	if len(segments) == 0 {
		segments = []string{cmd}
	}

	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}

		if res, ok := checkSegmentShellEdit(seg, strictMode); ok {
			return res
		}
	}

	// Fallback check for unclosed quotes or unconventional formatting that bypassed segmentation.
	if res, ok := fallbackShellEditCheck(cmd, strictMode); ok {
		return res
	}

	return ShellEditNudgeResult{}
}

// SuggestStructuredEditTool inspects cmd and returns whether it is a shell edit,
// the advisory suggestion, and whether execution should be blocked.
func SuggestStructuredEditTool(cmd string, strictMode bool) (isShellEdit bool, suggestion string, blocked bool) {
	res := CheckShellEditNudge(cmd, strictMode)
	return res.IsShellEdit, res.Suggestion, res.Blocked
}

func checkSegmentShellEdit(seg string, strictMode bool) (ShellEditNudgeResult, bool) {
	// 1. Check sed and awk / gawk via ExtractSedAwkMutations
	if muts, err := ExtractSedAwkMutations(seg); err == nil && len(muts) > 0 {
		for _, m := range muts {
			if m.InPlace {
				tool := m.Tool
				if tool == "" {
					tool = "sed"
				}
				return ShellEditNudgeResult{
					IsShellEdit:  true,
					Suggestion:   ShellEditNudgeAdvisory,
					Blocked:      strictMode,
					DetectedTool: tool,
					TargetPath:   m.TargetPath,
				}, true
			}
		}
	}

	words := streamWords(seg)
	if len(words) == 0 {
		return ShellEditNudgeResult{}, false
	}

	start := commandWordStart(words)
	if start >= len(words) {
		return ShellEditNudgeResult{}, false
	}

	start = skipElevationPrefix(words, start)
	if start >= len(words) {
		return ShellEditNudgeResult{}, false
	}

	head := normalizeExecutable(words[start].text)

	// Subshell unwrapping (e.g. bash -c "...", sh -c '...')
	if isSubshell(head) && start+2 < len(words) && words[start+1].text == "-c" {
		inner := words[start+2].text
		innerRes := CheckShellEditNudge(inner, strictMode)
		if innerRes.IsShellEdit {
			return innerRes, true
		}
	}

	// 2. Check perl in-place edits
	if head == "perl" || head == "perl5" {
		if isMutation, target := extractPerlMutation(words[start+1:]); isMutation {
			return ShellEditNudgeResult{
				IsShellEdit:  true,
				Suggestion:   ShellEditNudgeAdvisory,
				Blocked:      strictMode,
				DetectedTool: "perl",
				TargetPath:   target,
			}, true
		}
	}

	// 3. Check python in-place / inline write edits
	if head == "python" || hasNumericSuffix(head, "python") || head == "py" {
		if isMutation, target := extractPythonMutation(words[start+1:]); isMutation {
			return ShellEditNudgeResult{
				IsShellEdit:  true,
				Suggestion:   ShellEditNudgeAdvisory,
				Blocked:      strictMode,
				DetectedTool: "python",
				TargetPath:   target,
			}, true
		}
	}

	return ShellEditNudgeResult{}, false
}

func extractPerlMutation(args []shellWord) (bool, string) {
	inPlace := false
	var scripts []string
	var positional []string
	endOfOptions := false

	for i := 0; i < len(args); i++ {
		w := args[i]
		tok := w.text

		if endOfOptions || w.quoted {
			positional = append(positional, tok)
			continue
		}

		if tok == "--" {
			endOfOptions = true
			continue
		}

		if strings.HasPrefix(tok, "--in-place") {
			inPlace = true
			continue
		}

		if strings.HasPrefix(tok, "-") && len(tok) > 1 {
			// Check for -e or -E
			if tok == "-e" || tok == "-E" {
				if i+1 < len(args) {
					scripts = append(scripts, args[i+1].text)
					i++
				}
				continue
			}
			if strings.HasPrefix(tok, "-e") || strings.HasPrefix(tok, "-E") {
				scripts = append(scripts, tok[2:])
				continue
			}

			// Clustered flags ending in 'e' or 'E' (e.g., -pie, -i -pe, -ne)
			if strings.HasSuffix(tok, "e") || strings.HasSuffix(tok, "E") {
				flagPart := tok[:len(tok)-1]
				if strings.ContainsRune(flagPart, 'i') {
					inPlace = true
				}
				if i+1 < len(args) {
					scripts = append(scripts, args[i+1].text)
					i++
				}
				continue
			}

			// Do not confuse include directory (-I...) with in-place edit (-i)
			if strings.HasPrefix(tok, "-I") {
				continue
			}
			if tok == "-f" || tok == "-M" || tok == "-F" {
				if i+1 < len(args) {
					i++
				}
				continue
			}
			if strings.HasPrefix(tok, "-f") || strings.HasPrefix(tok, "-M") || strings.HasPrefix(tok, "-F") {
				continue
			}

			if strings.ContainsRune(tok, 'i') {
				inPlace = true
			}
			continue
		}

		positional = append(positional, tok)
	}

	if !inPlace {
		return false, ""
	}

	var targetFiles []string
	if len(scripts) > 0 {
		targetFiles = positional
	} else if len(positional) > 1 {
		targetFiles = positional[1:]
	} else if len(positional) == 1 {
		targetFiles = positional
	}

	targetPath := ""
	if len(targetFiles) > 0 {
		targetPath = cleanShellOperand(targetFiles[0])
	}
	return true, targetPath
}

func extractPythonMutation(args []shellWord) (bool, string) {
	var script string
	hasScript := false

	for i := 0; i < len(args); i++ {
		tok := args[i].text
		if tok == "-c" {
			if i+1 < len(args) {
				script = args[i+1].text
				hasScript = true
			}
			break
		}
		if strings.HasPrefix(tok, "-c") && len(tok) > 2 {
			script = tok[2:]
			hasScript = true
			break
		}
	}

	if !hasScript {
		return false, ""
	}

	return parsePythonScriptMutation(script)
}

func parsePythonScriptMutation(script string) (bool, string) {
	if m := pythonOpenWritePattern.FindStringSubmatch(script); len(m) >= 3 {
		return true, cleanShellOperand(m[1])
	}
	if m := pythonOpenWriteKeywordPattern.FindStringSubmatch(script); len(m) >= 3 {
		return true, cleanShellOperand(m[2])
	}
	if m := pythonOpenMethodWritePattern.FindStringSubmatch(script); len(m) >= 2 {
		return true, cleanShellOperand(m[1])
	}
	if m := pythonPathWritePattern.FindStringSubmatch(script); len(m) >= 2 {
		return true, cleanShellOperand(m[1])
	}

	return false, ""
}

func fallbackShellEditCheck(cmd string, strictMode bool) (ShellEditNudgeResult, bool) {
	lc := strings.ToLower(cmd)

	if strings.Contains(lc, "sed -i") || strings.Contains(lc, "sed --in-place") {
		return ShellEditNudgeResult{
			IsShellEdit:  true,
			Suggestion:   ShellEditNudgeAdvisory,
			Blocked:      strictMode,
			DetectedTool: "sed",
		}, true
	}

	if (strings.Contains(lc, "awk") || strings.Contains(lc, "gawk")) && strings.Contains(lc, "-i inplace") {
		tool := "awk"
		if strings.Contains(lc, "gawk") {
			tool = "gawk"
		}
		return ShellEditNudgeResult{
			IsShellEdit:  true,
			Suggestion:   ShellEditNudgeAdvisory,
			Blocked:      strictMode,
			DetectedTool: tool,
		}, true
	}

	if strings.Contains(lc, "perl") && (strings.Contains(lc, "-pi") || strings.Contains(lc, "-i")) {
		return ShellEditNudgeResult{
			IsShellEdit:  true,
			Suggestion:   ShellEditNudgeAdvisory,
			Blocked:      strictMode,
			DetectedTool: "perl",
		}, true
	}

	if strings.Contains(lc, "python") && strings.Contains(lc, "-c") && strings.Contains(lc, "open(") &&
		(strings.Contains(lc, "'w'") || strings.Contains(lc, "\"w\"") || strings.Contains(lc, ".write(")) {
		return ShellEditNudgeResult{
			IsShellEdit:  true,
			Suggestion:   ShellEditNudgeAdvisory,
			Blocked:      strictMode,
			DetectedTool: "python",
		}, true
	}

	return ShellEditNudgeResult{}, false
}
