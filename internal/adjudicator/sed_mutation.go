package adjudicator

import (
	"fmt"
	"strings"
)

// StreamMutationTarget records an inspected stream-editor (sed, awk, gawk) invocation target.
type StreamMutationTarget struct {
	Tool         string `json:"tool"`
	InPlace      bool   `json:"in_place"`
	TargetPath   string `json:"target_path"`
	Script       string `json:"script"`
	TreeKnown    bool   `json:"tree_known"`
	DeclaredTree string `json:"declared_tree,omitempty"`
}

// MatchDeclaredTree matches TargetPath against declared lane tree globs
// when TreeKnown is true.
func (m StreamMutationTarget) MatchDeclaredTree(globs []string) string {
	if !m.TreeKnown || m.TargetPath == "" {
		return ""
	}
	if g := matchGlob(m.TargetPath, globs); g != "" {
		return g
	}
	for _, g := range globs {
		if matchLaneGlob(m.TargetPath, g) {
			return g
		}
	}
	return ""
}

// ClassifyStreamMutationTarget returns the matched declared lane tree for a target.
func ClassifyStreamMutationTarget(target StreamMutationTarget, globs []string) string {
	return target.MatchDeclaredTree(globs)
}

// ExtractSedAwkMutations parses a shell command line and extracts stream editor
// invocation targets, identifying in-place file modifications and dynamic path expressions.
func ExtractSedAwkMutations(cmd string) ([]StreamMutationTarget, error) {
	return ExtractSedAwkMutationsWithTrees(cmd, nil)
}

// ExtractSedAwkMutationsWithTrees extracts stream editor targets and classifies
// each target path against the provided lane tree globs.
func ExtractSedAwkMutationsWithTrees(cmd string, globs []string) ([]StreamMutationTarget, error) {
	if q := checkUnterminatedQuote(cmd); q != 0 {
		return nil, fmt.Errorf("unterminated quote %c in command", q)
	}

	segments := shellSegments(cmd)
	var allTargets []StreamMutationTarget
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		targets, err := extractFromSegment(seg)
		if err != nil {
			return nil, err
		}
		allTargets = append(allTargets, targets...)
	}

	if len(globs) > 0 {
		for i := range allTargets {
			allTargets[i].DeclaredTree = allTargets[i].MatchDeclaredTree(globs)
		}
	}

	return allTargets, nil
}

func extractFromSegment(seg string) ([]StreamMutationTarget, error) {
	words := streamWords(seg)
	if len(words) == 0 {
		return nil, nil
	}

	start := commandWordStart(words)
	if start >= len(words) {
		return nil, nil
	}

	start = skipElevationPrefix(words, start)
	if start >= len(words) {
		return nil, nil
	}

	head := normalizeExecutable(words[start].text)
	if isSubshell(head) && start+2 < len(words) && words[start+1].text == "-c" {
		return ExtractSedAwkMutations(words[start+2].text)
	}

	tool := normalizeToolName(words[start].text)
	if tool == "" {
		return nil, nil
	}

	args := filterRedirectionWords(words[start+1:])
	switch tool {
	case "sed":
		return parseSedArgs(tool, args)
	case "awk", "gawk":
		return parseAwkArgs(tool, args)
	default:
		return nil, nil
	}
}

func parseSedArgs(tool string, words []shellWord) ([]StreamMutationTarget, error) {
	var inPlace bool
	var scripts []string
	var scriptFiles []string
	var positional []string

	i := 0
	endOfOptions := false

	for i < len(words) {
		w := words[i]
		tok := w.text

		if endOfOptions || w.quoted {
			positional = append(positional, tok)
			i++
			continue
		}

		if tok == "--" {
			endOfOptions = true
			i++
			continue
		}

		if strings.HasPrefix(tok, "--") {
			if tok == "--in-place" {
				inPlace = true
				if i+1 < len(words) && words[i+1].text == "" {
					i++
				}
			} else if strings.HasPrefix(tok, "--in-place=") {
				inPlace = true
			} else if tok == "--expression" {
				if i+1 < len(words) {
					scripts = append(scripts, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '--expression' requires an argument")
				}
			} else if strings.HasPrefix(tok, "--expression=") {
				scripts = append(scripts, strings.TrimPrefix(tok, "--expression="))
			} else if tok == "--file" {
				if i+1 < len(words) {
					scriptFiles = append(scriptFiles, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '--file' requires an argument")
				}
			} else if strings.HasPrefix(tok, "--file=") {
				scriptFiles = append(scriptFiles, strings.TrimPrefix(tok, "--file="))
			}
			i++
			continue
		}

		if strings.HasPrefix(tok, "-") && len(tok) > 1 {
			if tok == "-i" {
				inPlace = true
				if i+1 < len(words) {
					next := words[i+1].text
					if next == "" || (strings.HasPrefix(next, ".") && !strings.Contains(next, "/") && i+2 < len(words)) {
						i++
					}
				}
				i++
				continue
			}

			if strings.HasPrefix(tok, "-i") {
				inPlace = true
				i++
				continue
			}

			if tok == "-e" {
				if i+1 < len(words) {
					scripts = append(scripts, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '-e' requires an argument")
				}
				i++
				continue
			}
			if strings.HasPrefix(tok, "-e") {
				scripts = append(scripts, tok[2:])
				i++
				continue
			}

			if tok == "-f" {
				if i+1 < len(words) {
					scriptFiles = append(scriptFiles, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '-f' requires an argument")
				}
				i++
				continue
			}
			if strings.HasPrefix(tok, "-f") {
				scriptFiles = append(scriptFiles, tok[2:])
				i++
				continue
			}

			if strings.ContainsRune(tok, 'i') {
				inPlace = true
			}
			if strings.HasSuffix(tok, "e") {
				if i+1 < len(words) {
					scripts = append(scripts, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '-%c' requires an argument", tok[len(tok)-1])
				}
			} else if strings.HasSuffix(tok, "f") {
				if i+1 < len(words) {
					scriptFiles = append(scriptFiles, words[i+1].text)
					i++
				} else {
					return nil, fmt.Errorf("option '-%c' requires an argument", tok[len(tok)-1])
				}
			}
			i++
			continue
		}

		positional = append(positional, tok)
		i++
	}

	var targetPaths []string
	if len(scripts) > 0 || len(scriptFiles) > 0 {
		targetPaths = positional
	} else if len(positional) > 0 {
		scripts = append(scripts, positional[0])
		targetPaths = positional[1:]
	}

	script := formatScript(scripts, scriptFiles)

	if len(targetPaths) == 0 {
		if inPlace {
			return []StreamMutationTarget{
				{
					Tool:       tool,
					InPlace:    true,
					TargetPath: "",
					Script:     script,
					TreeKnown:  false,
				},
			}, nil
		}
		return nil, nil
	}

	var targets []StreamMutationTarget
	for _, p := range targetPaths {
		pClean := cleanShellOperand(p)
		targets = append(targets, StreamMutationTarget{
			Tool:       tool,
			InPlace:    inPlace,
			TargetPath: pClean,
			Script:     script,
			TreeKnown:  isStreamTreeKnown(p, pClean),
		})
	}
	return targets, nil
}

func parseAwkArgs(tool string, words []shellWord) ([]StreamMutationTarget, error) {
	var inPlace bool
	var scripts []string
	var scriptFiles []string
	var positional []string

	i := 0
	endOfOptions := false

	for i < len(words) {
		w := words[i]
		tok := w.text

		if endOfOptions || w.quoted {
			positional = append(positional, tok)
			i++
			continue
		}

		if tok == "--" {
			endOfOptions = true
			i++
			continue
		}

		if strings.HasPrefix(tok, "--") {
			if tok == "--include" {
				if i+1 < len(words) {
					val := words[i+1].text
					if isInplaceLib(val) {
						inPlace = true
					}
					i++
				}
			} else if strings.HasPrefix(tok, "--include=") {
				val := strings.TrimPrefix(tok, "--include=")
				if isInplaceLib(val) {
					inPlace = true
				}
			} else if tok == "--source" {
				if i+1 < len(words) {
					scripts = append(scripts, words[i+1].text)
					i++
				}
			} else if strings.HasPrefix(tok, "--source=") {
				scripts = append(scripts, strings.TrimPrefix(tok, "--source="))
			} else if tok == "--file" {
				if i+1 < len(words) {
					scriptFiles = append(scriptFiles, words[i+1].text)
					i++
				}
			} else if strings.HasPrefix(tok, "--file=") {
				scriptFiles = append(scriptFiles, strings.TrimPrefix(tok, "--file="))
			} else if tok == "--field-separator" || tok == "--assign" {
				if i+1 < len(words) {
					i++
				}
			}
			i++
			continue
		}

		if strings.HasPrefix(tok, "-") && len(tok) > 1 {
			if tok == "-i" {
				if i+1 < len(words) {
					val := words[i+1].text
					if isInplaceLib(val) {
						inPlace = true
					}
					i++
				}
				i++
				continue
			}
			if strings.HasPrefix(tok, "-i") {
				val := tok[2:]
				if isInplaceLib(val) {
					inPlace = true
				}
				i++
				continue
			}
			if tok == "-e" {
				if i+1 < len(words) {
					scripts = append(scripts, words[i+1].text)
					i++
				}
				i++
				continue
			}
			if strings.HasPrefix(tok, "-e") {
				scripts = append(scripts, tok[2:])
				i++
				continue
			}
			if tok == "-f" {
				if i+1 < len(words) {
					scriptFiles = append(scriptFiles, words[i+1].text)
					i++
				}
				i++
				continue
			}
			if strings.HasPrefix(tok, "-f") {
				scriptFiles = append(scriptFiles, tok[2:])
				i++
				continue
			}
			if tok == "-F" || tok == "-v" || tok == "-W" {
				if i+1 < len(words) {
					i++
				}
				i++
				continue
			}
			i++
			continue
		}

		positional = append(positional, tok)
		i++
	}

	var targetPaths []string
	if len(scripts) > 0 || len(scriptFiles) > 0 {
		for _, op := range positional {
			if !isAssignment(op) {
				targetPaths = append(targetPaths, op)
			}
		}
	} else if len(positional) > 0 {
		scripts = append(scripts, positional[0])
		for _, op := range positional[1:] {
			if !isAssignment(op) {
				targetPaths = append(targetPaths, op)
			}
		}
	}

	script := formatScript(scripts, scriptFiles)

	if len(targetPaths) == 0 {
		if inPlace {
			return []StreamMutationTarget{
				{
					Tool:       tool,
					InPlace:    true,
					TargetPath: "",
					Script:     script,
					TreeKnown:  false,
				},
			}, nil
		}
		return nil, nil
	}

	var targets []StreamMutationTarget
	for _, p := range targetPaths {
		pClean := cleanShellOperand(p)
		targets = append(targets, StreamMutationTarget{
			Tool:       tool,
			InPlace:    inPlace,
			TargetPath: pClean,
			Script:     script,
			TreeKnown:  isStreamTreeKnown(p, pClean),
		})
	}
	return targets, nil
}

func isInplaceLib(val string) bool {
	val = strings.ToLower(cleanShellOperand(val))
	return val == "inplace" || strings.HasPrefix(val, "inplace.") || strings.HasPrefix(val, "inplace=")
}

func formatScript(scripts, scriptFiles []string) string {
	if len(scripts) > 0 {
		return strings.Join(scripts, "; ")
	}
	if len(scriptFiles) > 0 {
		return "-f " + strings.Join(scriptFiles, " -f ")
	}
	return ""
}

func streamWords(cmd string) []shellWord {
	var out []shellWord
	var b strings.Builder
	var quote byte
	var subQuote byte
	quoted := false
	escaped := false
	parenDepth := 0
	braceDepth := 0

	flush := func() {
		if b.Len() == 0 && !quoted {
			return
		}
		out = append(out, shellWord{text: b.String(), quoted: quoted})
		b.Reset()
		quoted = false
	}

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' && subQuote != '\'' {
			escaped = true
			continue
		}

		if parenDepth > 0 || braceDepth > 0 {
			b.WriteByte(c)
			if c == '\'' || c == '"' {
				if subQuote == c {
					subQuote = 0
				} else if subQuote == 0 {
					subQuote = c
				}
			} else if subQuote == 0 {
				if c == '(' {
					parenDepth++
				} else if c == ')' {
					parenDepth--
				} else if c == '{' {
					braceDepth++
				} else if c == '}' {
					braceDepth--
				}
			}
			continue
		}

		if quote != 0 {
			if c == quote {
				if quote == '`' {
					b.WriteByte('`')
				}
				quote = 0
				quoted = true
				continue
			}
			b.WriteByte(c)
			continue
		}

		switch c {
		case '\'', '"':
			quote = c
			quoted = true
		case '`':
			b.WriteByte('`')
			quote = '`'
			quoted = true
		case '$':
			b.WriteByte(c)
			if i+1 < len(cmd) && cmd[i+1] == '(' {
				parenDepth++
				i++
				b.WriteByte('(')
			} else if i+1 < len(cmd) && cmd[i+1] == '{' {
				braceDepth++
				i++
				b.WriteByte('{')
			}
		case ' ', '\t', '\n', '\r':
			flush()
		default:
			b.WriteByte(c)
		}
	}
	flush()
	return out
}

func filterRedirectionWords(words []shellWord) []shellWord {
	var filtered []shellWord
	for i := 0; i < len(words); i++ {
		w := words[i]
		if w.quoted {
			filtered = append(filtered, w)
			continue
		}
		tok := w.text
		if tok == ">" || tok == ">>" || tok == "<" || tok == ">|" ||
			tok == "1>" || tok == "2>" || tok == "1>>" || tok == "2>>" ||
			tok == "&>" || tok == "&>>" {
			if i+1 < len(words) {
				i++
			}
			continue
		}
		if tok == "2>&1" || tok == "1>&2" || tok == ">&2" || tok == ">&1" ||
			tok == "2>&-" || tok == "1>&-" || tok == "&>1" || tok == "&>2" {
			continue
		}
		if strings.HasPrefix(tok, ">") || strings.HasPrefix(tok, "<") ||
			strings.HasPrefix(tok, "2>") || strings.HasPrefix(tok, "1>") ||
			strings.HasPrefix(tok, "&>") {
			continue
		}
		filtered = append(filtered, w)
	}
	return filtered
}

func isStreamTreeKnown(raw, cleaned string) bool {
	cleaned = strings.TrimSpace(cleaned)
	if cleaned == "" || isNullSink(cleaned) {
		return false
	}
	if !isTreeKnown(raw, cleaned) {
		return false
	}
	if strings.ContainsAny(cleaned, "{}[]<>|;") {
		return false
	}
	return true
}

func matchLaneGlob(path, glob string) bool {
	if glob == "" || path == "" {
		return false
	}
	p := strings.TrimPrefix(path, "./")
	cleanGlob := strings.TrimSuffix(glob, "/**")
	cleanGlob = strings.TrimSuffix(cleanGlob, "/*")
	cleanGlob = strings.TrimPrefix(cleanGlob, "./")
	if cleanGlob != glob {
		if strings.HasPrefix(p, cleanGlob+"/") || p == cleanGlob {
			return true
		}
		if strings.Contains(p, cleanGlob+"/") {
			return true
		}
	}
	return false
}

func checkUnterminatedQuote(s string) byte {
	var quote byte
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		if quote != 0 {
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '\'' || c == '"' || c == '`' {
			quote = c
		}
	}
	return quote
}

func skipElevationPrefix(words []shellWord, start int) int {
	for start < len(words) {
		wLower := strings.ToLower(words[start].text)
		if wLower == "sudo" || wLower == "doas" {
			start++
			for start < len(words) {
				tok := words[start].text
				if tok == "--" {
					start++
					break
				}
				if strings.HasPrefix(tok, "-") {
					if tok == "-u" || tok == "-g" || tok == "-h" || tok == "-p" {
						start += 2
						continue
					}
					start++
					continue
				}
				break
			}
			continue
		}
		break
	}
	return start
}

func normalizeExecutable(head string) string {
	head = strings.TrimPrefix(strings.ToLower(head), "./")
	if slash := strings.LastIndexAny(head, "/\\"); slash >= 0 {
		head = head[slash+1:]
	}
	return strings.TrimSuffix(head, ".exe")
}

func isSubshell(name string) bool {
	return name == "sh" || name == "bash" || name == "zsh" || name == "dash"
}

func normalizeToolName(head string) string {
	name := normalizeExecutable(head)
	switch name {
	case "sed", "gsed":
		return "sed"
	case "awk", "nawk", "mawk":
		return "awk"
	case "gawk":
		return "gawk"
	default:
		return ""
	}
}
