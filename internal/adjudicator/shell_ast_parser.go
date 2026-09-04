package adjudicator

import (
	"fmt"
	"strings"
)

// BashWriteTarget records a file system write operation extracted from a shell command.
type BashWriteTarget struct {
	Path      string `json:"path"`
	Op        string `json:"op"`
	TreeKnown bool   `json:"tree_known"`
}

// ParseBashWriteTargets inspects a shell command line and extracts all mutation targets,
// unwrapping subshells (bash -c, sh -c), coreutils file operations (cp, mv, rm),
// stream edits (sed -i), pipeline writers (tee, tee -a), redirections (>, >>, 1>, 1>>),
// and inline here-documents. It evaluates treeKnown=true when all mutation targets
// are resolved to concrete static paths.
func ParseBashWriteTargets(cmd string) ([]BashWriteTarget, bool, error) {
	trimmed := strings.TrimSpace(cmd)
	if trimmed == "" {
		return []BashWriteTarget{}, true, nil
	}

	if q := checkUnterminatedQuote(cmd); q != 0 {
		return nil, false, fmt.Errorf("unterminated quote %c in command", q)
	}

	// 1. Strip inert here-document bodies so payload lines are not parsed as commands.
	cleanedCmd := stripHeredocBodies(cmd)

	// 2. Strip comments outside quotes.
	cleanedCmd = stripShellComments(cleanedCmd)

	// 3. Split into statements (;, \n, &&, ||, &).
	statements := splitShellStatements(cleanedCmd)
	if len(statements) == 0 {
		return []BashWriteTarget{}, true, nil
	}

	var allTargets []BashWriteTarget
	treeKnown := true

	for _, stmt := range statements {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}

		// Check for subshell parenthesized block: ( cmd1 ; cmd2 ) [> out.txt]
		if strings.HasPrefix(stmt, "(") {
			parenTargets, pKnown, pHandled, err := parseSubshellParenStatement(stmt)
			if err != nil {
				return nil, false, err
			}
			if pHandled {
				allTargets = append(allTargets, parenTargets...)
				if !pKnown {
					treeKnown = false
				}
				continue
			}
		}

		// 4. Split statement into pipeline stages (|).
		stages := splitShellPipeline(stmt)
		for _, stage := range stages {
			stage = strings.TrimSpace(stage)
			if stage == "" {
				continue
			}

			// 5. Extract redirections from this stage and replace them with spaces.
			nonRedirStage, redirTargets := extractRedirections(stage)

			// 6. Tokenize remaining words and parse command verbs.
			cmdTargets, stageKnown, err := parseCommandStage(nonRedirStage)
			if err != nil {
				return nil, false, err
			}
			if !stageKnown {
				treeKnown = false
			}

			allTargets = append(allTargets, cmdTargets...)
			allTargets = append(allTargets, redirTargets...)
		}
	}

	if allTargets == nil {
		allTargets = []BashWriteTarget{}
	}

	// Evaluate treeKnown: all mutation targets must be resolved to concrete static paths.
	for _, t := range allTargets {
		if !t.TreeKnown || t.Path == "" {
			treeKnown = false
			break
		}
	}

	return allTargets, treeKnown, nil
}

// parseSubshellParenStatement handles grouped subshell statements like `( cmd1 ; cmd2 ) > out.txt`.
func parseSubshellParenStatement(stmt string) ([]BashWriteTarget, bool, bool, error) {
	nonRedir, outerRedirs := extractRedirections(stmt)
	trimmed := strings.TrimSpace(nonRedir)
	if !strings.HasPrefix(trimmed, "(") || !strings.HasSuffix(trimmed, ")") {
		return nil, true, false, nil
	}

	inner := trimmed[1 : len(trimmed)-1]
	innerTargets, innerKnown, err := ParseBashWriteTargets(inner)
	if err != nil {
		return nil, false, true, err
	}

	var results []BashWriteTarget
	results = append(results, innerTargets...)
	results = append(results, outerRedirs...)

	known := innerKnown
	for _, t := range outerRedirs {
		if !t.TreeKnown || t.Path == "" {
			known = false
			break
		}
	}

	return results, known, true, nil
}

// parseCommandStage parses the command words in a single pipeline stage.
func parseCommandStage(stage string) ([]BashWriteTarget, bool, error) {
	words := tokenizeShellWords(stage)
	if len(words) == 0 {
		return nil, true, nil
	}

	start := commandWordStart(words)
	if start >= len(words) {
		return nil, true, nil
	}

	start = skipElevationPrefix(words, start)
	if start >= len(words) {
		return nil, true, nil
	}

	headWord := words[start]
	rawHead := headWord.text
	if strings.HasPrefix(rawHead, "$") {
		// Dynamic command invocation (e.g. $CMD arg)
		return []BashWriteTarget{
			{
				Path:      rawHead,
				Op:        "write",
				TreeKnown: false,
			},
		}, false, nil
	}

	head := normalizeExecutable(rawHead)
	args := words[start+1:]

	// Subshell unwrapping: bash -c "...", sh -c '...'
	if isSubshell(head) {
		for i := 0; i < len(args); i++ {
			tok := args[i].text
			if !args[i].quoted && (tok == "-c" || (strings.HasPrefix(tok, "-") && strings.HasSuffix(tok, "c") && len(tok) > 1)) {
				if i+1 < len(args) {
					innerCmd := args[i+1].text
					subTargets, subKnown, err := ParseBashWriteTargets(innerCmd)
					if err != nil {
						return nil, false, err
					}
					return subTargets, subKnown, nil
				}
				return nil, false, fmt.Errorf("option '-c' requires an argument")
			}
		}
		return nil, true, nil
	}

	switch head {
	case "tee":
		return parseTeeStage(args), true, nil
	case "cp":
		return parseCpStage(args), true, nil
	case "mv":
		return parseMvStage(args), true, nil
	case "rm":
		return parseRmStage(args), true, nil
	case "sed":
		return parseSedStage(args), true, nil
	default:
		return nil, true, nil
	}
}

// parseTeeStage extracts write targets from tee [-a|--append] invocations.
func parseTeeStage(args []shellWord) []BashWriteTarget {
	appendMode := false
	var fileOperands []shellWord
	endOfOptions := false

	for _, w := range args {
		if !endOfOptions && !w.quoted {
			if w.text == "--" {
				endOfOptions = true
				continue
			}
			if strings.HasPrefix(w.text, "-") && w.text != "-" {
				if w.text == "-a" || w.text == "--append" {
					appendMode = true
				} else if !strings.HasPrefix(w.text, "--") && strings.ContainsRune(w.text, 'a') {
					appendMode = true
				}
				continue
			}
		}
		fileOperands = append(fileOperands, w)
	}

	var targets []BashWriteTarget
	op := "write"
	if appendMode {
		op = "append"
	}

	for _, w := range fileOperands {
		raw := w.text
		cleaned := cleanShellPath(raw)
		if cleaned == "" || isNullSink(cleaned) {
			continue
		}
		targets = append(targets, BashWriteTarget{
			Path:      cleaned,
			Op:        op,
			TreeKnown: isBashTreeKnown(raw, cleaned),
		})
	}
	return targets
}

// parseCpStage extracts the copy destination target from cp src dest.
func parseCpStage(args []shellWord) []BashWriteTarget {
	var targetDir string
	var positional []shellWord
	endOfOptions := false

	for i := 0; i < len(args); i++ {
		w := args[i]
		if !endOfOptions && !w.quoted {
			if w.text == "--" {
				endOfOptions = true
				continue
			}
			if strings.HasPrefix(w.text, "-") && w.text != "-" {
				if w.text == "-t" || w.text == "--target-directory" {
					if i+1 < len(args) {
						targetDir = args[i+1].text
						i++
					}
					continue
				}
				if strings.HasPrefix(w.text, "-t") {
					targetDir = strings.TrimPrefix(w.text, "-t")
					continue
				}
				if strings.HasPrefix(w.text, "--target-directory=") {
					targetDir = strings.TrimPrefix(w.text, "--target-directory=")
					continue
				}
				if w.text == "-S" || w.text == "--suffix" || w.text == "--backup" {
					if i+1 < len(args) && !strings.HasPrefix(args[i+1].text, "-") {
						i++
					}
					continue
				}
				continue
			}
		}
		positional = append(positional, w)
	}

	if targetDir != "" {
		cleaned := cleanShellPath(targetDir)
		if cleaned != "" && !isNullSink(cleaned) {
			return []BashWriteTarget{
				{
					Path:      cleaned,
					Op:        "copy",
					TreeKnown: isBashTreeKnown(targetDir, cleaned),
				},
			}
		}
		return nil
	}

	if len(positional) >= 2 {
		destWord := positional[len(positional)-1]
		raw := destWord.text
		cleaned := cleanShellPath(raw)
		if cleaned != "" && !isNullSink(cleaned) {
			return []BashWriteTarget{
				{
					Path:      cleaned,
					Op:        "copy",
					TreeKnown: isBashTreeKnown(raw, cleaned),
				},
			}
		}
	} else if len(positional) == 1 {
		return []BashWriteTarget{
			{
				Path:      "",
				Op:        "copy",
				TreeKnown: false,
			},
		}
	}
	return nil
}

// parseMvStage extracts the destination target from mv src dest.
func parseMvStage(args []shellWord) []BashWriteTarget {
	var targetDir string
	var positional []shellWord
	endOfOptions := false

	for i := 0; i < len(args); i++ {
		w := args[i]
		if !endOfOptions && !w.quoted {
			if w.text == "--" {
				endOfOptions = true
				continue
			}
			if strings.HasPrefix(w.text, "-") && w.text != "-" {
				if w.text == "-t" || w.text == "--target-directory" {
					if i+1 < len(args) {
						targetDir = args[i+1].text
						i++
					}
					continue
				}
				if strings.HasPrefix(w.text, "-t") {
					targetDir = strings.TrimPrefix(w.text, "-t")
					continue
				}
				if strings.HasPrefix(w.text, "--target-directory=") {
					targetDir = strings.TrimPrefix(w.text, "--target-directory=")
					continue
				}
				if w.text == "-S" || w.text == "--suffix" || w.text == "--backup" {
					if i+1 < len(args) && !strings.HasPrefix(args[i+1].text, "-") {
						i++
					}
					continue
				}
				continue
			}
		}
		positional = append(positional, w)
	}

	if targetDir != "" {
		cleaned := cleanShellPath(targetDir)
		if cleaned != "" && !isNullSink(cleaned) {
			return []BashWriteTarget{
				{
					Path:      cleaned,
					Op:        "move",
					TreeKnown: isBashTreeKnown(targetDir, cleaned),
				},
			}
		}
		return nil
	}

	if len(positional) >= 2 {
		destWord := positional[len(positional)-1]
		raw := destWord.text
		cleaned := cleanShellPath(raw)
		if cleaned != "" && !isNullSink(cleaned) {
			return []BashWriteTarget{
				{
					Path:      cleaned,
					Op:        "move",
					TreeKnown: isBashTreeKnown(raw, cleaned),
				},
			}
		}
	} else if len(positional) == 1 {
		return []BashWriteTarget{
			{
				Path:      "",
				Op:        "move",
				TreeKnown: false,
			},
		}
	}
	return nil
}

// parseRmStage extracts path targets from rm [-rf] path...
func parseRmStage(args []shellWord) []BashWriteTarget {
	var paths []shellWord
	endOfOptions := false

	for i := 0; i < len(args); i++ {
		w := args[i]
		if !endOfOptions && !w.quoted {
			if w.text == "--" {
				endOfOptions = true
				continue
			}
			if strings.HasPrefix(w.text, "-") && w.text != "-" {
				continue
			}
		}
		paths = append(paths, w)
	}

	var targets []BashWriteTarget
	for _, p := range paths {
		raw := p.text
		cleaned := cleanShellPath(raw)
		if cleaned == "" {
			continue
		}
		targets = append(targets, BashWriteTarget{
			Path:      cleaned,
			Op:        "remove",
			TreeKnown: isBashTreeKnown(raw, cleaned),
		})
	}
	return targets
}

// parseSedStage extracts in-place stream editing targets from sed -i.
func parseSedStage(args []shellWord) []BashWriteTarget {
	inPlace := false
	var scripts []string
	var positional []shellWord
	endOfOptions := false

	for i := 0; i < len(args); i++ {
		w := args[i]
		tok := w.text
		if !endOfOptions && !w.quoted {
			if tok == "--" {
				endOfOptions = true
				continue
			}
			if tok == "--in-place" || strings.HasPrefix(tok, "--in-place=") {
				inPlace = true
				continue
			}
			if tok == "-i" {
				inPlace = true
				if i+1 < len(args) {
					next := args[i+1].text
					// macOS sed -i '' or backup suffix like -i .bak
					if next == "" || (strings.HasPrefix(next, ".") && !strings.Contains(next, "/") && i+2 < len(args)) {
						i++
					}
				}
				continue
			}
			if strings.HasPrefix(tok, "-i") {
				inPlace = true
				continue
			}
			if strings.HasPrefix(tok, "-") && len(tok) > 1 {
				if strings.ContainsRune(tok, 'i') {
					inPlace = true
				}
				if tok == "-e" || tok == "-f" {
					if i+1 < len(args) {
						scripts = append(scripts, args[i+1].text)
						i++
					}
					continue
				}
				if strings.HasPrefix(tok, "-e") || strings.HasPrefix(tok, "-f") {
					scripts = append(scripts, tok[2:])
					continue
				}
				if strings.HasPrefix(tok, "--expression=") || strings.HasPrefix(tok, "--file=") {
					scripts = append(scripts, tok)
					continue
				}
				if tok == "--expression" || tok == "--file" {
					if i+1 < len(args) {
						scripts = append(scripts, args[i+1].text)
						i++
					}
					continue
				}
				continue
			}
		}
		positional = append(positional, w)
	}

	if !inPlace {
		return nil
	}

	var targetFiles []shellWord
	if len(scripts) > 0 {
		targetFiles = positional
	} else if len(positional) > 0 {
		targetFiles = positional[1:]
	}

	if len(targetFiles) == 0 {
		return []BashWriteTarget{
			{
				Path:      "",
				Op:        "stream_edit",
				TreeKnown: false,
			},
		}
	}

	var targets []BashWriteTarget
	for _, tf := range targetFiles {
		raw := tf.text
		cleaned := cleanShellPath(raw)
		if cleaned == "" {
			continue
		}
		targets = append(targets, BashWriteTarget{
			Path:      cleaned,
			Op:        "stream_edit",
			TreeKnown: isBashTreeKnown(raw, cleaned),
		})
	}
	return targets
}

// extractRedirections scans a command segment for output redirections (>, >>, 1>, 1>>, >|, &>, &>>),
// extracts their targets, and replaces the redirection operator and target with whitespace.
func extractRedirections(seg string) (string, []BashWriteTarget) {
	var targets []BashWriteTarget
	runes := []rune(seg)
	var quote rune
	escaped := false
	parenDepth := 0
	doubleBracketDepth := 0

	i := 0
	for i < len(runes) {
		r := runes[i]
		if escaped {
			escaped = false
			i++
			continue
		}
		if r == '\\' && quote != '\'' {
			escaped = true
			i++
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			}
			i++
			continue
		}
		if r == '\'' || r == '"' || r == '`' {
			quote = r
			i++
			continue
		}
		if r == '(' {
			parenDepth++
			i++
			continue
		}
		if r == ')' {
			if parenDepth > 0 {
				parenDepth--
			}
			i++
			continue
		}
		if r == '[' && i+1 < len(runes) && runes[i+1] == '[' {
			doubleBracketDepth++
			i += 2
			continue
		}
		if r == ']' && i+1 < len(runes) && runes[i+1] == ']' {
			if doubleBracketDepth > 0 {
				doubleBracketDepth--
			}
			i += 2
			continue
		}

		if parenDepth == 0 && doubleBracketDepth == 0 {
			isRedir := false
			op := "write"
			startIdx := i
			targetStart := i

			if r == '>' {
				isRedir = true
				if i > 0 && runes[i-1] == '1' {
					if i-1 == 0 || runes[i-2] == ' ' || runes[i-2] == '\t' || runes[i-2] == ';' || runes[i-2] == '&' || runes[i-2] == '|' {
						startIdx = i - 1
					}
				} else if i > 0 && runes[i-1] == '2' {
					if i-1 == 0 || runes[i-2] == ' ' || runes[i-2] == '\t' || runes[i-2] == ';' || runes[i-2] == '&' || runes[i-2] == '|' {
						isRedir = false
					}
				} else if i > 0 && runes[i-1] == '&' {
					startIdx = i - 1
				}

				if isRedir {
					opEnd := i + 1
					if i+1 < len(runes) && runes[i+1] == '>' {
						op = "append"
						opEnd = i + 2
					} else if i+1 < len(runes) && runes[i+1] == '|' {
						op = "write"
						opEnd = i + 2
					}
					targetStart = opEnd
				}
			}

			if isRedir {
				k := targetStart
				for k < len(runes) && (runes[k] == ' ' || runes[k] == '\t') {
					k++
				}
				if k < len(runes) && (runes[k] == '&' || runes[k] == '(') {
					// fd duplication (> &2) or process substitution (>(...))
					i++
					continue
				}

				targetIdx := k
				endIdx := targetIdx
				if targetIdx >= len(runes) {
					targets = append(targets, BashWriteTarget{
						Path:      "",
						Op:        op,
						TreeKnown: false,
					})
					for idx := startIdx; idx < len(runes); idx++ {
						runes[idx] = ' '
					}
					break
				}

				if runes[targetIdx] == '\'' || runes[targetIdx] == '"' {
					tq := runes[targetIdx]
					endIdx = targetIdx + 1
					for endIdx < len(runes) && runes[endIdx] != tq {
						if tq == '"' && runes[endIdx] == '\\' && endIdx+1 < len(runes) {
							endIdx++
						}
						endIdx++
					}
					if endIdx < len(runes) {
						endIdx++
					}
				} else {
					for endIdx < len(runes) {
						if isShellDelimRune(runes[endIdx]) {
							numSlashes := 0
							for k := endIdx - 1; k >= targetIdx && runes[k] == '\\'; k-- {
								numSlashes++
							}
							if numSlashes%2 == 0 {
								break
							}
						}
						endIdx++
					}
				}

				rawTarget := string(runes[targetIdx:endIdx])
				cleaned := cleanShellPath(rawTarget)
				if !isNullSink(cleaned) {
					known := isBashTreeKnown(rawTarget, cleaned)
					targets = append(targets, BashWriteTarget{
						Path:      cleaned,
						Op:        op,
						TreeKnown: known,
					})
				}

				for idx := startIdx; idx < endIdx; idx++ {
					runes[idx] = ' '
				}
				i = endIdx
				continue
			}
		}
		i++
	}

	return string(runes), targets
}

func isShellDelimRune(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r', ';', '&', '|', '<', '>', '(', ')':
		return true
	}
	return false
}

// tokenizeShellWords splits cmd into shellWord tokens, respecting quotes and escapes.
func tokenizeShellWords(cmd string) []shellWord {
	var words []shellWord
	var b strings.Builder
	var quote byte
	quoted := false
	escaped := false

	flush := func() {
		if b.Len() == 0 && !quoted {
			return
		}
		words = append(words, shellWord{text: b.String(), quoted: quoted})
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
		if c == '\\' && quote != '\'' {
			if quote == '"' {
				if i+1 < len(cmd) {
					next := cmd[i+1]
					if next == '"' || next == '\\' || next == '$' || next == '`' || next == '\n' {
						escaped = true
						continue
					}
				}
				b.WriteByte(c)
				continue
			}
			if i+1 < len(cmd) {
				next := cmd[i+1]
				if isShellEscapeChar(next) {
					escaped = true
					continue
				}
			}
			b.WriteByte(c)
			continue
		}
		if quote != 0 {
			if c == quote {
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
		case ' ', '\t', '\n', '\r':
			flush()
		default:
			b.WriteByte(c)
		}
	}
	flush()
	return words
}

// splitShellStatements splits a shell command line at unquoted command separators
// (;, \n, &&, ||, &).
func splitShellStatements(cmd string) []string {
	var statements []string
	var b strings.Builder
	var quote byte
	escaped := false
	parenDepth := 0

	flush := func() {
		s := strings.TrimSpace(b.String())
		if s != "" {
			statements = append(statements, s)
		}
		b.Reset()
	}

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			b.WriteByte(c)
			continue
		}
		if quote != 0 {
			b.WriteByte(c)
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '\'' || c == '"' || c == '`' {
			quote = c
			b.WriteByte(c)
			continue
		}
		if c == '(' {
			parenDepth++
			b.WriteByte(c)
			continue
		}
		if c == ')' {
			if parenDepth > 0 {
				parenDepth--
			}
			b.WriteByte(c)
			continue
		}

		if parenDepth == 0 {
			if c == ';' || c == '\n' {
				flush()
				continue
			}
			if c == '&' {
				if i+1 < len(cmd) && cmd[i+1] == '&' {
					flush()
					i++
					continue
				}
				if i+1 < len(cmd) && cmd[i+1] == '>' {
					b.WriteByte(c)
					continue
				}
				if b.Len() > 0 && b.String()[b.Len()-1] == '>' {
					b.WriteByte(c)
					continue
				}
				flush()
				continue
			}
			if c == '|' {
				if i+1 < len(cmd) && cmd[i+1] == '|' {
					flush()
					i++
					continue
				}
				b.WriteByte(c)
				continue
			}
		}
		b.WriteByte(c)
	}
	flush()
	return statements
}

// splitShellPipeline splits a shell statement into pipeline stages at unquoted | or |&.
func splitShellPipeline(stmt string) []string {
	var stages []string
	var b strings.Builder
	var quote byte
	escaped := false
	parenDepth := 0

	flush := func() {
		s := strings.TrimSpace(b.String())
		if s != "" {
			stages = append(stages, s)
		}
		b.Reset()
	}

	for i := 0; i < len(stmt); i++ {
		c := stmt[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			b.WriteByte(c)
			continue
		}
		if quote != 0 {
			b.WriteByte(c)
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '\'' || c == '"' || c == '`' {
			quote = c
			b.WriteByte(c)
			continue
		}
		if c == '(' {
			parenDepth++
			b.WriteByte(c)
			continue
		}
		if c == ')' {
			if parenDepth > 0 {
				parenDepth--
			}
			b.WriteByte(c)
			continue
		}
		if parenDepth == 0 && c == '|' {
			if b.Len() > 0 && b.String()[b.Len()-1] == '>' {
				b.WriteByte(c)
				continue
			}
			if i+1 < len(stmt) && stmt[i+1] == '|' {
				flush()
				i++
				continue
			}
			if i+1 < len(stmt) && stmt[i+1] == '&' {
				flush()
				i++
				continue
			}
			flush()
			continue
		}
		b.WriteByte(c)
	}
	flush()
	return stages
}

type heredocDelim struct {
	delim     string
	stripTabs bool
}

// stripHeredocBodies removes the bodies of inline here-documents so that payload bytes
// are not tokenized as commands.
func stripHeredocBodies(cmd string) string {
	if !strings.Contains(cmd, "<<") || !strings.Contains(cmd, "\n") {
		return cmd
	}
	lines := strings.Split(strings.ReplaceAll(cmd, "\r\n", "\n"), "\n")
	var pending []heredocDelim
	var out []string

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if len(pending) > 0 {
			cur := pending[0]
			matched := false
			if cur.stripTabs {
				matched = strings.TrimLeft(line, "\t") == cur.delim || strings.TrimSpace(line) == cur.delim
			} else {
				matched = strings.TrimSpace(line) == cur.delim
			}
			if matched {
				pending = pending[1:]
			}
			continue
		}

		out = append(out, line)

		delims := findHeredocDelims(line)
		if len(delims) > 0 {
			pending = append(pending, delims...)
		}
	}
	return strings.Join(out, "\n")
}

func findHeredocDelims(line string) []heredocDelim {
	var delims []heredocDelim
	var quote byte
	escaped := false

	for i := 0; i < len(line); i++ {
		c := line[i]
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
		if c == '\'' || c == '"' {
			quote = c
			continue
		}
		if c == '<' && i+1 < len(line) && line[i+1] == '<' {
			// Skip here-string <<<
			if i+2 < len(line) && line[i+2] == '<' {
				i += 2
				continue
			}
			i += 2
			stripTabs := false
			if i < len(line) && line[i] == '-' {
				stripTabs = true
				i++
			}
			for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
				i++
			}
			if i >= len(line) {
				break
			}
			var delim strings.Builder
			if line[i] == '\'' || line[i] == '"' {
				dq := line[i]
				i++
				for i < len(line) && line[i] != dq {
					delim.WriteByte(line[i])
					i++
				}
			} else if line[i] == '\\' {
				i++
				for i < len(line) && !isShellDelimRune(rune(line[i])) {
					delim.WriteByte(line[i])
					i++
				}
			} else {
				for i < len(line) && !isShellDelimRune(rune(line[i])) {
					delim.WriteByte(line[i])
					i++
				}
			}
			if delim.Len() > 0 {
				delims = append(delims, heredocDelim{
					delim:     delim.String(),
					stripTabs: stripTabs,
				})
			}
		}
	}
	return delims
}

// stripShellComments strips shell comments (# ...) outside quotes.
func stripShellComments(cmd string) string {
	var b strings.Builder
	var quote byte
	escaped := false
	inComment := false

	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if inComment {
			if c == '\n' {
				inComment = false
				b.WriteByte(c)
			}
			continue
		}
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			b.WriteByte(c)
			continue
		}
		if quote != 0 {
			b.WriteByte(c)
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '\'' || c == '"' || c == '`' {
			quote = c
			b.WriteByte(c)
			continue
		}
		if c == '#' {
			if i == 0 || cmd[i-1] == ' ' || cmd[i-1] == '\t' || cmd[i-1] == '\n' || cmd[i-1] == ';' {
				inComment = true
				continue
			}
		}
		b.WriteByte(c)
	}
	return b.String()
}

// isWindowsVolume reports whether s starts with a Windows drive letter path matching ^[A-Za-z]:[\\/].
func isWindowsVolume(s string) bool {
	if len(s) >= 3 && s[1] == ':' && (s[2] == '\\' || s[2] == '/') {
		c := s[0]
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
	}
	return false
}

// hasWindowsDrivePrefix reports whether s starts with a Windows drive specifier ^[A-Za-z]:.
func hasWindowsDrivePrefix(s string) bool {
	if len(s) >= 2 && s[1] == ':' {
		c := s[0]
		return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
	}
	return false
}

// isUNCPath reports whether s begins with a Windows UNC share prefix (\\server\share).
func isUNCPath(s string) bool {
	return len(s) >= 3 && s[0] == '\\' && s[1] == '\\' && s[2] != '\\' && s[2] != '/'
}

// isShellEscapeChar reports whether c is a character commonly escaped with a backslash in POSIX shells.
func isShellEscapeChar(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\'', '"', '`', '$', '&', '|', ';', '<', '>', '(', ')', '*', '?', '[', ']', '{', '}', '!', '#', '~', '^', '=', '\\':
		return true
	default:
		return false
	}
}

// cleanShellPath cleans quotes and escapes from a target path operand while preserving
// Windows volume paths (^[A-Za-z]:[\\/]) and directory separators in Windows-style paths.
func cleanShellPath(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if len(s) >= 2 && ((s[0] == '\'' && s[len(s)-1] == '\'') || (s[0] == '"' && s[len(s)-1] == '"')) {
		quote := s[0]
		inner := s[1 : len(s)-1]
		if quote == '\'' {
			return inner
		}
		inner = strings.ReplaceAll(inner, `\"`, `"`)
		inner = strings.ReplaceAll(inner, `\\`, `\`)
		return inner
	}

	isWinVolume := isWindowsVolume(s) || hasWindowsDrivePrefix(s)
	isUNC := isUNCPath(s)

	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if i == 0 && isUNC && len(s) >= 2 && s[0] == '\\' && s[1] == '\\' {
			b.WriteString(`\\`)
			i++
			continue
		}
		if c == '\\' {
			if i+1 >= len(s) {
				b.WriteByte(c)
				continue
			}
			next := s[i+1]
			if isWinVolume && next == '$' {
				b.WriteByte('\\')
				continue
			}
			if isShellEscapeChar(next) {
				b.WriteByte(next)
				i++
				continue
			}
			b.WriteByte('\\')
			continue
		}
		if c != '\'' && c != '"' {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// isBashTreeKnown evaluates whether raw and cleaned resolve to a static, concrete path.
func isBashTreeKnown(raw, cleaned string) bool {
	cleaned = strings.TrimSpace(cleaned)
	if cleaned == "" || isNullSink(cleaned) {
		return false
	}
	trimmedRaw := strings.TrimSpace(raw)

	// Single-quoted literal: in bash, single quotes inhibit all expansions.
	if len(trimmedRaw) >= 2 && trimmedRaw[0] == '\'' && trimmedRaw[len(trimmedRaw)-1] == '\'' && strings.Count(trimmedRaw, "'") == 2 {
		return true
	}

	if strings.Count(trimmedRaw, "'")%2 != 0 {
		return false
	}

	unescapedDQuotes := 0
	for i := 0; i < len(trimmedRaw); i++ {
		if trimmedRaw[i] == '\\' && i+1 < len(trimmedRaw) {
			i++
			continue
		}
		if trimmedRaw[i] == '"' {
			unescapedDQuotes++
		}
	}
	if unescapedDQuotes%2 != 0 {
		return false
	}

	if strings.ContainsAny(cleaned, "$`()") {
		return false
	}
	if strings.ContainsAny(cleaned, "*?[]") {
		return false
	}
	if strings.ContainsAny(cleaned, "{}<>|;\n\r\t") {
		return false
	}
	return true
}
