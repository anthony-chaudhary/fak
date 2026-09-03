package normgate

import (
	"errors"
	"strings"
)

// PowerShellTarget captures a resolved target path and mutation metadata
// from a PowerShell command.
type PowerShellTarget struct {
	Cmdlet    string
	Path      string
	Op        string
	TreeKnown bool
}

type cmdletSpec struct {
	canonical string
	defaultOp string
}

var knownPowerShellCmdlets = map[string]cmdletSpec{
	"set-content": {canonical: "Set-Content", defaultOp: "write"},
	"sc":          {canonical: "Set-Content", defaultOp: "write"},
	"out-file":    {canonical: "Out-File", defaultOp: "write"},
	"add-content": {canonical: "Add-Content", defaultOp: "append"},
	"ac":          {canonical: "Add-Content", defaultOp: "append"},
	"remove-item": {canonical: "Remove-Item", defaultOp: "delete"},
	"rm":          {canonical: "Remove-Item", defaultOp: "delete"},
	"ri":          {canonical: "Remove-Item", defaultOp: "delete"},
	"del":         {canonical: "Remove-Item", defaultOp: "delete"},
	"erase":       {canonical: "Remove-Item", defaultOp: "delete"},
	"rd":          {canonical: "Remove-Item", defaultOp: "delete"},
	"rmdir":       {canonical: "Remove-Item", defaultOp: "delete"},
	"new-item":    {canonical: "New-Item", defaultOp: "create"},
	"ni":          {canonical: "New-Item", defaultOp: "create"},
}

// ExtractPowerShellTargets scans command for recognized PowerShell file-mutating cmdlets
// and extracts target paths, accounting for flag variations, positional arguments,
// pipelines, and quotes.
func ExtractPowerShellTargets(command string) ([]PowerShellTarget, error) {
	cmd := unwrapPowerShellInvocation(command)
	if strings.TrimSpace(cmd) == "" {
		return []PowerShellTarget{}, nil
	}

	statements, err := splitPowerShellStatements(cmd)
	if err != nil {
		return nil, err
	}

	var results []PowerShellTarget
	for _, stmt := range statements {
		stages, err := splitPowerShellPipeline(stmt)
		if err != nil {
			return nil, err
		}

		for stageIdx, stage := range stages {
			targets, err := parsePowerShellStage(stage, stageIdx > 0)
			if err != nil {
				return nil, err
			}
			results = append(results, targets...)
		}
	}

	if results == nil {
		return []PowerShellTarget{}, nil
	}
	return results, nil
}

func parsePowerShellStage(stage string, isPipelineDestination bool) ([]PowerShellTarget, error) {
	words, err := tokenizePowerShellWords(stage)
	if err != nil {
		return nil, err
	}
	if len(words) == 0 {
		return nil, nil
	}

	startIdx := 0
	if words[0] == "&" {
		startIdx = 1
	}
	if startIdx >= len(words) {
		return nil, nil
	}

	cmdWord := words[startIdx]
	cmdWord = cleanPathToken(cmdWord)
	if idx := strings.LastIndex(cmdWord, `\`); idx != -1 {
		cmdWord = cmdWord[idx+1:]
	}
	if idx := strings.LastIndex(cmdWord, "/"); idx != -1 {
		cmdWord = cmdWord[idx+1:]
	}
	cmdWordLower := strings.ToLower(cmdWord)

	spec, ok := knownPowerShellCmdlets[cmdWordLower]
	if !ok {
		return nil, nil
	}

	var explicitPaths []string
	var positionalWords []string
	var nameArg string
	isAppend := false

	i := startIdx + 1
	for i < len(words) {
		w := words[i]
		if strings.HasPrefix(w, "-") {
			flag := w
			attachedVal := ""
			hasAttached := false
			if idx := strings.IndexAny(w, ":="); idx != -1 {
				flag = w[:idx]
				attachedVal = w[idx+1:]
				hasAttached = true
			}

			flagLower := strings.ToLower(flag)
			if isPathFlag(flagLower, spec.canonical) {
				if hasAttached {
					explicitPaths = append(explicitPaths, splitCommaPaths(attachedVal)...)
				} else {
					joined, nextIdx := collectCommaTokens(words, i+1)
					if joined != "" {
						explicitPaths = append(explicitPaths, splitCommaPaths(joined)...)
					}
					i = nextIdx
					continue
				}
			} else if isAppendFlag(flagLower) {
				if !hasAttached || isTruthy(attachedVal) {
					isAppend = true
				}
			} else if isNameFlag(flagLower) {
				if hasAttached {
					nameArg = cleanPathToken(attachedVal)
				} else if i+1 < len(words) && !strings.HasPrefix(words[i+1], "-") {
					i++
					nameArg = cleanPathToken(words[i])
				}
			} else if flagConsumesValue(flagLower) {
				if !hasAttached && i+1 < len(words) && !strings.HasPrefix(words[i+1], "-") {
					i++
				}
			}
			i++
			continue
		}

		positionalWords = append(positionalWords, w)
		i++
	}

	op := spec.defaultOp
	if spec.canonical == "Out-File" && isAppend {
		op = "append"
	}

	resolvedPaths := explicitPaths
	if len(resolvedPaths) == 0 {
		switch spec.canonical {
		case "Set-Content", "Add-Content":
			if len(positionalWords) > 0 {
				resolvedPaths = splitCommaPaths(positionalWords[0])
			}
		case "Out-File":
			if len(positionalWords) > 0 {
				resolvedPaths = splitCommaPaths(positionalWords[0])
			}
		case "Remove-Item":
			if len(positionalWords) > 0 {
				joined := strings.Join(positionalWords, " ")
				resolvedPaths = splitCommaPaths(joined)
			}
		case "New-Item":
			if len(positionalWords) > 0 {
				base := splitCommaPaths(positionalWords[0])
				if nameArg != "" {
					for _, b := range base {
						resolvedPaths = append(resolvedPaths, joinPathAndName(b, nameArg))
					}
				} else {
					resolvedPaths = base
				}
			} else if nameArg != "" {
				resolvedPaths = []string{nameArg}
			}
		}
	} else if spec.canonical == "New-Item" && nameArg != "" {
		var combined []string
		for _, p := range resolvedPaths {
			combined = append(combined, joinPathAndName(p, nameArg))
		}
		resolvedPaths = combined
	}

	if len(resolvedPaths) == 0 {
		return []PowerShellTarget{
			{
				Cmdlet:    spec.canonical,
				Path:      "",
				Op:        op,
				TreeKnown: false,
			},
		}, nil
	}

	var targets []PowerShellTarget
	for _, p := range resolvedPaths {
		raw := p
		cleaned := cleanPathToken(p)
		known := isTreeKnown(raw, cleaned)
		targets = append(targets, PowerShellTarget{
			Cmdlet:    spec.canonical,
			Path:      cleaned,
			Op:        op,
			TreeKnown: known,
		})
	}

	return targets, nil
}

func isPathFlag(flagLower, cmdletCanonical string) bool {
	switch flagLower {
	case "-path", "-literalpath", "-filepath", "-lp":
		return true
	}
	if strings.HasPrefix(flagLower, "-path") || strings.HasPrefix(flagLower, "-literal") || strings.HasPrefix(flagLower, "-filepath") {
		return true
	}
	if cmdletCanonical == "Out-File" && (flagLower == "-f" || flagLower == "-file") {
		return true
	}
	if (cmdletCanonical == "Set-Content" || cmdletCanonical == "Add-Content" || cmdletCanonical == "Remove-Item" || cmdletCanonical == "New-Item") &&
		(flagLower == "-p" || flagLower == "-pa") {
		return true
	}
	return false
}

func isAppendFlag(flagLower string) bool {
	return flagLower == "-append" || flagLower == "-a" || strings.HasPrefix(flagLower, "-append")
}

func isNameFlag(flagLower string) bool {
	return flagLower == "-name" || flagLower == "-n"
}

func flagConsumesValue(flagLower string) bool {
	switch flagLower {
	case "-value", "-val", "-v",
		"-itemtype", "-type", "-t",
		"-encoding", "-enc", "-e",
		"-width",
		"-filter",
		"-include",
		"-exclude",
		"-credential",
		"-target",
		"-inputobject",
		"-stream":
		return true
	default:
		return false
	}
}

func isTruthy(v string) bool {
	vLower := strings.ToLower(cleanPathToken(v))
	return vLower == "" || vLower == "true" || vLower == "$true" || vLower == "1"
}

func isTreeKnown(raw, cleaned string) bool {
	if cleaned == "" {
		return false
	}
	trimmed := strings.TrimSpace(raw)
	if len(trimmed) >= 2 && trimmed[0] == '\'' && trimmed[len(trimmed)-1] == '\'' {
		return true
	}
	if strings.HasPrefix(cleaned, "$") || strings.HasPrefix(cleaned, "(") || strings.HasPrefix(cleaned, "{") {
		return false
	}
	if strings.Contains(cleaned, "$") {
		return false
	}
	return true
}

func cleanPathToken(token string) string {
	token = strings.TrimSpace(token)
	if token == "" {
		return ""
	}
	if len(token) >= 2 && ((token[0] == '"' && token[len(token)-1] == '"') || (token[0] == '\'' && token[len(token)-1] == '\'')) {
		quote := token[0]
		inner := token[1 : len(token)-1]
		if quote == '\'' {
			return strings.ReplaceAll(inner, "''", "'")
		}
		inner = strings.ReplaceAll(inner, `\"`, `"`)
		inner = strings.ReplaceAll(inner, "`\"", `"`)
		inner = strings.ReplaceAll(inner, `""`, `"`)
		return inner
	}
	return token
}

func joinPathAndName(dir, name string) string {
	dir = cleanPathToken(dir)
	name = cleanPathToken(name)
	if dir == "" {
		return name
	}
	if name == "" {
		return dir
	}
	if strings.HasSuffix(dir, "/") || strings.HasSuffix(dir, `\`) {
		return dir + name
	}
	if strings.Contains(dir, `\`) {
		return dir + `\` + name
	}
	return dir + "/" + name
}

func splitCommaPaths(s string) []string {
	var paths []string
	var cur strings.Builder
	var inSingle, inDouble bool

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '\'' && !inDouble {
			if inSingle && i+1 < len(s) && s[i+1] == '\'' {
				cur.WriteByte('\'')
				i++
				continue
			}
			inSingle = !inSingle
			cur.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			cur.WriteByte(ch)
			continue
		}
		if ch == ',' && !inSingle && !inDouble {
			p := strings.TrimSpace(cur.String())
			if p != "" {
				paths = append(paths, p)
			}
			cur.Reset()
			continue
		}
		cur.WriteByte(ch)
	}

	p := strings.TrimSpace(cur.String())
	if p != "" {
		paths = append(paths, p)
	}
	return paths
}

func collectCommaTokens(words []string, start int) (string, int) {
	var parts []string
	i := start
	for i < len(words) {
		w := words[i]
		if strings.HasPrefix(w, "-") && len(parts) > 0 && !strings.HasSuffix(parts[len(parts)-1], ",") {
			break
		}
		parts = append(parts, w)
		i++
		if !strings.HasSuffix(w, ",") {
			if i < len(words) && words[i] == "," {
				parts = append(parts, words[i])
				i++
				continue
			}
			break
		}
	}
	return strings.Join(parts, " "), i
}

func unwrapPowerShellInvocation(cmd string) string {
	trimmed := strings.TrimSpace(cmd)
	lower := strings.ToLower(trimmed)

	var rest string
	if strings.HasPrefix(lower, "powershell.exe") {
		rest = trimmed[len("powershell.exe"):]
	} else if strings.HasPrefix(lower, "powershell") {
		rest = trimmed[len("powershell"):]
	} else if strings.HasPrefix(lower, "pwsh.exe") {
		rest = trimmed[len("pwsh.exe"):]
	} else if strings.HasPrefix(lower, "pwsh") {
		rest = trimmed[len("pwsh"):]
	} else {
		return cmd
	}

	if len(rest) > 0 && rest[0] != ' ' && rest[0] != '\t' {
		return cmd
	}

	tokens, err := tokenizePowerShellWords(rest)
	if err != nil {
		return cmd
	}

	for i := 0; i < len(tokens); i++ {
		t := strings.ToLower(tokens[i])
		if t == "-c" || t == "-command" || t == "/c" || t == "/command" {
			idx := strings.Index(strings.ToLower(rest), t)
			if idx != -1 {
				inner := strings.TrimSpace(rest[idx+len(t):])
				if len(inner) >= 2 {
					if (inner[0] == '"' && inner[len(inner)-1] == '"') || (inner[0] == '\'' && inner[len(inner)-1] == '\'') {
						inner = cleanPathToken(inner)
					} else if inner[0] == '{' && inner[len(inner)-1] == '}' {
						inner = strings.TrimSpace(inner[1 : len(inner)-1])
					}
				}
				return inner
			}
		}
	}

	return cmd
}

func splitPowerShellStatements(cmd string) ([]string, error) {
	var statements []string
	var current strings.Builder
	var inSingle, inDouble bool
	var parenDepth, braceDepth, bracketDepth int
	var escaped bool

	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		if escaped {
			current.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '`' && !inSingle {
			escaped = true
			current.WriteByte(ch)
			continue
		}
		if ch == '\\' && inDouble {
			escaped = true
			current.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			if inSingle && i+1 < len(cmd) && cmd[i+1] == '\'' {
				current.WriteString("''")
				i++
				continue
			}
			inSingle = !inSingle
			current.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			current.WriteByte(ch)
			continue
		}
		if !inSingle && !inDouble {
			switch ch {
			case '(':
				parenDepth++
			case ')':
				if parenDepth > 0 {
					parenDepth--
				}
			case '{':
				braceDepth++
			case '}':
				if braceDepth > 0 {
					braceDepth--
				}
			case '[':
				bracketDepth++
			case ']':
				if bracketDepth > 0 {
					bracketDepth--
				}
			case ';', '\n', '\r':
				if parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 {
					stmt := strings.TrimSpace(current.String())
					if stmt != "" {
						statements = append(statements, stmt)
					}
					current.Reset()
					continue
				}
			case '&':
				if parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 && i+1 < len(cmd) && cmd[i+1] == '&' {
					i++
					stmt := strings.TrimSpace(current.String())
					if stmt != "" {
						statements = append(statements, stmt)
					}
					current.Reset()
					continue
				}
			case '|':
				if parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 && i+1 < len(cmd) && cmd[i+1] == '|' {
					i++
					stmt := strings.TrimSpace(current.String())
					if stmt != "" {
						statements = append(statements, stmt)
					}
					current.Reset()
					continue
				}
			}
		}
		current.WriteByte(ch)
	}

	if inSingle || inDouble {
		return nil, errors.New("powershell: unclosed quote in command")
	}

	stmt := strings.TrimSpace(current.String())
	if stmt != "" {
		statements = append(statements, stmt)
	}
	return statements, nil
}

func splitPowerShellPipeline(statement string) ([]string, error) {
	var stages []string
	var current strings.Builder
	var inSingle, inDouble bool
	var parenDepth, braceDepth, bracketDepth int
	var escaped bool

	for i := 0; i < len(statement); i++ {
		ch := statement[i]
		if escaped {
			current.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '`' && !inSingle {
			escaped = true
			current.WriteByte(ch)
			continue
		}
		if ch == '\\' && inDouble {
			escaped = true
			current.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			if inSingle && i+1 < len(statement) && statement[i+1] == '\'' {
				current.WriteString("''")
				i++
				continue
			}
			inSingle = !inSingle
			current.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			current.WriteByte(ch)
			continue
		}
		if !inSingle && !inDouble {
			switch ch {
			case '(':
				parenDepth++
			case ')':
				if parenDepth > 0 {
					parenDepth--
				}
			case '{':
				braceDepth++
			case '}':
				if braceDepth > 0 {
					braceDepth--
				}
			case '[':
				bracketDepth++
			case ']':
				if bracketDepth > 0 {
					bracketDepth--
				}
			case '|':
				if parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 {
					stages = append(stages, strings.TrimSpace(current.String()))
					current.Reset()
					continue
				}
			}
		}
		current.WriteByte(ch)
	}

	if inSingle || inDouble {
		return nil, errors.New("powershell: unclosed quote in command")
	}

	stages = append(stages, strings.TrimSpace(current.String()))
	return stages, nil
}

func tokenizePowerShellWords(s string) ([]string, error) {
	var words []string
	var cur strings.Builder
	var inSingle, inDouble bool
	var parenDepth, braceDepth, bracketDepth int
	var escaped bool

	flush := func() {
		if cur.Len() > 0 {
			words = append(words, cur.String())
			cur.Reset()
		}
	}

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			cur.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '`' && !inSingle {
			escaped = true
			cur.WriteByte(ch)
			continue
		}
		if ch == '\\' && inDouble {
			escaped = true
			cur.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			if inSingle && i+1 < len(s) && s[i+1] == '\'' {
				cur.WriteString("''")
				i++
				continue
			}
			inSingle = !inSingle
			cur.WriteByte(ch)
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			cur.WriteByte(ch)
			continue
		}
		if !inSingle && !inDouble {
			switch ch {
			case '(':
				parenDepth++
			case ')':
				if parenDepth > 0 {
					parenDepth--
				}
			case '{':
				braceDepth++
			case '}':
				if braceDepth > 0 {
					braceDepth--
				}
			case '[':
				bracketDepth++
			case ']':
				if bracketDepth > 0 {
					bracketDepth--
				}
			case ' ', '\t', '\r', '\n':
				if parenDepth == 0 && braceDepth == 0 && bracketDepth == 0 {
					flush()
					continue
				}
			}
		}
		cur.WriteByte(ch)
	}

	if inSingle || inDouble {
		return nil, errors.New("powershell: unclosed quote in command")
	}

	flush()
	return words, nil
}
