package codetools

import (
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ensureGoTmpDir resolves and creates the GOTMPDIR directory, pointing to
// <root>/_scratch/gotmp, falling back to an OS temporary directory if scratch cannot be created.
func (t *Toolset) ensureGoTmpDir() string {
	gotmp := filepath.Join(t.root, "_scratch", "gotmp")
	if err := os.MkdirAll(gotmp, 0o755); err == nil {
		return gotmp
	}
	fallback := filepath.Join(os.TempDir(), "fak-gotmp")
	_ = os.MkdirAll(fallback, 0o755)
	return fallback
}

// enforceGoTmpEnv ensures GOTMPDIR is set in the environment slice.
func enforceGoTmpEnv(base []string, gotmpDir string) []string {
	out := make([]string, 0, len(base)+1)
	for _, entry := range base {
		key, _, _ := strings.Cut(entry, "=")
		if strings.EqualFold(key, "GOTMPDIR") {
			continue
		}
		out = append(out, entry)
	}
	out = append(out, "GOTMPDIR="+gotmpDir)
	return out
}

type commandPart struct {
	text string
	isOp bool
}

type cmdToken struct {
	raw string
	val string
}

func splitCommandPipeline(s string) []commandPart {
	var parts []commandPart
	var cur strings.Builder
	inSingle := false
	inDouble := false
	escaped := false

	flush := func() {
		if cur.Len() > 0 {
			parts = append(parts, commandPart{text: cur.String(), isOp: false})
			cur.Reset()
		}
	}

	i := 0
	for i < len(s) {
		ch := s[i]
		if escaped {
			cur.WriteByte(ch)
			escaped = false
			i++
			continue
		}
		if ch == '\\' && !inSingle {
			escaped = true
			cur.WriteByte(ch)
			i++
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			cur.WriteByte(ch)
			i++
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			cur.WriteByte(ch)
			i++
			continue
		}
		if !inSingle && !inDouble {
			if i+1 < len(s) && (s[i:i+2] == "&&" || s[i:i+2] == "||") {
				flush()
				parts = append(parts, commandPart{text: s[i : i+2], isOp: true})
				i += 2
				continue
			}
			if ch == ';' || ch == '|' || ch == '&' || ch == '\n' {
				flush()
				parts = append(parts, commandPart{text: string(ch), isOp: true})
				i++
				continue
			}
		}
		cur.WriteByte(ch)
		i++
	}
	flush()
	return parts
}

func splitWhitespace(s string) (leading, core, trailing string) {
	trimmedLeft := strings.TrimLeft(s, " \t\r\n")
	leading = s[:len(s)-len(trimmedLeft)]
	trimmed := strings.TrimRight(trimmedLeft, " \t\r\n")
	trailing = trimmedLeft[len(trimmed):]
	core = trimmed
	return
}

func tokenizeCommand(s string) []cmdToken {
	var tokens []cmdToken
	var raw strings.Builder
	var val strings.Builder
	inSingle := false
	inDouble := false
	escaped := false
	inToken := false

	flush := func() {
		if inToken {
			tokens = append(tokens, cmdToken{raw: raw.String(), val: val.String()})
			raw.Reset()
			val.Reset()
			inToken = false
		}
	}

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			raw.WriteByte(ch)
			val.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && !inSingle {
			escaped = true
			raw.WriteByte(ch)
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
			raw.WriteByte(ch)
			inToken = true
			continue
		}
		if ch == '"' && !inSingle {
			inDouble = !inDouble
			raw.WriteByte(ch)
			inToken = true
			continue
		}
		if !inSingle && !inDouble && (ch == ' ' || ch == '\t') {
			flush()
			continue
		}
		inToken = true
		raw.WriteByte(ch)
		val.WriteByte(ch)
	}
	flush()
	return tokens
}

func reconstructTokens(tokens []cmdToken) string {
	var sb strings.Builder
	for i, t := range tokens {
		if i > 0 {
			sb.WriteByte(' ')
		}
		sb.WriteString(t.raw)
	}
	return sb.String()
}

var flagsWithArg = map[string]bool{
	"-o": true, "--o": true,
	"-cpuprofile": true, "--cpuprofile": true,
	"-memprofile": true, "--memprofile": true,
	"-coverprofile": true, "--coverprofile": true,
	"-blockprofile": true, "--blockprofile": true,
	"-mutexprofile": true, "--mutexprofile": true,
	"-trace": true, "--trace": true,
	"-run": true, "--run": true,
	"-skip": true, "--skip": true,
	"-bench": true, "--bench": true,
	"-benchtime": true, "--benchtime": true,
	"-count": true, "--count": true,
	"-timeout": true, "--timeout": true,
	"-parallel": true, "--parallel": true,
	"-tags": true, "--tags": true,
	"-vet": true, "--vet": true,
	"-exec": true, "--exec": true,
	"-shuffle": true, "--shuffle": true,
	"-ldflags": true, "--ldflags": true,
	"-gcflags": true, "--gcflags": true,
	"-asmflags": true, "--asmflags": true,
	"-coverpkg": true, "--coverpkg": true,
	"-outputdir": true, "--outputdir": true,
}

func (t *Toolset) rewriteGoTestTokens(tokens []cmdToken, cwd, gotmpDir string) []cmdToken {
	cmdIdx := 0
	for cmdIdx < len(tokens) {
		tokVal := tokens[cmdIdx].val
		if strings.Contains(tokVal, "=") && !strings.HasPrefix(tokVal, "-") {
			if strings.HasPrefix(strings.ToUpper(tokVal), "GOTMPDIR=") {
				tokens[cmdIdx] = cmdToken{
					raw: "GOTMPDIR=" + quoteIfNecessary(gotmpDir),
					val: "GOTMPDIR=" + gotmpDir,
				}
			}
			cmdIdx++
			continue
		}
		break
	}

	if cmdIdx >= len(tokens) {
		return tokens
	}
	exe := filepath.Base(tokens[cmdIdx].val)
	if !strings.EqualFold(exe, "go") && !strings.EqualFold(exe, "go.exe") {
		return tokens
	}
	if cmdIdx+1 >= len(tokens) || tokens[cmdIdx+1].val != "test" {
		return tokens
	}

	var hasCFlag bool
	oTokenIdx := -1
	oArgIdx := -1
	var targets []string

	profFlagNames := []string{
		"-cpuprofile", "--cpuprofile",
		"-memprofile", "--memprofile",
		"-coverprofile", "--coverprofile",
		"-blockprofile", "--blockprofile",
		"-mutexprofile", "--mutexprofile",
		"-trace", "--trace",
	}

	i := cmdIdx + 2
	for i < len(tokens) {
		tok := tokens[i]
		if tok.val == "--" {
			break
		}
		if tok.val == "-c" || tok.val == "--c" || tok.val == "-c=true" || tok.val == "--c=true" {
			hasCFlag = true
			i++
			continue
		}
		if tok.val == "-o" || tok.val == "--o" {
			oTokenIdx = i
			if i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1].val, "-") {
				i++
				oArgIdx = i
			}
			i++
			continue
		}
		if strings.HasPrefix(tok.val, "-o=") || strings.HasPrefix(tok.val, "--o=") {
			oTokenIdx = i
			i++
			continue
		}
		isProf := false
		for _, prof := range profFlagNames {
			if tok.val == prof {
				isProf = true
				if i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1].val, "-") {
					pathVal := tokens[i+1].val
					if isBarePath(pathVal, t.root) {
						newTarget := filepath.Join(gotmpDir, filepath.Base(pathVal))
						tokens[i+1] = cmdToken{raw: quoteIfNecessary(newTarget), val: newTarget}
					}
					i++
				}
				break
			} else if strings.HasPrefix(tok.val, prof+"=") {
				isProf = true
				prefix := tok.val[:len(prof)+1]
				pathVal := tok.val[len(prof)+1:]
				if isBarePath(pathVal, t.root) {
					newTarget := filepath.Join(gotmpDir, filepath.Base(pathVal))
					tokens[i] = cmdToken{
						raw: prefix + quoteIfNecessary(newTarget),
						val: prefix + newTarget,
					}
				}
				break
			}
		}
		if isProf {
			i++
			continue
		}
		if strings.HasPrefix(tok.val, "-") {
			if !strings.Contains(tok.val, "=") && flagsWithArg[tok.val] {
				if i+1 < len(tokens) && !strings.HasPrefix(tokens[i+1].val, "-") {
					i++
				}
			}
			i++
			continue
		}
		targets = append(targets, tok.val)
		i++
	}

	if oTokenIdx != -1 {
		if oArgIdx != -1 {
			pathVal := tokens[oArgIdx].val
			if isBarePath(pathVal, t.root) {
				newTarget := filepath.Join(gotmpDir, filepath.Base(pathVal))
				tokens[oArgIdx] = cmdToken{raw: quoteIfNecessary(newTarget), val: newTarget}
			}
		} else {
			eqIdx := strings.Index(tokens[oTokenIdx].val, "=")
			if eqIdx != -1 {
				prefix := tokens[oTokenIdx].val[:eqIdx+1]
				pathVal := tokens[oTokenIdx].val[eqIdx+1:]
				if isBarePath(pathVal, t.root) {
					newTarget := filepath.Join(gotmpDir, filepath.Base(pathVal))
					tokens[oTokenIdx] = cmdToken{
						raw: prefix + quoteIfNecessary(newTarget),
						val: prefix + newTarget,
					}
				}
			}
		}
	} else if hasCFlag {
		targetPkg := ""
		if len(targets) > 0 {
			targetPkg = targets[0]
		}
		pkgName := detectPackageName(cwd, targetPkg)
		outBinary := filepath.Join(gotmpDir, pkgName+".test")
		insertTokens := []cmdToken{
			{raw: "-o", val: "-o"},
			{raw: quoteIfNecessary(outBinary), val: outBinary},
		}
		tokens = append(tokens[:cmdIdx+2], append(insertTokens, tokens[cmdIdx+2:]...)...)
	}

	return tokens
}

func (t *Toolset) rewriteCommandForContainment(command, cwd, gotmpDir string) string {
	parts := splitCommandPipeline(command)
	if len(parts) == 0 {
		return command
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.isOp {
			sb.WriteString(p.text)
			continue
		}
		leading, core, trailing := splitWhitespace(p.text)
		if core == "" {
			sb.WriteString(p.text)
			continue
		}
		tokens := tokenizeCommand(core)
		rewrittenTokens := t.rewriteGoTestTokens(tokens, cwd, gotmpDir)
		sb.WriteString(leading)
		sb.WriteString(reconstructTokens(rewrittenTokens))
		sb.WriteString(trailing)
	}
	return sb.String()
}

func isBarePath(p, root string) bool {
	p = strings.Trim(p, `"'`)
	if p == "" {
		return false
	}
	if filepath.IsAbs(p) {
		rel, err := filepath.Rel(root, p)
		if err == nil && !strings.HasPrefix(rel, "..") {
			cleanRel := filepath.ToSlash(rel)
			if !strings.HasPrefix(cleanRel, "_scratch/") && cleanRel != "_scratch" {
				return true
			}
		}
		return false
	}
	clean := filepath.ToSlash(filepath.Clean(p))
	if strings.HasPrefix(clean, "_scratch/") || clean == "_scratch" {
		return false
	}
	return true
}

func detectPackageName(dir, targetPkg string) string {
	cleanTarget := strings.TrimSuffix(strings.TrimSuffix(targetPkg, "/..."), `\...`)
	if cleanTarget != "" && cleanTarget != "." {
		clean := filepath.Base(filepath.Clean(cleanTarget))
		if clean != "." && clean != "/" && clean != "\\" && clean != "" {
			return sanitizeName(clean)
		}
	}
	if entries, err := os.ReadDir(dir); err == nil {
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
				content, err := os.ReadFile(filepath.Join(dir, e.Name()))
				if err != nil {
					continue
				}
				for _, line := range strings.Split(string(content), "\n") {
					line = strings.TrimSpace(line)
					if strings.HasPrefix(line, "package ") {
						pkg := strings.TrimSpace(strings.TrimPrefix(line, "package "))
						pkg = strings.TrimSuffix(pkg, ";")
						if pkg != "" && pkg != "main" {
							return sanitizeName(pkg)
						}
						if pkg == "main" {
							if modBytes, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
								for _, modLine := range strings.Split(string(modBytes), "\n") {
									modLine = strings.TrimSpace(modLine)
									if strings.HasPrefix(modLine, "module ") {
										m := strings.TrimSpace(strings.TrimPrefix(modLine, "module "))
										return sanitizeName(filepath.Base(m))
									}
								}
							}
							return "main"
						}
					}
				}
			}
		}
	}
	base := filepath.Base(dir)
	if base != "" && base != "." && base != "/" && base != "\\" {
		return sanitizeName(base)
	}
	return "test"
}

func sanitizeName(name string) string {
	s := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, name)
	if s == "" {
		return "test"
	}
	return s
}

func quoteIfNecessary(s string) string {
	if strings.ContainsAny(s, " \t\r\n\"") {
		return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
	}
	return s
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

// containmentCleanup performs a post-execution scan of the working tree outside
// _scratch and .git. Any *.test or *.test.exe binary found is cleaned and moved
// to gotmpDir.
func (t *Toolset) containmentCleanup(gotmpDir string) {
	if fi, err := os.Stat(t.root); err != nil || !fi.IsDir() {
		return
	}
	_ = filepath.WalkDir(t.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if name == ".git" || name == "_scratch" || (path != t.root && strings.HasPrefix(name, ".")) {
				return fs.SkipDir
			}
			return nil
		}
		lower := strings.ToLower(name)
		if strings.HasSuffix(lower, ".test") || strings.HasSuffix(lower, ".test.exe") {
			dest := filepath.Join(gotmpDir, name)
			if filepath.Clean(dest) != filepath.Clean(path) {
				if err := os.Rename(path, dest); err != nil {
					_ = os.Remove(dest)
					if err := os.Rename(path, dest); err != nil {
						if copyFile(path, dest) == nil {
							_ = os.Remove(path)
						} else {
							_ = os.Remove(path)
						}
					}
				}
			}
		}
		return nil
	})
}
