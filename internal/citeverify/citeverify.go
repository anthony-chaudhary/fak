// Package citeverify mechanically checks source-code path:line citations.
package citeverify

import (
	"bufio"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"

	"strconv"
	"strings"
)

// Status is the conservative result of checking one or more source citations.
type Status string

const (
	Supports    Status = "supports"
	Contradicts Status = "contradicts"
	Unknown     Status = "unknown"
	Mixed       Status = "mixed"
)

const maxSourceBytes int64 = 1 << 20

var (
	citationRE = regexp.MustCompile(`(?m)([A-Za-z]:[\\/][^\s:]+|(?:\.?\.?[\\/])?[^\s:]+):(\d+)`)
	quotedRE   = regexp.MustCompile("`([^`]+)`")
	identRE    = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_.-]*`)
)

// Verify opens every path:line citation in evidence under root and compares the
// cited line with symbols named by claim. Strong answers are deliberately
// asymmetric: an out-of-range line or a resolved, non-empty line containing no
// claimed symbol contradicts; ambiguity and every unsafe/unreadable input are
// unknown. A mixture of supports and contradicts returns Mixed.
func Verify(claim string, evidence []string, root string) Status {
	symbols := claimSymbols(claim)
	hasMatch, hasConflict := false, false
	for _, item := range evidence {
		matches := citationRE.FindAllStringSubmatch(item, -1)
		for _, match := range matches {
			lineNo, err := strconv.Atoi(match[2])
			if err != nil {
				continue
			}
			line, status := readCitation(root, match[1], lineNo)
			switch status {
			case Contradicts:
				hasConflict = true
			case Supports:
				if len(symbols) == 0 || strings.TrimSpace(line) == "" {
					continue
				}
				if containsSymbol(line, symbols) {
					hasMatch = true
				} else {
					hasConflict = true
				}
			}
		}
	}
	if hasMatch && hasConflict {
		return Mixed
	}
	if hasConflict {
		return Contradicts
	}
	if hasMatch {
		return Supports
	}
	return Unknown
}

func readCitation(root, cited string, lineNo int) (string, Status) {
	if lineNo < 1 {
		return "", Contradicts
	}
	path, ok := resolve(root, cited)
	if !ok {
		return "", Unknown
	}
	f, err := os.Open(path)
	if err != nil {
		return "", Unknown
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || info.Size() > maxSourceBytes {
		return "", Unknown
	}
	reader := bufio.NewReader(io.LimitReader(f, maxSourceBytes+1))
	for n := 1; ; n++ {
		line, err := reader.ReadString('\n')
		if n == lineNo {
			if errors.Is(err, io.EOF) || err == nil {
				return strings.TrimRight(line, "\r\n"), Supports
			}
			return "", Unknown
		}
		if errors.Is(err, io.EOF) {
			return "", Contradicts
		}
		if err != nil {
			return "", Unknown
		}
	}
}

func resolve(root, cited string) (string, bool) {
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", false
	}
	cited = filepath.FromSlash(strings.ReplaceAll(cited, `\`, `/`))
	if unsafePath(cited) || !codeExtension(cited) {
		return "", false
	}
	var candidates []string
	if filepath.IsAbs(cited) {
		candidates = append(candidates, cited)
	} else {
		direct := filepath.Join(rootAbs, cited)
		if regularWithin(rootAbs, direct) {
			candidates = append(candidates, direct)
		} else if filepath.Base(cited) == cited {
			_ = filepath.WalkDir(rootAbs, func(path string, d os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return nil
				}
				if d.Type()&os.ModeSymlink != 0 && d.IsDir() {
					return filepath.SkipDir
				}
				if d.IsDir() && path != rootAbs && strings.HasPrefix(d.Name(), ".") {
					return filepath.SkipDir
				}
				if !d.IsDir() && d.Name() == cited && regularWithin(rootAbs, path) {
					candidates = append(candidates, path)
				}
				return nil
			})
		}
	}
	if len(candidates) != 1 || !regularWithin(rootAbs, candidates[0]) {
		return "", false
	}
	return candidates[0], true
}

func regularWithin(root, path string) bool {
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return false
	}
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(realRoot, realPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	info, err := os.Stat(realPath)
	return err == nil && info.Mode().IsRegular()
}

func unsafePath(path string) bool {
	for _, part := range strings.FieldsFunc(filepath.ToSlash(path), func(r rune) bool { return r == '/' }) {
		low := strings.ToLower(part)
		if strings.HasPrefix(part, ".") || low == "id_rsa" || low == "id_ed25519" || strings.Contains(low, "secret") || strings.Contains(low, "credential") || strings.HasSuffix(low, ".pem") || strings.HasSuffix(low, ".key") {
			return true
		}
	}
	return false
}

func codeExtension(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".java", ".kt", ".js", ".jsx", ".ts", ".tsx", ".py", ".rb", ".sh", ".ps1", ".cs", ".swift":
		return true
	default:
		return false
	}
}

func claimSymbols(claim string) []string {
	var symbols []string
	for _, m := range quotedRE.FindAllStringSubmatch(claim, -1) {
		for _, symbol := range identRE.FindAllString(m[1], -1) {
			symbols = appendUnique(symbols, symbol)
		}
	}
	if len(symbols) == 0 {
		for _, symbol := range identRE.FindAllString(claim, -1) {
			if len(symbol) > 2 {
				symbols = appendUnique(symbols, symbol)
			}
		}
	}
	return symbols
}

func appendUnique(xs []string, x string) []string {
	for _, old := range xs {
		if old == x {
			return xs
		}
	}
	return append(xs, x)
}
func containsSymbol(line string, symbols []string) bool {
	for _, s := range symbols {
		if strings.Contains(line, s) {
			return true
		}
	}
	return false
}
