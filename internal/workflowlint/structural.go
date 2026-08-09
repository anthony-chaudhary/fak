package workflowlint

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

type Finding struct {
	Path    string `json:"path"`
	Line    int    `json:"line"`
	Code    string `json:"code"`
	Message string `json:"message"`
}
type line struct {
	n, indent int
	text      string
}

// CheckTree walks .github/workflows rather than relying on a maintained file list.
func CheckTree(root string) ([]Finding, error) {
	dir := filepath.Join(root, ".github", "workflows")
	var files []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext == ".yml" || ext == ".yaml" {
			files = append(files, path)
		}
		return nil
	})
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var out []Finding
	for _, p := range files {
		f, e := CheckFile(p)
		if e != nil {
			return nil, e
		}
		out = append(out, f...)
	}
	return out, nil
}

func CheckFile(path string) ([]Finding, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	return Check(path, string(b)), nil
}

var keyRE = regexp.MustCompile(`^([A-Za-z0-9_.-]+):(?:\s|$)`)

func Check(path, src string) []Finding {
	var findings []Finding
	lines := structuralLines(src)
	add := func(l line, code, msg string) { findings = append(findings, Finding{path, l.n, code, msg}) }
	// All scanners consume structuralLines; block bodies and comments never enter here.
	siblings := map[int]int{}
	var jobsIndent = -1
	jobs := map[string]line{}
	var needs []struct {
		l   line
		ids []string
	}
	for _, l := range lines {
		if strings.Contains(l.text, "\t") {
			add(l, "tab-indent", "tabs are not valid workflow indentation")
		}
		t := strings.TrimSpace(l.text)
		if t == "" {
			continue
		}
		if unbalanced(t) {
			add(l, "unbalanced-delimiter", "unbalanced quote or bracket")
		}
		if looksLikeKey(t) && !keyRE.MatchString(t) {
			add(l, "missing-colon-space", "mapping key must use ': ' or end at ':'")
		}
		if m := keyRE.FindStringSubmatch(t); m != nil {
			if prev, ok := siblings[l.indent]; ok && prev != l.indent {
				add(l, "sibling-indent", "inconsistent sibling indentation")
			}
			siblings[l.indent] = l.indent
			key := m[1]
			if key == "jobs" {
				jobsIndent = l.indent
				continue
			}
			if jobsIndent >= 0 && l.indent == jobsIndent+2 {
				if old, ok := jobs[key]; ok {
					add(l, "duplicate-job", fmt.Sprintf("job %q duplicates line %d", key, old.n))
				} else {
					jobs[key] = l
				}
			}
			if key == "needs" {
				needs = append(needs, struct {
					l   line
					ids []string
				}{l, parseNeeds(strings.TrimSpace(strings.TrimPrefix(t, m[0])))})
			}
		}
	}
	for _, n := range needs {
		for _, id := range n.ids {
			if _, ok := jobs[id]; !ok {
				add(n.l, "unknown-needs", fmt.Sprintf("needs references unknown job %q", id))
			}
		}
	}
	sort.SliceStable(findings, func(i, j int) bool {
		if findings[i].Path != findings[j].Path {
			return findings[i].Path < findings[j].Path
		}
		if findings[i].Line != findings[j].Line {
			return findings[i].Line < findings[j].Line
		}
		return findings[i].Code < findings[j].Code
	})
	return findings
}

func structuralLines(src string) []line {
	s := bufio.NewScanner(strings.NewReader(src))
	n := 0
	blockIndent := -1
	var out []line
	for s.Scan() {
		n++
		raw := strings.TrimSuffix(s.Text(), "\r")
		indent := len(raw) - len(strings.TrimLeft(raw, " "))
		if blockIndent >= 0 {
			if strings.TrimSpace(raw) == "" || indent > blockIndent {
				continue
			}
			blockIndent = -1
		}
		text := stripComment(raw)
		trim := strings.TrimSpace(text)
		if trim == "" {
			continue
		}
		if regexp.MustCompile(`:\s*[>|][+-]?\s*$`).MatchString(trim) {
			blockIndent = indent
		}
		out = append(out, line{n, indent, text})
	}
	return out
}
func stripComment(s string) string {
	var q rune
	esc := false
	for i, r := range s {
		if esc {
			esc = false
			continue
		}
		if r == '\\' && q == '"' {
			esc = true
			continue
		}
		if q == 0 && (r == '"' || r == '\'') {
			q = r
			continue
		}
		if q == r {
			q = 0
			continue
		}
		if q == 0 && r == '#' && (i == 0 || s[i-1] == ' ') {
			return strings.TrimRight(s[:i], " ")
		}
	}
	return s
}
func unbalanced(s string) bool {
	stack := []rune{}
	var q rune
	esc := false
	for _, r := range s {
		if esc {
			esc = false
			continue
		}
		if r == '\\' && q == '"' {
			esc = true
			continue
		}
		if q != 0 {
			if r == q {
				q = 0
			}
			continue
		}
		if r == '"' || r == '\'' {
			q = r
			continue
		}
		if strings.ContainsRune("[{(", r) {
			stack = append(stack, r)
		}
		if strings.ContainsRune("]})", r) {
			if len(stack) == 0 {
				return true
			}
			o := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if (o == '[' && r != ']') || (o == '{' && r != '}') || (o == '(' && r != ')') {
				return true
			}
		}
	}
	return q != 0 || len(stack) != 0
}
func looksLikeKey(s string) bool {
	if strings.HasPrefix(s, "-") {
		return false
	}
	i := strings.IndexByte(s, ':')
	return i > 0 && !strings.ContainsAny(s[:i], " []{}")
}
func parseNeeds(s string) []string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "[]")
	fs := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' })
	var out []string
	for _, f := range fs {
		f = strings.Trim(f, "\"'")
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

// Actionlint runs the external semantic checker when installed. found=false is a clean skip.
func Actionlint(root string) (found bool, output []byte, err error) {
	p, e := exec.LookPath("actionlint")
	if e != nil {
		return false, nil, nil
	}
	c := exec.Command(p)
	c.Dir = root
	o, e := c.CombinedOutput()
	return true, o, e
}
