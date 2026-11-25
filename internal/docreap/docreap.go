// Package docreap ranks stale, orphaned docs and archives them without breaking links.
//
// Rank scores every tracked doc under docs/ and lists reap candidates worst-first.
// Archive moves a doc into docs/archive/, stamps the banner docs/archive/README.md
// requires and repoints every inbound link. Check verifies that every archived doc
// carries that banner.
//
// The rank score is a triage queue, not a verdict. Index files are excluded from
// inbound counts because they list everything and would hide orphans.
package docreap

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

var (
	indexFiles = []string{
		"INDEX.md", "llms.txt", "llms-full.txt",
		"docs/index.md", "docs/extended-index.md",
	}
	nonReferencePrefixes = []string{
		"docs/archive/", "docs/_witnesses/", "docs/research/inventory/",
		"docs/nightrun/", "docs/generated/",
	}
	// Front doors are kept current by their own gates; never queue them for reaping.
	protected = map[string]bool{
		"docs/index.md": true, "docs/FAQ.md": true, "docs/extended-index.md": true,
		"docs/glossary.md": true, "docs/architecture.md": true,
	}
	codeSuffixes = []string{".go", ".py", ".yml", ".yaml", ".ps1", ".sh", ".json", ".toml"}

	// Reasons is the closed vocabulary an archive banner may name.
	Reasons = []string{"superseded", "retired", "completed", "rejected", "orphaned", "stale-snapshot"}

	stampRE         = regexp.MustCompile(`<!--\s*[a-z0-9-]+:\s*(\d{4}-\d{2}-\d{2})\s*·\s*process:\s*(\S+)`)
	archiveBannerRE = regexp.MustCompile(`(?m)^> \*\*Archived (\d{4}-\d{2}-\d{2})\*\* · reason: ([\w-]+) · replacement: (.+?) · scope: (.+)$`)
	linkRE          = regexp.MustCompile(`(\]\()([^)\s#]+)((?:#[^)\s]*)?\))`)
	schemeRE        = regexp.MustCompile(`^[a-z]+:`)
	mdNameRE        = `[A-Za-z0-9_.-]+\.md`
)

const dateLayout = "2006-01-02"

func git(root string, args ...string) (string, error) {
	cmd := windowgate.Command("git", append([]string{"-C", root}, args...)...)
	out, err := cmd.Output()
	return string(out), err
}

// RepoRoot is the top level of the git checkout containing dir.
func RepoRoot(dir string) (string, error) {
	out, err := git(dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", err
	}
	return filepath.FromSlash(strings.TrimSpace(out)), nil
}

func lines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimRight(l, "\r"); l != "" {
			out = append(out, l)
		}
	}
	return out
}

func validReason(r string) bool {
	for _, x := range Reasons {
		if x == r {
			return true
		}
	}
	return false
}

func trackedDocs(root string, recursive bool) ([]string, error) {
	raw, err := git(root, "ls-files", "docs")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, p := range lines(raw) {
		if !strings.HasSuffix(p, ".md") || strings.HasPrefix(p, "docs/archive/") {
			continue
		}
		if !recursive && strings.Count(p, "/") != 1 {
			continue
		}
		out = append(out, p)
	}
	return out, nil
}

// LastCommitDates maps each path under docs/ to the date of its newest commit.
func LastCommitDates(root string) (map[string]string, error) {
	raw, err := git(root, "log", "--format=@%ad", "--date=short", "--name-only", "--", "docs")
	if err != nil {
		return nil, err
	}
	dates := map[string]string{}
	cur := ""
	for _, line := range lines(raw) {
		if strings.HasPrefix(line, "@") {
			cur = line[1:]
		} else if _, seen := dates[line]; !seen {
			dates[line] = cur
		}
	}
	return dates, nil
}

type refSet struct{ doc, code map[string]bool }

func excludes(paths ...string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		out = append(out, ":!"+p)
	}
	return out
}

// grepOut tolerates git grep's exit 1 on no match, which is an empty result here.
func grepOut(root string, args ...string) (string, error) {
	out, err := git(root, args...)
	var exit *exec.ExitError
	if errors.As(err, &exit) && exit.ExitCode() == 1 {
		return out, nil
	}
	return out, err
}

func inboundRefs(root string) (map[string]*refSet, error) {
	args := []string{"grep", "-I", "-o", "-E", mdNameRE, "--", "."}
	args = append(args, excludes(indexFiles...)...)
	args = append(args, excludes(nonReferencePrefixes...)...)
	args = append(args, ":!*.jsonl")
	raw, err := grepOut(root, args...)
	if err != nil {
		return nil, err
	}
	refs := map[string]*refSet{}
	for _, line := range lines(raw) {
		i := strings.LastIndex(line, ":")
		if i <= 0 {
			continue
		}
		src, name := line[:i], line[i+1:]
		if src[strings.LastIndex(src, "/")+1:] == name {
			continue
		}
		r := refs[name]
		if r == nil {
			r = &refSet{doc: map[string]bool{}, code: map[string]bool{}}
			refs[name] = r
		}
		isCode := false
		for _, suf := range codeSuffixes {
			if strings.HasSuffix(src, suf) {
				isCode = true
				break
			}
		}
		if isCode {
			r.code[src] = true
		} else {
			r.doc[src] = true
		}
	}
	return refs, nil
}

// Row is one reap candidate.
type Row struct {
	Path        string   `json:"path"`
	Score       float64  `json:"score"`
	LastCommit  string   `json:"last_commit"`
	AgeDays     int      `json:"age_days"`
	DocRefs     int      `json:"doc_refs"`
	CodeRefs    int      `json:"code_refs"`
	Appeal      *float64 `json:"appeal"`
	GeneratedBy *string  `json:"generated_by"`
	Action      string   `json:"action"`
	Reasons     []string `json:"reasons"`
}

// AppealFunc returns a doc's 0-100 prose score, or false when it has none.
type AppealFunc func(rel, text string) (float64, bool)

func daysBetween(today time.Time, date string) (int, bool) {
	d, err := time.Parse(dateLayout, date)
	if err != nil {
		return 0, false
	}
	return int(math.Round(today.Sub(d).Hours() / 24)), true
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }

// Rank scores tracked docs under docs/ worst-first. appeal may be nil.
func Rank(root string, recursive bool, today time.Time, appeal AppealFunc) ([]Row, error) {
	today = time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, time.UTC)
	dates, err := LastCommitDates(root)
	if err != nil {
		return nil, err
	}
	refs, err := inboundRefs(root)
	if err != nil {
		return nil, err
	}
	docs, err := trackedDocs(root, recursive)
	if err != nil {
		return nil, err
	}
	var rows []Row
	for _, rel := range docs {
		if protected[rel] {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		text := string(raw)
		docRefs, codeRefs := 0, 0
		if r := refs[rel[strings.LastIndex(rel, "/")+1:]]; r != nil {
			docRefs, codeRefs = len(r.doc), len(r.code)
		}
		last := dates[rel]
		age, _ := daysBetween(today, last)
		stamp := stampRE.FindStringSubmatch(text)
		stampAge, stale := 0, false
		if stamp != nil {
			if a, ok := daysBetween(today, stamp[1]); ok {
				stampAge, stale = a, a > 60
			}
		}

		row := Row{
			Path: rel, LastCommit: last, AgeDays: age, DocRefs: docRefs, CodeRefs: codeRefs,
			Action: "archive", Reasons: []string{},
		}
		score := math.Min(float64(age)/180.0, 1.0) * 35
		switch docRefs + codeRefs {
		case 0:
			score += 25
		case 1:
			score += 15
		case 2:
			score += 8
		}
		if appeal != nil {
			if a, ok := appeal(rel, text); ok {
				score += (100.0 - a) / 100.0 * 25
				rounded := round1(a)
				row.Appeal = &rounded
			}
		}
		if stale {
			score += 15
		}
		if age >= 90 {
			row.Reasons = append(row.Reasons, fmt.Sprintf("untouched %dd", age))
		}
		if docRefs+codeRefs == 0 {
			row.Reasons = append(row.Reasons, "orphan")
		}
		if stale {
			row.Reasons = append(row.Reasons, fmt.Sprintf("snapshot %dd old (regen: %s)", stampAge, stamp[2]))
		}
		if row.Appeal != nil && *row.Appeal < 70 {
			row.Reasons = append(row.Reasons, fmt.Sprintf("appeal %.0f", *row.Appeal))
		}
		switch {
		case stamp != nil:
			row.Action = "regenerate"
			row.GeneratedBy = &stamp[2]
		case codeRefs > 0:
			row.Action = "repoint+archive"
		}
		row.Score = round1(score)
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Score != rows[j].Score {
			return rows[i].Score > rows[j].Score
		}
		return rows[i].Path < rows[j].Path
	})
	return rows, nil
}

func relSlash(target, start string) string {
	r, err := filepath.Rel(start, target)
	if err != nil {
		return filepath.ToSlash(target)
	}
	return filepath.ToSlash(r)
}

func external(target string) bool {
	return schemeRE.MatchString(target) || strings.HasPrefix(target, "/")
}

func rewriteLinks(text, fileDir, oldPath, newPath string) string {
	return linkRE.ReplaceAllStringFunc(text, func(m string) string {
		g := linkRE.FindStringSubmatch(m)
		if external(g[2]) {
			return m
		}
		if filepath.Join(fileDir, filepath.FromSlash(g[2])) == oldPath {
			return g[1] + relSlash(newPath, fileDir) + g[3]
		}
		return m
	})
}

func rebaseOwnLinks(text, oldDir, newDir string) string {
	return linkRE.ReplaceAllStringFunc(text, func(m string) string {
		g := linkRE.FindStringSubmatch(m)
		if external(g[2]) {
			return m
		}
		return g[1] + relSlash(filepath.Join(oldDir, filepath.FromSlash(g[2])), newDir) + g[3]
	})
}

func wordRune(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) }

// replaceBarePath swaps repo-relative mentions of old for new, leaving alone a match
// that is only the tail of a longer path or the head of a longer name.
func replaceBarePath(text, old, new string) string {
	var b strings.Builder
	for {
		i := strings.Index(text, old)
		if i < 0 {
			b.WriteString(text)
			return b.String()
		}
		end := i + len(old)
		prev, _ := utf8.DecodeLastRuneInString(text[:i])
		next, _ := utf8.DecodeRuneInString(text[end:])
		joinedBefore := i > 0 && (wordRune(prev) || prev == '.' || prev == '/' || prev == '-')
		joinedAfter := end < len(text) && (wordRune(next) || next == '-')
		b.WriteString(text[:i])
		if joinedBefore || joinedAfter {
			b.WriteString(old)
		} else {
			b.WriteString(new)
		}
		text = text[end:]
	}
}

func insertBanner(text, banner string) string {
	nl := "\n"
	if strings.HasPrefix(text, "---\r\n") {
		nl = "\r\n"
	}
	if strings.HasPrefix(text, "---"+nl) {
		fence := nl + "---" + nl
		if end := strings.Index(text[3:], fence); end != -1 {
			cut := 3 + end + len(fence)
			return text[:cut] + nl + banner + nl + text[cut:]
		}
	}
	return banner + nl + nl + text
}

// ArchiveOptions describes one archive move.
type ArchiveOptions struct {
	Doc       string
	Reason    string
	Successor string
	Scope     string
	Today     time.Time
	DryRun    bool
}

// ErrRefused marks an archive the tool declined for a stated reason.
var ErrRefused = errors.New("refuse")

func refuse(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrRefused, fmt.Sprintf(format, a...))
}

func exists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}

// Archive moves a doc into docs/archive/, stamps its banner and repoints inbound links.
func Archive(root string, o ArchiveOptions, w io.Writer) error {
	rel := filepath.ToSlash(o.Doc)
	oldPath := filepath.Join(root, filepath.FromSlash(rel))
	if st, err := os.Stat(oldPath); err != nil || !st.Mode().IsRegular() {
		return refuse("%s does not exist", rel)
	}
	if !validReason(o.Reason) {
		return refuse("reason must be one of %s", strings.Join(Reasons, ", "))
	}
	name := filepath.Base(oldPath)
	newPosix := "docs/archive/" + name
	newPath := filepath.Join(root, "docs", "archive", name)
	if exists(newPath) {
		return refuse("%s already exists", newPosix)
	}
	successorPath := filepath.Join(root, filepath.FromSlash(o.Successor))
	if o.Successor != "none" && !exists(successorPath) {
		return refuse("successor %s does not exist", o.Successor)
	}
	repl := "none"
	if o.Successor != "none" {
		repl = fmt.Sprintf("[%s](%s)", o.Successor, relSlash(successorPath, filepath.Dir(newPath)))
	}
	banner := fmt.Sprintf("> **Archived %s** · reason: %s · replacement: %s · scope: %s",
		o.Today.Format(dateLayout), o.Reason, repl, o.Scope)

	// Evidence records and generated files keep their original paths. Archived docs
	// are repointed too so links between archived docs survive.
	args := []string{"grep", "-I", "-l", "-F", name, "--", "."}
	for _, p := range nonReferencePrefixes {
		if p != "docs/archive/" {
			args = append(args, ":!"+p)
		}
	}
	args = append(args, ":!*.jsonl", ":!llms-full.txt")
	hits, err := grepOut(root, args...)
	if err != nil {
		return err
	}
	var touched []string
	for _, src := range lines(hits) {
		if src == rel {
			continue
		}
		p := filepath.Join(root, filepath.FromSlash(src))
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		text := string(raw)
		updated := replaceBarePath(rewriteLinks(text, filepath.Dir(p), oldPath, newPath), rel, newPosix)
		if updated == text {
			continue
		}
		touched = append(touched, src)
		if !o.DryRun {
			if err := os.WriteFile(p, []byte(updated), 0o644); err != nil {
				return err
			}
		}
	}

	raw, err := os.ReadFile(oldPath)
	if err != nil {
		return err
	}
	body := insertBanner(rebaseOwnLinks(string(raw), filepath.Dir(oldPath), filepath.Dir(newPath)), banner)
	fmt.Fprintf(w, "archive %s -> %s (%s); repointed %d file(s)\n", rel, newPosix, o.Reason, len(touched))
	for _, t := range touched {
		fmt.Fprintf(w, "  repoint %s\n", t)
	}
	if o.DryRun {
		return nil
	}
	if _, err := git(root, "mv", rel, newPosix); err != nil {
		return err
	}
	return os.WriteFile(newPath, []byte(body), 0o644)
}

// Check lists archived docs that lack the archive banner.
func Check(root string) ([]string, error) {
	raw, err := git(root, "ls-files", "docs/archive")
	if err != nil {
		return nil, err
	}
	var bad []string
	for _, rel := range lines(raw) {
		if !strings.HasSuffix(rel, ".md") || strings.HasSuffix(rel, "/README.md") {
			continue
		}
		text, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, err
		}
		m := archiveBannerRE.FindSubmatch(text)
		if m == nil || !validReason(string(m[2])) {
			bad = append(bad, rel)
		}
	}
	return bad, nil
}
