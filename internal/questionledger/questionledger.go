// Package questionledger is the deterministic labeling authority for the
// /question-loop skill's ledger, docs/questions/asked.jsonl.
//
// The skill's whole value rests on the ledger being TRUSTWORTHY: unambiguous
// ids, a closed category vocabulary, a closed status lifecycle, and no leaked
// host/path/email. Prose in a SKILL.md cannot enforce any of that; this package
// can. It is the shell/core split the repo insists on, with the *rules* here in
// code (LintRows/NextID/DedupeMatch/Stats over in-memory rows, plus a hermetic
// Run that owns the file I/O and the gh seam) and the *question generation* left
// to the agent.
//
// This is the Go port of the retired tools/question_ledger.py (fak pythongate):
// byte-faithful subcommands, exit codes, and row schema. The gh subprocess for
// `ensure-label` is behind the GHFunc seam so the whole CLI is unit-testable.
//
// Row schema (exactly these 8 keys — kept in lockstep with docs/questions/README.md):
//
//	id       "Q-YYYYMMDD-NNN"  unique
//	ts       ISO-8601 UTC      e.g. 2026-07-08T00:00:00Z
//	category UNASKED|AFRAID|CONTRARIAN|STEELMAN
//	target   non-empty string  (the claim/area interrogated)
//	question non-empty string, ends with '?'
//	why      non-empty string
//	status   open|answered|ticketed|dismissed
//	ticket   null unless status==ticketed, then a positive int
package questionledger

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

// DefaultLedger is the ledger path both loops of the /question-loop skill share.
const DefaultLedger = "docs/questions/asked.jsonl"

// The closed vocabularies the ledger is held to.
var (
	categories = map[string]bool{"UNASKED": true, "AFRAID": true, "CONTRARIAN": true, "STEELMAN": true}
	statuses   = map[string]bool{"open": true, "answered": true, "ticketed": true, "dismissed": true}
	keys       = []string{"id", "ts", "category", "target", "question", "why", "status", "ticket"}
)

var (
	idRE   = regexp.MustCompile(`^Q-(\d{8})-(\d{3})$`)
	tsRE   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$`)
	dateRE = regexp.MustCompile(`^\d{8}$`)

	reNonAlnumSpace = regexp.MustCompile(`[^a-z0-9 ]`)
	reWhitespace    = regexp.MustCompile(`\s+`)
)

// leakREs mirror the PUBLIC_LEAK class the fleet guards: a question or its
// rationale must never carry a machine-absolute path, a home dir, or an email.
var leakREs = []*regexp.Regexp{
	regexp.MustCompile(`[A-Za-z]:\\`),                                    // C:\ style Windows path
	regexp.MustCompile(`(?:^|[\s(])/(?:home|Users|root|mnt|tmp)/`),       // POSIX absolute
	regexp.MustCompile(`\\\\[A-Za-z0-9._-]+\\`),                          // UNC \\host\share
	regexp.MustCompile(`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`), // email
}

// Label provenance for the gh label ensure-label creates.
const (
	Label      = "question-loop"
	LabelColor = "8a63d2"
	LabelDesc  = "Filed by /question-loop from a witnessed provocative question"
)

// Row is one parsed ledger line: (lineno, decoded value, raw text). Value is nil
// when the line failed to decode (or decoded to a JSON null) — matching the
// Python read_rows, which folds both cases to None.
type Row struct {
	Line   int
	Raw    string
	parsed bool        // json.Unmarshal returned no error
	value  interface{} // decoded value; nil on error or literal null
}

// object returns the row as a JSON object and whether it is one.
func (r Row) object() (map[string]interface{}, bool) {
	m, ok := r.value.(map[string]interface{})
	return m, ok
}

// ParseRows returns a Row for every non-blank line of text, mirroring the
// Python read_rows contract: a decode error and a literal null both yield a
// value-nil row.
func ParseRows(text string) []Row {
	var out []Row
	for i, raw := range strings.Split(text, "\n") {
		raw = strings.TrimRight(raw, "\r")
		if strings.TrimSpace(raw) == "" {
			continue
		}
		var v interface{}
		err := json.Unmarshal([]byte(raw), &v)
		out = append(out, Row{Line: i + 1, Raw: raw, parsed: err == nil, value: v})
	}
	return out
}

func norm(s string) string {
	s = strings.ToLower(s)
	s = reNonAlnumSpace.ReplaceAllString(s, "")
	s = reWhitespace.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func leakHit(text string) (string, bool) {
	for _, rx := range leakREs {
		if loc := rx.FindStringIndex(text); loc != nil {
			return text[loc[0]:loc[1]], true
		}
	}
	return "", false
}

// str pulls key k off an object as a string ("" if absent or not a string).
func str(obj map[string]interface{}, k string) string {
	s, _ := obj[k].(string)
	return s
}

// LintRows returns a list of human-readable violation strings (empty == clean).
func LintRows(rows []Row) []string {
	var problems []string
	seenIDs := map[string]int{}
	for _, r := range rows {
		tag := fmt.Sprintf("line %d", r.Line)
		if r.value == nil {
			problems = append(problems, fmt.Sprintf("%s: not valid JSON", tag))
			continue
		}
		obj, isObj := r.object()
		if !isObj {
			problems = append(problems, fmt.Sprintf("%s: row is not a JSON object", tag))
			continue
		}
		if missing := missingKeys(obj); len(missing) > 0 {
			problems = append(problems, fmt.Sprintf("%s: missing key(s) %v", tag, missing))
		}
		if extra := extraKeys(obj); len(extra) > 0 {
			problems = append(problems, fmt.Sprintf("%s: unexpected key(s) %v", tag, extra))
		}
		if rid, ok := obj["id"].(string); ok {
			tag = fmt.Sprintf("%s (line %d)", rid, r.Line)
			if !idRE.MatchString(rid) {
				problems = append(problems, fmt.Sprintf("%s: id must match Q-YYYYMMDD-NNN", tag))
			}
			if prev, dup := seenIDs[rid]; dup {
				problems = append(problems, fmt.Sprintf("%s: duplicate id (also line %d)", tag, prev))
			}
			seenIDs[rid] = r.Line
		} else {
			problems = append(problems, fmt.Sprintf("%s: id missing or not a string", tag))
		}
		if !tsRE.MatchString(str(obj, "ts")) {
			problems = append(problems, fmt.Sprintf("%s: ts must be ISO-8601 UTC (…Z)", tag))
		}
		if cat, ok := obj["category"].(string); !ok || !categories[cat] {
			problems = append(problems, fmt.Sprintf("%s: category %#v not in %v", tag, obj["category"], sortedKeys(categories)))
		}
		status, _ := obj["status"].(string)
		if !statuses[status] {
			problems = append(problems, fmt.Sprintf("%s: status %#v not in %v", tag, obj["status"], sortedKeys(statuses)))
		}
		for _, field := range []string{"target", "why"} {
			if v := str(obj, field); strings.TrimSpace(v) == "" {
				problems = append(problems, fmt.Sprintf("%s: %s must be a non-empty string", tag, field))
			}
		}
		if q := str(obj, "question"); strings.TrimSpace(q) == "" {
			problems = append(problems, fmt.Sprintf("%s: question must be a non-empty string", tag))
		} else if !strings.HasSuffix(strings.TrimRight(q, " \t\n\r"), "?") {
			problems = append(problems, fmt.Sprintf("%s: question must end with '?'", tag))
		}
		// status/ticket coherence. JSON numbers decode to float64; a positive
		// integral value stands in for the Python "positive int" (a JSON bool
		// decodes to bool and a null to nil, so both correctly fail the guard).
		ticket := obj["ticket"]
		if status == "ticketed" {
			if !isPositiveInt(ticket) {
				problems = append(problems, fmt.Sprintf("%s: status 'ticketed' requires a positive int ticket, got %#v", tag, ticket))
			}
		} else if ticket != nil {
			problems = append(problems, fmt.Sprintf("%s: ticket must be null unless status is 'ticketed' (got %#v)", tag, ticket))
		}
		for _, field := range []string{"question", "why", "target"} {
			if hit, ok := leakHit(str(obj, field)); ok {
				problems = append(problems, fmt.Sprintf("%s: %s leaks %q (absolute path/host/email)", tag, field, hit))
			}
		}
	}
	return problems
}

func isPositiveInt(v interface{}) bool {
	f, ok := v.(float64)
	return ok && f > 0 && f == math.Trunc(f)
}

func missingKeys(obj map[string]interface{}) []string {
	var out []string
	for _, k := range keys {
		if _, ok := obj[k]; !ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func extraKeys(obj map[string]interface{}) []string {
	allowed := map[string]bool{}
	for _, k := range keys {
		allowed[k] = true
	}
	var out []string
	for k := range obj {
		if !allowed[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// NextID returns the next Q-YYYYMMDD-NNN id for dateStr, continuing that day's
// sequence. dateStr must be YYYYMMDD.
func NextID(rows []Row, dateStr string) (string, error) {
	if !dateRE.MatchString(dateStr) {
		return "", fmt.Errorf("--date must be YYYYMMDD, got %q", dateStr)
	}
	hi := 0
	for _, r := range rows {
		obj, ok := r.object()
		if !ok {
			continue
		}
		m := idRE.FindStringSubmatch(str(obj, "id"))
		if m != nil && m[1] == dateStr {
			var n int
			fmt.Sscanf(m[2], "%d", &n)
			if n > hi {
				hi = n
			}
		}
	}
	return fmt.Sprintf("Q-%s-%03d", dateStr, hi+1), nil
}

// DedupeMatch returns the first row whose question (or target+overlapping
// question) coarsely matches, or nil.
func DedupeMatch(rows []Row, question, target string) map[string]interface{} {
	nq := truncate(norm(question), 70)
	nt := truncate(norm(target), 40)
	for _, r := range rows {
		obj, ok := r.object()
		if !ok {
			continue
		}
		oq := truncate(norm(str(obj, "question")), 70)
		ot := truncate(norm(str(obj, "target")), 40)
		if nq != "" && nq == oq {
			return obj
		}
		if nt != "" && nt == ot && nq != "" && oq != "" && (strings.Contains(oq, nq) || strings.Contains(nq, oq)) {
			return obj
		}
	}
	return nil
}

// StatsResult is the JSON shape of the stats subcommand.
type StatsResult struct {
	Total      int            `json:"total"`
	ByCategory map[string]int `json:"by_category"`
	ByStatus   map[string]int `json:"by_status"`
}

// Stats folds category and status counts over the object rows.
func Stats(rows []Row) StatsResult {
	s := StatsResult{ByCategory: map[string]int{}, ByStatus: map[string]int{}}
	for _, r := range rows {
		obj, ok := r.object()
		if !ok {
			continue
		}
		s.Total++
		s.ByCategory[getOr(obj, "category", "?")]++
		s.ByStatus[getOr(obj, "status", "?")]++
	}
	return s
}

func getOr(obj map[string]interface{}, k, dflt string) string {
	if v, ok := obj[k].(string); ok {
		return v
	}
	return dflt
}

// GHFunc runs the gh CLI (args after "gh") and returns its stdout/stderr/err.
// The seam keeps ensure-label hermetic under test.
type GHFunc func(args []string) (stdout, stderr string, err error)

// Run is the CLI entry point. It owns argument parsing, file I/O, and exit codes
// (0 = OK, 1 = lint violation(s), 2 = harness error, 3 = dedupe hit). nowUTC
// supplies next-id's default date; gh may be nil unless ensure-label --apply.
func Run(stdout, stderr io.Writer, argv []string, nowUTCyyyymmdd string, gh GHFunc) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: question-ledger {lint|next-id|dedupe-check|stats|ensure-label} [flags]")
		return 2
	}
	cmd := argv[0]
	opt := parseFlags(argv[1:])
	if opt.date == "" {
		opt.date = nowUTCyyyymmdd
	}

	if cmd == "ensure-label" {
		args := []string{"label", "create", Label, "--color", LabelColor, "--description", LabelDesc}
		if opt.apply {
			if gh == nil {
				fmt.Fprintln(stderr, "question-ledger: ensure-label --apply requires a gh runner")
				return 2
			}
			out, errOut, _ := gh(append(args, "--force"))
			io.WriteString(stdout, out)
			io.WriteString(stderr, errOut)
			return 0
		}
		fmt.Fprintln(stdout, "gh "+strings.Join(args, " "))
		return 0
	}

	var rows []Row
	if data, err := os.ReadFile(opt.ledger); err == nil {
		rows = ParseRows(string(data))
	} else if !os.IsNotExist(err) {
		// An absent ledger is empty, not an error — the loop may run before its
		// first append. Any other read failure is a harness error.
		fmt.Fprintf(stderr, "question-ledger: cannot read %s: %v\n", opt.ledger, err)
		return 2
	}

	switch cmd {
	case "lint":
		problems := LintRows(rows)
		if len(problems) > 0 {
			for _, p := range problems {
				fmt.Fprintf(stdout, "FAIL %s\n", p)
			}
			fmt.Fprintf(stdout, "\nquestion-ledger: %d violation(s) in %s\n", len(problems), opt.ledger)
			return 1
		}
		fmt.Fprintf(stdout, "question-ledger: clean - %d row(s) in %s\n", Stats(rows).Total, opt.ledger)
		return 0
	case "next-id":
		id, err := NextID(rows, opt.date)
		if err != nil {
			fmt.Fprintf(stderr, "question-ledger: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, id)
		return 0
	case "dedupe-check":
		if opt.question == "" {
			fmt.Fprintln(stderr, "dedupe-check requires --question")
			return 2
		}
		if hit := DedupeMatch(rows, opt.question, opt.target); hit != nil {
			fmt.Fprintf(stdout, "DUPLICATE of %s: %s\n", getOr(hit, "id", ""), getOr(hit, "question", ""))
			return 3
		}
		fmt.Fprintln(stdout, "unique")
		return 0
	case "stats":
		b, _ := json.MarshalIndent(Stats(rows), "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}
	fmt.Fprintf(stderr, "question-ledger: unknown subcommand %q\n", cmd)
	return 2
}

type options struct {
	ledger, date, question, target string
	apply                          bool
}

// parseFlags reads the tool's --flag value / --flag pairs from args. It is
// deliberately permissive (positional cmd is already peeled off in Run) and
// mirrors the small argparse surface of the retired Python tool.
func parseFlags(args []string) options {
	opt := options{ledger: DefaultLedger}
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--apply":
			opt.apply = true
		case "--ledger", "--date", "--question", "--target":
			val := ""
			if i+1 < len(args) {
				val = args[i+1]
				i++
			}
			switch args[i-1] {
			case "--ledger":
				opt.ledger = val
			case "--date":
				opt.date = val
			case "--question":
				opt.question = val
			case "--target":
				opt.target = val
			}
		default:
			if strings.HasPrefix(args[i], "--") && strings.Contains(args[i], "=") {
				kv := strings.SplitN(args[i][2:], "=", 2)
				switch kv[0] {
				case "ledger":
					opt.ledger = kv[1]
				case "date":
					opt.date = kv[1]
				case "question":
					opt.question = kv[1]
				case "target":
					opt.target = kv[1]
				case "apply":
					opt.apply = true
				}
			}
		}
	}
	return opt
}
