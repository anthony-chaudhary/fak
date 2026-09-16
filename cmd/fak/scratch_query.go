package main

// scratch_query.go â€” `fak scratch query`, the READ half of the harness session
// scratchpad (public issue #13067), plus the PROMOTION-CANDIDATE surface that
// flags parked artifacts looking like unfiled session flags before the reap
// window closes (public issue #13069).
//
//	fak scratch query --session <id> --transcript <path> --root <r> [--pattern p] [--json]
//	fak scratch query --promotable --store <transcript-store> [--root <r>] [--json]
//
// The content match and the closed promotion taxonomy live in internal/scratchquery
// (a deterministic, stdlib-only leaf). This shell does only the I/O the leaf must
// not: it resolves each session's scratchpad address with the ALREADY-SHIPPED
// deriveScratchpadPath (no new address logic), orients the scan at the scratchpad
// root `fak resume scan` addresses, attaches each session's resume Diagnosis, and
// renders hits/candidates. It is READ-ONLY and ADVISORY: it never files an issue
// and never deletes a scratchpad. Any promotion/decision policy stays in the
// private factory plane.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/scratchjanitor"
	"github.com/anthony-chaudhary/fak/internal/scratchquery"
)

// cmdScratch is the `fak scratch` entry point; it dispatches the scratchpad
// subcommands and maps the testable core's exit code to the process exit code.
func cmdScratch(argv []string) { os.Exit(runScratch(os.Stdout, os.Stderr, argv)) }

// runScratch is the testable core: 0 ok, 1 a runtime error, 2 a usage error.
func runScratch(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		scratchUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "query":
		return runScratchQuery(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		scratchUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak scratch: unknown subcommand %q (want query)\n", argv[0])
		scratchUsage(stderr)
		return 2
	}
}

// runScratchQuery answers "what did session <id> leave in here?" (content mode)
// and "which parked artifacts look like unfiled session flags?" (--promotable).
func runScratchQuery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("scratch query", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "scratch")
	session := fs.String("session", "", "session UUID whose scratchpad to read (content mode)")
	transcript := fs.String("transcript", "", "the session's Claude Code transcript (.jsonl) â€” resolves the scratchpad via deriveScratchpadPath")
	root := fs.String("root", "", "scratchpad root: <root>/<project-slug>/<session-id>/scratchpad (default: the Claude scratchpad root)")
	store := fs.String("store", "", "directory of Claude Code transcripts â€” the --promotable census over dead/unreferenced sessions")
	pattern := fs.String("pattern", "", "literal substring to match in content mode")
	regex := fs.Bool("regex", false, "treat --pattern as a RE2 regular expression")
	limit := fs.Int("limit", 0, "cap returned content hits (0 = unlimited)")
	promotable := fs.Bool("promotable", false, "list promotion candidates across sessions instead of content hits")
	all := fs.Bool("all", false, "with --promotable, include clean/live sessions (default: dead/unreferenced only)")
	maxAge := fs.Duration("max-age", scratchjanitor.DefaultMaxAge, "with --promotable, the reap-window age a session must reach to be listed")
	jsonOut := fs.Bool("json", false, "emit the result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak scratch query: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	scratchpadRoot := strings.TrimSpace(*root)
	if scratchpadRoot == "" {
		scratchpadRoot = defaultClaudeScratchpadRoot()
	}
	if scratchpadRoot == "" {
		fmt.Fprintln(stderr, "fak scratch query: cannot resolve the scratchpad root; pass --root DIR")
		return 2
	}

	if *promotable {
		return runScratchQueryPromotable(stdout, stderr, scratchpadRoot, *store, *maxAge, *all, *jsonOut)
	}
	return runScratchQueryContent(stdout, stderr, scratchpadRoot, *session, *transcript, *pattern, *regex, *limit, *jsonOut)
}

// runScratchQueryContent resolves one session's scratchpad address and runs the
// bounded content match. A missing scratchpad is a clean empty result (exit 0).
func runScratchQueryContent(stdout, stderr io.Writer, scratchpadRoot, session, transcript, pattern string, regex bool, limit int, jsonOut bool) int {
	if strings.TrimSpace(session) == "" && strings.TrimSpace(transcript) == "" {
		fmt.Fprintln(stderr, "fak scratch query: content mode needs --session <id> and --transcript <path> (or use --promotable)")
		return 2
	}
	scratchpad, err := resolveScratchpad(scratchpadRoot, session, transcript)
	if err != nil {
		fmt.Fprintln(stderr, "fak scratch query:", err)
		return 1
	}
	result, err := scratchquery.Query(scratchpad, scratchquery.QueryOptions{Pattern: pattern, Regex: regex, Limit: limit})
	if err != nil {
		fmt.Fprintln(stderr, "fak scratch query:", err)
		return 1
	}
	result.Root = scratchpad
	if jsonOut {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":     "fak.scratch-query.v1",
			"session":    session,
			"scratchpad": scratchpad,
			"query":      result,
		}, "fak scratch query")
	}
	renderScratchContent(stdout, scratchpad, session, result)
	return 0
}

// scratchPromotableRow is one promotion candidate with its session context: the
// resume Diagnosis (why the session is dead/unresumed) and the session identity.
type scratchPromotableRow struct {
	Session        string           `json:"session_id"`
	ScratchpadPath string           `json:"scratchpad_path"`
	Kind           string           `json:"kind"`
	File           string           `json:"file"`
	Bytes          int              `json:"bytes"`
	AgeSeconds     int64            `json:"age_seconds,omitempty"`
	Diagnosis      resume.Diagnosis `json:"diagnosis"`
}

// runScratchQueryPromotable scans the transcript store, keeps the dead/
// unreferenced sessions (a resumable session that points at a scratchpad spares
// it â€” the janitor's own Referenced guard), and lists their parked artifacts that
// Classify flags as promotion candidates, each carrying the session's resume
// Diagnosis. Advisory only.
func runScratchQueryPromotable(stdout, stderr io.Writer, scratchpadRoot, store string, maxAge time.Duration, all, jsonOut bool) int {
	if strings.TrimSpace(store) == "" {
		fmt.Fprintln(stderr, "fak scratch query --promotable: need --store DIR (a directory of Claude Code .jsonl transcripts)")
		return 2
	}
	rows, code := diagnoseScratchStore(stderr, store, scratchpadRoot)
	if code != 0 {
		return code
	}
	if len(rows) == 0 {
		fmt.Fprintf(stderr, "fak scratch query --promotable: no .jsonl transcripts in %q\n", store)
		return 1
	}

	var candidates []scratchPromotableRow
	for _, row := range rows {
		if !all && !isReapWindowCandidate(row, maxAge) {
			continue
		}
		if strings.TrimSpace(row.ScratchpadPath) == "" {
			continue
		}
		result, err := scratchquery.ScanPromotable(row.ScratchpadPath, row.SessionID)
		if err != nil {
			continue // an unreadable scratchpad is skipped, never fatal
		}
		for _, c := range result.Candidates {
			candidates = append(candidates, scratchPromotableRow{
				Session:        row.SessionID,
				ScratchpadPath: row.ScratchpadPath,
				Kind:           string(c.Kind),
				File:           c.Path,
				Bytes:          c.Bytes,
				AgeSeconds:     row.IdleSeconds,
				Diagnosis:      row.Diagnosis,
			})
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Kind != candidates[j].Kind {
			return candidates[i].Kind < candidates[j].Kind
		}
		if candidates[i].Session != candidates[j].Session {
			return candidates[i].Session < candidates[j].Session
		}
		return candidates[i].File < candidates[j].File
	})

	if jsonOut {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"schema":     "fak.scratch-promotable.v1",
			"store":      store,
			"root":       scratchpadRoot,
			"taxonomy":   scratchquery.CandidateKinds,
			"candidates": candidates,
			"advisory":   true,
			"read_only":  true,
			"note":       "promotion candidates only; this query files no issue and deletes nothing",
		}, "fak scratch query --promotable")
	}
	renderScratchPromotable(stdout, store, candidates)
	return 0
}

// isReapWindowCandidate reports whether a session is a reap-window candidate:
// dead or unreferenced (its resume Diagnosis is not clean) OR aged past the reap
// window. This keeps the surface aligned with the janitor's own selection (an
// old, unreferenced session is the one whose flags would strand) without
// deleting anything.
func isReapWindowCandidate(row scanRow, maxAge time.Duration) bool {
	if row.Diagnosis.Unresumed || row.Diagnosis.Crash != resume.CrashNone {
		return true
	}
	if row.IdleSeconds >= 0 && maxAge > 0 {
		return row.IdleSeconds >= int64(maxAge/time.Second)
	}
	return false
}

// diagnoseScratchStore walks a Claude Code transcript store and diagnoses every
// session, reusing the SAME event classification and pure verdict `fak resume
// scan` uses (scanTranscriptToEvents + resume.Diagnose), so the promotion surface
// and the resume surface cannot drift. It resolves each session's scratchpad
// address with the shipped deriveScratchpadPath.
func diagnoseScratchStore(stderr io.Writer, store, scratchpadRoot string) ([]scanRow, int) {
	entries, err := os.ReadDir(store)
	if err != nil {
		fmt.Fprintf(stderr, "fak scratch query --promotable: read store %q: %v\n", store, err)
		return nil, 1
	}
	now := time.Now().Unix()
	var rows []scanRow
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		path := filepath.Join(store, e.Name())
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		events, model, lastUnix, limitMsg := scanTranscriptToEvents(f)
		f.Close()
		idle := int64(-1)
		if lastUnix > 0 {
			if idle = now - lastUnix; idle < 0 {
				idle = 0
			}
		}
		d := resume.Diagnose(events, resume.Input{IdleSeconds: idle})
		sessionID := strings.TrimSuffix(e.Name(), ".jsonl")
		rows = append(rows, scanRow{
			SessionID:      sessionID,
			ScratchpadPath: deriveScratchpadPath(path, sessionID, scratchpadRoot),
			Model:          model,
			IdleSeconds:    idle,
			LimitMessage:   limitMsg,
			Diagnosis:      d,
		})
	}
	return rows, 0
}

// resolveScratchpad resolves a session's scratchpad address with the
// ALREADY-SHIPPED deriveScratchpadPath when a transcript is supplied (no new
// address logic); otherwise it composes the address from the root + session id.
func resolveScratchpad(scratchpadRoot, session, transcript string) (string, error) {
	if strings.TrimSpace(transcript) != "" {
		if !filepath.IsAbs(scratchpadRoot) {
			return "", fmt.Errorf("scratchpad root %q must be absolute", scratchpadRoot)
		}
		sid := strings.TrimSpace(session)
		if sid == "" {
			sid = strings.TrimSuffix(filepath.Base(transcript), ".jsonl")
		}
		path := deriveScratchpadPath(transcript, sid, scratchpadRoot)
		if path == "" {
			return "", fmt.Errorf("transcript %q does not resolve to a scratchpad under %q", transcript, scratchpadRoot)
		}
		return path, nil
	}
	if !safeResumePathComponent(session) {
		return "", fmt.Errorf("bad --session %q", session)
	}
	if !filepath.IsAbs(scratchpadRoot) {
		return "", fmt.Errorf("scratchpad root %q must be absolute", scratchpadRoot)
	}
	return filepath.Join(scratchpadRoot, "unknown-project", session, "scratchpad"), nil
}

// renderScratchContent prints content hits (file:line + the matched line).
func renderScratchContent(w io.Writer, scratchpad, session string, result scratchquery.QueryResult) {
	fmt.Fprintf(w, "scratchpad %s (session %s)\n", scratchpad, shortID(session))
	fmt.Fprintf(w, "pattern %q: %d hit(s) across %d file(s)\n", result.Pattern, len(result.Hits), result.Matched)
	for _, h := range result.Hits {
		fmt.Fprintf(w, "  %s:%d  %s\n", h.File, h.Line, h.Match)
	}
	for _, s := range result.Skipped {
		fmt.Fprintf(w, "  SKIP %s (%s, %d bytes)\n", s.File, s.Reason, s.Bytes)
	}
}

// renderScratchPromotable prints the promotion candidates, grouped by taxonomy
// kind, each with its session's resume diagnosis and a promote-before-reap note.
func renderScratchPromotable(w io.Writer, store string, candidates []scratchPromotableRow) {
	fmt.Fprintf(w, "promotion candidates in %s\n", store)
	fmt.Fprintf(w, "taxonomy: %s\n\n", joinKinds(scratchquery.CandidateKinds))
	if len(candidates) == 0 {
		fmt.Fprintln(w, "no promotion candidates â€” nothing parked looks like an unfiled session flag.")
		return
	}
	for _, c := range candidates {
		fmt.Fprintf(w, "  [%s] session %s  %s  (%d bytes, idle %s)\n",
			c.Kind, shortID(c.Session), c.File, c.Bytes, humanIdle(c.AgeSeconds))
		fmt.Fprintf(w, "      diagnosis: crash=%s unresumed=%t needs_restart=%t\n",
			c.Diagnosis.Crash, c.Diagnosis.Unresumed, c.Diagnosis.NeedsRestart)
	}
	fmt.Fprintf(w, "\n%d candidate(s). READ-ONLY + ADVISORY: promote one to an issue before the reap window closes; this query files nothing and deletes nothing.\n", len(candidates))
}

func joinKinds(kinds []scratchquery.CandidateKind) string {
	parts := make([]string, 0, len(kinds))
	for _, k := range kinds {
		parts = append(parts, string(k))
	}
	return strings.Join(parts, ", ")
}

func scratchUsage(w io.Writer) {
	fmt.Fprint(w, `fak scratch â€” the harness session scratchpad surface

  fak scratch query --session <id> --transcript <file.jsonl> [--root DIR]
                    [--pattern P] [--regex] [--limit N] [--json]

  fak scratch query --promotable --store DIR [--root DIR] [--max-age D]
                    [--all] [--json]

  Content mode reports bounded literal matches in a session's scratchpad
  (resolved with the shipped deriveScratchpadPath). --promotable lists parked
  artifacts that look like unfiled session flags, with each session's resume
  diagnosis attached, before the reap window closes. Both modes are read-only.
`)
}
