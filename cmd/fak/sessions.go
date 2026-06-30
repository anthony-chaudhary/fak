package main

// `fak sessions` -- run the session-observability ladder over our OWN coding-session
// transcripts. This is the impure shell over internal/sessionobs: it discovers the
// transcripts on this host, folds each into a scrubbed Record (the pure
// sessionobs.FoldTranscript), witnesses each session's committed SHAs against git
// history, classifies the value-vs-waste outcome, and scores how observable the whole
// corpus is for RSI.
//
//	fak sessions discover [--project SUB] [--root DIR ...] [--since-days N]   list the transcripts found
//	fak sessions score    [--project SUB] [--root DIR ...] [--max N] [--corpus OUT] [--json]
//	                                                                          fold + witness + score the corpus
//
// It reads transcripts and shells `git` (read-only) off any hot path; it writes
// nothing unless --corpus is given (a scrubbed JSONL corpus, only structured signal,
// safe to commit). The score is the honest readout: over real sessions it reports the
// rungs already built (capture/structure/link) and the rungs still owed (a committed
// corpus, a consuming loop).

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/sessionobs"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdSessions(argv []string) { os.Exit(runSessions(os.Stdout, os.Stderr, argv)) }

func runSessions(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		sessionsUsage(stderr)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "discover", "ls", "list":
		return sessionsDiscover(stdout, stderr, rest)
	case "score":
		return sessionsScore(stdout, stderr, rest)
	case "learn":
		return sessionsLearn(stdout, stderr, rest)
	case "-h", "--help", "help":
		sessionsUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak sessions: unknown subcommand %q\n", sub)
		sessionsUsage(stderr)
		return 2
	}
}

func sessionsUsage(w io.Writer) {
	fmt.Fprint(w, `fak sessions -- score the RSI-observability of our own coding-session transcripts

usage:
  fak sessions discover [--project SUB] [--root DIR ...] [--since-days N]
  fak sessions score    [--project SUB] [--root DIR ...] [--max N] [--corpus OUT] [--json]
  fak sessions learn    [--corpus IN] [--project SUB] [--root DIR ...] [--max N] [--json]

Start here:
  fak sessions score        fold THIS host's fak sessions, witness their commits, and
                            print how far up the capture->structure->link->aggregate->learn
                            ladder the corpus has climbed (sessionobs_debt).
  fak sessions learn        the LEARN rung: contrast value-side (shipped) against
                            waste-side (stopped) sessions and surface which behaviors
                            separate them. Reads a committed corpus with --corpus, else
                            folds this host's sessions live.
`)
}

type sessionsFlags struct {
	project   string
	roots     []string
	sinceDays float64
	max       int
	corpus    string
	asJSON    bool
}

func parseSessionsFlags(name string, fs *flag.FlagSet, argv []string) (*sessionsFlags, error) {
	f := &sessionsFlags{}
	var rootList multiFlag
	fs.StringVar(&f.project, "project", "work-fak", "namespace substring to match (the repo's Claude project dir); --project '' scans all")
	fs.Var(&rootList, "root", "a projects dir or account home to scan (repeatable); default: all ~/.claude* homes")
	fs.Float64Var(&f.sinceDays, "since-days", 0, "only transcripts modified within N days (0 = all)")
	if name == "score" {
		fs.IntVar(&f.max, "max", 0, "cap the number of transcripts folded (0 = all; newest first)")
		fs.StringVar(&f.corpus, "corpus", "", "write the scrubbed Record corpus as JSONL to this path (default: write nothing)")
		fs.BoolVar(&f.asJSON, "json", false, "emit the machine-readable score envelope")
	}
	if name == "learn" {
		fs.IntVar(&f.max, "max", 0, "cap the number of transcripts folded when reading live (0 = all; newest first)")
		fs.StringVar(&f.corpus, "corpus", "", "read a pre-folded scrubbed corpus JSONL from this path instead of folding this host's transcripts live")
		fs.BoolVar(&f.asJSON, "json", false, "emit the machine-readable control-pane envelope")
	}
	if err := fs.Parse(argv); err != nil {
		return nil, err
	}
	f.roots = rootList
	return f, nil
}

// multiFlag collects a repeatable string flag.
type multiFlag []string

func (m *multiFlag) String() string { return strings.Join(*m, ",") }
func (m *multiFlag) Set(s string) error {
	*m = append(*m, s)
	return nil
}

// transcriptRef is one discovered transcript file plus the namespace/account it lives under.
type transcriptRef struct {
	path      string
	namespace string
	account   string
	mtime     time.Time
}

// discoverTranscripts enumerates session transcripts under the given roots (or all
// ~/.claude* homes by default), filtered to namespaces matching the project substring
// and the since-days window. Newest first. Subagent sub-transcripts (nested dirs) are
// skipped -- only top-level session files.
func discoverTranscripts(f *sessionsFlags) ([]transcriptRef, error) {
	roots := f.roots
	if len(roots) == 0 {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		entries, err := os.ReadDir(home)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			// account homes are ~/.claude, ~/.claude-<acct>, etc. Skip the deleted ones.
			if !e.IsDir() || !strings.HasPrefix(e.Name(), ".claude") {
				continue
			}
			if strings.Contains(e.Name(), ".DELETED") {
				continue
			}
			roots = append(roots, filepath.Join(home, e.Name(), "projects"))
		}
	}
	var cutoff time.Time
	if f.sinceDays > 0 {
		cutoff = time.Now().Add(-time.Duration(f.sinceDays * float64(24*time.Hour)))
	}
	var out []transcriptRef
	for _, root := range roots {
		account := accountFromRoot(root)
		nsEntries, err := os.ReadDir(root)
		if err != nil {
			continue // a missing/!dir root is fine -- just skip it
		}
		for _, ns := range nsEntries {
			if !ns.IsDir() {
				continue
			}
			if f.project != "" && !strings.Contains(ns.Name(), f.project) {
				continue
			}
			nsDir := filepath.Join(root, ns.Name())
			files, err := os.ReadDir(nsDir)
			if err != nil {
				continue
			}
			for _, fi := range files {
				if fi.IsDir() || !strings.HasSuffix(fi.Name(), ".jsonl") {
					continue
				}
				info, err := fi.Info()
				if err != nil {
					continue
				}
				if !cutoff.IsZero() && info.ModTime().Before(cutoff) {
					continue
				}
				out = append(out, transcriptRef{
					path:      filepath.Join(nsDir, fi.Name()),
					namespace: ns.Name(),
					account:   account,
					mtime:     info.ModTime(),
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].mtime.After(out[j].mtime) })
	return out, nil
}

// accountFromRoot extracts the account label from a ".../.claude-<acct>/projects" path
// (".claude" -> "default"). Best-effort, for per-worker attribution only.
func accountFromRoot(root string) string {
	parts := strings.Split(filepath.ToSlash(root), "/")
	for _, p := range parts {
		if strings.HasPrefix(p, ".claude") {
			acct := strings.TrimPrefix(p, ".claude")
			acct = strings.TrimPrefix(acct, "-")
			if acct == "" {
				return "default"
			}
			return acct
		}
	}
	return ""
}

func sessionsDiscover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessions discover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseSessionsFlags("discover", fs, argv)
	if err != nil {
		return 2
	}
	refs, err := discoverTranscripts(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions discover: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "%d transcripts (project filter %q)\n", len(refs), f.project)
	limit := len(refs)
	if limit > 40 {
		limit = 40
	}
	for _, r := range refs[:limit] {
		fmt.Fprintf(stdout, "  %s  %-14s  %s\n", r.mtime.Format("2006-01-02 15:04"), r.account, filepath.Base(r.path))
	}
	return 0
}

func sessionsScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessions score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseSessionsFlags("score", fs, argv)
	if err != nil {
		return 2
	}
	corpus, err := foldLiveCorpus(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions score: %v\n", err)
		return 1
	}

	if f.corpus != "" {
		if err := writeCorpus(f.corpus, corpus); err != nil {
			fmt.Fprintf(stderr, "fak sessions score: writing corpus: %v\n", err)
			return 1
		}
		fmt.Fprintf(stderr, "wrote %d records to %s\n", len(corpus), f.corpus)
	}

	root := repoRoot()
	pipe := sessionobs.Pipeline{
		CorpusCommitted: corpusIsCommitted(root),
		// LoopConsumes is true once a REGISTERED loop reads the committed corpus -- the
		// `sessions_learn` garden member (gardenbundle.Members). This is checked, not
		// asserted: remove the member and the rung honestly fails again.
		LoopConsumes: sessionsLearnRegistered(),
		// Registered tracks the sessionobs SCORECARD folding into the scorecard control
		// pane ratchet (a Python-side wiring); not done yet. SOFT, so it costs score, not debt.
		Registered: false,
	}
	rep := sessionobs.Score(corpus, pipe)
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, rep)
		return 0
	}
	sessionobs.Render(stdout, rep)
	return 0
}

// foldLiveCorpus discovers this host's transcripts (honoring the project/root/since/max
// filters), folds each into a scrubbed Record, and witnesses its commits against git
// history to classify the outcome. It is the shared engine behind `score` and a live
// `learn`. Read-only and off any hot path.
func foldLiveCorpus(f *sessionsFlags) ([]sessionobs.Record, error) {
	refs, err := discoverTranscripts(f)
	if err != nil {
		return nil, err
	}
	if f.max > 0 && len(refs) > f.max {
		refs = refs[:f.max]
	}
	root := repoRoot()
	wit := newGitWitness(root)
	corpus := make([]sessionobs.Record, 0, len(refs))
	for _, ref := range refs {
		fh, err := os.Open(ref.path)
		if err != nil {
			continue
		}
		rec, ev := sessionobs.FoldTranscript(fh, sessionobs.FoldMeta{
			SessionID: strings.TrimSuffix(filepath.Base(ref.path), ".jsonl"),
			Namespace: ref.namespace,
			Account:   ref.account,
		})
		fh.Close()
		rec.Outcome = sessionobs.Classify(ev, wit.anySurvived(ev.CommitSHAs))
		corpus = append(corpus, rec)
	}
	return corpus, nil
}

// learnSchema is the control-pane schema for the `learn --json` envelope.
const learnSchema = "fak.sessionobs.learn.v1"

// learnEnvelope wraps the pure ContrastReport in the standard control-pane fields a
// garden member reads (ok/verdict/finding/reason/next_action), so the learn pass can
// join the garden bundle as a registered member.
type learnEnvelope struct {
	Schema     string                    `json:"schema"`
	OK         bool                      `json:"ok"`
	Verdict    string                    `json:"verdict"`
	Finding    string                    `json:"finding"`
	Reason     string                    `json:"reason"`
	NextAction string                    `json:"next_action"`
	Source     string                    `json:"source"`
	Contrast   sessionobs.ContrastReport `json:"contrast"`
}

// sessionsLearn is the LEARN-rung consumer: it builds the corpus (from a committed
// JSONL via --corpus, else by folding this host's sessions live), runs the pure
// value-vs-waste Contrast, and renders the ranked discriminators + the one headline
// finding. With --json it emits a control-pane envelope so the garden bundle can run it.
func sessionsLearn(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessions learn", flag.ContinueOnError)
	fs.SetOutput(stderr)
	f, err := parseSessionsFlags("learn", fs, argv)
	if err != nil {
		return 2
	}

	var corpus []sessionobs.Record
	source := "live"
	if f.corpus != "" {
		source = f.corpus
		recs, rerr := readCorpus(f.corpus)
		if rerr != nil {
			// A missing/unreadable committed corpus is a SURFACED condition, never a
			// crash: under --json emit a well-formed advisory envelope (exit 0) so a
			// garden member treats it as advisory, not an errored member that reds the gate.
			if f.asJSON {
				_ = writeIndentedJSONNoEscape(stdout, learnEnvelope{
					Schema: learnSchema, OK: false, Verdict: "ACTION",
					Finding:    "sessionobs_corpus_missing",
					Reason:     fmt.Sprintf("corpus %s unreadable: %v", f.corpus, rerr),
					NextAction: "regenerate the committed corpus: fak sessions score --corpus " + f.corpus,
					Source:     source,
					Contrast:   sessionobs.Contrast(nil),
				})
				return 0
			}
			fmt.Fprintf(stderr, "fak sessions learn: reading corpus %s: %v\n", f.corpus, rerr)
			return 1
		}
		corpus = recs
	} else {
		recs, ferr := foldLiveCorpus(f)
		if ferr != nil {
			fmt.Fprintf(stderr, "fak sessions learn: %v\n", ferr)
			return 1
		}
		corpus = recs
	}

	rep := sessionobs.Contrast(corpus)
	if f.asJSON {
		_ = writeIndentedJSONNoEscape(stdout, buildLearnEnvelope(rep, source))
		return 0
	}
	sessionobs.RenderContrast(stdout, rep)
	return 0
}

// buildLearnEnvelope shapes a successful contrast into the control-pane envelope. A
// pass that read the corpus is OK even when it finds no discriminator yet -- surfacing
// "insufficient" or "flat" is the loop WORKING, not a failure (mirroring guard_route).
func buildLearnEnvelope(rep sessionobs.ContrastReport, source string) learnEnvelope {
	env := learnEnvelope{
		Schema: learnSchema, OK: true, Verdict: "OK",
		Reason: rep.Headline, NextAction: rep.NextAction, Source: source, Contrast: rep,
	}
	switch {
	case !rep.Separable:
		env.Finding = "sessionobs_learn_insufficient"
	case rep.TopFeature == "":
		env.Finding = "sessionobs_learn_flat"
	default:
		env.Finding = "sessionobs_learn"
	}
	return env
}

// readCorpus reads a scrubbed Record corpus JSONL (the inverse of writeCorpus). A torn
// line is skipped, never fatal; a missing file is an error the caller surfaces.
func readCorpus(path string) ([]sessionobs.Record, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	var out []sessionobs.Record
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r sessionobs.Record
		if json.Unmarshal([]byte(line), &r) != nil {
			continue
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// sessionsLearnRegistered reports whether the `sessions_learn` loop is registered in the
// garden bundle -- the honest witness for the sessionobs `loop_consumes` rung. It reads
// the live member registry, so the rung tracks reality: drop the member and it fails again.
func sessionsLearnRegistered() bool {
	for _, m := range gardenbundle.Members {
		if m.Key == "sessions_learn" {
			return true
		}
	}
	return false
}

// gitWitness answers "did this short SHA survive into HEAD's history?" by shelling
// `git merge-base --is-ancestor`. It caches per-SHA so a session's repeated SHAs and
// cross-session shared SHAs cost one git call each.
type gitWitness struct {
	root  string
	cache map[string]bool
}

func newGitWitness(root string) *gitWitness {
	return &gitWitness{root: root, cache: map[string]bool{}}
}

// anySurvived reports whether ANY of the session's committed SHAs is an ancestor of
// HEAD (survived, not reverted) -- the witnessed bit that lifts a Claimed outcome to
// Shipped.
func (g *gitWitness) anySurvived(shas []string) bool {
	for _, sha := range shas {
		if g.survived(sha) {
			return true
		}
	}
	return false
}

func (g *gitWitness) survived(sha string) bool {
	if v, ok := g.cache[sha]; ok {
		return v
	}
	// `git merge-base --is-ancestor <sha> HEAD` exits 0 iff sha is reachable from HEAD.
	// A bad/ambiguous short SHA errors (non-zero) and is treated as not-survived.
	cmd := exec.Command("git", "-C", g.root, "merge-base", "--is-ancestor", sha, "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Run()
	v := err == nil
	g.cache[sha] = v
	return v
}

// writeCorpus writes the scrubbed Records as JSONL (one Record per line). The Records
// carry only structured signal, so the file is safe to commit and fold across hosts.
func writeCorpus(path string, corpus []sessionobs.Record) error {
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fh.Close()
	w := bufio.NewWriter(fh)
	enc := json.NewEncoder(w)
	for _, r := range corpus {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return w.Flush()
}

// corpusIsCommitted reports whether a session-observability corpus is tracked in git
// (the CorpusCommitted pipeline fact). Reads `git ls-files` for the conventional path.
func corpusIsCommitted(root string) bool {
	cmd := exec.Command("git", "-C", root, "ls-files", "experiments/sessionobs/")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	return err == nil && len(strings.TrimSpace(string(out))) > 0
}
