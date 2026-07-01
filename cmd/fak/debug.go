package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/cdb"
	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// fak debug — the CONTEXT DEBUGGER. It attaches to a FINISHED session as if to a core
// dump and answers a follow-up by demand-paging only the working set the question
// touches, never by replaying the whole address space. With --session it ingests a
// REAL Claude Code transcript (driving every tool result back through the SHIPPED gate,
// so heavy results page out and an injection/secret result seals); with no --session it
// runs a hermetic demo over the committed synthetic fixture and emits cdb-report.json.
// stripFlags removes any `--name value` or `--name=value` occurrence (for the given bare
// names, with either one or two leading dashes) from argv, so a sub-dispatch's own
// flag.ExitOnError FlagSet — which does not declare the caller's already-consumed flags —
// does not fatally choke on them (e.g. cmdDebug's --cmd/--dir forwarded into
// cmdDebugContextPlanPreview's own, disjoint flag set).
func stripFlags(argv []string, names ...string) []string {
	strip := make(map[string]bool, len(names))
	for _, n := range names {
		strip["-"+n] = true
		strip["--"+n] = true
	}
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		if eq := strings.Index(a, "="); eq > 0 && strip[a[:eq]] {
			continue // --name=value: one token, no following value to skip
		}
		if strip[a] {
			i++ // --name value: also skip the following value token
			continue
		}
		out = append(out, a)
	}
	return out
}

func cmdDebug(argv []string) {
	// context-plan-preview (#1574) takes its own flag set (--intent/--pins/--budget-tokens/
	// etc., not the cdb ingest/attach flags below), so it must dispatch BEFORE the top-level
	// FlagSet below parses argv — that parse uses flag.ExitOnError and would otherwise kill
	// the process on the first preview-only flag it doesn't recognize (e.g. --intent).
	for i, a := range argv {
		isPreview := (a == "--cmd" && i+1 < len(argv) && argv[i+1] == "context-plan-preview") ||
			a == "-cmd=context-plan-preview" || a == "--cmd=context-plan-preview"
		if isPreview {
			cmdDebugContextPlanPreview(stripFlags(argv, "cmd", "dir", "session", "list"))
			return
		}
	}

	fs := flag.NewFlagSet("debug", flag.ExitOnError)
	list := fs.Bool("list", false, "discover real Claude Code session transcripts on this machine and print the `fak debug --session <path>` to run for each (most-recent first)")
	session := fs.String("session", "", "path to a Claude Code session .jsonl to ingest as a core image (default: the committed fixture)")
	dir := fs.String("dir", "cdb-image", "directory for the persisted core image (attached if it already holds one and --session is empty)")
	cmd := fs.String("cmd", "report", "report | html | info | bt | x | ws | grep | tombstone | context-query | context-diff | context-plan-preview")
	query := fs.String("query", "what refund fee did the user's account show?", "the follow-up question to demand-page a working set for (cmd=ws/report)")
	step := fs.Int("step", 0, "page step to examine (cmd=x)")
	grepPat := fs.String("grep", "", "descriptor pattern to search the page table for (cmd=grep)")
	k := fs.Int("k", 0, "max pages in the working set (0 = every referenced page)")
	pins := fs.String("pins", "", "comma-separated descriptor/role/digest patterns to force into cmd=context-query")
	excludes := fs.String("excludes", "", "comma-separated descriptor/role/digest patterns to refuse in cmd=context-query")
	query2 := fs.String("query2", "", "second query for cmd=context-diff (diffed against --query over the SAME image)")
	budgetBytes := fs.Int64("budget-bytes", 0, "max rendered bytes to materialize in cmd=context-query (0 = unbounded)")
	policyVersion := fs.String("policy-version", "", "policy version label to stamp on memory views in cmd=context-query")
	preferView := fs.String("prefer-view", "", "derived view type to prefer in cmd=context-query (e.g. summary); runs a cold-then-warm two-pass demo showing FAULT then HIT")
	assumptions := fs.String("assumptions", "", "path to JSON array of assumed context rows to include in cmd=report")
	reason := fs.String("reason", "requested context tombstone", "reason to record for cmd=tombstone")
	requestedBy := fs.String("requested-by", "operator", "requester identity to record for cmd=tombstone")
	out := fs.String("out", "cdb-report.json", "report output path (cmd=report)")
	sid := fs.String("session-id", "", "core-image session id (default: derived from the source)")
	_ = fs.Parse(argv)
	*dir = pathutil.ExpandTilde(*dir) // a leading ~ is never expanded by Go; do it so --dir ~/img works

	// Discovery: point an operator at their REAL transcripts instead of silently
	// running the synthetic demo. Read-only; no core image is touched.
	if *list {
		listTranscripts(os.Stdout)
		return
	}

	// Decide whether to ingest a fresh core image or attach to an existing one.
	attachExisting := *session == "" && imageExists(*dir)
	if !attachExisting {
		src := *session
		if src == "" {
			src = cdbFixturePath()
			fmt.Fprintln(os.Stderr, "fak debug: no --session given; ingesting the committed synthetic fixture (a demo). Run `fak debug --list` to find and attach your real Claude Code transcripts.")
		}
		id := *sid
		if id == "" {
			id = "cdb-" + strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))
		}
		rec, st, err := cdb.IngestSession(ctx(), src, id)
		must(err)
		must(rec.Persist(*dir))
		fmt.Printf("ingested %s -> core image %s/  (%d records, %d tool calls, %d pages, %d sealed)\n",
			src, *dir, st.Records, st.ToolUses, st.Pages, st.Sealed)
	}

	im, err := cdb.Attach(*dir)
	must(err)

	switch *cmd {
	case "info":
		fmt.Println(string(jsonIndent(im.Info())))
	case "bt":
		printBacktrace(im)
	case "x":
		b, err := im.Examine(ctx(), *step)
		if err != nil {
			fmt.Printf("page %d: REFUSED — %v\n", *step, err)
			return
		}
		fmt.Printf("page %d: RESOLVED %d bytes (%s)\n%s\n", *step, len(b), recall.Digest(b)[:12], previewBytes(b, 600))
	case "ws":
		printWorkingSet(im.WorkingSet(ctx(), *query, *k))
	case "grep":
		for _, f := range im.Grep(*grepPat) {
			fmt.Printf("  [%2d] %-14s %s\n", f.Step, f.Role, f.Descriptor)
		}
	case "tombstone":
		ch, err := im.RequestContextChange(recall.ContextChangeRequest{
			Action:      recall.ContextActionTombstone,
			Step:        *step,
			Reason:      *reason,
			RequestedBy: *requestedBy,
		})
		must(err)
		must(im.Persist())
		fmt.Printf("page %d tombstoned: %s requested_by=%s reason=%q\n",
			ch.Step, ch.ID, ch.RequestedBy, ch.Reason)
	case "context-query":
		req := contextq.Request{
			Query:         *query,
			K:             *k,
			BudgetBytes:   *budgetBytes,
			Pins:          splitCSV(*pins),
			Excludes:      splitCSV(*excludes),
			PolicyVersion: *policyVersion,
		}
		if v := strings.TrimSpace(*preferView); v != "" {
			// Two-pass demo: cold pass builds derived views (FAULT); warm pass with
			// the SAME shared cache serves them as HIT without paging raw bytes.
			cache := contextq.NewViewCache()
			req.PreferView = contextq.ViewType(v)
			req.ViewCache = cache
			cold := contextq.Query(ctx(), im, req)
			warm := contextq.Query(ctx(), im, req)
			fmt.Printf("\n== cold pass (build derived views) ==\n")
			printContextQuery(cold, "")
			fmt.Printf("\n== warm pass (reuse, same cache + policy) ==\n")
			printContextQuery(warm, "")
			fmt.Printf("view reuse: %d HIT(s), %d raw byte(s) paged on warm pass (cold paged %d)\n",
				warm.Stats.ViewHits, warm.Stats.BytesPagedIn, cold.Stats.BytesPagedIn)
			must(os.WriteFile(*out, jsonIndent(cold), 0o644))
			break
		}
		res := contextq.Query(ctx(), im, req)
		must(os.WriteFile(*out, jsonIndent(res), 0o644))
		printContextQuery(res, *out)
	case "context-diff":
		// Source-set diff: materialize the SAME image under two queries and report
		// which evidence handles the second added / dropped / kept, plus the
		// sealed/poisoned pages each side refused to expand (issue #427).
		baseReq := contextq.Request{
			Query: *query, K: *k, BudgetBytes: *budgetBytes,
			Pins: splitCSV(*pins), Excludes: splitCSV(*excludes), PolicyVersion: *policyVersion,
		}
		nextReq := baseReq
		nextReq.Query = *query2
		base := contextq.Query(ctx(), im, baseReq)
		next := contextq.Query(ctx(), im, nextReq)
		diff := contextq.DiffWorkingSets(base, next)
		must(os.WriteFile(*out, jsonIndent(diff), 0o644))
		fmt.Print(diff.Markdown())
		fmt.Printf("\n(full JSON diff -> %s)\n", *out)
	case "report":
		debugReport(im, *dir, *session, *query, *out, loadAssumedContext(*assumptions))
	case "html":
		debugHTMLReport(im, *dir, *session, *query, *out)
	default:
		fmt.Fprintf(os.Stderr, "fak debug: unknown --cmd %q\n", *cmd)
		os.Exit(2)
	}
}

// debugReport runs the full attach->inspect->demand-page demonstration and emits a
// committed-style JSON artifact plus a human summary — the cdb analogue of recall's
// report.
func debugReport(im *cdb.Image, dir, session, query, out string, assumed []contextq.AssumedContext) {
	info := im.Info()
	frames := im.Backtrace()
	ws := im.WorkingSet(ctx(), query, 0)
	contextRes := contextq.Query(ctx(), im, contextq.Request{Query: query})
	contextEvidence := contextq.RenderKnownUnknownAssumedContext(contextRes, assumed)

	// examine one benign page (resolves) and one sealed page (refused) to show the gate
	// still stands on every page-in from the reloaded image.
	type examined struct {
		Step    int    `json:"step"`
		Role    string `json:"role"`
		Sealed  bool   `json:"sealed"`
		OK      bool   `json:"resolved"`
		Outcome string `json:"outcome"`
	}
	var exs []examined
	examine := func(stp int) {
		f := frames[stp]
		b, err := im.Examine(ctx(), stp)
		e := examined{Step: stp, Role: f.Role, Sealed: f.Sealed}
		if err != nil {
			e.Outcome = "REFUSED: " + err.Error()
		} else {
			e.OK = true
			e.Outcome = fmt.Sprintf("RESOLVED %d bytes (%s)", len(b), recall.Digest(b)[:12])
		}
		exs = append(exs, e)
	}
	benignStep, sealedStep := -1, -1
	for _, f := range frames {
		if !f.Sealed && benignStep < 0 {
			benignStep = f.Step
		}
		if f.Sealed && sealedStep < 0 {
			sealedStep = f.Step
		}
	}
	if benignStep >= 0 {
		examine(benignStep)
	}
	if sealedStep >= 0 {
		examine(sealedStep)
	}

	// the working-set view, WITHOUT the paged-in bytes (steps/roles/descriptors only).
	wsPages := make([]map[string]any, 0, len(ws.Slices))
	for _, sl := range ws.Slices {
		wsPages = append(wsPages, map[string]any{"step": sl.Step, "role": sl.Role, "descriptor": sl.Descriptor})
	}
	source := session
	if source == "" {
		source = "synthetic committed fixture (testdata/cdb/session.jsonl)"
	}
	report := map[string]any{
		"app_version":      appversion.Current(),
		"demo":             "context-debugger: attach to a finished session as a core dump; demand-page only the working set",
		"source":           source,
		"image_dir":        dir,
		"info":             info,
		"query":            query,
		"context_evidence": contextEvidence,
		"working_set": map[string]any{
			"pages_touched": ws.PagesTouched, "pages_benign": ws.PagesBenign, "pages_total": ws.PagesTotal,
			"sealed_skipped": ws.SealedSkipped, "tombstoned_skipped": ws.TombstonedSkipped,
			"faults_avoided": ws.FaultsAvoided,
			"bytes_paged_in": ws.BytesPagedIn, "resident_bytes": ws.ResidentBytes,
			"residency_pct": ws.ResidencyPct, "poison_in_set": ws.PoisonInSet,
			"pages": wsPages,
		},
		"examine": exs,
		"witness": "benign pages page in byte-identical; sealed pages refused on page-in (gate survives reload); the working set is a small resident slice and carries no poison",
	}
	must(os.WriteFile(out, jsonIndent(report), 0o644))

	// human summary
	fmt.Printf("\n== fak debug: %s  (core image %s/) ==\n", info.SessionID, dir)
	fmt.Printf("core dump        : %d pages = %d benign + %d sealed; %d heavy (paged out)\n",
		info.Pages, info.Benign, info.Sealed, info.Heavy)
	fmt.Printf("page table       : %d B on disk (the map you always carry)\n", info.ManifestFileBytes)
	fmt.Printf("swap device      : %d B raw across %d distinct blobs (dedup saved %d B)\n",
		info.CASBytes, info.DistinctBlobs, info.DedupSaved)
	fmt.Println("\npage table (bt):")
	printBacktrace(im)
	fmt.Printf("\nfollow-up: %q\n", query)
	printWorkingSet(ws)
	fmt.Printf("\n")
	fmt.Print(contextEvidence.Markdown())
	fmt.Println("\nexamine (the gate still stands on every page-in):")
	for _, e := range exs {
		fmt.Printf("  step %d %-14s -> %s\n", e.Step, e.Role, e.Outcome)
	}
	fmt.Printf("\nreport written   : %s\n", out)
}

func loadAssumedContext(path string) []contextq.AssumedContext {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	b, err := os.ReadFile(pathutil.ExpandTilde(path))
	must(err)
	var rows []contextq.AssumedContext
	if err := json.Unmarshal(b, &rows); err != nil {
		must(fmt.Errorf("read assumptions %s: %w", path, err))
	}
	return rows
}

// debugHTMLReport emits the "context debugger as a product" artifact (#574): a
// self-contained, static HTML inspection report over the attached core image —
// the shareable thing a teammate opens in a browser (no fak install, no JS, no
// external CSS). One document covers the page table (bt), the quarantine/seal
// panel with honest reason codes + the evadable-detector fence, the
// working-set residency view for the follow-up query, and demand-paged
// examine previews of every benign page. Sealed pages are refused at the gate
// by construction, so the report can never echo poison.
//
// The default --out (cdb-report.json) is auto-swapped to cdb-report.html so an
// operator who runs `fak debug --cmd html` without --out still gets a usable
// file; an explicit --out is honored verbatim.
func debugHTMLReport(im *cdb.Image, dir, session, query, out string) {
	if out == "" || out == "cdb-report.json" {
		out = "cdb-report.html"
	}
	f, err := os.Create(out)
	must(err)
	defer f.Close()
	source := session
	if source == "" {
		source = "synthetic committed fixture (testdata/cdb/session.jsonl)"
	}
	must(im.HTMLReport(ctx(), query, dir, source, f))
	info := im.Info()
	fmt.Printf("\n== fak debug html: %s  (core image %s/) ==\n", info.SessionID, dir)
	fmt.Printf("page table        : %d pages = %d benign + %d sealed + %d tombstoned\n",
		info.Pages, info.Benign, info.Sealed, info.Tombstoned)
	fmt.Printf("swap device       : %d B raw, %d distinct blobs, dedup saved %d B\n",
		info.CASBytes, info.DistinctBlobs, info.DedupSaved)
	fmt.Printf("panels            : decomposition · backtrace timeline · quarantine/seal panel · working-set residency · examine\n")
	fmt.Printf("honest fence      : sealed decisions are inherited (detector is evadable); cdb makes them durable, not more correct\n")
	fmt.Printf("report written    : %s  (open in a browser)\n", out)
}

func printBacktrace(im *cdb.Image) {
	for _, f := range im.Backtrace() {
		tag := "     "
		if f.Tombstoned {
			tag = "TOMB "
		} else if f.Sealed {
			tag = "SEAL "
		} else if f.Heavy {
			tag = "heavy"
		}
		fmt.Printf("  [%2d] %s %-14s %7dB  %s\n", f.Step, tag, f.Role, f.Len, f.Descriptor)
	}
}

func printWorkingSet(ws cdb.WorkingSet) {
	fmt.Printf("  working set W(query): %d of %d benign page(s) referenced; %d sealed excluded; %d tombstoned skipped\n",
		ws.PagesTouched, ws.PagesBenign, ws.SealedSkipped, ws.TombstonedSkipped)
	fmt.Printf("  demand-paged %d B of %d resident B = %.2f%% residency  (%d page-fault(s) avoided; poison in set: %v)\n",
		ws.BytesPagedIn, ws.ResidentBytes, ws.ResidencyPct, ws.FaultsAvoided, ws.PoisonInSet)
}

func printContextQuery(res contextq.Result, out string) {
	fmt.Printf("\n== fak debug context-query ==\n")
	fmt.Printf("query            : %q\n", res.Query)
	fmt.Printf("frames           : %d page-table row(s)\n", len(res.Frames))
	fmt.Printf("materialized     : %d slice(s), %d view record(s), %d render item(s), ~%d token(s)\n",
		len(res.Slices), len(res.Views), len(res.RenderPlan.Items), res.RenderPlan.EstimatedTokens)
	fmt.Printf("verdicts         : %d HIT, %d RECOMPUTE; %d raw byte(s) paged, %d rendered\n",
		res.Stats.ViewHits, res.Stats.ViewRecomputes, res.Stats.BytesPagedIn, res.Stats.RenderedBytes)
	fmt.Printf("refused/omitted  : %d refused, %d omitted\n", len(res.Refused), len(res.Omissions))
	for _, r := range res.Refused {
		fmt.Printf("  REFUSE step %d %-14s %s\n", r.Step, r.Role, r.Reason)
	}
	for _, o := range res.Omissions {
		fmt.Printf("  OMIT   step %d %-14s %s\n", o.Step, o.Role, o.Reason)
	}
	if out != "" {
		fmt.Printf("report written   : %s\n", out)
	}
}

func previewBytes(b []byte, max int) string {
	s := strings.TrimSpace(string(b))
	if len(s) > max {
		return s[:max] + fmt.Sprintf("… (+%d B)", len(s)-max)
	}
	return s
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func imageExists(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "manifest.json"))
	return err == nil
}

// cdbFixturePath resolves the committed synthetic session fixture relative to cwd or
// the executable, mirroring traceDir().
func cdbFixturePath() string {
	rel := filepath.Join("testdata", "cdb", "session.jsonl")
	if _, err := os.Stat(rel); err == nil {
		return rel
	}
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), rel)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return rel
}

// transcriptCandidate is one discovered Claude Code session file.
type transcriptCandidate struct {
	path  string
	size  int64
	mtime time.Time
}

// listTranscripts discovers real Claude Code session transcripts on this machine
// and prints, most-recent first, the exact `fak debug --session <path>` to attach
// each one. It is the answer to "fak debug ran a demo — where is MY session?":
// transcripts live under <claude-home>/projects/<ns>/<uuid>.jsonl, a path no
// operator memorizes. Read-only and bounded (a fixed two-level glob per root,
// never a recursive walk).
func listTranscripts(w io.Writer) {
	var roots []string
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		// ~/.claude, ~/.claude-<variant> (the per-host state trees this fleet uses).
		if ms, _ := filepath.Glob(filepath.Join(home, ".claude*")); ms != nil {
			roots = append(roots, ms...)
		}
	}
	if cfg := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); cfg != "" {
		roots = append(roots, cfg)
	}

	seen := map[string]bool{}
	var found []transcriptCandidate
	for _, root := range roots {
		// <root>/projects/<ns>/<session>.jsonl — a fixed-depth glob, not a walk.
		matches, _ := filepath.Glob(filepath.Join(root, "projects", "*", "*.jsonl"))
		for _, p := range matches {
			if seen[p] {
				continue
			}
			seen[p] = true
			info, err := os.Stat(p)
			if err != nil || info.IsDir() {
				continue
			}
			found = append(found, transcriptCandidate{path: p, size: info.Size(), mtime: info.ModTime()})
		}
	}

	if len(found) == 0 {
		fmt.Fprintln(w, "fak debug --list: no Claude Code transcripts found.")
		fmt.Fprintln(w, "  looked under: ~/.claude*/projects/*/*.jsonl"+claudeConfigHint())
		fmt.Fprintln(w, "  set CLAUDE_CONFIG_DIR, or pass --session <path.jsonl> directly.")
		return
	}

	sort.Slice(found, func(i, j int) bool { return found[i].mtime.After(found[j].mtime) })

	const max = 15
	fmt.Fprintf(w, "found %d Claude Code transcript(s); most recent first", len(found))
	if len(found) > max {
		fmt.Fprintf(w, " (showing %d)", max)
	}
	fmt.Fprintln(w, ":")
	for i, c := range found {
		if i >= max {
			break
		}
		fmt.Fprintf(w, "  [%2d] %s  %7s  %s\n", i+1, c.mtime.Format("2006-01-02 15:04"), humanBytes(c.size), filepath.Base(c.path))
		fmt.Fprintf(w, "       fak debug --session %q\n", c.path)
	}
}

func claudeConfigHint() string {
	if cfg := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); cfg != "" {
		return " and " + filepath.Join(cfg, "projects", "*", "*.jsonl")
	}
	return ""
}

// humanBytes renders a byte count compactly for the transcript listing.
func humanBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1fM", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1fK", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%dB", n)
	}
}
