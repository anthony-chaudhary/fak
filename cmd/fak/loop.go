package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func cmdLoop(argv []string) { os.Exit(runLoop(os.Stdout, os.Stderr, argv)) }

func runLoop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		loopUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "append":
		return runLoopAppend(stdout, stderr, argv[1:])
	case "run":
		return runLoopRun(stdout, stderr, argv[1:])
	case "status":
		return runLoopStatus(stdout, stderr, argv[1:])
	case "rollup":
		return runLoopRollup(stdout, stderr, argv[1:])
	case "admit":
		return runLoopAdmit(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		loopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak loop: unknown subcommand %q\n", argv[0])
		loopUsage(stderr)
		return 2
	}
}

func runLoopAppend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop append", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "loop id")
	kind := fs.String("kind", "", "event kind: armed|fire|admit|start|heartbeat|end|witness|notify")
	runID := fs.String("run", "", "run id")
	source := fs.String("source", "", "event source, such as schedule|github|slack|task-scheduler")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	state := fs.String("state", "", "loop state")
	status := fs.String("status", "", "run/admission/witness status")
	reason := fs.String("reason", "", "bounded reason token or short refusal code")
	summary := fs.String("summary", "", "bounded human summary")
	asJSON := fs.Bool("json", false, "emit the appended event as JSON")
	var evidence loopKVList
	var metrics loopKVList
	fs.Var(&evidence, "evidence", "repeatable KIND=REF evidence ref")
	fs.Var(&metrics, "metric", "repeatable NAME=INT64 metric")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop append: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	ev := loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventKind(*kind),
		Source:       *source,
		Principal:    *principal,
		State:        loopmgr.LoopState(*state),
		Status:       loopmgr.RunStatus(*status),
		Reason:       *reason,
		Summary:      *summary,
		EvidenceRefs: parseLoopEvidence(evidence),
	}
	if len(metrics) > 0 {
		ev.Metrics = map[string]int64{}
		for _, item := range metrics {
			k, v, ok := strings.Cut(item, "=")
			if !ok || strings.TrimSpace(k) == "" {
				fmt.Fprintf(stderr, "fak loop append: --metric must be NAME=INT64, got %q\n", item)
				return 2
			}
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil {
				fmt.Fprintf(stderr, "fak loop append: --metric %q has invalid value: %v\n", item, err)
				return 2
			}
			ev.Metrics[strings.TrimSpace(k)] = n
		}
	}

	appended, err := loopmgr.Append(*ledger, ev)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop append: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(appended); err != nil {
			fmt.Fprintf(stderr, "fak loop append: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "appended loop event seq=%d kind=%s loop=%s ledger=%s\n",
		appended.Seq, appended.Kind, appended.LoopID, *ledger)
	return 0
}

func runLoopRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	loopID := fs.String("loop", "", "loop id")
	runID := fs.String("run", "", "run id")
	source := fs.String("source", "manual", "trigger source, such as cron|launchd|task-scheduler|manual")
	principal := fs.String("principal", "", "authenticated principal or producer id")
	asJSON := fs.Bool("json", false, "emit a JSON run report")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	cmdArgs := fs.Args()
	if strings.TrimSpace(*loopID) == "" {
		fmt.Fprintln(stderr, "fak loop run: --loop is required")
		return 2
	}
	if len(cmdArgs) == 0 {
		fmt.Fprintln(stderr, "fak loop run: command is required after --")
		return 2
	}
	if *runID == "" {
		*runID = defaultLoopRunID(*loopID)
	}

	baseEvidence := []loopmgr.EvidenceRef{{Kind: "command", Ref: filepath.Base(cmdArgs[0])}}
	baseMetrics := map[string]int64{"argc": int64(len(cmdArgs))}
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventFire,
		Source:       *source,
		Principal:    *principal,
		Summary:      "loop run requested",
		EvidenceRefs: baseEvidence,
		Metrics:      cloneLoopMetrics(baseMetrics),
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventAdmit,
		Source:       *source,
		Principal:    *principal,
		Status:       loopmgr.StatusAdmitted,
		Reason:       "WRAPPER_ADMITTED",
		Summary:      "loop wrapper admitted command",
		EvidenceRefs: baseEvidence,
		Metrics:      cloneLoopMetrics(baseMetrics),
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	started := time.Now()
	if err := cmd.Start(); err != nil {
		m := cloneLoopMetrics(baseMetrics)
		m["exit_code"] = 127
		_ = appendLoopRunEvent(*ledger, loopmgr.Event{
			LoopID:       *loopID,
			RunID:        *runID,
			Kind:         loopmgr.EventEnd,
			Source:       *source,
			Principal:    *principal,
			Status:       loopmgr.StatusFailed,
			Reason:       "START_FAILED",
			Summary:      err.Error(),
			EvidenceRefs: baseEvidence,
			Metrics:      m,
		})
		fmt.Fprintf(stderr, "fak loop run: start command: %v\n", err)
		return 127
	}
	mStart := cloneLoopMetrics(baseMetrics)
	mStart["pid"] = int64(cmd.Process.Pid)
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventStart,
		Source:       *source,
		Principal:    *principal,
		Status:       loopmgr.StatusRunning,
		Reason:       "STARTED",
		Summary:      "child process started",
		EvidenceRefs: baseEvidence,
		Metrics:      mStart,
	}); err != nil {
		_ = cmd.Process.Kill()
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		return 1
	}

	waitErr := cmd.Wait()
	durationMS := time.Since(started).Milliseconds()
	exitCode := 0
	status := loopmgr.StatusClaimedDone
	reason := "EXIT_0"
	if waitErr != nil {
		status = loopmgr.StatusFailed
		reason = "EXIT_NONZERO"
		if ee, ok := waitErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = 1
			reason = "WAIT_FAILED"
		}
	}
	mEnd := cloneLoopMetrics(baseMetrics)
	mEnd["pid"] = int64(cmd.Process.Pid)
	mEnd["exit_code"] = int64(exitCode)
	mEnd["duration_ms"] = durationMS
	if err := appendLoopRunEvent(*ledger, loopmgr.Event{
		LoopID:       *loopID,
		RunID:        *runID,
		Kind:         loopmgr.EventEnd,
		Source:       *source,
		Principal:    *principal,
		Status:       status,
		Reason:       reason,
		Summary:      fmt.Sprintf("child exited with code %d", exitCode),
		EvidenceRefs: baseEvidence,
		Metrics:      mEnd,
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop run: %v\n", err)
		if exitCode == 0 {
			return 1
		}
	}
	if *asJSON {
		rep := map[string]any{
			"schema":      "fak.loop-run-report.v1",
			"ledger_path": *ledger,
			"loop_id":     *loopID,
			"run_id":      *runID,
			"exit_code":   exitCode,
			"duration_ms": durationMS,
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak loop run: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "loop run %s exit=%d ledger=%s\n", *runID, exitCode, *ledger)
	}
	return exitCode
}

func runLoopStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	asJSON := fs.Bool("json", false, "emit the full JSON status")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	st, err := loopmgr.SnapshotFile(*ledger, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak loop status: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(st); err != nil {
			fmt.Fprintf(stderr, "fak loop status: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderLoopStatus(stdout, st)
	return 0
}

// runLoopRollup is the cross-node read-only fold (#769, Pillar 7): it ingests N
// nodes' loop ledgers and renders ONE fleet-wide "how often did every loop run"
// view — per-loop run counts, cadence, and last-run, the per-node columns reusing
// the aligned `fak ps` table idiom. It only reads the journals: it appends no
// event and issues no control verb, so adding a node's journal changes only this
// rollup, never any node's behavior. This is aggregation, explicitly NOT consensus
// (epic §5) — it has no write path that could influence another node's admission.
//
//	fak loop rollup --ledger node-a.jsonl --ledger node-b.jsonl   explicit per-node ledgers
//	fak loop rollup --ledger mac=/path/loops.jsonl                NODE=PATH labels the node
//	fak loop rollup --dir .fleet-journals [--glob '*.jsonl']      every match is one node
//	fak loop rollup ... --json                                    machine-readable fold
func runLoopRollup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop rollup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var ledgers loopKVList
	fs.Var(&ledgers, "ledger", "repeatable node ledger: PATH or NODE=PATH (node id defaults to the file basename)")
	dir := fs.String("dir", "", "directory of per-node ledgers (each file matching --glob is one node)")
	glob := fs.String("glob", "*.jsonl", "filename glob used with --dir")
	asJSON := fs.Bool("json", false, "emit the fleet rollup as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop rollup: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	nodes, err := loopRollupNodes(ledgers, *dir, *glob)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop rollup: %v\n", err)
		return 2
	}
	if len(nodes) == 0 {
		fmt.Fprintln(stderr, "fak loop rollup: no node ledgers given (--ledger PATH ... or --dir DIR)")
		return 2
	}

	rep := foldLoopRollup(nodes, time.Now())
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak loop rollup: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	renderLoopRollup(stdout, rep)
	return 0
}

// loopRollupNode names one node's ledger: an id (for the per-node attribution
// column) and the path to its loop JSONL ledger.
type loopRollupNode struct {
	ID   string
	Path string
}

// loopRollupNodes builds the node list from the repeatable --ledger flags and the
// optional --dir scan, de-duplicating by path. A --ledger value of NODE=PATH
// labels the node explicitly; a bare PATH derives the id from the file basename.
func loopRollupNodes(ledgers []string, dir, glob string) ([]loopRollupNode, error) {
	var nodes []loopRollupNode
	seen := map[string]bool{}
	add := func(id, path string) {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			return
		}
		seen[path] = true
		if id == "" {
			id = nodeIDFromPath(path)
		}
		nodes = append(nodes, loopRollupNode{ID: id, Path: path})
	}
	for _, item := range ledgers {
		if k, v, ok := strings.Cut(item, "="); ok && strings.TrimSpace(k) != "" && strings.TrimSpace(v) != "" {
			add(strings.TrimSpace(k), v)
			continue
		}
		add("", item)
	}
	if strings.TrimSpace(dir) != "" {
		matches, err := filepath.Glob(filepath.Join(dir, glob))
		if err != nil {
			return nil, fmt.Errorf("glob %q: %w", glob, err)
		}
		sort.Strings(matches)
		for _, m := range matches {
			add("", m)
		}
	}
	return nodes, nil
}

// nodeIDFromPath derives a node id from a ledger path: the file basename without
// its extension (so node-a.jsonl -> node-a), falling back to the raw path.
func nodeIDFromPath(path string) string {
	base := filepath.Base(path)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	if base == "" || base == "." {
		return path
	}
	return base
}

// loopRollupReport (schema fak.loop-rollup.v1) is the machine-readable fleet fold:
// the node ids that contributed, any node journal that could not be read, and one
// aggregated row per loop id seen across all nodes.
type loopRollupReport struct {
	Schema     string           `json:"schema"`
	TSUnixNano int64            `json:"ts_unix_nano"`
	Nodes      []string         `json:"nodes"`
	Skipped    []loopRollupSkip `json:"skipped,omitempty"`
	Loops      []loopRollupRow  `json:"loops"`
}

// loopRollupSkip records a node ledger that could not be folded (a corrupt or
// forked journal). A read-only fleet view must not go dark on one bad node, so
// the fold skips it and surfaces why rather than aborting the whole rollup.
type loopRollupSkip struct {
	Node  string `json:"node"`
	Path  string `json:"path"`
	Error string `json:"error"`
}

// loopRollupRow is one loop's fleet-wide fold: how many nodes ran it, the summed
// run/admit/start/end/witness counts, the merged-timeline cadence, and the most
// recent event across nodes. Runs is the fire count — the canonical "the loop was
// triggered" marker the operator's "how often did it run" question asks about.
type loopRollupRow struct {
	LoopID            string   `json:"loop_id"`
	Nodes             int      `json:"nodes"`
	NodeIDs           []string `json:"node_ids,omitempty"`
	Runs              uint64   `json:"runs"`
	Admitted          uint64   `json:"admitted"`
	Refused           uint64   `json:"refused"`
	Started           uint64   `json:"started"`
	Ended             uint64   `json:"ended"`
	Witnessed         uint64   `json:"witnessed"`
	CadenceSeconds    float64  `json:"cadence_seconds,omitempty"`
	LastEventUnixNano int64    `json:"last_event_unix_nano,omitempty"`
}

// loopRollupAcc accumulates one loop's cross-node fold while the nodes are walked.
type loopRollupAcc struct {
	runs, admitted, refused, started, ended, witnessed uint64
	last                                               int64
	fireTS                                             []int64
	nodes                                              map[string]bool
}

// foldLoopRollup is the pure cross-node aggregation: load each node's ledger,
// summarize it with the same fold `fak loop status` uses, and sum the per-loop
// counts fleet-wide. Cadence is the mean interval of every loop's fire events
// merged across nodes (the fleet's "how often"); last-run is the latest event
// across nodes. An unreadable node ledger is skipped (recorded in Skipped), never
// fatal. Read-only: it opens journals and writes nothing.
func foldLoopRollup(nodes []loopRollupNode, now time.Time) loopRollupReport {
	rep := loopRollupReport{
		Schema:     "fak.loop-rollup.v1",
		TSUnixNano: now.UTC().UnixNano(),
	}
	agg := map[string]*loopRollupAcc{}
	get := func(id string) *loopRollupAcc {
		a := agg[id]
		if a == nil {
			a = &loopRollupAcc{nodes: map[string]bool{}}
			agg[id] = a
		}
		return a
	}
	for _, n := range nodes {
		rep.Nodes = append(rep.Nodes, n.ID)
		events, err := loopmgr.Load(n.Path)
		if err != nil {
			rep.Skipped = append(rep.Skipped, loopRollupSkip{Node: n.ID, Path: n.Path, Error: err.Error()})
			continue
		}
		for _, ls := range loopmgr.Summarize(events, now).Loops {
			a := get(ls.LoopID)
			a.runs += ls.Fires
			a.admitted += ls.Admitted
			a.refused += ls.Refused
			a.started += ls.Started
			a.ended += ls.Ended
			a.witnessed += ls.Witnessed
			if ls.LastEventUnixNano > a.last {
				a.last = ls.LastEventUnixNano
			}
			a.nodes[n.ID] = true
		}
		for _, ev := range events {
			if ev.Kind == loopmgr.EventFire {
				get(ev.LoopID).fireTS = append(get(ev.LoopID).fireTS, ev.TSUnixNano)
			}
		}
	}

	ids := make([]string, 0, len(agg))
	for id := range agg {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		a := agg[id]
		nodeIDs := make([]string, 0, len(a.nodes))
		for nid := range a.nodes {
			nodeIDs = append(nodeIDs, nid)
		}
		sort.Strings(nodeIDs)
		rep.Loops = append(rep.Loops, loopRollupRow{
			LoopID:            id,
			Nodes:             len(a.nodes),
			NodeIDs:           nodeIDs,
			Runs:              a.runs,
			Admitted:          a.admitted,
			Refused:           a.refused,
			Started:           a.started,
			Ended:             a.ended,
			Witnessed:         a.witnessed,
			CadenceSeconds:    cadenceSeconds(a.fireTS),
			LastEventUnixNano: a.last,
		})
	}
	return rep
}

// cadenceSeconds is the mean interval between runs: the span of the merged fire
// timestamps divided by the gaps between them. Fewer than two fires (or a
// zero-span burst) has no measurable cadence and returns 0.
func cadenceSeconds(ts []int64) float64 {
	if len(ts) < 2 {
		return 0
	}
	sorted := append([]int64(nil), ts...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	span := sorted[len(sorted)-1] - sorted[0]
	if span <= 0 {
		return 0
	}
	return float64(span) / float64(len(sorted)-1) / 1e9
}

// renderLoopRollup prints the fleet rollup as one aligned row per loop, reusing
// the `fak ps` tabwriter idiom for the per-run columns. RUNS is the fleet-wide
// fire count, CADENCE the mean interval between runs, LAST the most recent event.
func renderLoopRollup(w io.Writer, rep loopRollupReport) {
	if len(rep.Loops) == 0 {
		fmt.Fprintf(w, "no loops found across %d node(s)\n", len(rep.Nodes))
		for _, s := range rep.Skipped {
			fmt.Fprintf(w, "skipped node %s (%s): %s\n", s.Node, s.Path, s.Error)
		}
		return
	}
	fmt.Fprintf(w, "fak loop rollup — %d loop(s) across %d node(s)\n\n", len(rep.Loops), len(rep.Nodes))
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "LOOP\tNODES\tRUNS\tSTARTED\tENDED\tWITNESSED\tREFUSED\tCADENCE\tLAST")
	for _, l := range rep.Loops {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n",
			l.LoopID, l.Nodes, l.Runs, l.Started, l.Ended, l.Witnessed, l.Refused,
			humanCadence(l.CadenceSeconds), formatLoopTime(l.LastEventUnixNano))
	}
	_ = tw.Flush()
	for _, s := range rep.Skipped {
		fmt.Fprintf(w, "skipped node %s (%s): %s\n", s.Node, s.Path, s.Error)
	}
}

// humanCadence renders a mean-interval (seconds) in the dominant unit: "-" when
// there is no measurable cadence, else "45s" / "12.0m" / "1.1h" / "2.0d".
func humanCadence(sec float64) string {
	if sec <= 0 {
		return "-"
	}
	switch {
	case sec >= 86400:
		return fmt.Sprintf("%.1fd", sec/86400)
	case sec >= 3600:
		return fmt.Sprintf("%.1fh", sec/3600)
	case sec >= 60:
		return fmt.Sprintf("%.1fm", sec/60)
	case sec >= 10:
		return fmt.Sprintf("%.0fs", sec)
	default:
		// Sub-10s: keep a decimal so a real-but-tiny interval reads distinct from
		// "-" (no measurable cadence) instead of rounding down to a misleading "0s".
		return fmt.Sprintf("%.1fs", sec)
	}
}

// runLoopAdmit applies the tunable loop-admission policy to the folded ledger
// and reports an admit/refuse verdict per loop. This is the governor surface
// that makes the always-on loop tunable: a scheduler line gates work on
// `fak loop admit --loop ID` (exit 0 admit, exit 3 refused), and the operator
// retunes the policy file — pause, cadence floor, refusal-storm backoff,
// witness-collapse hold — without re-registering the OS task. It only reads:
// it appends no event, so a refusal here is not itself a recorded refusal.
func runLoopAdmit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop admit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	policyPath := fs.String("policy", defaultLoopPolicy(), "loop admission policy JSON path")
	loopID := fs.String("loop", "", "evaluate one loop id (default: every loop in the ledger)")
	asJSON := fs.Bool("json", false, "emit the decisions as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop admit: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	policies, err := loopmgr.LoadPolicies(*policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop admit: %v\n", err)
		return 2
	}
	now := time.Now()
	st, err := loopmgr.SnapshotFile(*ledger, now)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop admit: %v\n", err)
		return 1
	}

	var decisions []loopmgr.Decision
	if id := strings.TrimSpace(*loopID); id != "" {
		// A named loop the ledger has never seen still gets a verdict: an empty
		// snapshot under its policy, so an operator can pre-pause a loop that has
		// not fired yet, and a first-ever fire is evaluated rather than skipped.
		decisions = []loopmgr.Decision{loopmgr.Admit(loopSnapshotForID(st, id), policies.PolicyFor(id), now)}
	} else {
		decisions = loopmgr.AdmitAll(st, policies, now)
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(map[string]any{
			"schema":      "fak.loop-admit.v1",
			"ledger_path": *ledger,
			"policy_path": *policyPath,
			"decisions":   decisions,
		}); err != nil {
			fmt.Fprintf(stderr, "fak loop admit: encode json: %v\n", err)
			return 1
		}
	} else {
		for _, d := range decisions {
			verdict := "ADMIT"
			if !d.Admit {
				verdict = "REFUSE"
			}
			fmt.Fprintf(stdout, "%-6s %-30s %-18s %s\n", verdict, d.LoopID, d.Reason, d.Summary)
		}
		if len(decisions) == 0 {
			fmt.Fprintf(stdout, "no loops to admit (ledger %s)\n", *ledger)
		}
	}

	// Exit 3 when any evaluated loop is refused, so a scheduler can gate on it:
	//   fak loop admit --loop ID && python tick.py ...
	for _, d := range decisions {
		if !d.Admit {
			return 3
		}
	}
	return 0
}

// loopSnapshotForID returns the folded snapshot for a loop id, or an empty
// snapshot bearing just that id when the ledger has never seen it — so a policy
// can still be evaluated against a not-yet-fired loop.
func loopSnapshotForID(st loopmgr.Status, id string) loopmgr.LoopSnapshot {
	for _, l := range st.Loops {
		if l.LoopID == id {
			return l
		}
	}
	return loopmgr.LoopSnapshot{LoopID: id}
}

func defaultLoopLedger() string {
	if v := os.Getenv("FAK_LOOP_LEDGER"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loops.jsonl")
}

func defaultLoopPolicy() string {
	if v := os.Getenv("FAK_LOOP_POLICY"); v != "" {
		return v
	}
	return filepath.Join(".fak", "loop-policy.json")
}

func appendLoopRunEvent(ledger string, ev loopmgr.Event) error {
	_, err := loopmgr.Append(ledger, ev)
	return err
}

func cloneLoopMetrics(in map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func defaultLoopRunID(loopID string) string {
	replacer := strings.NewReplacer("/", "-", "\\", "-", ":", "-", " ", "-")
	name := strings.Trim(replacer.Replace(loopID), "-")
	if name == "" {
		name = "loop"
	}
	return fmt.Sprintf("%s-%s-%d", name, time.Now().UTC().Format("20060102T150405Z"), os.Getpid())
}

type loopKVList []string

func (l *loopKVList) String() string {
	if l == nil {
		return ""
	}
	return strings.Join(*l, ",")
}

func (l *loopKVList) Set(v string) error {
	*l = append(*l, v)
	return nil
}

func parseLoopEvidence(items []string) []loopmgr.EvidenceRef {
	out := make([]loopmgr.EvidenceRef, 0, len(items))
	for _, item := range items {
		kind, ref, ok := strings.Cut(item, "=")
		if !ok {
			continue
		}
		kind = strings.TrimSpace(kind)
		ref = strings.TrimSpace(ref)
		if kind == "" || ref == "" {
			continue
		}
		out = append(out, loopmgr.EvidenceRef{Kind: kind, Ref: ref})
	}
	return out
}

func renderLoopStatus(w io.Writer, st loopmgr.Status) {
	if len(st.Loops) == 0 {
		fmt.Fprintf(w, "no loops found (ledger %s)\n", st.LedgerPath)
		return
	}
	fmt.Fprintf(w, "loop ledger=%s loops=%d\n", st.LedgerPath, len(st.Loops))
	for _, loop := range st.Loops {
		state := loop.State
		if state == "" {
			state = "-"
		}
		lastRun := "-"
		if loop.LastRun != nil {
			lastRun = string(loop.LastRun.Status)
			if loop.LastRun.RunID != "" {
				lastRun = loop.LastRun.RunID + ":" + lastRun
			}
		}
		fmt.Fprintf(w, "loop %-28s state=%-20s fires=%d admitted=%d refused=%d started=%d ended=%d witnessed=%d notify=%d last=%s last_run=%s\n",
			loop.LoopID,
			state,
			loop.Fires,
			loop.Admitted,
			loop.Refused,
			loop.Started,
			loop.Ended,
			loop.Witnessed,
			loop.Notifications,
			formatLoopTime(loop.LastEventUnixNano),
			lastRun,
		)
	}
}

func formatLoopTime(ts int64) string {
	if ts == 0 {
		return "-"
	}
	return time.Unix(0, ts).UTC().Format(time.RFC3339)
}

func loopUsage(w io.Writer) {
	fmt.Fprint(w, `fak loop - durable long-running loop ledger

  fak loop append --loop ID --kind KIND [--ledger FILE] [--run ID]
                  [--source NAME] [--principal ID] [--status STATUS]
                  [--reason CODE] [--summary TEXT] [--evidence KIND=REF]
                  [--metric NAME=INT64] [--json]
  fak loop run --loop ID [--ledger FILE] [--source cron|launchd|task-scheduler] -- CMD [ARG...]
  fak loop status [--ledger FILE] [--json]
  fak loop rollup [--ledger PATH|NODE=PATH ...] [--dir DIR] [--glob '*.jsonl'] [--json]
  fak loop admit [--loop ID] [--ledger FILE] [--policy FILE] [--json]

Append records one scheduler/script/control event in the canonical hash-chained
ledger. Run wraps an OS scheduler command and records fire/admit/start/end around it.
Status folds that ledger into the current loop/run view. Rollup folds MANY nodes'
ledgers into one fleet-wide "how often did every loop run" view — per-loop run
counts, cadence, and last-run — reusing the fak ps table format; it is a read-only
aggregation that ingests journals and writes nothing. Admit applies the tunable
admission policy (default .fak/loop-policy.json, FAK_LOOP_POLICY) to the fold and
prints admit/refuse per loop — exit 3 when any evaluated loop is refused, so a
scheduler line can gate work on it. The ledger records events; admission, scheduler
authority, and completion witnesses live in producers.
`)
}
