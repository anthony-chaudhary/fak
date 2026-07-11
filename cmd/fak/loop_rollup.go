package main

import (
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/maputil"
)

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

	ids := maputil.SortedKeys(agg)
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
