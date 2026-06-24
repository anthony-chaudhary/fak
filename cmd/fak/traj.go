package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/simhash"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/trajhook"
)

// cmdTraj handles `fak traj <subcommand>` over a TRAJECTORY CORPUS — the JSONL of
// per-turn Turn rows that a trajectory.Recorder exports (the gateway writes one when
// trajectory recording is enabled; a test/offline run produces one the same way).
// These are the GARDENING verbs an agent skill drives: fak ships the data plane, the
// reference similarity primitive, and the scorer seam, and the skill builds the
// semantic policy on top.
//
//	similar  --corpus C --query Q [--k N]   — the k past queries most like Q (simhash cosine)
//	cluster  --corpus C [--threshold T]      — group near-duplicate queries into clusters
//	score    --corpus C [--json]             — run the registered scorers, list findings worst-first
//	gc       --corpus C [--threshold T]      — propose prune candidates (later near-duplicates)
//	export   --corpus C                      — re-emit the corpus as JSONL (validate/normalize)
//
// Every verb reads a corpus file; none mutates it (gc PROPOSES, it never deletes) —
// the prune decision belongs to the skill/operator, not the kernel.
func cmdTraj(args []string) {
	if len(args) == 0 {
		trajUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "similar":
		cmdTrajSimilar(args[1:])
	case "cluster":
		cmdTrajCluster(args[1:])
	case "score":
		cmdTrajScore(args[1:])
	case "gc":
		cmdTrajGC(args[1:])
	case "export":
		cmdTrajExport(args[1:])
	case "-h", "--help", "help":
		trajUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak traj: unknown subcommand %q\n", args[0])
		trajUsage()
		os.Exit(2)
	}
}

func trajUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak traj similar --corpus <turns.jsonl> --query <text> [--k 5]   (k most-similar past queries)")
	fmt.Fprintln(os.Stderr, "       fak traj cluster --corpus <turns.jsonl> [--threshold 0.9]         (group near-duplicate queries)")
	fmt.Fprintln(os.Stderr, "       fak traj score   --corpus <turns.jsonl> [--json]                  (run scorers; findings worst-first)")
	fmt.Fprintln(os.Stderr, "       fak traj gc      --corpus <turns.jsonl> [--threshold 0.92]        (propose prune candidates)")
	fmt.Fprintln(os.Stderr, "       fak traj export  --corpus <turns.jsonl>                           (re-emit normalized JSONL)")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "A corpus is the JSONL a trajectory.Recorder exports. fak ships the data + similarity +")
	fmt.Fprintln(os.Stderr, "scorer seam; build your own trajectory analysis on top (see docs/observability/trajectory.md).")
}

// loadCorpus reads a trajectory JSONL file into a Recorder, exiting with a clear
// message on failure. Every verb starts here.
func loadCorpus(verb, path string) *trajectory.Recorder {
	if path == "" {
		fmt.Fprintf(os.Stderr, "fak traj %s: --corpus is required\n", verb)
		os.Exit(2)
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak traj %s: %v\n", verb, err)
		os.Exit(1)
	}
	defer f.Close()
	r, n, err := trajectory.ImportFrom(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak traj %s: read %s: %v\n", verb, path, err)
		os.Exit(1)
	}
	if n == 0 {
		fmt.Fprintf(os.Stderr, "fak traj %s: %s holds no turns\n", verb, path)
	}
	return r
}

// cmdTrajSimilar finds the k past queries most similar to --query by simhash cosine —
// the one call a gardening skill makes to ask "have we seen this before?".
func cmdTrajSimilar(args []string) {
	fs := flag.NewFlagSet("traj similar", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	query := fs.String("query", "", "query text to compare against the corpus")
	k := fs.Int("k", 5, "how many matches to return")
	asJSON := fs.Bool("json", false, "emit matches as JSON")
	_ = fs.Parse(args)

	if *query == "" {
		fmt.Fprintln(os.Stderr, "fak traj similar: --query is required")
		os.Exit(2)
	}
	r := loadCorpus("similar", *corpus)
	matches := r.Index().TopK(simhash.Embed(*query), *k)
	if *asJSON {
		emitJSON(matches)
		return
	}
	fmt.Printf("query: %q\n", *query)
	if len(matches) == 0 {
		fmt.Println("(no comparable queries in corpus)")
		return
	}
	for _, m := range matches {
		fmt.Printf("  %.3f  %s  %q\n", m.Score, m.ID, m.Meta)
	}
}

// cmdTrajCluster groups near-duplicate queries (cosine >= threshold) into clusters by
// single-link agglomeration — a coarse map of "the same work, done N times" a skill
// uses to find redundancy.
func cmdTrajCluster(args []string) {
	fs := flag.NewFlagSet("traj cluster", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	threshold := fs.Float64("threshold", trajhook.DefaultDuplicateThreshold, "cosine similarity to join a cluster")
	asJSON := fs.Bool("json", false, "emit clusters as JSON")
	_ = fs.Parse(args)

	r := loadCorpus("cluster", *corpus)
	turns := withQueries(r.Turns())
	clusters := clusterByQuery(turns, *threshold)

	if *asJSON {
		emitJSON(clusters)
		return
	}
	dupClusters := 0
	for _, c := range clusters {
		if len(c) > 1 {
			dupClusters++
		}
	}
	fmt.Printf("%d turns -> %d clusters (%d with duplicates) at cosine >= %.2f\n", len(turns), len(clusters), dupClusters, *threshold)
	for i, c := range clusters {
		if len(c) < 2 {
			continue // only show clusters that found redundancy
		}
		fmt.Printf("cluster %d (%d turns):\n", i+1, len(c))
		for _, t := range c {
			fmt.Printf("  %s:%d  %q\n", t.TraceID, t.Seq, t.Query)
		}
	}
}

// cmdTrajScore runs the registered reference scorers over the corpus and lists the
// findings worst-first — the bad-query / cost-outlier / high-deny-rate signals.
func cmdTrajScore(args []string) {
	fs := flag.NewFlagSet("traj score", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	_ = fs.Parse(args)

	r := loadCorpus("score", *corpus)
	findings := trajhook.Default().Run(r.Turns())
	if *asJSON {
		emitJSON(findings)
		return
	}
	if len(findings) == 0 {
		fmt.Println("no findings (corpus is clean by the reference scorers)")
		return
	}
	fmt.Printf("%d finding(s), worst first:\n", len(findings))
	for _, f := range findings {
		where := f.TraceID
		if f.Seq > 0 {
			where = fmt.Sprintf("%s:%d", f.TraceID, f.Seq)
		}
		fmt.Printf("  [%s] %.3f  %s  %s\n", f.Label, f.Score, where, f.Reason)
	}
}

// cmdTrajGC proposes prune candidates: each LATER near-duplicate of an earlier query
// (the duplicate_query scorer's Seq>0 findings). It only PROPOSES — printing the keys
// a skill or operator can then prune from its own memory store. fak never deletes a
// user's trajectory data.
func cmdTrajGC(args []string) {
	fs := flag.NewFlagSet("traj gc", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	threshold := fs.Float64("threshold", trajhook.DefaultDuplicateThreshold, "cosine similarity to treat as a duplicate")
	asJSON := fs.Bool("json", false, "emit prune candidates as JSON")
	_ = fs.Parse(args)

	r := loadCorpus("gc", *corpus)
	reg := trajhook.NewRegistry()
	reg.Register("duplicate_query", trajhook.DuplicateQuery(*threshold))
	findings := reg.Run(r.Turns())

	type candidate struct {
		Prune   string  `json:"prune"`  // the later duplicate's key (the prune target)
		KeepFor string  `json:"keep"`   // the earlier match it duplicates (keep this one)
		Cosine  float64 `json:"cosine"` // how similar
		Query   string  `json:"query"`  // the duplicate's query text
	}
	cands := make([]candidate, 0, len(findings))
	for _, f := range findings {
		cands = append(cands, candidate{
			Prune:   fmt.Sprintf("%s:%d", f.TraceID, f.Seq),
			KeepFor: f.Related,
			Cosine:  f.Score,
			Query:   f.Query,
		})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].Cosine > cands[j].Cosine })

	if *asJSON {
		emitJSON(cands)
		return
	}
	if len(cands) == 0 {
		fmt.Printf("no prune candidates at cosine >= %.2f (no redundant trajectories)\n", *threshold)
		return
	}
	fmt.Printf("%d prune candidate(s) at cosine >= %.2f (proposal only — fak deletes nothing):\n", len(cands), *threshold)
	for _, c := range cands {
		fmt.Printf("  prune %s (dup of %s, cos=%.3f)  %q\n", c.Prune, c.KeepFor, c.Cosine, c.Query)
	}
}

// cmdTrajExport re-emits the corpus as normalized JSONL on stdout — a sound copy that
// has been parsed and re-serialized (drops malformed lines, stable field order).
func cmdTrajExport(args []string) {
	fs := flag.NewFlagSet("traj export", flag.ExitOnError)
	corpus := fs.String("corpus", "", "trajectory JSONL corpus file")
	_ = fs.Parse(args)

	r := loadCorpus("export", *corpus)
	if _, err := r.ExportTo(os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "fak traj export: %v\n", err)
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func withQueries(turns []trajectory.Turn) []trajectory.Turn {
	out := turns[:0:0]
	for _, t := range turns {
		if t.Query != "" {
			out = append(out, t)
		}
	}
	return out
}

// clusterByQuery does single-link agglomerative clustering over query embeddings: a
// turn joins a cluster if it is within threshold cosine of ANY member. O(n^2) — fine
// for a trajectory corpus (thousands of turns), and dependency-free.
func clusterByQuery(turns []trajectory.Turn, threshold float64) [][]trajectory.Turn {
	vecs := make([]simhash.Vector, len(turns))
	for i, t := range turns {
		if len(t.QueryEmbedding) > 0 {
			vecs[i] = t.QueryEmbedding
		} else {
			vecs[i] = simhash.Embed(t.Query)
		}
	}
	assigned := make([]int, len(turns))
	for i := range assigned {
		assigned[i] = -1
	}
	var clusters [][]int // clusters of turn indices, so similarity reuses precomputed vecs
	for i := range turns {
		if assigned[i] >= 0 {
			continue
		}
		cid := len(clusters)
		assigned[i] = cid
		clusters = append(clusters, []int{i})
		// Single-link: pull in any later turn within threshold of any current member.
		for j := i + 1; j < len(turns); j++ {
			if assigned[j] >= 0 {
				continue
			}
			for _, mi := range clusters[cid] {
				if simhash.Cosine(vecs[j], vecs[mi]) >= threshold {
					assigned[j] = cid
					clusters[cid] = append(clusters[cid], j)
					break
				}
			}
		}
	}
	out := make([][]trajectory.Turn, len(clusters))
	for ci, members := range clusters {
		group := make([]trajectory.Turn, len(members))
		for gi, mi := range members {
			group[gi] = turns[mi]
		}
		out[ci] = group
	}
	return out
}

func emitJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "fak traj: json: %v\n", err)
		os.Exit(1)
	}
}
