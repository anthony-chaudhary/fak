package trajhook

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/simhash"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// Finding is one scorer's verdict about a turn (or a trace): a label naming what was
// found, a numeric score (higher = more notable), the turn's identity, and a human
// reason. A gardening skill collects Findings and acts on them — surface the worst,
// propose prunes for the duplicates, alert on the cost outliers.
type Finding struct {
	Scorer  string  `json:"scorer"`            // which scorer produced this
	Label   string  `json:"label"`             // e.g. "duplicate_query", "cost_outlier", "high_deny_rate"
	Score   float64 `json:"score"`             // notability, higher = more notable
	TraceID string  `json:"trace_id"`          // the turn's trace
	Seq     int     `json:"seq,omitempty"`     // the turn within the trace (0 for trace-level findings)
	Query   string  `json:"query,omitempty"`   // the query the finding is about
	Reason  string  `json:"reason,omitempty"`  // human-readable explanation
	Related string  `json:"related,omitempty"` // a related turn key (e.g. the duplicate it matched)
}

// Scorer scores ONE turn in the context of the whole corpus, returning zero or more
// findings. The corpus is passed so a per-turn scorer can compare against its
// neighbors (the near-duplicate scorer needs this); a scorer that ignores it is a
// pure per-turn function. A Scorer must not mutate either argument.
type Scorer func(t trajectory.Turn, corpus []trajectory.Turn) []Finding

// CorpusScorer scores the WHOLE corpus at once, for signals that are inherently
// cross-turn or per-trace (deny rate over a trace, cost distribution). It returns all
// findings for the corpus.
type CorpusScorer func(corpus []trajectory.Turn) []Finding

// Registry holds named scorers an application registers. The zero value is ready;
// register scorers, then Run over a corpus. Not safe for concurrent mutation — build
// it once, then run.
type Registry struct {
	perTurn map[string]Scorer
	corpus  map[string]CorpusScorer
	order   []string // registration order, for deterministic Run output
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{perTurn: map[string]Scorer{}, corpus: map[string]CorpusScorer{}}
}

// Register adds a per-turn scorer under name (replacing any prior one of that name).
func (r *Registry) Register(name string, s Scorer) {
	if _, dup := r.perTurn[name]; !dup {
		if _, dup2 := r.corpus[name]; !dup2 {
			r.order = append(r.order, name)
		}
	}
	r.perTurn[name] = s
}

// RegisterCorpus adds a corpus-level scorer under name.
func (r *Registry) RegisterCorpus(name string, s CorpusScorer) {
	if _, dup := r.corpus[name]; !dup {
		if _, dup2 := r.perTurn[name]; !dup2 {
			r.order = append(r.order, name)
		}
	}
	r.corpus[name] = s
}

// Names returns the registered scorer names in registration order.
func (r *Registry) Names() []string { return append([]string(nil), r.order...) }

// Run executes every registered scorer over the corpus and returns all findings,
// sorted by descending score (ties broken by trace id then seq) for a stable,
// worst-first list. Per-turn scorers see the whole corpus as context.
func (r *Registry) Run(corpus []trajectory.Turn) []Finding {
	var out []Finding
	for _, name := range r.order {
		if s, ok := r.perTurn[name]; ok {
			for _, t := range corpus {
				for _, f := range s(t, corpus) {
					f.Scorer = name
					out = append(out, f)
				}
			}
		}
		if s, ok := r.corpus[name]; ok {
			for _, f := range s(corpus) {
				f.Scorer = name
				out = append(out, f)
			}
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		if out[i].TraceID != out[j].TraceID {
			return out[i].TraceID < out[j].TraceID
		}
		return out[i].Seq < out[j].Seq
	})
	return out
}

// DefaultDuplicateThreshold is the cosine at which the reference embedder treats two
// queries as near-duplicates. It is calibrated to simhash's ACTUAL discrimination
// (a hashing-trick sketch, not a learned model): genuine paraphrases that share most
// words land ~0.70–0.78, while same-intent queries phrased with DIFFERENT vocabulary
// ("delete rows" vs "drop records") land ~0.35 — below this. So 0.70 catches the
// real redundancy a lexical ranker misses without firing on distinct work. A
// deployment that swaps in semantic embeddings raises this toward 0.9; that tuning
// freedom is the point — fak ships the seam, not the policy.
const DefaultDuplicateThreshold = 0.70

// Default returns a Registry preloaded with the three reference scorers — the
// day-one gardening toolkit. An application calls Register/RegisterCorpus to add its
// own on top, or builds a NewRegistry from scratch to use only its own.
func Default() *Registry {
	r := NewRegistry()
	r.Register("duplicate_query", DuplicateQuery(DefaultDuplicateThreshold))
	r.RegisterCorpus("cost_outlier", CostOutlier(0.95))
	r.RegisterCorpus("deny_rate", DenyRate(0.5, 2))
	return r
}

// turnKey is the corpus-stable identity of a turn (matches trajectory's index key).
func turnKey(t trajectory.Turn) string { return t.TraceID + ":" + itoa(t.Seq) }

// ---------------------------------------------------------------------------
// Reference scorers (EXAMPLES, not policy).
// ---------------------------------------------------------------------------

// DuplicateQuery flags a turn whose query is a near-duplicate (cosine >= threshold)
// of an EARLIER turn's query — redundant work a gardening skill can dedupe or cache.
// It uses simhash so it catches paraphrases the lexical ranker misses. The Score is
// the cosine to the closest earlier match; Related names that match.
func DuplicateQuery(threshold float64) Scorer {
	return func(t trajectory.Turn, corpus []trajectory.Turn) []Finding {
		if t.Query == "" {
			return nil
		}
		qv := t.QueryEmbedding
		if len(qv) == 0 {
			qv = simhash.Embed(t.Query)
		}
		best := -1.0
		bestKey := ""
		bestQuery := ""
		for _, o := range corpus {
			if o.Query == "" || sameTurn(o, t) {
				continue
			}
			// "earlier" = strictly before t in corpus order; corpus is trace-then-seq
			// ordered, so compare by (trace, seq).
			if !before(o, t) {
				continue
			}
			ov := o.QueryEmbedding
			if len(ov) == 0 {
				ov = simhash.Embed(o.Query)
			}
			if c := simhash.Cosine(qv, ov); c > best {
				best, bestKey, bestQuery = c, turnKey(o), o.Query
			}
		}
		if best < threshold {
			return nil
		}
		return []Finding{{
			Label:   "duplicate_query",
			Score:   best,
			TraceID: t.TraceID,
			Seq:     t.Seq,
			Query:   t.Query,
			Reason:  "near-duplicate of earlier query " + bestKey + " (" + truncate(bestQuery, 60) + ")",
			Related: bestKey,
		}}
	}
}

// CostOutlier flags turns whose TokenEstimate is at or above the given quantile of
// the corpus token distribution — the expensive turns worth investigating. Score is
// the turn's token estimate. A corpus with no token costs yields nothing.
func CostOutlier(quantile float64) CorpusScorer {
	return func(corpus []trajectory.Turn) []Finding {
		costs := make([]int, 0, len(corpus))
		for _, t := range corpus {
			if t.TokenEstimate > 0 {
				costs = append(costs, t.TokenEstimate)
			}
		}
		if len(costs) < 2 {
			return nil
		}
		cutoff := quantileInt(costs, quantile)
		var out []Finding
		for _, t := range corpus {
			if t.TokenEstimate >= cutoff && t.TokenEstimate > 0 {
				out = append(out, Finding{
					Label:   "cost_outlier",
					Score:   float64(t.TokenEstimate),
					TraceID: t.TraceID,
					Seq:     t.Seq,
					Query:   t.Query,
					Reason:  "token cost " + itoa(t.TokenEstimate) + " >= p" + itoa(int(quantile*100)) + " cutoff " + itoa(cutoff),
				})
			}
		}
		return out
	}
}

// DenyRate flags a TRACE whose fraction of DENY/QUARANTINE turns is at or above
// minRate (over at least minTurns turns) — a trajectory the kernel kept refusing,
// often a sign of a confused or adversarial agent loop worth a human look. Score is
// the deny fraction; the finding is trace-level (Seq 0).
func DenyRate(minRate float64, minTurns int) CorpusScorer {
	return func(corpus []trajectory.Turn) []Finding {
		type acc struct{ denied, total int }
		byTrace := map[string]*acc{}
		order := []string{}
		for _, t := range corpus {
			a, ok := byTrace[t.TraceID]
			if !ok {
				a = &acc{}
				byTrace[t.TraceID] = a
				order = append(order, t.TraceID)
			}
			a.total++
			if t.Verdict == "DENY" || t.Verdict == "QUARANTINE" {
				a.denied++
			}
		}
		var out []Finding
		for _, id := range order {
			a := byTrace[id]
			if a.total < minTurns {
				continue
			}
			rate := float64(a.denied) / float64(a.total)
			if rate >= minRate {
				out = append(out, Finding{
					Label:   "high_deny_rate",
					Score:   rate,
					TraceID: id,
					Reason:  "trace refused " + itoa(a.denied) + "/" + itoa(a.total) + " turns",
				})
			}
		}
		return out
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func sameTurn(a, b trajectory.Turn) bool {
	return a.TraceID == b.TraceID && a.Seq == b.Seq
}

// before reports whether a precedes b in corpus order (trace then seq). Same trace =>
// lower seq is earlier; different traces are ordered by trace id (matching the
// trace-then-seq corpus layout used by trajectory.Turns).
func before(a, b trajectory.Turn) bool {
	if a.TraceID != b.TraceID {
		return a.TraceID < b.TraceID
	}
	return a.Seq < b.Seq
}

// quantileInt returns the value at the given quantile (0..1) of xs using the
// nearest-rank method. xs is copied before sorting (the caller's slice is untouched).
func quantileInt(xs []int, q float64) int {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]int(nil), xs...)
	sort.Ints(cp)
	if q <= 0 {
		return cp[0]
	}
	if q >= 1 {
		return cp[len(cp)-1]
	}
	rank := int(q*float64(len(cp)-1) + 0.5)
	if rank >= len(cp) {
		rank = len(cp) - 1
	}
	return cp[rank]
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
