package trajctl

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	QuoteSchema         = "trajctl-quote.v1"
	QuoteBacktestSchema = "trajctl-quote-backtest.v1"
	RepoQuestionClass   = "repo_question"
	MinQuoteSamples     = 2
)

var ErrUnsupportedColdStart = errors.New("unsupported cold-start quote")

type CapabilitySnapshot struct {
	Version           string `json:"version"`
	RepoRead          bool   `json:"repo_read"`
	Search            bool   `json:"search"`
	ToolPolicyVersion string `json:"tool_policy_version"`
}

type IndexSnapshot struct {
	Version  string  `json:"version"`
	Coverage float64 `json:"coverage"`
}

type QualityContract struct {
	Metric  string  `json:"metric"`
	Minimum float64 `json:"minimum"`
	Witness string  `json:"witness"`
}

type RouteTemplate struct {
	ModelClass string   `json:"model_class"`
	Tools      []string `json:"tools"`
	MaxTurns   int      `json:"max_turns"`
}

type CostEnvelope struct {
	P50  float64 `json:"p50"`
	P80  float64 `json:"p80"`
	P95  float64 `json:"p95"`
	Unit string  `json:"unit"`
}

type Quote struct {
	Schema       string             `json:"schema"`
	QuoteID      string             `json:"quote_id"`
	RequestClass string             `json:"request_class"`
	CreatedAt    string             `json:"created_at"`
	Capability   CapabilitySnapshot `json:"capability"`
	Index        IndexSnapshot      `json:"index"`
	Quality      QualityContract    `json:"quality"`
	Route        RouteTemplate      `json:"route"`
	Envelope     CostEnvelope       `json:"envelope"`
	SampleCount  int                `json:"sample_count"`
}

type QuoteRevision struct {
	Schema             string        `json:"schema"`
	QuoteID            string        `json:"quote_id"`
	Revision           int           `json:"revision"`
	CreatedAt          string        `json:"created_at"`
	Reason             string        `json:"reason"`
	Route              RouteTemplate `json:"route"`
	Envelope           CostEnvelope  `json:"envelope"`
	SupersedesRevision int           `json:"supersedes_revision"`
}

type QuoteCompletion struct {
	Schema          string          `json:"schema"`
	QuoteID         string          `json:"quote_id"`
	CreatedAt       string          `json:"created_at"`
	QualityScore    float64         `json:"quality_score"`
	QualityWitness  string          `json:"quality_witness"`
	Quality         QualityContract `json:"quality"`
	RawRealizedCost float64         `json:"raw_realized_cost"`
	CostUnit        string          `json:"cost_unit"`
	Censored        bool            `json:"censored,omitempty"`
}

type QuoteLedgerRecord struct {
	Kind       string           `json:"kind"`
	Quote      *Quote           `json:"quote,omitempty"`
	Revision   *QuoteRevision   `json:"revision_record,omitempty"`
	Completion *QuoteCompletion `json:"completion,omitempty"`
}

type QuoteObservation struct {
	ID              string             `json:"id"`
	At              string             `json:"at"`
	RequestClass    string             `json:"request_class"`
	Capability      CapabilitySnapshot `json:"capability"`
	Index           IndexSnapshot      `json:"index"`
	Quality         QualityContract    `json:"quality"`
	Route           RouteTemplate      `json:"route"`
	QualityScore    float64            `json:"quality_score"`
	QualityWitness  string             `json:"quality_witness"`
	RawRealizedCost float64            `json:"raw_realized_cost"`
	CostUnit        string             `json:"cost_unit"`
	Censored        bool               `json:"censored,omitempty"`
	Failure         string             `json:"failure,omitempty"`
}

type Coverage struct {
	Quantile        string  `json:"quantile"`
	Target          float64 `json:"target"`
	EmpiricalLower  float64 `json:"empirical_lower"`
	EmpiricalUpper  float64 `json:"empirical_upper"`
	KnownCovered    int     `json:"known_covered"`
	UnknownCensored int     `json:"unknown_censored"`
	Total           int     `json:"total"`
}

type QuoteBacktestReport struct {
	Schema            string     `json:"schema"`
	RequestClass      string     `json:"request_class"`
	CorpusSize        int        `json:"corpus_size"`
	Tested            int        `json:"tested"`
	ColdStartRefusals int        `json:"cold_start_refusals"`
	Censored          int        `json:"censored"`
	Coverage          []Coverage `json:"coverage"`
	Verdict           string     `json:"verdict"`
}

func validateQuoteInputs(cap CapabilitySnapshot, idx IndexSnapshot, q QualityContract, route RouteTemplate) error {
	if cap.Version == "" || idx.Version == "" || q.Metric == "" || q.Witness == "" || route.ModelClass == "" {
		return errors.New("capability, index, quality witness, and route snapshots must be versioned")
	}
	if !cap.RepoRead || !cap.Search || idx.Coverage <= 0 || q.Minimum <= 0 {
		return errors.New("repo_question prerequisites are not satisfied")
	}
	return nil
}

func NewRepoQuestionQuote(at time.Time, cap CapabilitySnapshot, idx IndexSnapshot, q QualityContract, route RouteTemplate, history []QuoteObservation) (Quote, error) {
	if err := validateQuoteInputs(cap, idx, q, route); err != nil {
		return Quote{}, err
	}
	costs := make([]float64, 0, len(history))
	unit := "cost_units"
	for _, h := range history {
		if h.RequestClass != RepoQuestionClass || h.QualityScore < q.Minimum || h.Capability.RepoRead != cap.RepoRead || h.Capability.Search != cap.Search {
			continue
		}
		if h.Censored || h.RawRealizedCost <= 0 {
			continue
		}
		costs = append(costs, h.RawRealizedCost)
		if h.CostUnit != "" {
			unit = h.CostUnit
		}
	}
	if len(costs) < MinQuoteSamples {
		return Quote{}, fmt.Errorf("%w: need %d comparable witnessed outcomes, got %d", ErrUnsupportedColdStart, MinQuoteSamples, len(costs))
	}
	sort.Float64s(costs)
	env := CostEnvelope{P50: empiricalQuantile(costs, .50), P80: empiricalQuantile(costs, .80), P95: empiricalQuantile(costs, .95), Unit: unit}
	seed, _ := json.Marshal(struct {
		At  string
		Cap CapabilitySnapshot
		Idx IndexSnapshot
		Q   QualityContract
		R   RouteTemplate
	}{at.UTC().Format(time.RFC3339Nano), cap, idx, q, route})
	sum := sha256.Sum256(seed)
	return Quote{Schema: QuoteSchema, QuoteID: "rq-" + hex.EncodeToString(sum[:8]), RequestClass: RepoQuestionClass, CreatedAt: at.UTC().Format(time.RFC3339Nano), Capability: cap, Index: idx, Quality: q, Route: route, Envelope: env, SampleCount: len(costs)}, nil
}

func empiricalQuantile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	n := int(math.Ceil(p*float64(len(sorted)))) - 1
	if n < 0 {
		n = 0
	}
	if n >= len(sorted) {
		n = len(sorted) - 1
	}
	return sorted[n]
}

func ReviseForCapabilityFailure(q Quote, revision int, at time.Time, reason string) QuoteRevision {
	route := q.Route
	route.ModelClass += "+fallback"
	route.MaxTurns = int(math.Ceil(float64(max(route.MaxTurns, 1)) * 1.5))
	return QuoteRevision{Schema: QuoteSchema, QuoteID: q.QuoteID, Revision: revision, CreatedAt: at.UTC().Format(time.RFC3339Nano), Reason: reason, Route: route, Envelope: CostEnvelope{P50: q.Envelope.P50 * 1.25, P80: q.Envelope.P80 * 1.5, P95: q.Envelope.P95 * 1.75, Unit: q.Envelope.Unit}, SupersedesRevision: revision - 1}
}

func AppendQuoteRecord(path string, rec QuoteLedgerRecord) error {
	if rec.Kind == "" {
		return errors.New("record kind is required")
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err = f.Write(append(b, '\n')); err != nil {
		return err
	}
	return f.Sync()
}

func ReadQuoteCorpus(r io.Reader) ([]QuoteObservation, error) {
	var out []QuoteObservation
	s := bufio.NewScanner(r)
	line := 0
	for s.Scan() {
		line++
		if strings.TrimSpace(s.Text()) == "" {
			continue
		}
		var v QuoteObservation
		if err := json.Unmarshal(s.Bytes(), &v); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, v)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].At < out[j].At })
	return out, nil
}

func BacktestRepoQuestion(obs []QuoteObservation) QuoteBacktestReport {
	rep := QuoteBacktestReport{Schema: QuoteBacktestSchema, RequestClass: RepoQuestionClass, CorpusSize: len(obs)}
	type tally struct{ known, unknown int }
	ts := []tally{{}, {}, {}}
	for i, o := range obs {
		q, err := NewRepoQuestionQuote(parseTime(o.At), o.Capability, o.Index, o.Quality, o.Route, obs[:i])
		if err != nil {
			if errors.Is(err, ErrUnsupportedColdStart) {
				rep.ColdStartRefusals++
			}
			continue
		}
		rep.Tested++
		if o.Censored {
			rep.Censored++
		}
		vals := []float64{q.Envelope.P50, q.Envelope.P80, q.Envelope.P95}
		for j, v := range vals {
			if o.Censored && o.RawRealizedCost <= v {
				ts[j].unknown++
			} else if !o.Censored && o.RawRealizedCost <= v {
				ts[j].known++
			}
		}
	}
	names := []string{"p50", "p80", "p95"}
	targets := []float64{.5, .8, .95}
	for i, t := range ts {
		lower, upper := 0.0, 0.0
		if rep.Tested > 0 {
			lower = float64(t.known) / float64(rep.Tested)
			upper = float64(t.known+t.unknown) / float64(rep.Tested)
		}
		rep.Coverage = append(rep.Coverage, Coverage{Quantile: names[i], Target: targets[i], EmpiricalLower: lower, EmpiricalUpper: upper, KnownCovered: t.known, UnknownCensored: t.unknown, Total: rep.Tested})
	}
	if rep.Tested == 0 {
		rep.Verdict = "INCONCLUSIVE"
	} else {
		rep.Verdict = "OBSERVED"
	}
	return rep
}

func parseTime(s string) time.Time { t, _ := time.Parse(time.RFC3339, s); return t }
