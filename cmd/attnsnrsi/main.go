// Command attnsnrsi closes the attention-S/N RSI loop in SHADOW (#867).
//
// It replays candidate ctxplan Forecasts over a RECORDED session and folds each
// through the rsiloop keep-bit (internal/rsiloop -> internal/shipgate), so a candidate
// forecast is KEPT only when it raises the witnessed attention-S/N over the running
// baseline AND the suite is green AND the worktree is truth-clean — a real, dos-verified
// S/N gain, never a self-report. This is the wiring snfitness.go documents as "a driver
// wires it in one line", made runnable: it connects ctxplan.WitnessedSNFitness (the
// #867 fitness scalar, 3227b7a) to the rsiloop Harness whose Journal already tracks the
// metric against main over time and whose breaker early-exits a bad streak — the
// multi-session S/N trend #867's acceptance calls for.
//
// SHADOW BY CONSTRUCTION. The replay is a pure offline scoring of recorded turns; it
// changes no live plan. Adopting a kept forecast into the live planner is the separate
// #858 two-posture flag flip, gated by experiment #866 (reward vs exact-eviction
// leave-one-out). This driver PRECEDES that flip — it proves the loop closes on a
// witnessed reward, it does not enforce a keep onto the planning path.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// sessionDoc is the on-disk shape attnsnrsi replays: the baseline Forecast (scored as
// the running bar to beat), the candidate Forecasts to try in order, and the recorded
// session turns every fitness is measured over. The witnessed per-span Attribution and
// demand-page Faults live on each Turn (ctxplan.Turn) — the model-witnessed reward the
// keep-bit closes on.
type sessionDoc struct {
	Baseline    ctxplan.Forecast   `json:"baseline"`
	Candidates  []ctxplan.Forecast `json:"candidates"`
	Session     []ctxplan.Turn     `json:"session"`
	BaselineRef string             `json:"baseline_ref,omitempty"`
}

// attentionSNHarness wires ctxplan.WitnessedSNFitness into an rsiloop.Harness — the
// one-line driver snfitness.go documents, made real.
//
// MetricName labels the journal KPI; LowerBetter is false because a HIGHER witnessed
// attention-S/N is better. BaselineMetric scores the baseline forecast; Measure scores a
// candidate. The keep-bit (shipgate.Evaluate, inside rsiloop.Run) then KEEPS a candidate
// only on a STRICT metric gain that is ALSO suiteGreen and truthClean — the "real,
// dos-verified S/N gain" #867 requires. suiteGreen/truthClean are supplied by the
// caller's real gates (`make ci` and `dos verify`), never asserted by the candidate, so
// the keep stays non-forgeable: a forecast that merely CLAIMS a gain cannot move the bit.
func attentionSNHarness(doc sessionDoc, suiteGreen, truthClean bool) rsiloop.Harness {
	ref := doc.BaselineRef
	if ref == "" {
		ref = "main"
	}
	return rsiloop.Harness{
		MetricName:      "attention_sn",
		LowerBetter:     false,
		BaselineRefName: ref,
		BaselineMetric: func() (float64, string, error) {
			return ctxplan.WitnessedSNFitness(doc.Baseline, doc.Session), ref, nil
		},
		Candidates: func() []rsiloop.Candidate {
			cands := make([]rsiloop.Candidate, len(doc.Candidates))
			for i, f := range doc.Candidates {
				cands[i] = rsiloop.Candidate{Label: fmt.Sprintf("forecast#%d", i+1), Payload: f}
			}
			return cands
		},
		Measure: func(c rsiloop.Candidate) (rsiloop.Measurement, error) {
			f, ok := c.Payload.(ctxplan.Forecast)
			if !ok {
				return rsiloop.Measurement{}, fmt.Errorf("attnsnrsi: candidate %q payload is %T, want ctxplan.Forecast", c.Label, c.Payload)
			}
			return rsiloop.Measurement{
				Metric:     ctxplan.WitnessedSNFitness(f, doc.Session),
				SuiteGreen: suiteGreen,
				TruthClean: truthClean,
			}, nil
		},
	}
}

// loadSession decodes a sessionDoc from r, rejecting unknown fields so a malformed
// session (a typo'd key, a stale schema) fails loudly rather than silently scoring an
// empty plan.
func loadSession(r io.Reader) (sessionDoc, error) {
	var doc sessionDoc
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&doc); err != nil {
		return sessionDoc{}, fmt.Errorf("decode session: %w", err)
	}
	return doc, nil
}

func main() {
	sessionPath := flag.String("session", "", "path to the recorded session JSON (baseline+candidates+turns); '-' reads stdin")
	journalPath := flag.String("journal", "-", "append-only JSONL journal path ('-' = stdout) — the S/N trend over time")
	suiteGreen := flag.Bool("suite-green", false, "the suite verdict for candidates (a real run wires this from `make ci`)")
	truthClean := flag.Bool("truth-clean", false, "the truth-syscall verdict (a real run wires this from `dos verify`)")
	breaker := flag.Int("breaker", 3, "consecutive non-keeps before the breaker ESCALATEs (early-exits a bad streak)")
	flag.Parse()

	if *sessionPath == "" {
		fmt.Fprintln(os.Stderr, "attnsnrsi: -session is required (use '-' for stdin)")
		os.Exit(2)
	}

	var in io.Reader = os.Stdin
	if *sessionPath != "-" {
		f, err := os.Open(*sessionPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "attnsnrsi:", err)
			os.Exit(1)
		}
		defer f.Close()
		in = f
	}
	doc, err := loadSession(in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "attnsnrsi:", err)
		os.Exit(1)
	}

	j, err := rsiloop.NewJournal(*journalPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "attnsnrsi:", err)
		os.Exit(1)
	}
	defer j.Close()

	h := attentionSNHarness(doc, *suiteGreen, *truthClean)
	res, err := rsiloop.Run(h, j, *breaker, 0)
	if err != nil {
		fmt.Fprintln(os.Stderr, "attnsnrsi:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "attnsnrsi: metric=attention_sn cycles=%d kept=%d final=%s final_baseline=%.4f escalated=%v\n",
		res.Cycles, res.Kept, res.Final, res.FinalBaseline, res.Escalated)
}
