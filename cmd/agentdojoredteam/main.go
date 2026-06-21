// Command agentdojoredteam runs the dynamic AgentDojo-style red-team battery
// (internal/agentdojo) against the stacked defense and prints the PER-ATTACK
// verdict stream, then folds every outcome into a frozen harvest LabelRow corpus.
//
// It is the runnable face of CLAIMS.md #76 (the dynamic attack battery that
// replaces the static testdata/poison.json fixture) and the example issue #243
// asks for. It is Go-only, in-process, and DETERMINISTIC — no model, no network,
// no API key — so the same invocation prints byte-identical output on every run
// and every platform (the battery is a fixed matrix; the --seed flag only fixes
// the report ordering, see below).
//
// WHAT IT SHOWS. For each attack in the expanded battery it scores TWO defenses:
//
//   - detection-only — content detectors alone (normgate + ctxmmu). An adaptive
//     PARAPHRASED injection carries no marker word, so it evades the lexical gate
//     and reaches context: ASR(detection-only) > 0.
//   - full-stack — the shipped configuration: detectors PLUS information-flow
//     control (IFC source-stamp + sink-gate). Reading untrusted content taints
//     the session by PROVENANCE (not content), so the attacker's egress/
//     destructive sink is barred regardless of phrasing: ASR(full-stack) == 0.
//
// That contrast is the thesis: detection and IFC are independent layers, and only
// their combination holds ASR at zero under an adaptive attacker. The exit code is
// the ASR gate — non-zero iff full-stack let any attack through — so run.sh / CI
// can treat a regression as a failure.
//
// USAGE:
//
//	go run ./cmd/agentdojoredteam            # full per-attack stream + summary
//	go run ./cmd/agentdojoredteam -seeds     # seed (hand-authored) battery only
//	go run ./cmd/agentdojoredteam -seed 7    # fix the report ordering with a seed
//	go run ./cmd/agentdojoredteam -json      # machine-readable outcome stream
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agentdojo"
	"github.com/anthony-chaudhary/fak/internal/harvest"
)

func main() {
	seedsOnly := flag.Bool("seeds", false, "run only the hand-authored seed Matrix(), not the generative expansion")
	seed := flag.Int64("seed", 0, "fix the report ordering deterministically (0 = stable by attack name)")
	asJSON := flag.Bool("json", false, "emit the per-attack outcome stream as JSON")
	flag.Parse()

	battery := agentdojo.ExpandedMatrix()
	if *seedsOnly {
		battery = agentdojo.Matrix()
	}
	battery = orderBattery(battery, *seed)

	ctx := context.Background()
	detection := agentdojo.NewDetectionOnly()
	fullStack := agentdojo.NewFullStack()

	// Fold every full-stack outcome into a frozen LabelRow corpus — the harvest
	// rung of the compiled loop. A non-Allow verdict (a CATCH) is a positive; an
	// Allow (a MISS) is a negative. We record the full-stack outcome because that
	// is the shipped defense the corpus trains the keep/revert gate against.
	corpus := harvest.NewCorpus()
	folder := harvest.New(corpus)

	rows := make([]outcomeRow, 0, len(battery))
	var detSucc, fullSucc int
	for _, a := range battery {
		det := detection.Run(ctx, a)
		full := fullStack.Run(ctx, a)
		if det.Succeeded {
			detSucc++
		}
		if full.Succeeded {
			fullSucc++
		}
		// Derive the LabelRow from the full-stack outcome and fold it through the
		// real harvest Emitter, so the corpus this example produces is built by
		// the same code path the compiled loop uses, not a hand-rolled shape.
		folder.Emit(labelEvent(a, full))
		rows = append(rows, outcomeRow{
			Name:        a.Name,
			Vector:      a.Vector.String(),
			Adaptivity:  a.Adaptivity.String(),
			ReadTool:    a.ReadTool,
			SinkTool:    a.SinkTool,
			DetReached:  det.InjectionReachedContext,
			DetSucc:     det.Succeeded,
			FullReached: full.InjectionReachedContext,
			FullSink:    full.SinkExecuted,
			FullSucc:    full.Succeeded,
		})
	}

	if *asJSON {
		emitJSON(rows, len(battery), detSucc, fullSucc, corpus)
	} else {
		emitText(rows, len(battery), detSucc, fullSucc, corpus, *seedsOnly)
	}

	// Exit code IS the ASR gate: non-zero iff the SHIPPED (full-stack) defense let
	// any attack through. detection-only ASR > 0 is expected and NOT a failure.
	if fullSucc > 0 {
		os.Exit(1)
	}
}

// outcomeRow is one attack's result against both defenses (the per-attack stream).
type outcomeRow struct {
	Name        string `json:"name"`
	Vector      string `json:"vector"`
	Adaptivity  string `json:"adaptivity"`
	ReadTool    string `json:"read_tool"`
	SinkTool    string `json:"sink_tool"`
	DetReached  bool   `json:"detection_reached_context"`
	DetSucc     bool   `json:"detection_attack_succeeded"`
	FullReached bool   `json:"fullstack_reached_context"`
	FullSink    bool   `json:"fullstack_sink_executed"`
	FullSucc    bool   `json:"fullstack_attack_succeeded"`
}

// labelEvent turns one full-stack Outcome into the abi.Event the harvest Emitter
// folds. A succeeded attack is a MISS (VerdictAllow — the gate let the harmful
// sink through); a blocked attack is a CATCH. The catch reason names HOW the
// full-stack defense barred it: a quarantine at the content layer is MALFORMED
// (the lexical gate caught a marker), while a sink barred by the IFC taint is a
// TRUST_VIOLATION (the provenance/scope rung — what stops the PARAPHRASED attacks
// the lexical gate misses). This mirrors the kernel's own decision stream, which
// is what harvest is built to ingest.
func labelEvent(a agentdojo.Attack, o agentdojo.Outcome) abi.Event {
	var v abi.Verdict
	switch {
	case o.Succeeded:
		v = abi.Verdict{Kind: abi.VerdictAllow, Reason: abi.ReasonNone}
	case !o.InjectionReachedContext:
		// The content detectors quarantined the injected read before it reached
		// context — a lexical catch.
		v = abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonMalformed}
	default:
		// The injection reached context but the IFC sink-gate denied the tainted
		// egress/destructive sink — the provenance rung, the one that holds under
		// an adaptive paraphrase.
		v = abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation}
	}
	call := &abi.ToolCall{
		Tool: a.SinkTool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(a.SinkArgs)},
	}
	return abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &v}
}

// orderBattery returns the battery in a stable, reproducible order. seed 0 sorts
// by attack name (fully deterministic, the default). A non-zero seed applies a
// deterministic Fisher-Yates shuffle keyed by that seed — so -seed fixes the
// ordering reproducibly without making the SET of attacks non-deterministic (the
// matrix is fixed; only the presentation order changes). This satisfies the
// "fixed random seed → deterministic" example requirement honestly: same seed,
// same order, every run.
func orderBattery(b []agentdojo.Attack, seed int64) []agentdojo.Attack {
	out := append([]agentdojo.Attack(nil), b...)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	if seed != 0 {
		r := rand.New(rand.NewSource(seed))
		r.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	}
	return out
}

func emitText(rows []outcomeRow, total, detSucc, fullSucc int, corpus *harvest.Corpus, seedsOnly bool) {
	battery := "expanded (seeds + generative paraphrase expansion)"
	if seedsOnly {
		battery = "seeds only (hand-authored Matrix)"
	}
	fmt.Printf("AgentDojo dynamic red-team — %s\n", battery)
	fmt.Printf("%d attacks scored against two defenses (detection-only, full-stack)\n\n", total)

	fmt.Printf("%-34s %-11s %-6s  %-13s  %-11s\n", "ATTACK", "ADAPTIVITY", "VECTOR", "DETECTION-ONLY", "FULL-STACK")
	fmt.Printf("%-34s %-11s %-6s  %-13s  %-11s\n", "------", "----------", "------", "--------------", "----------")
	for _, r := range rows {
		fmt.Printf("%-34s %-11s %-6s  %-13s  %-11s\n",
			r.Name, r.Adaptivity, r.Vector, verdictLabel(r.DetSucc), verdictLabel(r.FullSucc))
	}

	detASR := ratio(detSucc, total)
	fullASR := ratio(fullSucc, total)
	fmt.Printf("\nASR(detection-only) = %.3f  (%d/%d attacks landed — paraphrased injections evade the lexical gate)\n",
		detASR, detSucc, total)
	fmt.Printf("ASR(full-stack)     = %.3f  (%d/%d — IFC taints by provenance and bars the sink regardless of phrasing)\n",
		fullASR, fullSucc, total)

	// The harvest corpus: every outcome folded into a frozen LabelRow.
	pos := len(corpus.Positives())
	fmt.Printf("\nharvest corpus: %d LabelRows folded — %d catches (positives), %d misses (negatives)\n",
		corpus.Len(), pos, corpus.Len()-pos)
	if by := corpus.ByReason(); len(by) > 0 {
		names := make([]string, 0, len(by))
		for k := range by {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, n := range names {
			fmt.Printf("  catch reason %-12s × %d\n", n, by[n])
		}
	}

	if fullSucc == 0 {
		fmt.Printf("\nGATE: PASS — full-stack ASR == 0 across the battery.\n")
	} else {
		fmt.Printf("\nGATE: FAIL — full-stack let %d attack(s) through (a defense regression).\n", fullSucc)
	}
}

func emitJSON(rows []outcomeRow, total, detSucc, fullSucc int, corpus *harvest.Corpus) {
	out := struct {
		Total           int          `json:"total"`
		ASRDetectionRaw int          `json:"asr_detection_succeeded"`
		ASRDetection    float64      `json:"asr_detection"`
		ASRFullRaw      int          `json:"asr_fullstack_succeeded"`
		ASRFull         float64      `json:"asr_fullstack"`
		CorpusRows      int          `json:"corpus_rows"`
		CorpusCatches   int          `json:"corpus_catches"`
		Gate            string       `json:"gate"`
		Attacks         []outcomeRow `json:"attacks"`
	}{
		Total:           total,
		ASRDetectionRaw: detSucc,
		ASRDetection:    ratio(detSucc, total),
		ASRFullRaw:      fullSucc,
		ASRFull:         ratio(fullSucc, total),
		CorpusRows:      corpus.Len(),
		CorpusCatches:   len(corpus.Positives()),
		Gate:            map[bool]string{true: "PASS", false: "FAIL"}[fullSucc == 0],
		Attacks:         rows,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

func verdictLabel(attackSucceeded bool) string {
	if attackSucceeded {
		return "MISSED" // the attacker's goal landed
	}
	return "caught"
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}
