// Command ctxbench runs the fak security gates over a corpus of tool calls and
// tool results. The public default is the committed synthetic poison fixture;
// operators can pass a private transcript-derived corpus with -corpus.
//
//   - RESULT side : ctxmmu.Admit  — the write-time context-admission gate
//     ("uninjectable context"). Allow / Quarantine / Transform.
//   - CALL  side  : preflight.Adjudicate — the rung ladder that catches a
//     malformed call before it fires.
//
// It is the operator-side answer to "run the security benchmarks on this corpus":
// feed captured bytes through the same gate the poison.json fixture exercises,
// and report what the gate would have done.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the blob PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/preflight"
	_ "github.com/anthony-chaudhary/fak/internal/registrations" // built-in driver list (ctxmmu + normgate + ...)
)

// foldAdmit mirrors kernel.admitResult: run the registered ResultAdmitter chain in
// rank order (each does its own page-out side-effect), and report the
// most-restrictive verdict. This measures the REAL composed chain, so enabling
// normgate is exactly one import line in internal/registrations.
func foldAdmit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(abi.VerdictAllow)
	for _, ra := range abi.ResultAdmittersFor(c) {
		v := ra.Admit(ctx, c, r)
		if v.Kind == abi.VerdictDefer { // no opinion — does not override Allow
			continue
		}
		if rk := abi.FoldRank(v.Kind); rk > bestRank {
			bestRank, best = rk, v
		}
	}
	return best
}

type corpus struct {
	Sources []string `json:"sources"`
	Calls   []struct {
		Tool string `json:"tool"`
		Args string `json:"args"`
	} `json:"calls"`
	Results []struct {
		Name    string `json:"name"`
		Tool    string `json:"tool"`
		Payload string `json:"payload"`
		Bytes   int    `json:"bytes"`
	} `json:"results"`
}

// Reporting copies of the ctxmmu detectors — used ONLY to explain WHICH trigger
// fired (the gate itself is authoritative; these label the cause for the report).
var secretPattern = regexp.MustCompile(`(?i)(sk-[a-z0-9]{16,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})`)
var injectionMarkers = []string{
	"ignore previous instructions", "ignore all previous", "disregard the above",
	"you are now", "system override", "new instructions:", "###system",
	"reveal your system prompt", "exfiltrate",
}

func firedMarkers(s string) []string {
	low := strings.ToLower(s)
	var hit []string
	for _, m := range injectionMarkers {
		if strings.Contains(low, m) {
			hit = append(hit, m)
		}
	}
	return hit
}

type resultRow struct {
	Name        string   `json:"name"`
	Tool        string   `json:"tool"`
	Bytes       int      `json:"bytes"`
	Verdict     string   `json:"verdict"`
	Reason      string   `json:"reason"`
	Quarantined bool     `json:"quarantined"`
	Markers     []string `json:"injection_markers,omitempty"`
	SecretShape bool     `json:"secret_shape,omitempty"`
	LeakedAfter bool     `json:"trigger_bytes_present_after_admit"`
}

type callRow struct {
	Tool    string `json:"tool"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	}
	return fmt.Sprintf("KIND_%d", k)
}

func main() {
	in := flag.String("corpus", "testdata/poison.json", "corpus JSON from extract_context_corpus.py or a hand-authored fixture")
	out := flag.String("out", "", "optional JSON report path")
	chain := flag.Bool("chain", false, "fold the registered ResultAdmitter chain (normgate+ctxmmu) instead of a bare ctxmmu")
	flag.Parse()

	raw, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read corpus:", err)
		os.Exit(1)
	}
	var cp corpus
	if err := json.Unmarshal(raw, &cp); err != nil {
		fmt.Fprintln(os.Stderr, "parse corpus:", err)
		os.Exit(1)
	}
	ctx := context.Background()

	// ---- RESULT side: the context-MMU write-time admission gate ----
	m := ctxmmu.New()
	var rows []resultRow
	byVerdict := map[string]int{}
	byReason := map[string]int{}
	leaked := 0
	for _, r := range cp.Results {
		c := &abi.ToolCall{Tool: r.Tool, Args: abi.Ref{Kind: abi.RefInline}, Meta: map[string]string{}}
		res := &abi.Result{Call: c, Status: abi.StatusOK,
			Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(r.Payload)}}

		markers := firedMarkers(r.Payload)
		secret := secretPattern.MatchString(r.Payload)

		var v abi.Verdict
		if *chain {
			v = foldAdmit(ctx, c, res)
		} else {
			v = m.Admit(ctx, c, res)
		}

		// Resolve what now sits in context and check the trigger bytes are gone.
		var after []byte
		if res.Payload.Kind == abi.RefInline {
			after = res.Payload.Inline
		} else if rv := abi.ActiveResolver(); rv != nil {
			after, _ = rv.Resolve(ctx, res.Payload)
		}
		leakedAfter := false
		if secretPattern.Match(after) || len(firedMarkers(string(after))) > 0 {
			leakedAfter = true
			leaked++
		}

		row := resultRow{
			Name: r.Name, Tool: r.Tool, Bytes: r.Bytes,
			Verdict: verdictName(v.Kind), Reason: abi.ReasonName(v.Reason),
			Quarantined: ctxmmu.Quarantined(res),
			Markers:     markers, SecretShape: secret, LeakedAfter: leakedAfter,
		}
		rows = append(rows, row)
		byVerdict[row.Verdict]++
		byReason[row.Reason]++
	}
	total := int64(len(rows))
	q := int64(byVerdict["QUARANTINE"])
	rate := 0.0
	if total > 0 {
		rate = float64(q) / float64(total)
	}

	// ---- CALL side: the pre-flight rung ladder ----
	l := preflight.New()
	var crows []callRow
	callByVerdict := map[string]int{}
	for _, c := range cp.Calls {
		tc := &abi.ToolCall{Tool: c.Tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(c.Args)}}
		v := l.Adjudicate(ctx, tc)
		cr := callRow{Tool: c.Tool, Verdict: verdictName(v.Kind), Reason: abi.ReasonName(v.Reason)}
		crows = append(crows, cr)
		callByVerdict[cr.Verdict]++
	}
	caught, ctotal, crate := l.CatchRate()

	// ---- report ----
	fmt.Printf("== ctxbench: fak security gates over corpus ==\n")
	fmt.Printf("sources: %v\n\n", cp.Sources)

	fmt.Printf("RESULT side — ctxmmu.Admit (write-time context-admission gate)\n")
	fmt.Printf("  results admitted : %d  (%d bytes total)\n", total, sumBytes(cp.Results))
	for _, k := range sortedKeys(byVerdict) {
		fmt.Printf("    %-11s %d\n", k, byVerdict[k])
	}
	fmt.Printf("  quarantine reasons:\n")
	for _, k := range sortedKeys(byReason) {
		if k == "NONE" {
			continue
		}
		fmt.Printf("    %-15s %d\n", k, byReason[k])
	}
	fmt.Printf("  pollution rate   : %d/%d = %.1f%%\n", q, total, rate*100)
	fmt.Printf("  trigger bytes still in context after admit (LEAK): %d  <-- must be 0\n\n", leaked)

	fmt.Printf("  per-result (non-ALLOW only):\n")
	for _, r := range rows {
		if r.Verdict == "ALLOW" {
			continue
		}
		tag := ""
		if len(r.Markers) > 0 {
			tag = " markers=" + strings.Join(r.Markers, ",")
		}
		if r.SecretShape {
			tag += " secret-shape"
		}
		fmt.Printf("    %-22s %-9s %-15s %6dB%s\n", r.Name, r.Verdict, r.Reason, r.Bytes, tag)
	}

	fmt.Printf("\nCALL side — preflight.Adjudicate (rung ladder)\n")
	fmt.Printf("  calls adjudicated: %d\n", ctotal)
	for _, k := range sortedKeys(callByVerdict) {
		fmt.Printf("    %-11s %d\n", k, callByVerdict[k])
	}
	fmt.Printf("  catch rate       : %d/%d = %.1f%% (malformed calls caught pre-fire)\n", caught, ctotal, crate*100)

	if *out != "" {
		report := map[string]any{
			"app_version": appversion.Current(),
			"sources":     cp.Sources,
			"result_side": map[string]any{
				"admitted": total, "quarantined": q, "pollution_rate": rate,
				"by_verdict": byVerdict, "by_reason": byReason,
				"leak_after_admit": leaked, "rows": rows,
			},
			"call_side": map[string]any{
				"adjudicated": ctotal, "caught": caught, "catch_rate": crate,
				"by_verdict": callByVerdict, "rows": crows,
			},
		}
		b, _ := json.MarshalIndent(report, "", "  ")
		_ = os.WriteFile(*out, b, 0644)
		fmt.Printf("\nwrote %s\n", *out)
	}
}

func sumBytes(rs []struct {
	Name    string `json:"name"`
	Tool    string `json:"tool"`
	Payload string `json:"payload"`
	Bytes   int    `json:"bytes"`
}) int {
	n := 0
	for _, r := range rs {
		n += r.Bytes
	}
	return n
}

func sortedKeys(m map[string]int) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
