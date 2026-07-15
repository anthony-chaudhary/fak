package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/loopindex"
	"os"
)

type receiptEnvelope struct {
	Receipt struct {
		Subject struct {
			IssueNumber int `json:"issue_number"`
		} `json:"subject"`
		Verdict      string `json:"verdict"`
		Severity     string `json:"severity"`
		Independence struct {
			Verdict string `json:"verdict"`
		} `json:"independence"`
		Author struct {
			Model string `json:"model"`
		} `json:"author"`
		Auditor struct {
			Model string `json:"model"`
		} `json:"auditor"`
		PolicyVersion string `json:"policy_version"`
		Timing        *struct {
			DurationNanos int64 `json:"duration_nanos"`
		} `json:"timing"`
		Usage *struct {
			Measured      bool   `json:"measured"`
			InputTokens   int64  `json:"input_tokens"`
			OutputTokens  int64  `json:"output_tokens"`
			TotalTokens   int64  `json:"total_tokens"`
			CostMicrosUSD int64  `json:"cost_micros_usd"`
			Basis         string `json:"basis"`
		} `json:"usage"`
	} `json:"receipt"`
}

func deriveScorecard(envs []receiptEnvelope) ([]byte, error) {
	classes := map[int]string{3853: "accidental-corpus", 3854: "calibration", 4852: "guard-transport"}
	records := make([]loopindex.AuditRecord, 0, len(envs))
	for _, x := range envs {
		r := x.Receipt
		rec := loopindex.AuditRecord{IssueNumber: r.Subject.IssueNumber, Class: classes[r.Subject.IssueNumber], Outcome: r.Verdict, Severity: r.Severity, Independence: r.Independence.Verdict, AuthorModel: r.Author.Model, AuditorModel: r.Auditor.Model, CalibrationVersion: r.PolicyVersion}
		if r.Timing != nil {
			rec.DurationNanos = r.Timing.DurationNanos
		}
		if r.Usage != nil {
			rec.CostMeasured = r.Usage.Measured
			rec.InputTokens = r.Usage.InputTokens
			rec.OutputTokens = r.Usage.OutputTokens
			rec.TotalTokens = r.Usage.TotalTokens
			rec.CostMicrosUSD = r.Usage.CostMicrosUSD
			rec.CostBasis = r.Usage.Basis
		}
		records = append(records, rec)
	}
	input := loopindex.CrossAuditInput{EligibleIssues: 5, Records: records, Loop: loopindex.LoopHealth{Present: true, Running: false, PendingIssues: 2, Providers: []loopindex.ProviderHealth{{Name: "anthropic", Available: true}, {Name: "openai", Available: true}, {Name: "local", Available: false}}}}
	sc := loopindex.ScoreCrossAudit(input)
	rendered, err := json.MarshalIndent(sc, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(rendered, '\n'), nil
}

func main() {
	in := flag.String("receipts", "", "JSON receipt envelopes")
	out := flag.String("out", "", "scorecard output")
	flag.Parse()
	if *in == "" || *out == "" {
		os.Exit(2)
	}
	b, e := os.ReadFile(*in)
	if e != nil {
		panic(e)
	}
	var envs []receiptEnvelope
	if e = json.Unmarshal(b, &envs); e != nil {
		panic(e)
	}
	rendered, e := deriveScorecard(envs)
	if e != nil {
		panic(e)
	}
	if e = os.WriteFile(*out, rendered, 0o644); e != nil {
		panic(e)
	}
	fmt.Printf("wrote %s\n", *out)
}
