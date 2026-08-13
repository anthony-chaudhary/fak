package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

const auditDatasetSchema = "fak-decision-outcome/1"

type auditDatasetRow struct {
	Schema        string `json:"schema"`
	TraceID       string `json:"trace_id,omitempty"`
	CallSeq       uint64 `json:"call_seq"`
	Seq           uint64 `json:"seq"`
	TSUnixNano    int64  `json:"ts_unix_nano"`
	Tool          string `json:"tool,omitempty"`
	Verdict       string `json:"verdict,omitempty"`
	Reason        string `json:"reason,omitempty"`
	By            string `json:"by,omitempty"`
	ArgsDigest    string `json:"args_digest,omitempty"`
	ArgsLabel     string `json:"args_label,omitempty"`
	Witness       string `json:"witness,omitempty"`
	ResultVerdict string `json:"result_verdict,omitempty"`
	ResultReason  string `json:"result_reason,omitempty"`
	Taint         string `json:"taint,omitempty"`
}

type auditDatasetProblem struct {
	Seq     uint64 `json:"seq"`
	Kind    string `json:"kind"`
	Problem string `json:"problem"`
}

// foldAuditDataset is pure: callers provide verified journal rows and receive
// deterministic per-call rows plus every outcome row that could not be joined.
func foldAuditDataset(rows []journal.Row) ([]auditDatasetRow, []auditDatasetProblem) {
	ordered := append([]journal.Row(nil), rows...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Seq < ordered[j].Seq })
	out := make([]auditDatasetRow, 0)
	byCall := make(map[uint64]int)
	var outcomes []journal.Row
	var problems []auditDatasetProblem
	for _, r := range ordered {
		if replayIsAdjudication(r) {
			if r.CallSeq == 0 {
				problems = append(problems, auditDatasetProblem{r.Seq, r.Kind, "adjudication row has no call_seq"})
				continue
			}
			if _, exists := byCall[r.CallSeq]; exists {
				problems = append(problems, auditDatasetProblem{r.Seq, r.Kind, "duplicate adjudication call_seq"})
				continue
			}
			byCall[r.CallSeq] = len(out)
			out = append(out, auditDatasetRow{
				Schema: auditDatasetSchema, TraceID: r.TraceID, CallSeq: r.CallSeq,
				Seq: r.Seq, TSUnixNano: r.TSUnixNano, Tool: r.Tool,
				Verdict: r.Verdict, Reason: r.Reason, By: r.By,
				ArgsDigest: r.ArgsDigest, ArgsLabel: r.ArgsLabel, Witness: r.Witness,
			})
			continue
		}
		if r.Kind == "RESULT_DENY" || r.Kind == "QUARANTINE" {
			outcomes = append(outcomes, r)
		}
	}
	for _, r := range outcomes {
		if r.CallSeq == 0 {
			problems = append(problems, auditDatasetProblem{r.Seq, r.Kind, "result outcome has no call_seq"})
			continue
		}
		i, ok := byCall[r.CallSeq]
		if !ok {
			problems = append(problems, auditDatasetProblem{r.Seq, r.Kind, "result outcome has no adjudication row"})
			continue
		}
		out[i].ResultVerdict = r.Verdict
		out[i].ResultReason = r.Reason
		out[i].Taint = r.Taint
	}
	return out, problems
}

func cmdAuditDataset(args []string) {
	fs := flag.NewFlagSet("audit dataset", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: fak audit dataset <journal.jsonl>")
		os.Exit(2)
	}
	// Segment-aware (#6488): the dataset is the WHOLE journal, so both the
	// refuse-if-unverified gate and the read follow the cut anchors across every
	// archived segment. (Verify/ReadRows on a rotated live file would refuse a sound
	// chain, and, past that gate, export only the tail.)
	segs, serr := journal.Segments(fs.Arg(0))
	if serr != nil {
		fmt.Fprintf(os.Stderr, "audit dataset: %v\n", serr)
		os.Exit(1)
	}
	if _, err := journal.VerifySegments(segs...); err != nil {
		fmt.Fprintf(os.Stderr, "audit dataset: refusing unverified journal: %v\n", err)
		os.Exit(1)
	}
	rows, err := journal.ReadAllSegments(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(os.Stderr, "audit dataset: %v\n", err)
		os.Exit(1)
	}
	dataset, problems := foldAuditDataset(journal.WithoutCutAnchors(rows))
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	for _, row := range dataset {
		if err := enc.Encode(row); err != nil {
			fmt.Fprintf(os.Stderr, "audit dataset: %v\n", err)
			os.Exit(1)
		}
	}
	if len(problems) > 0 {
		for _, p := range problems {
			fmt.Fprintf(os.Stderr, "audit dataset: unkeyable seq=%d kind=%s: %s\n", p.Seq, p.Kind, p.Problem)
		}
		os.Exit(1)
	}
}

var _ io.Writer
