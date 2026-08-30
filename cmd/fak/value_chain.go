package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/valuechain"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func cmdValueChain(args []string) { os.Exit(runValueChain(os.Stdout, os.Stderr, args)) }

func runValueChain(out, errOut io.Writer, args []string) int {
	if len(args) > 0 && args[0] == "usage" {
		return runValueChainUsage(out, errOut, args[1:])
	}
	if len(args) == 0 || args[0] != "audit" {
		fmt.Fprintln(errOut, "usage: fak value-chain audit --manifest M --observations O [--json] [--ledger PATH]\n       fak value-chain usage [--ledger PATH]")
		return 2
	}
	fs := flag.NewFlagSet("value-chain audit", flag.ContinueOnError)
	fs.SetOutput(errOut)
	manifest := fs.String("manifest", "", "manifest JSON")
	observations := fs.String("observations", "", "observations JSON")
	asJSON := fs.Bool("json", false, "emit JSON")
	agenticPacket := fs.String("agentic-packet", "", "graduated AgenticBench result packet")
	agenticStage := fs.String("agentic-stage", "benchmark", "manifest stage for AgenticBench observations")
	selfcheck := fs.Bool("selfcheck", false, "verify the report against the checked-in expected witness")
	expect := fs.String("expect", "", "expected text witness (required with --selfcheck)")
	ledger := fs.String("ledger", defaultValueChainLedger(), "durable invocation ledger (empty disables)")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	outcome := "error"
	defer func() {
		if err := appendValueChainUsage(*ledger, time.Now().UTC(), outcome); err != nil {
			fmt.Fprintf(errOut, "value-chain usage ledger: %v\n", err)
		}
	}()
	if *manifest == "" || *observations == "" {
		fmt.Fprintln(errOut, "--manifest and --observations are required")
		return 2
	}
	if *selfcheck && *expect == "" {
		fmt.Fprintln(errOut, "--expect is required with --selfcheck")
		return 2
	}
	m, in, err := valuechain.Read(*manifest, *observations)
	if err == nil && *agenticPacket != "" {
		var packet valuechain.Input
		packet, err = valuechain.ReadAgenticPacket(*agenticPacket, *agenticStage)
		if err == nil {
			in.Observations = append(in.Observations, packet.Observations...)
		}
	}
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	rep, err := valuechain.Audit(m, in)
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(errOut, err)
			return 1
		}
		outcome = "success"
		return 0
	}
	rendered := formatValueChainReport(rep)
	if *selfcheck {
		want, err := os.ReadFile(*expect)
		if err != nil {
			fmt.Fprintln(errOut, err)
			return 1
		}
		if string(want) != rendered {
			fmt.Fprintln(errOut, "value-chain selfcheck mismatch")
			return 1
		}
	}
	if _, err := io.WriteString(out, rendered); err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	outcome = "success"
	return 0
}

const valueChainUsageSchema = "fak-value-chain-usage/1"

type valueChainUsageRow struct {
	Schema  string    `json:"schema"`
	At      time.Time `json:"at"`
	Outcome string    `json:"outcome"`
}

const dispatchValueChainUsageLedger = ".dispatch-runs/value-chain-usage.jsonl"

func recordDispatchValueChainUsage(root string, payload map[string]any, at time.Time) (map[string]any, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		root = "."
	}
	outcome := strings.TrimSpace(dispatchMapString(payload, "action"))
	if outcome == "" {
		outcome = "unknown"
	}
	path := filepath.Join(root, filepath.FromSlash(dispatchValueChainUsageLedger))
	if err := appendValueChainUsage(path, at, outcome); err != nil {
		return nil, err
	}
	return map[string]any{
		"schema": valueChainUsageSchema, "ledger": path, "outcome": outcome, "automatic": true,
	}, nil
}

func defaultValueChainLedger() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "fak", "value-chain-usage.jsonl")
}

func appendValueChainUsage(path string, at time.Time, outcome string) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(valueChainUsageRow{Schema: valueChainUsageSchema, At: at, Outcome: outcome})
}

func runValueChainUsage(out, errOut io.Writer, args []string) int {
	fs := flag.NewFlagSet("value-chain usage", flag.ContinueOnError)
	fs.SetOutput(errOut)
	ledger := fs.String("ledger", defaultValueChainLedger(), "durable invocation ledger")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	f, err := os.Open(*ledger)
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	defer f.Close()
	counts := map[string]int{}
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		var row valueChainUsageRow
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil || row.Schema != valueChainUsageSchema || row.At.IsZero() {
			fmt.Fprintln(errOut, "invalid value-chain usage ledger row")
			return 1
		}
		y, w := row.At.ISOWeek()
		counts[fmt.Sprintf("%04d-W%02d", y, w)]++
	}
	if err := scan.Err(); err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	weeks := make([]string, 0, len(counts))
	for week := range counts {
		weeks = append(weeks, week)
	}
	sort.Strings(weeks)
	for _, week := range weeks {
		fmt.Fprintf(out, "week=%s invocations=%d\n", week, counts[week])
	}
	return 0
}
func formatValueChainReport(rep valuechain.Report) string {
	var out strings.Builder
	fmt.Fprintf(&out, "VALUE CHAIN %s\n", rep.Name)
	for _, a := range rep.Arms {
		fmt.Fprintf(&out, "arm=%s traces=%d sessions=%d turns=%d billing_evidence=%d/%d", a.Arm, a.Traces, a.Sessions, a.Turns, a.BillingEvidence.Covered, a.BillingEvidence.Total)
		if a.CostPerTurn != nil {
			fmt.Fprintf(&out, " $/turn=%.6f", *a.CostPerTurn)
		} else {
			fmt.Fprint(&out, " $/turn=UNKNOWN")
		}
		outcomeIDs := make([]string, 0, len(a.CostPerOutcome))
		for id := range a.CostPerOutcome {
			outcomeIDs = append(outcomeIDs, id)
		}
		sort.Strings(outcomeIDs)
		for _, id := range outcomeIDs {
			fmt.Fprintf(&out, " $/%s=%.6f", id, a.CostPerOutcome[id])
		}
		fmt.Fprintln(&out)
	}
	for _, stage := range rep.Inventory {
		fmt.Fprintf(&out, "stage=%s kind=%s status=%s observations=%d\n", stage.Stage, stage.Kind, stage.Status, stage.Observations)
	}
	if rep.Comparison != nil {
		fmt.Fprintf(&out, "comparison=%s->%s design=%s paired=%d\n", rep.Comparison.Baseline, rep.Comparison.Candidate, rep.Comparison.Design, rep.Comparison.PairedTraces)
	}
	return out.String()
}
