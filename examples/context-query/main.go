// context-query is a no-model selfcheck for bounded derivation over addressable records.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/contextq"
)

func main() {
	os.Exit(run(os.Stdout, os.Stderr, os.Args[1:]))
}

type selfcheckReceipt struct {
	Schema          string                     `json:"schema"`
	View            contextq.DerivedRecordView `json:"view"`
	Result          json.RawMessage            `json:"result"`
	ModelRoundTrips int                        `json:"model_round_trips"`
	NetworkCalls    int                        `json:"network_calls"`
}

func run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("context-query", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run the deterministic addressable-record query witness")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfcheck || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: context-query -selfcheck")
		return 2
	}

	body, err := selfcheckSource()
	if err != nil {
		fmt.Fprintf(stderr, "context-query: build source: %v\n", err)
		return 1
	}
	store := blob.New()
	source, err := store.Put(context.Background(), body)
	if err != nil {
		fmt.Fprintf(stderr, "context-query: store source: %v\n", err)
		return 1
	}
	source.Taint = abi.TaintTrusted
	source.Scope = abi.ScopeAgent

	view, err := contextq.DeriveRecords(context.Background(), store, source, contextq.RecordPlan{
		Schema:    contextq.RecordPlanSchema,
		Operation: contextq.RecordOperationGroupCount,
		Filter:    &contextq.RecordEqualFilter{Field: "status", Value: "failed"},
		GroupBy:   "owner",
	}, contextq.RecordLimits{
		MaxSourceBytes: 1 << 20,
		MaxOutputBytes: 1 << 14,
		MaxRecords:     100,
		MaxWorkUnits:   1000,
	})
	if err != nil {
		fmt.Fprintf(stderr, "context-query: derive: %v\n", err)
		return 1
	}
	result, err := store.Resolve(context.Background(), view.Output)
	if err != nil {
		fmt.Fprintf(stderr, "context-query: resolve view: %v\n", err)
		return 1
	}

	receipt := selfcheckReceipt{
		Schema:          "fak.context-query-selfcheck/1",
		View:            view,
		Result:          json.RawMessage(result),
		ModelRoundTrips: 0,
		NetworkCalls:    0,
	}
	enc := json.NewEncoder(stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(receipt); err != nil {
		fmt.Fprintf(stderr, "context-query: encode receipt: %v\n", err)
		return 1
	}
	return 0
}

func selfcheckSource() ([]byte, error) {
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	owners := []string{"alice", "bob", "carol"}
	for i := 0; i < 30; i++ {
		status := "ok"
		if i%5 == 0 {
			status = "failed"
		}
		record := struct {
			ID     string `json:"id"`
			Status string `json:"status"`
			Owner  string `json:"owner"`
			Detail string `json:"detail"`
		}{
			ID:     fmt.Sprintf("ticket-%02d", i),
			Status: status,
			Owner:  owners[i%len(owners)],
			Detail: fmt.Sprintf("source-only-detail-%02d-is-not-emitted-in-the-receipt", i),
		}
		if err := enc.Encode(record); err != nil {
			return nil, err
		}
	}
	return out.Bytes(), nil
}
