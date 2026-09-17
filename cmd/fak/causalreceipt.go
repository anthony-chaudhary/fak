package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/causalreceipt"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func cmdCausalReceipt(argv []string) {
	os.Exit(runCausalReceipt(os.Stdout, os.Stderr, os.Stdin, argv))
}

type causalReceiptOutput struct {
	Schema  string                 `json:"schema"`
	Valid   bool                   `json:"valid"`
	Error   string                 `json:"error,omitempty"`
	Metrics *causalreceipt.Metrics `json:"metrics,omitempty"`
	Labels  map[string]string      `json:"labels,omitempty"`
	Answers []string               `json:"incident_answers,omitempty"`
}

func runCausalReceipt(stdout, stderr io.Writer, stdin io.Reader, argv []string) int {
	fs := flag.NewFlagSet("causal-receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)

	selfTest := fs.Bool("self-test", false, "run self-test validation on representative fixtures")
	validateOnly := fs.Bool("validate", false, "validate receipt schema, invariants, and privacy guards only")
	asJSON := fs.Bool("json", false, "emit structured JSON output")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *selfTest {
		return runCausalReceiptSelfTest(stdout, stderr, *asJSON)
	}

	args := fs.Args()
	if len(args) > 0 && (args[0] == "produce" || args[0] == "fold") {
		return runCausalReceiptProduce(stdout, stderr, *asJSON, args[1:])
	}
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak causal-receipt [flags] <receipt.json|- >")
		fmt.Fprintln(stderr, "       fak causal-receipt produce [--trace <compute-trace.json>] [--attr k=v ...] [--json]")
		fmt.Fprintln(stderr, "       fak causal-receipt --self-test [--json]")
		return 2
	}

	var data []byte
	var err error
	if args[0] == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(args[0])
	}
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt: read input: %v\n", err)
		return 1
	}

	var receipt causalreceipt.Receipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		fmt.Fprintf(stderr, "causal-receipt: parse JSON: %v\n", err)
		return 1
	}

	valErr := causalreceipt.Validate(receipt)
	if valErr != nil {
		if *asJSON {
			out := causalReceiptOutput{
				Schema: causalreceipt.Schema,
				Valid:  false,
				Error:  valErr.Error(),
			}
			_ = writeIndentedJSONNoEscape(stdout, out)
		} else {
			fmt.Fprintf(stderr, "causal-receipt: invalid: %v\n", valErr)
		}
		return 1
	}

	if *validateOnly {
		if *asJSON {
			out := causalReceiptOutput{
				Schema: causalreceipt.Schema,
				Valid:  true,
			}
			_ = writeIndentedJSONNoEscape(stdout, out)
		} else {
			fmt.Fprintln(stdout, "causal-receipt: OK (valid)")
		}
		return 0
	}

	metrics, err := causalreceipt.DeriveMetrics(receipt)
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt: derive metrics: %v\n", err)
		return 1
	}

	labels := causalreceipt.MetricLabels(receipt)
	answers := causalreceipt.IncidentAnswers(receipt)

	if *asJSON {
		out := causalReceiptOutput{
			Schema:  causalreceipt.Schema,
			Valid:   true,
			Metrics: &metrics,
			Labels:  labels,
			Answers: answers,
		}
		if err := writeIndentedJSONNoEscape(stdout, out); err != nil {
			fmt.Fprintf(stderr, "causal-receipt: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "causal-receipt: VALID (%s)\n", receipt.Schema)
	fmt.Fprintf(stdout, "  phases: %d, tokens: %d, bytes: %d, cache_reuse: %d, overhead_ns: %d\n",
		metrics.PhaseCount, metrics.Tokens, metrics.Bytes, metrics.CacheReuseBytes, metrics.OverheadNS)
	if len(answers) > 0 {
		fmt.Fprintf(stdout, "  incident answers: %s\n", strings.Join(answers, ", "))
	}
	return 0
}

func runCausalReceiptSelfTest(stdout, stderr io.Writer, asJSON bool) int {
	sample := causalreceipt.Receipt{
		Schema: causalreceipt.Schema,
		IDs: causalreceipt.IDs{
			Work:         "w-self",
			Turn:         "t-self",
			Graph:        "g-self",
			Request:      "r-self",
			ModelSession: "s-self",
		},
		Phases: []causalreceipt.Phase{
			{
				ID:      "root",
				Kind:    "agent",
				Engine:  "fak-native",
				Backend: "offline",
				Outcome: "completed",
			},
		},
		Resources: []causalreceipt.Resource{
			{
				ID:       "res-1",
				Kind:     "weights",
				State:    "released",
				Released: true,
			},
		},
		Decisions: []causalreceipt.Decision{
			{
				ID:     "dec-1",
				Kind:   "policy",
				Actual: "allow",
			},
		},
	}

	if err := causalreceipt.Validate(sample); err != nil {
		fmt.Fprintf(stderr, "self-test fixture validation failed: %v\n", err)
		return 1
	}

	metrics, err := causalreceipt.DeriveMetrics(sample)
	if err != nil {
		fmt.Fprintf(stderr, "self-test metrics derivation failed: %v\n", err)
		return 1
	}

	if asJSON {
		out := causalReceiptOutput{
			Schema:  causalreceipt.Schema,
			Valid:   true,
			Metrics: &metrics,
			Labels:  causalreceipt.MetricLabels(sample),
			Answers: causalreceipt.IncidentAnswers(sample),
		}
		_ = writeIndentedJSONNoEscape(stdout, out)
	} else {
		fmt.Fprintln(stdout, "causal-receipt: self-test passed (valid fixture verified)")
	}
	return 0
}

// runCausalReceiptProduce folds a bounded compute-trace artifact plus optional
// launch-context attributes into a validated causal receipt. It is the producer
// seam for the fak.causal-receipt/1 schema: it reuses causalreceipt.Produce and
// reports the receipt's metrics so the caller can confirm non-zero overhead.
func runCausalReceiptProduce(stdout, stderr io.Writer, asJSON bool, argv []string) int {
	fs := flag.NewFlagSet("causal-receipt produce", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tracePath := fs.String("trace", "", "compute-trace artifact JSON path (- for stdin)")
	attrFlags := multiFlag{}
	fs.Var(&attrFlags, "attr", "launch-context attribute k=v (repeatable)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *tracePath == "" && len(fs.Args()) > 0 {
		*tracePath = fs.Args()[0]
	}
	if *tracePath == "" {
		fmt.Fprintln(stderr, "usage: fak causal-receipt produce [--trace <compute-trace.json>] [--attr k=v ...] [--json]")
		return 2
	}

	var data []byte
	var err error
	if *tracePath == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(*tracePath)
	}
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt produce: read trace: %v\n", err)
		return 1
	}

	artifact, err := computetrace.Read(bytes.NewReader(data))
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt produce: parse compute trace: %v\n", err)
		return 1
	}

	attributes := map[string]string{}
	for _, kv := range attrFlags {
		key, value, ok := strings.Cut(kv, "=")
		if !ok || strings.TrimSpace(key) == "" {
			fmt.Fprintf(stderr, "causal-receipt produce: attribute %q must be k=v\n", kv)
			return 2
		}
		attributes[key] = value
	}

	receipt, err := causalreceipt.Produce(artifact, attributes)
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt produce: %v\n", err)
		return 1
	}
	metrics, err := causalreceipt.DeriveMetrics(receipt)
	if err != nil {
		fmt.Fprintf(stderr, "causal-receipt produce: derive metrics: %v\n", err)
		return 1
	}

	if err := writeIndentedJSONNoEscape(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "causal-receipt produce: encode json: %v\n", err)
		return 1
	}
	if !asJSON {
		fmt.Fprintf(stderr, "causal-receipt produce: VALID (%s) phases=%d overhead_ns=%d bytes=%d\n",
			receipt.Schema, metrics.PhaseCount, metrics.OverheadNS, metrics.Bytes)
	}
	return 0
}
