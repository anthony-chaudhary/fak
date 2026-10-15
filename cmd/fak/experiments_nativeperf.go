package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/experiments"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	nativePerfDecisionLedgerRel    = ".fak/nativeperf-decisions.jsonl"
	nativePerfDefaultDecisionLimit = 20
	nativePerfMaxDecisionLimit     = 100
	nativePerfDefaultDecisionBytes = 1 << 20
	nativePerfMaxDecisionBytes     = 16 << 20
)

type nativePerfInputScope struct {
	EvidenceScope      string `json:"evidence_scope"`
	HardwareVerified   bool   `json:"hardware_verified"`
	DefaultObservation string `json:"default_observation"`
	ServingMutated     bool   `json:"serving_mutated"`
}

func nativePerfSuppliedScope() nativePerfInputScope {
	return nativePerfInputScope{EvidenceScope: "supplied-files", DefaultObservation: "caller-supplied"}
}

type nativePerfCLIResult struct {
	nativePerfInputScope
	Journal  experiments.NativePerfDecisionJournal `json:"journal"`
	Ledger   string                                `json:"ledger,omitempty"`
	Replayed bool                                  `json:"replayed,omitempty"`
}

type nativePerfDecisionReadback struct {
	nativePerfInputScope
	Ledger             string                                  `json:"ledger"`
	Present            bool                                    `json:"present"`
	FileBytes          *int64                                  `json:"file_bytes"`
	WindowStart        int64                                   `json:"window_start"`
	OmittedBytes       int64                                   `json:"omitted_bytes"`
	Decisions          []experiments.NativePerfDecisionJournal `json:"decisions"`
	RowsRead           int                                     `json:"rows_read"`
	Returned           int                                     `json:"returned"`
	RowLimit           int                                     `json:"row_limit"`
	BytesRead          int                                     `json:"bytes_read"`
	ByteLimit          int                                     `json:"byte_limit"`
	ByteTruncated      bool                                    `json:"byte_truncated"`
	RowTruncated       bool                                    `json:"row_truncated"`
	Truncated          bool                                    `json:"truncated"`
	TotalRows          *int                                    `json:"total_rows"`
	OmittedRows        *int                                    `json:"omitted_rows"`
	OmittedRowsAtLeast int                                     `json:"omitted_rows_at_least"`
}

func experimentsNativePerformanceUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak experiments native-performance evaluate --file REQUEST [--ledger FILE] [--json]
  fak experiments native-performance replay --file REQUEST --decision JOURNAL [--json]
  fak experiments native-performance decisions [--ledger FILE] [--limit 20] [--max-bytes 1048576] [--json]

Evaluate records a KEEP/REJECT decision from supplied evidence. KEEP applies the
existing non-regression policy; it does not prove improvement or promotion.
Replay recomputes one supplied journal. Neither runs a model or changes serving. Hardware remains
unverified and the default observation is caller-supplied. Current transaction
envelopes cover Qwen3.8 M3 Pro Metal/A100 CUDA, not DS/Vulkan.
The default ledger is .fak/nativeperf-decisions.jsonl under the repository root.
`)
}

func runExperimentsNativePerformance(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		experimentsNativePerformanceUsage(stderr)
		return 2
	}
	mode := argv[0]
	if mode == "help" || mode == "--help" || mode == "-h" {
		experimentsNativePerformanceUsage(stdout)
		return 0
	}
	if mode != "evaluate" && mode != "replay" && mode != "decisions" {
		experimentsNativePerformanceUsage(stderr)
		return 2
	}
	fs := flag.NewFlagSet("experiments native-performance "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { experimentsNativePerformanceUsage(stderr) }
	asJSON := fs.Bool("json", false, "emit machine-readable result and evidence scope")
	var input, decision, ledger string
	limit, maxBytes := nativePerfDefaultDecisionLimit, nativePerfDefaultDecisionBytes
	if mode != "decisions" {
		fs.StringVar(&input, "file", "", "supplied NativePerfTransactionRequest JSON")
	}
	if mode == "replay" {
		fs.StringVar(&decision, "decision", "", "supplied NativePerfDecisionJournal JSON")
	} else {
		fs.StringVar(&ledger, "ledger", "", "decision ledger path")
	}
	if mode == "decisions" {
		fs.IntVar(&limit, "limit", limit, "latest complete rows returned in ledger order (1..100)")
		fs.IntVar(&maxBytes, "max-bytes", maxBytes, "ledger tail bytes inspected (1..16777216), plus one boundary probe byte")
	}
	if err := fs.Parse(argv[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || (mode != "decisions" && input == "") || (mode == "replay" && decision == "") ||
		limit < 1 || limit > nativePerfMaxDecisionLimit || maxBytes < 1 || maxBytes > nativePerfMaxDecisionBytes {
		fs.Usage()
		return 2
	}
	if mode != "replay" && ledger == "" {
		ledger = filepath.Join(repoRoot(), filepath.FromSlash(nativePerfDecisionLedgerRel))
	}
	if mode == "decisions" {
		return experimentsNativePerfDecisions(stdout, stderr, ledger, limit, maxBytes, *asJSON)
	}
	var request experiments.NativePerfTransactionRequest
	requestData, err := decodeNativePerfJSONFile(input, &request)
	if err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: request: %v\n", err)
		return 2
	}
	type receiptPresence struct {
		Execution struct {
			FallbackCount *int `json:"fallback_count"`
		} `json:"execution"`
	}
	var rawRequest struct {
		Profile     json.RawMessage `json:"profile"`
		GateRequest struct {
			LastAccepted receiptPresence `json:"last_accepted"`
			Candidate    receiptPresence `json:"candidate"`
		} `json:"gate_request"`
	}
	if err := json.Unmarshal(requestData, &rawRequest); err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: request: %v\n", err)
		return 2
	}
	request.Profile, err = nativeperf.DecodeProfile(rawRequest.Profile)
	if err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: profile: %v\n", err)
		return 2
	}
	if rawRequest.GateRequest.LastAccepted.Execution.FallbackCount == nil || rawRequest.GateRequest.Candidate.Execution.FallbackCount == nil {
		fmt.Fprintln(stderr, "fak experiments native-performance: both gate receipts require an explicit non-null execution.fallback_count")
		return 2
	}
	result := nativePerfCLIResult{nativePerfInputScope: nativePerfSuppliedScope()}
	if mode == "replay" {
		if _, err := decodeNativePerfJSONFile(decision, &result.Journal); err != nil {
			fmt.Fprintf(stderr, "fak experiments native-performance: journal: %v\n", err)
			return 2
		}
		if err := experiments.ReplayNativePerfTransaction(request, result.Journal); err != nil {
			fmt.Fprintf(stderr, "fak experiments native-performance: replay: %v\n", err)
			return 1
		}
		result.Replayed = true
	} else {
		journal, err := experiments.RunNativePerfTransaction(request)
		if err != nil {
			fmt.Fprintf(stderr, "fak experiments native-performance: evaluate: %v\n", err)
			return 1
		}
		if err := experiments.AppendNativePerfDecision(ledger, journal); err != nil {
			fmt.Fprintf(stderr, "fak experiments native-performance: append: %v\n", err)
			return 1
		}
		result.Journal, result.Ledger = journal, ledger
	}
	if *asJSON {
		return encodeNativePerfCLIJSON(stdout, stderr, result)
	}
	fmt.Fprintf(stdout, "%s candidate=%s gate=%s replay_digest=%s replayed=%t\n", result.Journal.Decision,
		result.Journal.SelectedCandidateID, result.Journal.GateOutput.Classification, result.Journal.ReplayDigest, result.Replayed)
	printNativePerfSuppliedScope(stdout)
	if result.Ledger != "" {
		fmt.Fprintf(stdout, "Ledger: %s\n", result.Ledger)
	}
	return 0
}

func decodeNativePerfJSONFile(path string, dst any) ([]byte, error) {
	data, truncated, err := readNativePerfPrefix(path, experiments.MaxReceiptLineBytes)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("input exceeds %d bytes", experiments.MaxReceiptLineBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, errors.New("input must contain exactly one JSON object")
	}
	return data, nil
}

func readNativePerfPrefix(path string, maxBytes int) ([]byte, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, int64(maxBytes)+1))
	return data, len(data) > maxBytes, err
}

type nativePerfTailWindow struct {
	data        []byte
	present     bool
	bytesRead   int
	fileBytes   int64
	windowStart int64
}

func readNativePerfTail(path string, maxBytes int) (nativePerfTailWindow, error) {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nativePerfTailWindow{}, nil
	}
	if err != nil {
		return nativePerfTailWindow{}, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nativePerfTailWindow{}, err
	}
	if !info.Mode().IsRegular() {
		return nativePerfTailWindow{}, errors.New("ledger must be a regular file")
	}
	window := nativePerfTailWindow{present: true, fileBytes: info.Size()}
	window.windowStart = max(0, window.fileBytes-int64(maxBytes))
	readStart := window.windowStart
	if readStart > 0 {
		readStart--
	}
	if _, err := file.Seek(readStart, io.SeekStart); err != nil {
		return nativePerfTailWindow{}, err
	}
	data := make([]byte, int(window.fileBytes-readStart))
	if _, err := io.ReadFull(file, data); err != nil {
		return nativePerfTailWindow{}, err
	}
	window.bytesRead = len(data)
	if len(data) > 0 && data[len(data)-1] != '\n' {
		return nativePerfTailWindow{}, errors.New("nativeperf decision ledger must end with a newline")
	}
	if window.windowStart > 0 {
		previous := data[0]
		data = data[1:]
		if previous != '\n' {
			boundary := bytes.IndexByte(data, '\n')
			if boundary < 0 || boundary+1 == len(data) {
				return nativePerfTailWindow{}, errors.New("latest decision does not fit; increase --max-bytes")
			}
			data = data[boundary+1:]
			window.windowStart += int64(boundary + 1)
		}
	}
	window.data = data
	return window, nil
}

func experimentsNativePerfDecisions(stdout, stderr io.Writer, ledger string, limit, maxBytes int, asJSON bool) int {
	window, err := readNativePerfTail(ledger, maxBytes)
	if err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: read ledger: %v\n", err)
		return 1
	}
	result := nativePerfDecisionReadback{nativePerfInputScope: nativePerfSuppliedScope(), Ledger: ledger,
		Present: window.present, WindowStart: window.windowStart, OmittedBytes: window.windowStart,
		RowLimit: limit, ByteLimit: maxBytes, BytesRead: window.bytesRead, ByteTruncated: window.windowStart > 0}
	if window.present {
		result.FileBytes = &window.fileBytes
	}
	rows, err := experiments.ParseNativePerfDecisionLedger(string(window.data))
	if err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: parse ledger tail: %v\n", err)
		return 1
	}
	result.RowsRead = len(rows)
	result.Returned = min(limit, len(rows))
	result.Decisions = append([]experiments.NativePerfDecisionJournal{}, rows[len(rows)-result.Returned:]...)
	result.OmittedRowsAtLeast = len(rows) - result.Returned
	result.RowTruncated = result.OmittedRowsAtLeast > 0
	result.Truncated = result.ByteTruncated || result.RowTruncated
	if result.Present && !result.ByteTruncated {
		result.TotalRows = &result.RowsRead
		result.OmittedRows = &result.OmittedRowsAtLeast
	}
	if asJSON {
		return encodeNativePerfCLIJSON(stdout, stderr, result)
	}
	printNativePerfSuppliedScope(stdout)
	fmt.Fprintf(stdout, "present=%t window_start=%d omitted_bytes=%d ", result.Present, result.WindowStart, result.OmittedBytes)
	if result.FileBytes == nil {
		fmt.Fprintln(stdout, "file_bytes=unknown")
	} else {
		fmt.Fprintf(stdout, "file_bytes=%d\n", *result.FileBytes)
	}
	fmt.Fprintf(stdout, "returned=%d rows_read=%d bytes_read=%d byte_limit=%d byte_truncated=%t row_truncated=%t\n",
		result.Returned, result.RowsRead, result.BytesRead, result.ByteLimit, result.ByteTruncated, result.RowTruncated)
	if result.TotalRows == nil {
		fmt.Fprintf(stdout, "total_rows=unknown omitted_rows=unknown omitted_rows_at_least=%d\n", result.OmittedRowsAtLeast)
	} else {
		fmt.Fprintf(stdout, "total_rows=%d omitted_rows=%d\n", *result.TotalRows, *result.OmittedRows)
	}
	for _, row := range result.Decisions {
		fmt.Fprintf(stdout, "%s %s %s\n", row.Decision, row.SelectedCandidateID, row.ReplayDigest)
	}
	return 0
}

func printNativePerfSuppliedScope(w io.Writer) {
	fmt.Fprintln(w, "Evidence: supplied-files; hardware_verified=false; default_observation=caller-supplied; serving_mutated=false")
}

func encodeNativePerfCLIJSON(stdout, stderr io.Writer, value any) int {
	if err := json.NewEncoder(stdout).Encode(value); err != nil {
		fmt.Fprintf(stderr, "fak experiments native-performance: encode result: %v\n", err)
		return 1
	}
	return 0
}
