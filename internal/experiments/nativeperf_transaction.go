package experiments

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

const NativePerfDecisionSchema = "fak-nativeperf-decision-journal/1"

const (
	NativePerfKeep   = "KEEP"
	NativePerfReject = "REJECT"
)

// DefaultReadback records which matched receipt remains selected after the gate.
type DefaultReadback struct {
	Selected string `json:"selected"`
	Revision string `json:"revision"`
}

// NativePerfTransactionRequest binds one accepted profile and compiled graph to
// one matched baseline/candidate gate request.
type NativePerfTransactionRequest struct {
	Profile               nativeperf.ProfileBundle
	ObligationGraph       newmodel.NativeObligationGraph
	ObligationGraphDigest string
	ObligationID          string
	CandidateID           string
	GateRequest           nativeperf.GateRequest
	DefaultReadback       DefaultReadback
}

// NativePerfDecisionJournal is the immutable, replayable result of one transaction.
type NativePerfDecisionJournal struct {
	Schema                 string                              `json:"schema"`
	ProfileDigest          string                              `json:"profile_digest"`
	ProfileClassification  nativeperf.BottleneckClassification `json:"profile_classification"`
	ObligationGraphDigest  string                              `json:"obligation_graph_digest"`
	SelectedObligationID   string                              `json:"selected_obligation_id"`
	SelectedCandidateID    string                              `json:"selected_candidate_id"`
	BaselineReceiptDigest  string                              `json:"baseline_receipt_digest"`
	CandidateReceiptDigest string                              `json:"candidate_receipt_digest"`
	GateRequestDigest      string                              `json:"gate_request_digest"`
	GateOutput             nativeperf.GateVerdict              `json:"gate_output"`
	Decision               string                              `json:"decision"`
	DefaultStateReadback   DefaultReadback                     `json:"default_state_readback"`
	ReplayDigest           string                              `json:"replay_digest"`
}

// RunNativePerfTransaction validates every input, delegates classification and
// promotion policy to nativeperf, and returns one content-addressed journal row.
func RunNativePerfTransaction(r NativePerfTransactionRequest) (NativePerfDecisionJournal, error) {
	graph := nativeperf.ActiveGraph()
	classification, err := nativeperf.ClassifyProfile(graph, r.Profile)
	if err != nil {
		return NativePerfDecisionJournal{}, fmt.Errorf("classify profile: %w", err)
	}
	graphDigest, err := digestJSON(r.ObligationGraph)
	if err != nil {
		return NativePerfDecisionJournal{}, err
	}
	if r.ObligationGraphDigest == "" || r.ObligationGraphDigest != graphDigest {
		return NativePerfDecisionJournal{}, fmt.Errorf("obligation graph digest mismatch: got %q want %q", r.ObligationGraphDigest, graphDigest)
	}
	if r.ObligationGraph.Engine != "fak-native" || r.ObligationGraph.ExternalRuntimeFallback {
		return NativePerfDecisionJournal{}, fmt.Errorf("obligation graph must use fak-native with no external fallback")
	}
	if err := validateSelection(r, classification); err != nil {
		return NativePerfDecisionJournal{}, err
	}
	verdict, err := nativeperf.Gate(r.GateRequest)
	if err != nil {
		return NativePerfDecisionJournal{}, fmt.Errorf("nativeperf gate: %w", err)
	}
	decision := NativePerfReject
	expected := DefaultReadback{Selected: "baseline", Revision: r.GateRequest.LastAccepted.Revision}
	if verdict.Classification == nativeperf.GatePass {
		decision = NativePerfKeep
		expected = DefaultReadback{Selected: "candidate", Revision: r.GateRequest.Candidate.Revision}
	}
	if r.DefaultReadback.Selected == "" || r.DefaultReadback.Revision == "" {
		return NativePerfDecisionJournal{}, fmt.Errorf("default-state readback is required")
	}
	if r.DefaultReadback != expected {
		return NativePerfDecisionJournal{}, fmt.Errorf("default-state readback mismatch: got %+v want %+v", r.DefaultReadback, expected)
	}

	profileDigest, _ := digestJSON(r.Profile)
	baselineDigest, _ := digestJSON(r.GateRequest.LastAccepted)
	candidateDigest, _ := digestJSON(r.GateRequest.Candidate)
	gateRequestDigest, _ := digestJSON(r.GateRequest)
	j := NativePerfDecisionJournal{
		Schema: NativePerfDecisionSchema, ProfileDigest: profileDigest,
		ProfileClassification: classification, ObligationGraphDigest: graphDigest,
		SelectedObligationID: r.ObligationID, SelectedCandidateID: r.CandidateID,
		BaselineReceiptDigest: baselineDigest, CandidateReceiptDigest: candidateDigest,
		GateRequestDigest: gateRequestDigest, GateOutput: verdict, Decision: decision,
		DefaultStateReadback: expected,
	}
	j.ReplayDigest, err = digestDecision(j)
	if err != nil {
		return NativePerfDecisionJournal{}, err
	}
	return j, nil
}

func validateSelection(r NativePerfTransactionRequest, c nativeperf.BottleneckClassification) error {
	if r.ObligationID == "" || r.CandidateID == "" {
		return fmt.Errorf("selected obligation and candidate IDs are required")
	}
	matches := 0
	for _, n := range r.ObligationGraph.Nodes {
		if n.ID != r.ObligationID || n.PromotionWitness.ID != r.CandidateID {
			continue
		}
		matches++
		if !n.Eligible || len(n.Blockers) != 0 {
			return fmt.Errorf("selected obligation %q is not eligible", n.ID)
		}
		if n.Backend.Engine != "fak-native" {
			return fmt.Errorf("selected obligation %q is not fak-native", n.ID)
		}
		if n.Operation != c.RecommendedLeverID && n.Backend.Operation != c.RecommendedLeverID {
			return fmt.Errorf("selected obligation %q does not implement classified lever %q", n.ID, c.RecommendedLeverID)
		}
	}
	if matches != 1 {
		return fmt.Errorf("selected obligation/candidate must identify exactly one graph node; matched %d", matches)
	}
	return nil
}

// ReplayNativePerfTransaction recomputes the transaction and requires byte-stable identity.
func ReplayNativePerfTransaction(r NativePerfTransactionRequest, recorded NativePerfDecisionJournal) error {
	got, err := RunNativePerfTransaction(r)
	if err != nil {
		return err
	}
	wantBytes, err := json.Marshal(recorded)
	if err != nil {
		return err
	}
	gotBytes, err := json.Marshal(got)
	if err != nil {
		return err
	}
	if !bytes.Equal(gotBytes, wantBytes) {
		return fmt.Errorf("nativeperf decision replay mismatch")
	}
	return nil
}

// ParseNativePerfDecisionLedger strictly decodes and verifies immutable JSONL rows.
func ParseNativePerfDecisionLedger(content string) ([]NativePerfDecisionJournal, error) {
	if content != "" && !strings.HasSuffix(content, "\n") {
		return nil, fmt.Errorf("nativeperf decision ledger must end with a newline")
	}
	s := bufio.NewScanner(strings.NewReader(content))
	s.Buffer(make([]byte, receiptScannerInitSize), MaxReceiptLineBytes)
	var rows []NativePerfDecisionJournal
	for line := 1; s.Scan(); line++ {
		if len(bytes.TrimSpace(s.Bytes())) == 0 {
			return nil, fmt.Errorf("line %d: blank rows are not allowed", line)
		}
		dec := json.NewDecoder(bytes.NewReader(s.Bytes()))
		dec.DisallowUnknownFields()
		var row NativePerfDecisionJournal
		if err := dec.Decode(&row); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if dec.More() {
			return nil, fmt.Errorf("line %d: trailing JSON", line)
		}
		if err := validateDecision(row); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// AppendNativePerfDecision appends one verified row and refuses duplicate identities.
func AppendNativePerfDecision(path string, row NativePerfDecisionJournal) error {
	if err := validateDecision(row); err != nil {
		return err
	}
	line, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if len(line) > MaxReceiptLineBytes {
		return fmt.Errorf("nativeperf decision row exceeds %d bytes", MaxReceiptLineBytes)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lock, err := acquireLedgerLock(path+".lock", time.Now().UTC())
	if err != nil {
		return err
	}
	defer lock.release()
	content, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	rows, err := ParseNativePerfDecisionLedger(string(content))
	if err != nil {
		return err
	}
	for _, existing := range rows {
		if existing.ReplayDigest == row.ReplayDigest {
			return fmt.Errorf("nativeperf decision %s already exists", row.ReplayDigest)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err = f.Write(append(line, '\n')); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func validateDecision(j NativePerfDecisionJournal) error {
	if j.Schema != NativePerfDecisionSchema {
		return fmt.Errorf("schema must be %q", NativePerfDecisionSchema)
	}
	if j.ProfileDigest == "" || j.ObligationGraphDigest == "" || j.BaselineReceiptDigest == "" || j.CandidateReceiptDigest == "" || j.GateRequestDigest == "" {
		return fmt.Errorf("all input digests are required")
	}
	if j.SelectedObligationID == "" || j.SelectedCandidateID == "" {
		return fmt.Errorf("selected IDs are required")
	}
	if j.Decision != NativePerfKeep && j.Decision != NativePerfReject {
		return fmt.Errorf("decision must be KEEP or REJECT")
	}
	if j.DefaultStateReadback.Selected == "" || j.DefaultStateReadback.Revision == "" {
		return fmt.Errorf("default-state readback is required")
	}
	if j.Decision == NativePerfKeep && j.DefaultStateReadback.Selected != "candidate" || j.Decision == NativePerfReject && j.DefaultStateReadback.Selected != "baseline" {
		return fmt.Errorf("default-state readback is inconsistent with decision")
	}
	want, err := digestDecision(j)
	if err != nil {
		return err
	}
	if j.ReplayDigest == "" || j.ReplayDigest != want {
		return fmt.Errorf("replay digest mismatch")
	}
	return nil
}

func digestDecision(j NativePerfDecisionJournal) (string, error) {
	j.ReplayDigest = ""
	return digestJSON(j)
}

func digestJSON(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
