package routing

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// This file is the READER of the fak.route-decision.v1 receipt ledger: the
// schema, its location, and a crash-tolerant decode that the ops projection
// (platform/ops/opsevent/project, reached by `fak-sync ops api --stats`) folds
// into the route_decisions plane. The router core stays pure: Route() performs
// no I/O and writes nothing.
//
// The in-repo writer (DecisionLedger, NewRoutedReceipt, NewUnroutedReceipt) was
// deleted in RA-01 leaf R3b: no production entrypoint reached it (`fak-flow
// route placement` prints its Assignment and never wrote a receipt), and the
// router's decision record is the routing journal (gateway --routing-journal),
// which writes a line for every routed request, including ones that made no
// upstream attempt. An external writer appends one JSON receipt per line.

// DecisionSchema is the fak.route-decision.v1 record identifier.
const DecisionSchema = "fak.route-decision.v1"

// DefaultDecisionLedgerRel is the repo-relative location of the decision ledger.
// It lives beside the other Ops ledgers under .fak/ so one directory listing
// shows every durable Ops stream.
const DefaultDecisionLedgerRel = ".fak/route-decisions.jsonl"

// DecisionPath returns <root>/.fak/route-decisions.jsonl.
func DecisionPath(root string) string {
	return filepath.Join(root, filepath.FromSlash(DefaultDecisionLedgerRel))
}

// DecisionReceipt is one durable routing decision. It is deliberately a flat
// projection of Assignment rather than the full struct: the audit prose and the
// rejected-engine detail are summarized to bounded counts and closed tokens so
// the ledger stays small and aggregable, and free text never becomes a grouping
// key.
type DecisionReceipt struct {
	Schema string `json:"schema"`
	// At is when the decision was made.
	At time.Time `json:"at"`
	// TicketID / Lane identify the work the decision was made for.
	TicketID string `json:"ticket_id,omitempty"`
	Lane     string `json:"lane,omitempty"`
	// Role is the routing role the decision served.
	Role string `json:"role,omitempty"`
	// Routed is false when the request could not be placed. An unrouted decision
	// is a real, recorded outcome, not a missing row.
	Routed bool `json:"routed"`
	// EngineID / Surface / Model are the chosen target. Empty when Routed=false.
	EngineID string  `json:"engine_id,omitempty"`
	Surface  Surface `json:"surface,omitempty"`
	Model    string  `json:"model,omitempty"`
	// Reason is the closed reason the winner won, or the closed refusal reason
	// when Routed=false.
	Reason string `json:"reason"`
	// Rejected is the per-engine rejection detail, bounded to the closed reason
	// vocabulary so it aggregates.
	Rejected []Rejection `json:"rejected,omitempty"`
	// AuditSteps is the count of audit lines, not the lines themselves: the
	// prose stays on the console, the count stays queryable.
	AuditSteps int `json:"audit_steps,omitempty"`
	// Source names what produced the decision (a CLI verb, a dispatch path) so a
	// surprising distribution can be traced.
	Source string `json:"source,omitempty"`
}

// ReadDecisions reads every receipt from a decision ledger. A partial trailing
// line (a crash mid-append) is tolerated and skipped rather than failing the
// read, matching the other Ops ledgers.
func ReadDecisions(path string) ([]DecisionReceipt, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("routing: read decision ledger: %w", err)
	}
	defer f.Close()
	return decodeDecisions(f)
}

func decodeDecisions(r io.Reader) ([]DecisionReceipt, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var out []DecisionReceipt
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec DecisionReceipt
		if err := json.Unmarshal(line, &rec); err != nil {
			// A trailing partial write is a tolerated crash artifact. Any other
			// malformed line is reported so it is not silently lost.
			if !json.Valid(line) {
				continue
			}
			return out, fmt.Errorf("routing: decode decision receipt: %w", err)
		}
		if rec.Schema != DecisionSchema {
			continue
		}
		out = append(out, rec)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("routing: scan decision ledger: %w", err)
	}
	return out, nil
}
