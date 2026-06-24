package advmodel

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// floorManifest is a FROZEN copy of examples/customer-support-readonly-policy.json
// — the shipped, real deployable capability floor. The training corpus's labels
// are the verdicts THIS floor emits, so the model learns the floor's actual
// decision surface (the same (call, verdict) stream internal/harvest folds into
// abi.LabelRow). It is embedded rather than read from disk so the corpus is
// self-contained and reproducible from this package alone. corpus_test re-derives
// every label from it, so a drift between this copy and the shipped example is a
// caught regression, not a silent training bug.
const floorManifest = `{
  "version": "fak-policy/v1",
  "posture": "fail_closed",
  "allow": [
    "create_support_ticket",
    "read_customer_record",
    "read_corp_kb",
    "transfer_to_human_agents"
  ],
  "allow_prefix": [
    "read_",
    "get_",
    "search_",
    "list_",
    "lookup_",
    "find_"
  ],
  "deny": {
    "delete_account": "POLICY_BLOCK",
    "export_customer_data": "SECRET_EXFIL",
    "refund_payment": "POLICY_BLOCK",
    "rotate_credentials": "POLICY_BLOCK",
    "send_customer_email": "POLICY_BLOCK",
    "transfer_funds": "POLICY_BLOCK"
  },
  "self_modify_globs": [
    ".git/",
    ".dos/",
    "policy.json",
    "customer-support-readonly-policy.json",
    "/etc/",
    "id_rsa"
  ],
  "redact_fields": [
    "password",
    "secret",
    "api_key",
    "token",
    "authorization",
    "ssn"
  ],
  "safe_sinks": [
    "transfer_to_human_agents"
  ],
  "sources": {
    "fetch_url": "untrusted",
    "read_corp_kb": "trusted_local",
    "read_customer_record": "trusted_local"
  },
  "arg_rules": [
    {
      "tool": "create_support_ticket",
      "arg": "body",
      "max_bytes": 4000,
      "reason": "OVERSIZE"
    }
  ]
}`

// FloorPolicy returns the real adjudicator floor the corpus is labeled against.
// It is the parsed frozen manifest above — a deployable capability floor, not a
// hand-rolled approximation. panics are impossible (the manifest is a const and
// policy_examples_test already proves the equivalent file parses), but the error
// is surfaced for honesty.
func FloorPolicy() adjudicator.Policy {
	p, err := policy.Parse([]byte(floorManifest))
	if err != nil {
		// A const manifest that fails to parse is a build-time programming error,
		// not a runtime condition — surface it loudly.
		panic(fmt.Sprintf("advmodel: frozen floor manifest did not parse: %v", err))
	}
	return p
}

// CorpusRow is one labeled training example: the call (tool + raw args JSON) and
// the floor's verdict on it. deny=true means the floor returned a non-Allow
// verdict (the deny-worthy positive class); reason is the floor's ReasonCode
// name. This is the content-bearing form of the harvest corpus — the same
// (call, verdict) stream internal/harvest folds into abi.LabelRow, with the call
// content retained (harvest hashes it away for de-dup identity).
type CorpusRow struct {
	Tool   string `json:"tool"`
	Args   string `json:"args"`
	Deny   bool   `json:"deny"`
	Reason string `json:"reason,omitempty"`
}

// FloorLabel runs the real adjudicator floor over a corpus row's call and
// returns the verdict + the derived (deny, reason) label. deny is true iff the
// floor did not affirmatively Allow the call (Deny/Transform/Defer-to-default —
// under a fail-closed posture, anything not allowed is denied). A Transform
// (secret redaction) is treated as ALLOWED-but-rewritten, hence NOT deny-worthy,
// so the model learns the floor's admit/deny boundary, not its redaction policy.
func FloorLabel(p adjudicator.Policy, r CorpusRow) (verdict abi.Verdict, deny bool, reason string) {
	adj := adjudicator.New(p)
	verdict = adj.Adjudicate(context.Background(), &abi.ToolCall{
		Tool: r.Tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(r.Args)},
	})
	deny = verdict.Kind == abi.VerdictDeny
	reason = abi.ReasonName(verdict.Reason)
	return
}

// LoadCorpus reads newline-delimited CorpusRows. Blank lines are skipped. It is
// the shared reader for corpus_test (honesty witness) and could back a Go-side
// trainer; train.py reads the same file directly.
func LoadCorpus(r io.Reader) ([]CorpusRow, error) {
	dec := json.NewDecoder(r)
	var rows []CorpusRow
	for dec.More() {
		var row CorpusRow
		if err := dec.Decode(&row); err != nil {
			return nil, fmt.Errorf("advmodel: decode corpus row: %w", err)
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("advmodel: corpus is empty")
	}
	return rows, nil
}

// CorpusJSON renders rows as newline-delimited JSON (the on-disk shape of
// testdata/corpus.jsonl). Used by the regeneration helper + tests.
func CorpusJSON(rows []CorpusRow) []byte {
	var b strings.Builder
	for _, r := range rows {
		j, _ := json.Marshal(r)
		b.Write(j)
		b.WriteByte('\n')
	}
	return []byte(b.String())
}
