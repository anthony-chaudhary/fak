package gateway

import (
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/trajquery"
)

// mcp_trajquery.go exposes the trajquery scoped-view query surface over MCP (#3550).
// Before this, trajquery was CLI-only (`fak trajquery run|validate`) and the gateway's
// MCP server exposed only the kernel/index/feature tools — so an MCP client could not
// query its own trajectory corpus through the SAME scope-enforcing rewrite the CLI uses.
//
// The tool routes the identical validate -> rewrite -> execute path as cmdTrajQueryRun:
// a user SELECT is validated against an operator-published View (its Columns allowlist,
// per-column Enums literal-sets, and non-removable Scope predicates), rewritten to inline
// the scope against the base relation, and executed over the supplied corpus. A query that
// would escape the scope (targeting the base, projecting a hidden column, or a WHERE value
// outside an enum's literals) comes back as a first-class refusal — {valid:false,
// violations:[...]} with NO rows — never as leaked data. The tool description and
// inputSchema advertise the scoped View schema (columns + enum literals) so a client knows
// exactly what it may project and filter, and the response echoes that schema back.

// TrajQueryRequest is the fak_trajquery argument body: the operator-published scoped view,
// the SELECT to run against it, the corpus to execute over, and an optional validate-only
// switch that checks scope soundness without returning any rows.
type TrajQueryRequest struct {
	View         trajquery.View  `json:"view"`
	SQL          string          `json:"sql"`
	Corpus       []trajquery.Row `json:"corpus,omitempty"`
	ValidateOnly bool            `json:"validate_only,omitempty"`
}

// TrajQueryResponse carries the view's advertised schema (so the caller always sees the
// columns/enums/scope that bound the query), the scope verdict, the scope-enforced
// rewritten query (for audit), and the projected rows. Rows is empty on a refusal or a
// validate-only call.
type TrajQueryResponse struct {
	Schema     trajquery.ViewSchema `json:"schema"`
	Valid      bool                 `json:"valid"`
	Violations []string             `json:"violations,omitempty"`
	Rewritten  string               `json:"rewritten,omitempty"`
	Rows       []trajquery.Row      `json:"rows,omitempty"`
}

// trajQuery runs one scoped trajectory query. It mirrors cmdTrajQueryRun: parse, validate
// (refuse-if-unsound), rewrite to enforce scope, then execute. A parse/rewrite/execute
// fault is a tool error; a scope escape is NOT an error — it is returned as {valid:false,
// violations} so the model sees exactly why it was refused, with no rows leaked.
func (s *Server) trajQuery(req TrajQueryRequest) (TrajQueryResponse, error) {
	q, err := trajquery.Parse(req.SQL)
	if err != nil {
		return TrajQueryResponse{}, fmt.Errorf("fak_trajquery: parse error: %w", err)
	}
	resp := TrajQueryResponse{Schema: req.View.Schema()}
	// Validate with the corpus so the dynamic no-leak check runs too when rows are present.
	rep := req.View.Validate(q, req.Corpus)
	resp.Valid = rep.Valid
	resp.Violations = rep.Violations
	if !rep.Valid {
		return resp, nil // scope refusal is data, not a fault — never leak rows
	}
	rewritten, err := req.View.Rewrite(q)
	if err != nil {
		return TrajQueryResponse{}, fmt.Errorf("fak_trajquery: rewrite error: %w", err)
	}
	resp.Rewritten = rewritten.String()
	if req.ValidateOnly {
		return resp, nil
	}
	rows, err := rewritten.Execute(req.Corpus)
	if err != nil {
		return TrajQueryResponse{}, fmt.Errorf("fak_trajquery: execute error: %w", err)
	}
	resp.Rows = rows
	return resp, nil
}

// trajqueryInputSchema advertises the scoped View schema — the column allowlist and the
// per-column enum literals — inside the tool contract, so a client can construct a legal
// query without probing. It is the MCP-visible statement of what trajquery confines.
var trajqueryInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "view": {
      "type": "object",
      "description": "the operator-published scoped view: the ONLY sanctioned query target (the base relation is never queried directly)",
      "properties": {
        "name": {"type": "string", "description": "the relation name your SQL must target; querying 'base' instead is a scope escape"},
        "base": {"type": "string", "description": "the underlying corpus relation the rewrite reads (not a legal query target)"},
        "columns": {"type": "array", "items": {"type": "string"}, "description": "allowlist of columns you may project (SELECT) or filter (WHERE); any other column is refused"},
        "enums": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}, "description": "optional per-column closed set of allowed WHERE literals, {\"column\": [\"lit\", ...]}. A WHERE value on an enum column must be one of these literals on ANY operator — documenting the legal values and blocking one-literal-at-a-time probing"},
        "scope": {"type": "array", "description": "non-removable row predicates ANDed into every query", "items": {"type": "object", "properties": {"field": {"type": "string"}, "op": {"type": "string", "enum": ["=", "!=", "<", ">", "<=", ">=", "LIKE"]}, "value": {"type": "string"}}, "required": ["field", "op", "value"]}}
      },
      "required": ["name", "base", "columns"]
    },
    "sql": {"type": "string", "description": "a SELECT over the view: SELECT <cols|*> FROM <view.name> [WHERE <field op literal> AND ...] [LIMIT n]"},
    "corpus": {"type": "array", "items": {"type": "object"}, "description": "the trajectory rows to execute over; each row is an object of column -> value"},
    "validate_only": {"type": "boolean", "description": "check scope soundness WITHOUT executing (no rows returned)"}
  },
  "required": ["view", "sql"]
}`)

// trajqueryToolDescriptor is the tools/list entry for fak_trajquery. Split out of
// toolDescriptors (mcp.go) so the descriptor lives beside the handler it documents.
func trajqueryToolDescriptor() map[string]any {
	return map[string]any{
		"name":        "fak_trajquery",
		"description": "Run a read-only SQL SELECT over your trajectory corpus inside an operator-published scoped view. Only allowlisted columns and declared enums are valid; the server adds immutable scope predicates, and scope escapes return valid:false with no rows. Pass view, sql, optional corpus or validate_only. Returns schema, rewritten query, violations, and rows.",
		"inputSchema": trajqueryInputSchema,
		"annotations": map[string]any{"readOnlyHint": true, "read_only_hint": true, "idempotentHint": true, "idempotent_hint": true},
	}
}
