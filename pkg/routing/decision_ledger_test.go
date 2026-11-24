package routing

// RA-01 leaf R3b, the DecisionLedger decision
// (docs/tickets/router-architecture/RA-01-one-routing-decision-authority.md,
// "routing.DecisionLedger ... has no writer on the router path"; INDEX.md item
// 5 "decide whether the ledger is written or deleted").
//
// DECISION (by reachability, docs/PRESENCE-VS-INVOKABILITY.md): DELETE the
// writer half; keep the fak.route-decision.v1 reader.
//   - The writer has no production caller. A repo-wide search of *.go for
//     NewDecisionLedger / NewDecisionLedgerAt / DecisionLedger.Append /
//     NewRoutedReceipt / NewUnroutedReceipt finds only tests
//     (this file, platform/ops/opsevent/project/project_test.go,
//     cmd/fak-sync/ops_api_stats_test.go). The one production caller of the
//     fabric, `fak-flow route placement` (cmd/fak-flow/route.go), prints its
//     Assignment and never wrote a receipt.
//   - The router's decision record is the routing journal: after leaf R3b
//     every routed request that resolves a decision writes a journal line
//     (config_hash, rule, skipped[], error_code), including no-attempt and
//     refused-spend ends (platform/gateway/route_decision_record_test.go).
//     Writing a fabric placement receipt per HTTP request from the router
//     would be a synchronous open/append/close on the hot path with
//     ticket/lane/role/surface fields the router does not have.
//   - The READER stays: platform/ops/opsevent/project (reached by
//     `fak-sync ops api --stats`, cmd/fak-sync/ops_api_stats.go) projects
//     .fak/route-decisions.jsonl into the route_decisions plane and reports an
//     absent ledger as Unavailable, not zero.
//
// CONTRACT the implementer must satisfy (platform/routing/decision_ledger.go):
//   - Delete from non-test code: type DecisionLedger and every method on it
//     (Path, Append), func NewDecisionLedger, func NewDecisionLedgerAt,
//     func NewRoutedReceipt, func NewUnroutedReceipt, and the writer-side
//     method DecisionReceipt.Validate (its only caller was Append).
//   - Keep unchanged: const DecisionSchema, const DefaultDecisionLedgerRel,
//     func DecisionPath, type DecisionReceipt (same JSON tags), func
//     ReadDecisions and its tolerant decoding (partial trailing line skipped,
//     foreign schema skipped, missing file = empty, no error).
//   - Record the decision and its reachability evidence in RA-01 (Progress)
//     and INDEX.md item 5, and name the follow-up: the route_decisions ops
//     plane has no in-repo writer; next step is to have `fak-flow route
//     placement` emit fak.route-decision.v1 receipts, or retire the plane.
//
// Witness: go test ./platform/routing -count=1

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestDecisionLedgerWriterIsDeleted pins the reachability decision: the
// receipt writer no production entrypoint called is gone, and the reader the
// ops projection calls is still here.
func TestDecisionLedgerWriterIsDeleted(t *testing.T) {
	forbiddenFuncs := map[string]bool{
		"NewDecisionLedger": true, "NewDecisionLedgerAt": true,
		"NewRoutedReceipt": true, "NewUnroutedReceipt": true,
	}
	required := map[string]bool{
		"type DecisionReceipt": false, "func ReadDecisions": false,
		"func DecisionPath": false, "const DecisionSchema": false,
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	fset := token.NewFileSet()
	var found []string
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, name, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil {
					if forbiddenFuncs[d.Name.Name] {
						found = append(found, name+": func "+d.Name.Name)
					}
					if _, ok := required["func "+d.Name.Name]; ok {
						required["func "+d.Name.Name] = true
					}
					continue
				}
				recv := receiverTypeName(d.Recv.List[0].Type)
				if recv == "DecisionLedger" || (recv == "DecisionReceipt" && d.Name.Name == "Validate") {
					found = append(found, name+": method ("+recv+")."+d.Name.Name)
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if s.Name.Name == "DecisionLedger" {
							found = append(found, name+": type DecisionLedger")
						}
						if _, ok := required["type "+s.Name.Name]; ok {
							required["type "+s.Name.Name] = true
						}
					case *ast.ValueSpec:
						for _, id := range s.Names {
							if _, ok := required[d.Tok.String()+" "+id.Name]; ok {
								required[d.Tok.String()+" "+id.Name] = true
							}
						}
					}
				}
			}
		}
	}
	if len(found) > 0 {
		sort.Strings(found)
		t.Errorf("pkg/routing still declares the DecisionLedger writer, which no production entrypoint reaches (the routing journal is the router's decision record):\n  %s",
			strings.Join(found, "\n  "))
	}
	var missing []string
	for decl, ok := range required {
		if !ok {
			missing = append(missing, decl)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		t.Errorf("the fak.route-decision.v1 reader the ops projection calls must stay; missing: %v", missing)
	}
}

// receiverTypeName is the base type name of a method receiver (T or *T).
func receiverTypeName(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.StarExpr:
		return receiverTypeName(e.X)
	case *ast.Ident:
		return e.Name
	case *ast.IndexExpr:
		return receiverTypeName(e.X)
	}
	return ""
}

// writeDecisionLines writes receipts as NDJSON exactly as an external writer
// of fak.route-decision.v1 would, then appends each raw suffix verbatim.
func writeDecisionLines(t *testing.T, path string, recs []DecisionReceipt, raw ...string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, r := range recs {
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(data)
		b.WriteByte('\n')
	}
	for _, s := range raw {
		b.WriteString(s)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func sampleReceipts(at time.Time) []DecisionReceipt {
	return []DecisionReceipt{
		{
			Schema: DecisionSchema, At: at, TicketID: "T-1", Role: "worker", Routed: true,
			EngineID: "strix-local", Surface: SurfaceLocal, Model: "deepseek-v4-flash",
			Reason:   "engine declares role worker and has headroom",
			Rejected: []Rejection{{EngineID: "hosted-1", Reason: ReasonHostedRefused}}, AuditSteps: 2, Source: "test",
		},
		{
			Schema: DecisionSchema, At: at, TicketID: "T-2", Lane: "lane-a", Role: "judge", Routed: false,
			Reason: "no engine satisfies RequireLocal", Source: "test",
		},
	}
}

func TestReadDecisionsReadsReceipts(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".fak", "route-decisions.jsonl")
	at := time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)
	// A line of another schema in the same file is skipped, not an error.
	writeDecisionLines(t, path, sampleReceipts(at), `{"schema":"fak.other.v1","at":"2026-09-24T12:00:00Z","reason":"x"}`+"\n")
	got, err := ReadDecisions(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 receipts, got %d: %+v", len(got), got)
	}
	if !got[0].Routed || got[0].EngineID != "strix-local" || got[0].Surface != SurfaceLocal || !got[0].At.Equal(at) {
		t.Fatalf("routed receipt lost fields: %+v", got[0])
	}
	if len(got[0].Rejected) != 1 || got[0].Rejected[0].Reason != ReasonHostedRefused || got[0].AuditSteps != 2 {
		t.Fatalf("routed receipt lost rejections/audit count: %+v", got[0])
	}
	if got[1].Routed || got[1].EngineID != "" || got[1].Reason == "" || got[1].Lane != "lane-a" {
		t.Fatalf("unrouted receipt: %+v", got[1])
	}
}

func TestReadDecisionsToleratesPartialTrailingLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route-decisions.jsonl")
	writeDecisionLines(t, path, sampleReceipts(time.Now().UTC())[:1], `{"schema":"fak.route-decision.v1","at":"2026-`)
	got, err := ReadDecisions(path)
	if err != nil {
		t.Fatalf("a partial trailing write must be tolerated, got %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("want the 1 complete receipt, got %d", len(got))
	}
}

func TestReadDecisionsMissingFileIsEmptyNotError(t *testing.T) {
	got, err := ReadDecisions(filepath.Join(t.TempDir(), "nope.jsonl"))
	if err != nil {
		t.Fatalf("missing ledger must read as empty, not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("want 0, got %d", len(got))
	}
}

func TestDecisionPathIsUnderFak(t *testing.T) {
	p := DecisionPath("/root/repo")
	if !strings.Contains(p, ".fak") || !strings.HasSuffix(p, "route-decisions.jsonl") {
		t.Fatalf("unexpected path %q", p)
	}
}
