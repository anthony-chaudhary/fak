package routing

// RA-01 leaf 6 witness, fabric side
// (docs/tickets/router-architecture/RA-01-one-routing-decision-authority.md,
// "AttemptBudget in the decision; Requeue/Retry consume it instead of their
// private ceilings. Delete the orphaned routing.RouteTable/TableHolder path").
//
// CONTRACT the implementer must satisfy (package routing):
//
//  1. The fabric owns the ONE total attempt budget, next to the per-class
//     same-target budget it already owns (Failure.RetryBudget,
//     docs/ROUTER-FAILOVER-AUTHORITY.md "Budget" plane):
//
//       // DefaultMaxAttempts is the total upstream attempts one client
//       // request may spend when the routing config declares no
//       // max_attempts. Finite and positive; 16 is the suggested value
//       // (these tests only require >= 4 so every existing walk test,
//       // e.g. a primary plus three fallbacks, stays green).
//       const DefaultMaxAttempts = 16
//
//       // AttemptBudget is the ONE bound on the upstream attempts a single
//       // client request may spend across every layer that re-executes it:
//       // the chain walk, same-backend key retry (still capped per class by
//       // Failure.RetryBudget), the durable zero-output requeue, lease-expiry
//       // reclaim, and client POST .../retry.
//       type AttemptBudget struct {
//           Max int `json:"max"` // total upstream attempts; <= 0 means DefaultMaxAttempts
//       }
//
//       // NewAttemptBudget returns {Max: max}, or {Max: DefaultMaxAttempts}
//       // when max <= 0.
//       func NewAttemptBudget(max int) AttemptBudget
//
//       // Remaining is how many further upstream attempts the budget
//       // authorizes after `spent` have been made: limit - spent floored at
//       // 0, where limit is Max, or DefaultMaxAttempts when Max <= 0 (the
//       // zero value is bounded by the default: never unbounded, never 0).
//       func (b AttemptBudget) Remaining(spent int) int
//
//  2. The orphaned snapshot path is DELETED, not folded (reachability proof,
//     2026-09-26, rg over every non-test .go file in the repo): the only
//     non-test reference to routing.RouteTable/TableHolder outside
//     platform/routing/route_table.go is platform/resbroker.RouteChainWithTable,
//     and RouteChainWithTable has zero non-test callers; cmd/fak-server serves
//     from gateway.RouteTable. Delete platform/routing/route_table.go (every
//     declaration named in TestOrphanedRouteTableIsDeleted) and
//     resbroker.RouteChainWithTable. Their tests were already removed by the
//     test plane (platform/routing/route_table_test.go and the two
//     TestRouteChainWithTable* tests in platform/resbroker). Record the
//     decision ("deleted, not folded", with this evidence) in the RA-01 ticket.
//
// Witness: go test ./platform/routing -count=1

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// TestAttemptBudgetIsOneFiniteTotal pins the budget value itself: a configured
// bound is kept verbatim, an unset or negative one becomes the fabric's finite
// default, and Remaining never goes negative or unbounded.
func TestAttemptBudgetIsOneFiniteTotal(t *testing.T) {
	if DefaultMaxAttempts < 4 {
		t.Fatalf("DefaultMaxAttempts = %d, want >= 4: an unconfigured router must still walk a primary plus three fallbacks", DefaultMaxAttempts)
	}
	for _, tc := range []struct {
		in, want int
	}{
		{in: 3, want: 3},
		{in: 1, want: 1},
		{in: 0, want: DefaultMaxAttempts},
		{in: -5, want: DefaultMaxAttempts},
	} {
		if got := NewAttemptBudget(tc.in).Max; got != tc.want {
			t.Errorf("NewAttemptBudget(%d).Max = %d, want %d", tc.in, got, tc.want)
		}
	}

	three := NewAttemptBudget(3)
	for _, tc := range []struct {
		spent, want int
	}{
		{spent: 0, want: 3},
		{spent: 1, want: 2},
		{spent: 2, want: 1},
		{spent: 3, want: 0},
		{spent: 9, want: 0},
	} {
		if got := three.Remaining(tc.spent); got != tc.want {
			t.Errorf("NewAttemptBudget(3).Remaining(%d) = %d, want %d", tc.spent, got, tc.want)
		}
	}

	// The zero value is a caller that forgot to resolve a decision: it is
	// still bounded by the default, never unbounded and never zero.
	var zero AttemptBudget
	if got := zero.Remaining(0); got != DefaultMaxAttempts {
		t.Errorf("AttemptBudget{}.Remaining(0) = %d, want DefaultMaxAttempts (%d)", got, DefaultMaxAttempts)
	}
	if got := zero.Remaining(DefaultMaxAttempts); got != 0 {
		t.Errorf("AttemptBudget{}.Remaining(DefaultMaxAttempts) = %d, want 0", got)
	}
}

// TestOrphanedRouteTableIsDeleted pins the RA-01 leaf 6 delete-or-fold
// decision: the fabric's second RouteTable (engines + policy + TableHolder
// hot-swap) had no production entrypoint, so it is deleted and
// gateway.RouteTable is the only routing table. A declaration reappearing here
// is a second table the live router would again not read.
func TestOrphanedRouteTableIsDeleted(t *testing.T) {
	forbidden := map[string]bool{
		"RouteTable": true, "NewRouteTable": true, "RouteTableSchema": true,
		"TableHolder": true, "NewTableHolder": true,
		"TableSource": true, "ManualTableSource": true, "NewManualTableSource": true,
		"EndpointTableSource": true, "NewEndpointTableSource": true,
		"EndpointFetcher": true, "TableDecoder": true,
		"RefreshOutcome": true, "RefreshResult": true, "RefreshResultSchema": true,
		"RefreshAdopted": true, "RefreshUnchanged": true, "RefreshRejectedStale": true, "RefreshFailed": true,
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
				if d.Recv == nil && forbidden[d.Name.Name] {
					found = append(found, name+": func "+d.Name.Name)
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if forbidden[s.Name.Name] {
							found = append(found, name+": type "+s.Name.Name)
						}
					case *ast.ValueSpec:
						for _, id := range s.Names {
							if forbidden[id.Name] {
								found = append(found, name+": "+d.Tok.String()+" "+id.Name)
							}
						}
					}
				}
			}
		}
	}
	if len(found) > 0 {
		t.Fatalf("pkg/routing still declares the orphaned RouteTable/TableHolder path (no production entrypoint reaches it; gateway.RouteTable is the live table):\n  %s",
			strings.Join(found, "\n  "))
	}
}
