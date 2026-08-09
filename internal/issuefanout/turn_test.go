package issuefanout

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

var turnAt = time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)

// ship is a shipped spine with the paths its commit changed.
func ship(issue int, sha string, paths ...string) Ship {
	return Ship{Issue: issue, SpineRef: sha, Paths: paths}
}

// ledgerRows decodes the durable rows a turn appended, in order.
func ledgerRows(t *testing.T, buf *bytes.Buffer) []LedgerRow {
	t.Helper()
	var rows []LedgerRow
	for _, line := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r LedgerRow
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("ledger line %q: %v", line, err)
		}
		rows = append(rows, r)
	}
	return rows
}

// The default in the pipeline (#2523): a turn fans out every leaf its ships touched,
// exactly once each, and leaves one durable ledger row per invocation behind. The
// once-each part is the issue's own confusion risk — a second pass over the same leaf
// would plan the same fanout-<leaf>-<slug> keys twice, and filed they would double-file.
func TestTurnFansOutEveryShippedLeafOnce(t *testing.T) {
	var ledger bytes.Buffer
	res := Turn([]Ship{
		ship(4001, "abc1234", "internal/issuefanout/turn.go", "cmd/fak/dispatch_fanout.go"),
		// A second commit in the SAME leaf plus one in a new leaf: the repeat must not
		// earn a second row, the newcomer must.
		ship(4002, "def5678", "internal/issuefanout/ledger.go", "internal/dispatchtick/witness.go"),
	}, turnAt, &ledger)

	if res.Schema != TurnSchema {
		t.Fatalf("schema = %q, want %q", res.Schema, TurnSchema)
	}
	if res.Ships != 2 {
		t.Fatalf("ships = %d, want 2", res.Ships)
	}
	if len(res.Rows) != 2 {
		t.Fatalf("want one row per distinct leaf (issuefanout, dispatchtick), got %d: %+v", len(res.Rows), res.Rows)
	}
	if res.Rows[0].Leaf != "issuefanout" || res.Rows[0].SpineRef != "abc1234" || res.Rows[0].Issue != 4001 {
		t.Fatalf("first row should bind the leaf to the commit that shipped it: %+v", res.Rows[0])
	}
	if res.Rows[1].Leaf != "dispatchtick" || res.Rows[1].SpineRef != "def5678" {
		t.Fatalf("second row wrong: %+v", res.Rows[1])
	}
	for _, r := range res.Rows {
		if r.Outcome != OutcomeSuccess {
			t.Fatalf("row %+v: outcome %q, want success", r, r.Outcome)
		}
		if r.Candidates < MinFanout {
			t.Fatalf("row %+v planned %d candidates, below the fan-out floor %d", r, r.Candidates, MinFanout)
		}
	}
	if res.Counts.Success != 2 || res.Counts.Total() != 2 {
		t.Fatalf("counts = %+v, want 2 successes", res.Counts)
	}
	if res.NoLeaf != 0 || len(res.Dropped) != 0 {
		t.Fatalf("nothing should have been dropped: no_leaf=%d dropped=%v", res.NoLeaf, res.Dropped)
	}

	// The durable half: one row per invocation, carrying the lane and the outcome and
	// nothing that could leak a boundary.
	rows := ledgerRows(t, &ledger)
	if len(rows) != 2 {
		t.Fatalf("want 2 durable ledger rows, got %d: %+v", len(rows), rows)
	}
	for i, want := range []string{"issuefanout", "dispatchtick"} {
		if rows[i].Leaf != want || rows[i].Outcome != OutcomeSuccess {
			t.Fatalf("ledger row %d = %+v, want leaf %q success", i, rows[i], want)
		}
		if rows[i].At != turnAt.Format(time.RFC3339) {
			t.Fatalf("ledger row %d stamped %q, want the caller's clock %q", i, rows[i].At, turnAt.Format(time.RFC3339))
		}
		if strings.Contains(rows[i].At, "abc1234") || rows[i].Schema != LedgerSchema {
			t.Fatalf("ledger row %d malformed: %+v", i, rows[i])
		}
	}
}

// A turn that could not fan out says so on the same artifact instead of reading like a
// turn that had nothing to fan out. Both halves are counted: a commit that named no leaf,
// and a leaf refused by the per-turn cap.
func TestTurnCountsWhatItCouldNotFanOut(t *testing.T) {
	var ledger bytes.Buffer
	// A docs/tools-only commit: real, shipped, and outside every internal/<leaf>/.
	res := Turn([]Ship{ship(5001, "aaa1111", "docs/spine-first-defaults.md", "tools/x.py")}, turnAt, &ledger)
	if res.NoLeaf != 1 || len(res.Rows) != 0 {
		t.Fatalf("a commit outside every leaf must be counted, not skipped: %+v", res)
	}
	if ledger.Len() != 0 {
		t.Fatalf("nothing was planned, so nothing should have been logged: %q", ledger.String())
	}

	// More leaves than one turn will expand.
	var paths []string
	for i := 0; i < TurnLeafCap+2; i++ {
		paths = append(paths, fmt.Sprintf("internal/leaf%02d/x.go", i))
	}
	capped := Turn([]Ship{ship(5002, "bbb2222", paths...)}, turnAt, &ledger)
	if len(capped.Rows) != TurnLeafCap {
		t.Fatalf("want %d rows at the cap, got %d", TurnLeafCap, len(capped.Rows))
	}
	if len(capped.Dropped) != 2 {
		t.Fatalf("the cap must name every leaf it refused, got %v", capped.Dropped)
	}
	if capped.Dropped[0] != fmt.Sprintf("leaf%02d", TurnLeafCap) {
		t.Fatalf("dropped leaves should be the tail of the sorted set, got %v", capped.Dropped)
	}
	if len(ledgerRows(t, &ledger)) != TurnLeafCap {
		t.Fatal("a dropped leaf must not appear in the durable ledger — it was never planned")
	}
}

// A planner refusal is a ROW, not a failed turn: the loop keeps dispatching and the
// reason travels on the artifact. A spine with no witness is the refusal the planner
// exists to make, so it is the one exercised here.
func TestTurnRefusalIsARowNotAFailedTurn(t *testing.T) {
	var ledger bytes.Buffer
	res := Turn([]Ship{
		ship(6001, "   ", "internal/issuefanout/turn.go"),
		ship(6002, "ccc3333", "internal/dispatchtick/witness.go"),
	}, turnAt, &ledger)

	if len(res.Rows) != 2 {
		t.Fatalf("want both rows, got %+v", res.Rows)
	}
	if res.Rows[0].Outcome != OutcomeRefused || !strings.Contains(res.Rows[0].Reason, "spine_ref is required") {
		t.Fatalf("a spine-less ship must refuse with its reason on the row: %+v", res.Rows[0])
	}
	if res.Rows[0].Candidates != 0 {
		t.Fatalf("a refused row planned nothing: %+v", res.Rows[0])
	}
	if res.Rows[1].Outcome != OutcomeSuccess {
		t.Fatalf("the refusal must not stop the rest of the turn: %+v", res.Rows[1])
	}
	if res.Counts.Refused != 1 || res.Counts.Success != 1 {
		t.Fatalf("counts = %+v, want one of each", res.Counts)
	}
	rows := ledgerRows(t, &ledger)
	if len(rows) != 2 || rows[0].Outcome != OutcomeRefused {
		t.Fatalf("the refusal is durable too: %+v", rows)
	}
}

// A caller that could not open its ledger still gets the plan: the nil writer is a no-op
// append, so the fan-out default degrades to plan-only rather than to nothing.
func TestTurnWithNoLedgerStillPlans(t *testing.T) {
	res := Turn([]Ship{ship(7001, "ddd4444", "internal/issuefanout/turn.go")}, turnAt, nil)
	if len(res.Rows) != 1 || res.Rows[0].Outcome != OutcomeSuccess || res.Rows[0].Candidates < MinFanout {
		t.Fatalf("a nil ledger must not change the plan: %+v", res)
	}
}
