package flowcredit

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"sync"
	"testing"
	"unicode"
)

// laneA is the lane under test; laneB exists to prove isolation.
var (
	laneA = Lane{Receiver: "decode-1", Sender: "prefill-1", Class: "bulk"}
	laneB = Lane{Receiver: "decode-1", Sender: "prefill-2", Class: "bulk"}
)

// mustInvariant asserts the ledger's core conservation law for a lane:
// reserved never exceeds granted, and available is exactly the difference.
func mustInvariant(t *testing.T, g *Ledger, l Lane) Snapshot {
	t.Helper()
	v := g.View(l)
	if v.Reserved > v.Granted {
		t.Fatalf("invariant violated: reserved %d > granted %d", v.Reserved, v.Granted)
	}
	if v.Available != v.Granted-v.Reserved {
		t.Fatalf("available %d != granted %d - reserved %d", v.Available, v.Granted, v.Reserved)
	}
	return v
}

// TestFlowCreditZeroCreditsZeroTransmission: before any grant the window is
// closed — every reserve refuses and the sender transmits nothing.
func TestFlowCreditZeroCreditsZeroTransmission(t *testing.T) {
	g := NewLedger()
	sent := 0
	for i := 0; i < 8; i++ {
		if g.TryReserve(laneA, 1) {
			sent++
		}
	}
	if sent != 0 {
		t.Fatalf("sender transmitted %d blocks with zero credit; want 0", sent)
	}
	if v := mustInvariant(t, g, laneA); v.Granted != 0 || v.Reserved != 0 {
		t.Fatalf("zero-credit lane mutated: %+v", v)
	}
}

// TestFlowCreditSenderStopsAtGrantThenFreshGrantUnblocks: the sender drains
// exactly the granted window and STOPS (no over-send), a fresh grant
// unblocks exactly the widened amount, and cumulative sent equals cumulative
// granted — credit conservation.
func TestFlowCreditSenderStopsAtGrantThenFreshGrantUnblocks(t *testing.T) {
	g := NewLedger()
	if !g.Grant(laneA, 1, 5) {
		t.Fatal("first grant (seq=1, cum=5) refused; want applied")
	}

	send := func(attempts int) (sent int) {
		for i := 0; i < attempts; i++ {
			if g.TryReserve(laneA, 1) {
				sent++
			}
			mustInvariant(t, g, laneA)
		}
		return sent
	}

	if got := send(12); got != 5 {
		t.Fatalf("sender transmitted %d of 12 attempted blocks under cum=5; want exactly 5", got)
	}
	if v := g.View(laneA); v.Available != 0 || v.Reserved != 5 {
		t.Fatalf("after draining the window: %+v; want available=0 reserved=5", v)
	}
	if g.TryReserve(laneA, 1) {
		t.Fatal("reserve past the granted cumulative credit succeeded; want refused (backpressure)")
	}

	// A fresh, wider cumulative grant unblocks the sender for the delta only.
	if !g.Grant(laneA, 2, 8) {
		t.Fatal("fresh grant (seq=2, cum=8) refused; want applied")
	}
	if got := send(12); got != 3 {
		t.Fatalf("sender transmitted %d more blocks after cum 5->8; want exactly 3", got)
	}
	v := mustInvariant(t, g, laneA)
	if v.Reserved != 8 || v.Reserved != v.Granted {
		t.Fatalf("conservation broken: cumulative sent %d, cumulative granted %d; want both 8", v.Reserved, v.Granted)
	}
}

// TestFlowCreditStaleGrantsIgnored: replayed sequence numbers and
// lower-cumulative values never re-widen or narrow the window (monotonic,
// idempotent by seq).
func TestFlowCreditStaleGrantsIgnored(t *testing.T) {
	g := NewLedger()
	if !g.Grant(laneA, 4, 10) {
		t.Fatal("grant (seq=4, cum=10) refused; want applied")
	}
	table := []struct {
		name     string
		seq, cum uint64
	}{
		{"replayed seq, same cum", 4, 10},
		{"replayed seq, inflated cum", 4, 100},
		{"older seq, inflated cum", 2, 100},
		{"newer seq, lower cum", 5, 7},
		{"newer seq, equal cum", 6, 10},
	}
	for _, tc := range table {
		if g.Grant(laneA, tc.seq, tc.cum) {
			t.Errorf("%s: grant applied; want ignored", tc.name)
		}
		if v := mustInvariant(t, g, laneA); v.Granted != 10 {
			t.Fatalf("%s: granted = %d; want 10 unchanged", tc.name, v.Granted)
		}
	}
	// The watermark advanced past the anomalous newer messages, and a
	// genuinely newer, wider grant still applies.
	if !g.Grant(laneA, 7, 12) {
		t.Fatal("grant (seq=7, cum=12) refused; want applied")
	}
	if v := g.View(laneA); v.Granted != 12 || v.LastSeq != 7 {
		t.Fatalf("after seq=7 cum=12: %+v; want granted=12 lastSeq=7", v)
	}
}

// TestFlowCreditOverReserveRefusedAtomically: an all-or-nothing reserve that
// does not fit leaves the lane byte-identical — no partial take.
func TestFlowCreditOverReserveRefusedAtomically(t *testing.T) {
	g := NewLedger()
	g.Grant(laneA, 1, 4)
	if !g.TryReserve(laneA, 3) {
		t.Fatal("reserve 3 of 4 refused; want granted")
	}
	before := g.View(laneA)
	if g.TryReserve(laneA, 2) {
		t.Fatal("reserve 2 with only 1 available succeeded; want refused")
	}
	if after := g.View(laneA); after != before {
		t.Fatalf("refused reserve mutated the lane: before %+v, after %+v", before, after)
	}
	if !g.TryReserve(laneA, 1) {
		t.Fatal("boundary reserve of the exact remaining credit refused; want granted")
	}
}

// TestFlowCreditReserveRollbackRestoresExactly: rollback returns exactly the
// reserved units, over-rollback clamps and never mints credit.
func TestFlowCreditReserveRollbackRestoresExactly(t *testing.T) {
	g := NewLedger()
	g.Grant(laneA, 1, 6)
	if !g.TryReserve(laneA, 4) {
		t.Fatal("reserve 4 of 6 refused; want granted")
	}
	if restored := g.Rollback(laneA, 4); restored != 4 {
		t.Fatalf("rollback restored %d; want 4", restored)
	}
	if v := mustInvariant(t, g, laneA); v.Available != 6 || v.Reserved != 0 {
		t.Fatalf("after reserve 4 + rollback 4: %+v; want available=6 reserved=0", v)
	}
	// Over-rollback clamps at what is held: no minting past the grant.
	g.TryReserve(laneA, 2)
	if restored := g.Rollback(laneA, 99); restored != 2 {
		t.Fatalf("over-rollback restored %d; want clamped to 2", restored)
	}
	if v := mustInvariant(t, g, laneA); v.Available != 6 {
		t.Fatalf("over-rollback minted credit: %+v; want available=6", v)
	}
}

// TestFlowCreditLaneIsolation: credit granted to one (receiver, sender,
// class) lane is invisible to every other lane.
func TestFlowCreditLaneIsolation(t *testing.T) {
	g := NewLedger()
	g.Grant(laneA, 1, 3)
	if g.TryReserve(laneB, 1) {
		t.Fatal("lane B reserved against lane A's grant; want refused")
	}
	if !g.TryReserve(laneA, 3) {
		t.Fatal("lane A reserve refused; want granted")
	}
	if v := g.View(laneB); v.Granted != 0 || v.Reserved != 0 {
		t.Fatalf("lane B mutated by lane A traffic: %+v", v)
	}
}

// TestFlowCreditOutOfOrderOpsHoldInvariant drives a deterministic
// interleaving of out-of-order grants, reserves, and rollbacks and checks
// reserved <= granted after every single step.
func TestFlowCreditOutOfOrderOpsHoldInvariant(t *testing.T) {
	g := NewLedger()
	steps := []struct {
		name string
		op   func() bool
		want bool
	}{
		{"reserve before any grant refused", func() bool { return g.TryReserve(laneA, 2) }, false},
		{"newest grant arrives first (seq=5, cum=6)", func() bool { return g.Grant(laneA, 5, 6) }, true},
		{"reserve 4 of 6", func() bool { return g.TryReserve(laneA, 4) }, true},
		{"delayed older grant (seq=2, cum=9) deduped", func() bool { return g.Grant(laneA, 2, 9) }, false},
		{"reserve 3 with 2 available refused", func() bool { return g.TryReserve(laneA, 3) }, false},
		{"rollback 1 restores 1", func() bool { return g.Rollback(laneA, 1) == 1 }, true},
		{"reserve 3 now fits exactly", func() bool { return g.TryReserve(laneA, 3) }, true},
		{"window drained: reserve 1 refused", func() bool { return g.TryReserve(laneA, 1) }, false},
		{"fresh wider grant (seq=6, cum=10)", func() bool { return g.Grant(laneA, 6, 10) }, true},
		{"zero-unit reserve is a no-op success", func() bool { return g.TryReserve(laneA, 0) }, true},
		{"reserve the new delta of 4", func() bool { return g.TryReserve(laneA, 4) }, true},
		{"replay of seq=6 with huge cum deduped", func() bool { return g.Grant(laneA, 6, 1000) }, false},
		{"reserve past the deduped replay refused", func() bool { return g.TryReserve(laneA, 1) }, false},
	}
	for i, s := range steps {
		if got := s.op(); got != s.want {
			t.Fatalf("step %d (%s) = %v; want %v", i, s.name, got, s.want)
		}
		mustInvariant(t, g, laneA)
	}
	v := g.View(laneA)
	if v.Granted != 10 || v.Reserved != 10 || v.Available != 0 {
		t.Fatalf("final window %+v; want granted=10 reserved=10 available=0", v)
	}
}

func TestFlowCreditZeroAmountOperations(t *testing.T) {
	g := NewLedger()
	if !g.TryReserve(laneA, 0) {
		t.Fatal("TryReserve 0 on empty lane failed; want true")
	}
	if v := mustInvariant(t, g, laneA); v.Granted != 0 || v.Reserved != 0 {
		t.Fatalf("lane mutated after zero reserve: %+v", v)
	}
	if restored := g.Rollback(laneA, 0); restored != 0 {
		t.Fatalf("Rollback 0 returned %d; want 0", restored)
	}

	g.Grant(laneA, 1, 10)
	g.TryReserve(laneA, 4)
	before := g.View(laneA)
	if !g.TryReserve(laneA, 0) {
		t.Fatal("TryReserve 0 on active lane failed; want true")
	}
	if after := g.View(laneA); after != before {
		t.Fatalf("lane mutated after zero reserve: before %+v, after %+v", before, after)
	}
	if restored := g.Rollback(laneA, 0); restored != 0 {
		t.Fatalf("Rollback 0 returned %d; want 0", restored)
	}
	if after := g.View(laneA); after != before {
		t.Fatalf("lane mutated after zero rollback: before %+v, after %+v", before, after)
	}
	mustInvariant(t, g, laneA)
}

func TestFlowCreditConcurrentSafety(t *testing.T) {
	g := NewLedger()
	const workers = 8
	const iterations = 500
	var wg sync.WaitGroup

	g.Grant(laneA, 1, 1000)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				switch (id + i) % 4 {
				case 0:
					g.TryReserve(laneA, 5)
				case 1:
					g.Rollback(laneA, 3)
				case 2:
					seq := uint64(2 + i)
					cum := uint64(1000 + i*10)
					g.Grant(laneA, seq, cum)
				case 3:
					snap := g.View(laneA)
					if snap.Reserved > snap.Granted {
						t.Errorf("concurrent violation: reserved %d > granted %d", snap.Reserved, snap.Granted)
					}
					if snap.Available != snap.Granted-snap.Reserved {
						t.Errorf("concurrent violation: available %d != %d - %d", snap.Available, snap.Granted, snap.Reserved)
					}
				}
			}
		}(w)
	}

	wg.Wait()
	mustInvariant(t, g, laneA)
}

func TestFlowCreditCommentHygieneAndNoFormulaicNoise(t *testing.T) {
	fset := token.NewFileSet()
	filename := "flowcredit.go"
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed to read %s: %v", filename, err)
	}

	node, err := parser.ParseFile(fset, filename, content, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", filename, err)
	}

	codeLines := countNonEmptyLines(content)
	commentLines := 0
	formulaicCount := 0
	hasFiller := false

	for _, cg := range node.Comments {
		for _, c := range cg.List {
			commentLines += strings.Count(c.Text, "\n") + 1
		}
		isForm, isFill := checkFormulaicComment(cg)
		if isForm {
			formulaicCount++
			t.Logf("%s: detected formulaic comment: %q", filename, strings.TrimSpace(cg.Text()))
		}
		if isFill {
			hasFiller = true
		}
	}

	commentRatio := float64(commentLines) / float64(codeLines)
	if codeLines > 30 && commentRatio > 0.35 {
		t.Errorf("%s: comment bloat ratio %.3f exceeds 0.35 (comments: %d, code: %d)",
			filename, commentRatio, commentLines, codeLines)
	}

	if formulaicCount > 0 || hasFiller {
		t.Errorf("%s: formulaic comments detected: count=%d, filler=%v",
			filename, formulaicCount, hasFiller)
	}

	exportedCount := 0
	documentedCount := 0
	for _, decl := range node.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if ast.IsExported(d.Name.Name) {
				exportedCount++
				if isSubstantiveDoc(d.Name.Name, d.Doc) {
					documentedCount++
				} else {
					t.Errorf("exported func %s missing substantive doc", d.Name.Name)
				}
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				if s, ok := spec.(*ast.TypeSpec); ok && ast.IsExported(s.Name.Name) {
					exportedCount++
					doc := s.Doc
					if doc == nil {
						doc = d.Doc
					}
					if isSubstantiveDoc(s.Name.Name, doc) {
						documentedCount++
					} else {
						t.Errorf("exported type %s missing substantive doc", s.Name.Name)
					}
				}
			}
		}
	}

	if exportedCount > 0 {
		ratio := float64(documentedCount) / float64(exportedCount)
		if ratio < 0.75 {
			t.Errorf("documented exports ratio %.2f < 0.75 (%d/%d)", ratio, documentedCount, exportedCount)
		}
	}
}

func countNonEmptyLines(b []byte) int {
	scanner := bufio.NewScanner(strings.NewReader(string(b)))
	lines := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) != "" {
			lines++
		}
	}
	return lines
}

func checkFormulaicComment(cg *ast.CommentGroup) (bool, bool) {
	if cg == nil {
		return false, false
	}
	text := strings.TrimSpace(cg.Text())
	lower := strings.ToLower(text)

	hasMarker := strings.Contains(lower, "invariant:") ||
		strings.Contains(lower, "invariants:") ||
		strings.Contains(lower, "key invariant:") ||
		strings.Contains(lower, "contract:") ||
		strings.Contains(lower, "fail-closed:") ||
		strings.Contains(lower, "fail-closed guard:") ||
		strings.HasPrefix(lower, "invariant") ||
		strings.HasPrefix(lower, "guard") ||
		strings.HasPrefix(lower, "contract") ||
		strings.HasPrefix(lower, "fail-closed")

	if !hasMarker {
		return false, false
	}

	words := strings.Fields(lower)
	if len(words) <= 3 {
		return true, true
	}

	keywordCount := 0
	for _, w := range words {
		clean := strings.Trim(w, ":,.-*#")
		if clean == "invariant" || clean == "invariants" || clean == "assumption" ||
			clean == "assumptions" || clean == "guard" || clean == "fail-closed" ||
			clean == "contract" || clean == "precondition" || clean == "postcondition" {
			keywordCount++
		}
	}
	if float64(keywordCount)/float64(len(words)) > 0.25 || keywordCount >= 3 {
		return true, true
	}

	return true, false
}

func isSubstantiveDoc(name string, doc *ast.CommentGroup) bool {
	if doc == nil || len(doc.List) == 0 {
		return false
	}
	text := strings.TrimSpace(doc.Text())
	if len(text) < 12 {
		return false
	}
	return !isTautologicalDoc(name, text)
}

func splitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func isTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := splitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}
