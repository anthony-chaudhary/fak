package scorecardpane

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestRewriteGoRunFakUsesCurrentExecutable(t *testing.T) {
	got := rewriteGoRunFak(
		[]string{"go", "run", "./cmd/fak", "guard-rsi-scorecard", "--json"},
		"/tmp/fak",
	)
	want := []string{"/tmp/fak", "guard-rsi-scorecard", "--json"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rewriteGoRunFak = %v, want %v", got, want)
	}
}

func TestRewriteGoRunFakLeavesOtherCommandsAlone(t *testing.T) {
	in := []string{"go", "test", "./cmd/fak"}
	got := rewriteGoRunFak(in, "/tmp/fak")
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("non go-run command changed: got %v want %v", got, in)
	}

	in = []string{"python", "tools/x.py", "--json"}
	got = rewriteGoRunFak(in, "/tmp/fak")
	if !reflect.DeepEqual(got, in) {
		t.Fatalf("python command changed: got %v want %v", got, in)
	}
}

func TestIsCmdFakTargetToleratesSlashForms(t *testing.T) {
	for _, target := range []string{"./cmd/fak", "cmd/fak", `.\\cmd\\fak`} {
		if !isCmdFakTarget(target) {
			t.Fatalf("%q should be recognized as cmd/fak", target)
		}
	}
	if isCmdFakTarget("./cmd/other") {
		t.Fatal("./cmd/other must not be recognized as cmd/fak")
	}
}

func TestCollectBudgetedMarksUnrunCardsWhenBudgetAlreadyExhausted(t *testing.T) {
	got := CollectBudgeted(t.TempDir(), "python", time.Second, -time.Nanosecond)
	if len(got) != len(Cards) {
		t.Fatalf("metrics len = %d, want %d", len(got), len(Cards))
	}
	if got[0].Debt != nil || got[0].Verdict != "ERROR" {
		t.Fatalf("first metric should be an error row with nil debt: %+v", got[0])
	}
	if !strings.Contains(got[0].Error, "budget exhausted") {
		t.Fatalf("budget error not recorded: %+v", got[0])
	}
}

func TestCollectBudgetedParallelPreservesCardOrderWhenBudgetExhausted(t *testing.T) {
	got := CollectBudgetedParallel(t.TempDir(), "python", time.Second, -time.Nanosecond, 4)
	if len(got) != len(Cards) {
		t.Fatalf("metrics len = %d, want %d", len(got), len(Cards))
	}
	for i := range Cards {
		if got[i].Key != Cards[i].Key {
			t.Fatalf("metric[%d] key = %q, want %q", i, got[i].Key, Cards[i].Key)
		}
		if got[i].Verdict != "ERROR" || !strings.Contains(got[i].Error, "budget exhausted") {
			t.Fatalf("metric[%d] should be a budget error row, got %+v", i, got[i])
		}
	}
}
