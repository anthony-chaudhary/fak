package cachevaluereport

import (
	"path/filepath"
	"testing"
	"time"
)

func TestRecallInjectionLedgerNumbersOnlyAndFold(t *testing.T) {
	p := filepath.Join(t.TempDir(), "recall.jsonl")
	if err := AppendRecallInjection(p, 2, 400, time.Unix(1, 0)); err != nil {
		t.Fatal(err)
	}
	got, err := ReadRecallInjectionDebit(p)
	if err != nil {
		t.Fatal(err)
	}
	if got.Injections != 1 || got.Records != 2 || got.EstimatedTokens != 400 || got.EstimatedUSD <= 0 {
		t.Fatalf("unexpected debit: %+v", got)
	}
}
