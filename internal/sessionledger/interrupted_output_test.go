package sessionledger

import (
	"reflect"
	"strings"
	"testing"
)

func TestInterruptedTurnSurvivesReopenWithExactLongChunks(t *testing.T) {
	dir := t.TempDir()
	ledger, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	chunks := []string{
		"first ",
		strings.Repeat("long-visible-prefix-", MaxContentBytes),
		" last",
	}
	reason := TerminalReason{Cause: "canceled", Evidence: "request context ended"}
	receipt, err := ledger.AppendInterruptedTurn("trace-long", InterruptedTurn{
		Turn: 3, Chunks: chunks, Reason: reason,
	})
	if err != nil {
		t.Fatalf("AppendInterruptedTurn: %v", err)
	}
	if receipt.Assistant == nil {
		t.Fatal("assistant receipt = nil, want durable non-empty prefix")
	}
	if receipt.Assistant.Chunks.SHA256 == "" {
		t.Fatal("assistant chunks object hash is empty")
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	record, reopenedReceipt, err := reopened.ReconstructInterruptedAssistant("trace-long", 3)
	if err != nil {
		t.Fatalf("ReconstructInterruptedAssistant: %v", err)
	}
	if record.Role != "assistant" || record.Turn != 3 || !reflect.DeepEqual(record.Chunks, chunks) || record.Reason != reason {
		t.Fatalf("reconstructed record = %+v, want exact role, turn, chunks, and reason", record)
	}
	if reopenedReceipt.ChunkCount != len(chunks) || reopenedReceipt.ContentBytes <= MaxContentBytes {
		t.Fatalf("reopened receipt = %+v, want %d chunks and >%d content bytes", reopenedReceipt, len(chunks), MaxContentBytes)
	}
	chain, err := reopened.Chain("trace-long")
	if err != nil {
		t.Fatalf("Chain: %v", err)
	}
	if got := entryKinds(chain); !reflect.DeepEqual(got, []string{KindInterruptedAssistant, KindTurnTerminal}) {
		t.Fatalf("ledger kinds = %v, want interrupted assistant followed by terminal", got)
	}
}

func TestInterruptedTurnWithEmptyPrefixAppendsOnlyTerminal(t *testing.T) {
	ledger, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	reason := TerminalReason{Cause: "canceled", Evidence: "request context ended"}
	receipt, err := ledger.AppendInterruptedTurn("trace-empty", InterruptedTurn{
		Turn: 1, Chunks: []string{""}, Reason: reason,
	})
	if err != nil {
		t.Fatalf("AppendInterruptedTurn: %v", err)
	}
	if receipt.Assistant != nil {
		t.Fatalf("assistant receipt = %+v, want no assistant record for empty prefix", receipt.Assistant)
	}
	if receipt.Terminal.Turn != 1 || receipt.Terminal.Status != TurnStatusInterrupted || receipt.Terminal.Reason != reason {
		t.Fatalf("terminal = %+v, want explicit interrupted terminal", receipt.Terminal)
	}
	chain, err := ledger.Chain("trace-empty")
	if err != nil {
		t.Fatalf("Chain: %v", err)
	}
	if got := entryKinds(chain); !reflect.DeepEqual(got, []string{KindTurnTerminal}) {
		t.Fatalf("ledger kinds = %v, want only terminal", got)
	}
}

func entryKinds(entries []Entry) []string {
	kinds := make([]string, 0, len(entries))
	for _, entry := range entries {
		kinds = append(kinds, entry.Kind)
	}
	return kinds
}
