package sessionctl

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNextQueueWitnessesEveryMoveFromDurableJSONL(t *testing.T) {
	q, err := NewNextQueue(SessionInteractive)
	if err != nil {
		t.Fatal(err)
	}
	kinds := []MoveKind{MoveContinue, MoveRedirect, MoveAnnotate, MoveReanchor, MoveHalt}
	for _, kind := range kinds {
		render, ok := DefaultRender(kind, SessionInteractive)
		if !ok {
			t.Fatalf("no render mapping for %q", kind)
		}
		if err := q.Enqueue(Move{Kind: kind, Render: render, Gate: "test-gate", Source: string(kind), Payload: "payload"}); err != nil {
			t.Fatalf("enqueue %q: %v", kind, err)
		}
	}

	path := filepath.Join(t.TempDir(), "next.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	err = q.Drain(f, func(move Move) (ApplyResult, error) {
		calls++
		if move.Kind == MoveAnnotate {
			return ApplyResult{Refusal: "annotation target vanished"}, nil
		}
		return ApplyResult{Applied: true}, nil
	})
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	if calls != len(kinds) {
		t.Fatalf("applier calls=%d want %d", calls, len(kinds))
	}

	in, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	records, err := ReadNextRecords(in)
	_ = in.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != len(kinds) {
		t.Fatalf("records=%d want %d", len(records), len(kinds))
	}
	for i, record := range records {
		if record.Move.Kind != kinds[i] || !record.Move.Kind.Valid() || record.Move.Gate == "" {
			t.Fatalf("record[%d]=%+v", i, record)
		}
		if record.Sequence != uint64(i+1) {
			t.Fatalf("sequence[%d]=%d", i, record.Sequence)
		}
	}
	if records[2].Applied || records[2].Refusal == "" {
		t.Fatalf("refusal was not witnessed: %+v", records[2])
	}
}

func TestNextVocabularyCompleteness(t *testing.T) {
	kinds := []MoveKind{MoveContinue, MoveRedirect, MoveAnnotate, MoveReanchor, MoveHalt}
	classes := []SessionClass{SessionInteractive, SessionAutonomous}
	for _, kind := range kinds {
		if !kind.Valid() {
			t.Fatalf("listed move %q is invalid", kind)
		}
		for _, class := range classes {
			render, ok := DefaultRender(kind, class)
			if !ok || !render.Valid() || !class.AllowsRender(render) {
				t.Fatalf("move=%q class=%q render=%q ok=%v", kind, class, render, ok)
			}
		}
	}
}

func TestNextQueueRejectsInvalidGateVocabularyAndMixedRenderClass(t *testing.T) {
	q, _ := NewNextQueue(SessionInteractive)
	for name, move := range map[string]Move{
		"empty gate":     {Kind: MoveContinue, Render: RenderUserSplice},
		"unknown kind":   {Kind: MoveKind("ask"), Render: RenderUserSplice, Gate: "g"},
		"unknown render": {Kind: MoveContinue, Render: RenderKind("chat"), Gate: "g"},
		"wrong class":    {Kind: MoveContinue, Render: RenderSystemDirective, Gate: "g"},
		"claimed class":  {Kind: MoveContinue, Render: RenderUserSplice, Session: SessionAutonomous, Gate: "g"},
	} {
		t.Run(name, func(t *testing.T) {
			if err := q.Enqueue(move); err == nil {
				t.Fatal("enqueue unexpectedly accepted invalid move")
			}
		})
	}
	if _, err := NewNextQueue(SessionClass("hybrid")); err == nil {
		t.Fatal("hybrid session class accepted")
	}
}

func TestNextQueueDrainsInSourceSiteOrderAcrossRenderClasses(t *testing.T) {
	q, _ := NewNextQueue(SessionInteractive)
	moves := []Move{
		{Kind: MoveContinue, Render: RenderUserSplice, Gate: "a", Source: "user-1"},
		{Kind: MoveReanchor, Render: RenderReopen, Gate: "b", Source: "system"},
		{Kind: MoveAnnotate, Render: RenderUserSplice, Gate: "c", Source: "user-2"},
	}
	for _, move := range moves {
		if err := q.Enqueue(move); err != nil {
			t.Fatal(err)
		}
	}
	var got []string
	var witness strings.Builder
	if err := q.Drain(&witness, func(move Move) (ApplyResult, error) {
		got = append(got, move.Source)
		return ApplyResult{Applied: true}, nil
	}); err != nil {
		t.Fatal(err)
	}
	want := []string{"user-1", "system", "user-2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
	if err := q.Drain(&witness, func(Move) (ApplyResult, error) { return ApplyResult{}, nil }); err == nil {
		t.Fatal("second boundary drain accepted")
	}
}

func TestWitnessMoveUsesSharedDrainAndReadback(t *testing.T) {
	move := Move{Kind: MoveContinue, Render: RenderReopen, Session: SessionInteractive, Gate: "guard", Source: "guard-stophook", Payload: "keep working"}
	record, err := WitnessMove(move, ApplyResult{Applied: false, Refusal: "fail-open"})
	if err != nil {
		t.Fatal(err)
	}
	if record.Move != move || record.Applied || record.Refusal != "fail-open" || record.Sequence != 1 {
		t.Fatalf("record = %+v, want durable refused continuation", record)
	}
}

func TestNextDrainHasSingleProductionAuthority(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	var definitions []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := filepath.Base(path)
			if base == ".git" || base == "_scratch" {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.Contains(string(b), "func (q *NextQueue) Drain(") {
			definitions = append(definitions, filepath.ToSlash(path))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(definitions) != 1 || !strings.HasSuffix(definitions[0], "internal/sessionctl/next.go") {
		t.Fatalf("NextQueue.Drain production authorities = %v, want only internal/sessionctl/next.go", definitions)
	}
}

func TestRecordSteerNextReadbackAndNoop(t *testing.T) {
	const trace = "next-steer-witness"
	ReadSteerNextRecords(trace)
	RecordSteerNext(trace, "  switch to plan B  ", ApplyResult{Applied: true})
	records := ReadSteerNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d, want 1", len(records))
	}
	r := records[0]
	if !r.Applied || r.Move.Kind != MoveAnnotate || r.Move.Render != RenderUserSplice || r.Move.Session != SessionInteractive {
		t.Fatalf("record=%+v", r)
	}
	if r.Move.Payload != "  switch to plan B  " {
		t.Fatalf("payload=%q", r.Move.Payload)
	}
	RecordSteerNext(trace, "", ApplyResult{Applied: true})
	if got := ReadSteerNextRecords(trace); len(got) != 0 {
		t.Fatalf("empty payload emitted records: %+v", got)
	}
	RecordSteerNext(trace, "", ApplyResult{Refusal: "quarantined"})
	refused := ReadSteerNextRecords(trace)
	if len(refused) != 1 || refused[0].Applied || refused[0].Refusal != "quarantined" {
		t.Fatalf("refused=%+v", refused)
	}
}

func TestRecordContextAdvisoryNextReadbackAndNoop(t *testing.T) {
	const trace = "next-context-advisory"
	ReadContextAdvisoryNextRecords(trace)
	RecordContextAdvisoryNext(trace, "  summarize, then continue  ")
	records := ReadContextAdvisoryNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d, want 1", len(records))
	}
	r := records[0]
	if !r.Applied || r.Move.Kind != MoveAnnotate || r.Move.Render != RenderUserSplice || r.Move.Session != SessionInteractive {
		t.Fatalf("record=%+v", r)
	}
	if r.Move.Payload != "  summarize, then continue  " || r.Move.Gate != "context-spike" || r.Move.Source != "agent-turn-boundary" {
		t.Fatalf("move=%+v", r.Move)
	}
	RecordContextAdvisoryNext(trace, "")
	if got := ReadContextAdvisoryNextRecords(trace); len(got) != 0 {
		t.Fatalf("empty advisory emitted records: %+v", got)
	}
}

func TestRecordStopWitnessNextReadbackAndNoop(t *testing.T) {
	const trace = "trace-stop-witness"
	const payload = "STOP_UNWITNESSED: missing declared witness: file:proof.sha256. Continue working until that witness exists."
	RecordStopWitnessNext(trace, payload)
	records := ReadStopWitnessNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d want 1", len(records))
	}
	r := records[0]
	if r.Move.Kind != MoveContinue || r.Move.Render != RenderUserSplice || r.Move.Session != SessionInteractive {
		t.Fatalf("move=%+v", r.Move)
	}
	if r.Move.Gate != "stop-witness" || r.Move.Source != "agent-turn-boundary" || r.Move.Payload != payload || !r.Applied {
		t.Fatalf("record=%+v", r)
	}
	if again := ReadStopWitnessNextRecords(trace); len(again) != 0 {
		t.Fatalf("read did not clear: %+v", again)
	}
	RecordStopWitnessNext("", payload)
	RecordStopWitnessNext(trace, "")
	if got := ReadStopWitnessNextRecords(trace); len(got) != 0 {
		t.Fatalf("noop records=%+v", got)
	}
}

func TestRecordBudgetResetNextReadbackAndNoop(t *testing.T) {
	const trace = "budget-reset-child"
	ReadBudgetResetNextRecords(trace)

	payload := "SESSION_RESET_RECAP: preserve exact bytes\nobjective continues"
	RecordBudgetResetNext(trace, payload)
	records := ReadBudgetResetNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d want 1", len(records))
	}
	got := records[0]
	if got.Move.Kind != MoveReanchor || got.Move.Render != RenderReopen || got.Move.Session != SessionInteractive {
		t.Fatalf("move=%+v", got.Move)
	}
	if got.Move.Gate != "served-session-reset" || got.Move.Source != "gateway-reset-hook" {
		t.Fatalf("provenance=%+v", got.Move)
	}
	if got.Move.Payload != payload || !got.Applied || got.Refusal != "" {
		t.Fatalf("record=%+v", got)
	}

	RecordBudgetResetNext("", payload)
	RecordBudgetResetNext(trace, "")
	if got := ReadBudgetResetNextRecords(trace); len(got) != 0 {
		t.Fatalf("no-op records=%+v", got)
	}
}

func TestRecordGuardRecoveryNextReadbackAndNoop(t *testing.T) {
	const trace = "guard-recovery-trace"
	ReadGuardRecoveryNextRecords(trace)
	payload := "[fak] resume recovery: do not retry unchanged"
	RecordGuardRecoveryNext(trace, payload)
	records := ReadGuardRecoveryNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("records=%d want 1", len(records))
	}
	got := records[0]
	if got.Move.Kind != MoveRedirect || got.Move.Render != RenderUserSplice || got.Move.Session != SessionInteractive {
		t.Fatalf("move=%+v", got.Move)
	}
	if got.Move.Gate != "guard-recovery" || got.Move.Source != "gateway-anthropic-boundary" {
		t.Fatalf("provenance=%+v", got.Move)
	}
	if got.Move.Payload != payload || !got.Applied || got.Refusal != "" {
		t.Fatalf("record=%+v", got)
	}
	RecordGuardRecoveryNext("", payload)
	RecordGuardRecoveryNext(trace, "")
	if got := ReadGuardRecoveryNextRecords(trace); len(got) != 0 {
		t.Fatalf("no-op records=%+v", got)
	}
}
