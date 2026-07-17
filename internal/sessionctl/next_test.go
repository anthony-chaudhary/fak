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
