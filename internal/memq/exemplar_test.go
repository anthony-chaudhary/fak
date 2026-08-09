package memq

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajquery"
)

type exemplarBackend struct {
	bodies map[string][]byte
	errs   map[string]error
}

func (b exemplarBackend) Cells(context.Context) ([]Cell, error) { return nil, nil }
func (b exemplarBackend) Materialize(_ context.Context, id string) ([]byte, error) {
	if err := b.errs[id]; err != nil {
		return nil, err
	}
	v, ok := b.bodies[id]
	if !ok {
		return nil, errors.New("missing")
	}
	return v, nil
}

func exemplarView(rows []trajquery.Row) trajquery.View {
	return trajquery.View{Name: "trajectory_exemplars", Base: "trajectory", Columns: []string{"trace_id", "cell_id", "task"}}
}

func TestRetrieveExemplarsRanksTrustGatesAndStampsProvenance(t *testing.T) {
	rows := []trajquery.Row{
		{"trace_id": "trace-build", "cell_id": "step:1", "task": "fix failing Go build compilation"},
		{"trace_id": "trace-css", "cell_id": "step:2", "task": "adjust CSS colors"},
		{"trace_id": "trace-stale", "cell_id": "step:3", "task": "fix Go build error"},
		{"trace_id": "trace-sealed", "cell_id": "step:4", "task": "fix Go compiler"},
	}
	backend := exemplarBackend{
		bodies: map[string][]byte{"step:1": []byte("run fleet-safe buildcheck"), "step:2": []byte("edit stylesheet")},
		errs:   map[string]error{"step:3": ErrStale, "step:4": ErrSealed},
	}
	got, err := RetrieveExemplars(context.Background(), ExemplarRequest{Task: "repair Go build failure", TopK: 4, ByteBudget: 400, Notes: []byte("notes"), View: exemplarView(rows), Corpus: rows, Backend: backend})
	if err != nil {
		t.Fatal(err)
	}
	text := string(got.Bytes)
	if !strings.Contains(text, "trace=\"trace-build\"") || !strings.Contains(text, "fleet-safe buildcheck") {
		t.Fatalf("missing concrete provenance-stamped exemplar: %q", text)
	}
	if strings.Contains(text, "trace-stale") || strings.Contains(text, "trace-sealed") {
		t.Fatalf("untrusted exemplar leaked: %q", text)
	}
	verdicts := map[string]ExemplarVerdict{}
	for _, d := range got.Decisions {
		verdicts[d.TraceID] = d.Verdict
	}
	if verdicts["trace-stale"] != ExemplarStale || verdicts["trace-sealed"] != ExemplarSealed {
		t.Fatalf("trust verdicts=%v", verdicts)
	}
}

func TestRetrieveExemplarsSharesBudgetWithNotes(t *testing.T) {
	rows := []trajquery.Row{{"trace_id": "trace-1", "cell_id": "step:1", "task": "same task"}}
	got, err := RetrieveExemplars(context.Background(), ExemplarRequest{Task: "same task", TopK: 1, ByteBudget: 17, Notes: []byte("notes consume 15"), View: exemplarView(rows), Corpus: rows, Backend: exemplarBackend{bodies: map[string][]byte{"step:1": []byte("secret body")}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Bytes) != 0 || len(got.Decisions) != 1 || got.Decisions[0].Verdict != ExemplarBudget {
		t.Fatalf("budget leak: %+v %q", got, got.Bytes)
	}
}

func TestRetrieveExemplarsFailsClosedOnUnconfinedView(t *testing.T) {
	rows := []trajquery.Row{{"trace_id": "trace-1", "cell_id": "step:1", "task": "task"}}
	view := trajquery.View{Name: "trajectory_exemplars", Base: "trajectory", Columns: []string{"trace_id", "task"}}
	_, err := RetrieveExemplars(context.Background(), ExemplarRequest{Task: "task", TopK: 1, ByteBudget: 100, View: view, Corpus: rows, Backend: exemplarBackend{}})
	if err == nil || !strings.Contains(err.Error(), "refused") {
		t.Fatalf("want confined-view refusal, got %v", err)
	}
}
