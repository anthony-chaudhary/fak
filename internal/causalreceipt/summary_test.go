package causalreceipt

import (
	"errors"
	"testing"
	"time"
)

func TestSummarizeValidMultiPhase(t *testing.T) {
	now := time.Now()
	r := Receipt{
		Schema: Schema,
		IDs: IDs{
			Work:         "work-100",
			Turn:         "turn-100",
			Graph:        "graph-100",
			Request:      "req-100",
			ModelSession: "sess-100",
		},
		Phases: []Phase{
			{
				ID:             "phase-1",
				Kind:           "agent",
				Engine:         "fak-native",
				Backend:        "cpu",
				Outcome:        "completed",
				Started:        now,
				Ended:          now.Add(5 * time.Millisecond),
				Tokens:         50,
				Bytes:          512,
				QueueNS:        100,
				LoadNS:         200,
				TransferNS:     300,
				RecomputeNS:    400,
				RecoveryNS:     500,
				VerificationNS: 600,
			},
			{
				ID:              "phase-2",
				ParentID:        "phase-1",
				Kind:            "model",
				Engine:          "fak-native",
				Backend:         "metal",
				Outcome:         "denied",
				Reason:          "policy_block",
				Started:         now.Add(5 * time.Millisecond),
				Ended:           now.Add(10 * time.Millisecond),
				Tokens:          75,
				Bytes:           1024,
				CacheReuseBytes: 256,
				QueueNS:         10,
				LoadNS:          20,
				TransferNS:      30,
				RecomputeNS:     40,
				RecoveryNS:      50,
				VerificationNS:  60,
			},
		},
		Resources: []Resource{
			{
				ID:              "res-1",
				Kind:            "kv_cache",
				State:           "released",
				PlannedLocality: "host",
				ActualLocality:  "host",
				Bytes:           256,
				Released:        true,
			},
		},
		Decisions: []Decision{
			{
				Kind:    "routing",
				ID:      "dec-1",
				Reason:  "default",
				Planned: "local",
				Actual:  "local",
			},
		},
	}

	m, err := Summarize(r)
	if err != nil {
		t.Fatalf("Summarize failed: %v", err)
	}

	if m.PhaseCount != 2 {
		t.Errorf("expected PhaseCount=2, got %d", m.PhaseCount)
	}
	if m.Tokens != 125 {
		t.Errorf("expected Tokens=125, got %d", m.Tokens)
	}
	if m.Bytes != 1536 {
		t.Errorf("expected Bytes=1536, got %d", m.Bytes)
	}
	if m.CacheReuseBytes != 256 {
		t.Errorf("expected CacheReuseBytes=256, got %d", m.CacheReuseBytes)
	}

	// phase-1 overhead: 100+200+300+400+500+600 = 2100
	// phase-2 overhead: 10+20+30+40+50+60 = 210
	expectedOverhead := int64(2310)
	if m.OverheadNS != expectedOverhead {
		t.Errorf("expected OverheadNS=%d, got %d", expectedOverhead, m.OverheadNS)
	}

	if m.Outcomes["completed"] != 1 || m.Outcomes["denied"] != 1 {
		t.Errorf("unexpected outcomes distribution: %+v", m.Outcomes)
	}
}

func TestSummarizeEmptyPhases(t *testing.T) {
	r := Receipt{
		Schema: Schema,
		IDs: IDs{
			Work:    "work-empty",
			Turn:    "turn-empty",
			Graph:   "graph-empty",
			Request: "req-empty",
		},
	}

	m, err := Summarize(r)
	if err != nil {
		t.Fatalf("Summarize failed on empty phases: %v", err)
	}
	if m.PhaseCount != 0 || m.Tokens != 0 || m.Bytes != 0 || m.OverheadNS != 0 {
		t.Errorf("expected zero metrics for empty phases, got %+v", m)
	}
	if len(m.Outcomes) != 0 {
		t.Errorf("expected empty outcomes map, got %+v", m.Outcomes)
	}
}

func TestSummarizeEdgeCasesFailClosed(t *testing.T) {
	valid := func() Receipt {
		return Receipt{
			Schema: Schema,
			IDs: IDs{
				Work:    "w",
				Turn:    "t",
				Graph:   "g",
				Request: "r",
			},
			Phases: []Phase{
				{ID: "p1", Kind: "k", Engine: "fak-native", Outcome: "completed"},
			},
			Resources: []Resource{
				{ID: "res1", Released: true},
			},
			Decisions: []Decision{
				{ID: "d1", Kind: "k", Actual: "a"},
			},
		}
	}

	tests := []struct {
		name   string
		modify func(*Receipt)
	}{
		{
			name: "wrong schema",
			modify: func(r *Receipt) {
				r.Schema = "fak.causal-receipt/2"
			},
		},
		{
			name: "missing work ID",
			modify: func(r *Receipt) {
				r.IDs.Work = ""
			},
		},
		{
			name: "missing turn ID",
			modify: func(r *Receipt) {
				r.IDs.Turn = ""
			},
		},
		{
			name: "missing graph ID",
			modify: func(r *Receipt) {
				r.IDs.Graph = ""
			},
		},
		{
			name: "missing request ID",
			modify: func(r *Receipt) {
				r.IDs.Request = ""
			},
		},
		{
			name: "content-bearing prompt attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"raw_prompt": "leak"}
			},
		},
		{
			name: "content-bearing output attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"model_output": "leak"}
			},
		},
		{
			name: "content-bearing argument attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"tool_argument": "leak"}
			},
		},
		{
			name: "content-bearing result attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"tool_result": "leak"}
			},
		},
		{
			name: "content-bearing screenshot attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"screenshot_png": "leak"}
			},
		},
		{
			name: "content-bearing file_path attribute",
			modify: func(r *Receipt) {
				r.Attributes = map[string]string{"file_path": "/tmp/secret"}
			},
		},
		{
			name: "empty phase ID",
			modify: func(r *Receipt) {
				r.Phases[0].ID = ""
			},
		},
		{
			name: "duplicate phase ID",
			modify: func(r *Receipt) {
				r.Phases = append(r.Phases, Phase{ID: "p1", Engine: "fak-native", Outcome: "completed"})
			},
		},
		{
			name: "empty phase engine",
			modify: func(r *Receipt) {
				r.Phases[0].Engine = ""
			},
		},
		{
			name: "empty phase outcome",
			modify: func(r *Receipt) {
				r.Phases[0].Outcome = ""
			},
		},
		{
			name: "orphan parent phase",
			modify: func(r *Receipt) {
				r.Phases = append(r.Phases, Phase{ID: "p2", ParentID: "nonexistent", Engine: "fak-native", Outcome: "completed"})
			},
		},
		{
			name: "ambiguous native engine",
			modify: func(r *Receipt) {
				r.Phases[0].Engine = "custom-native"
			},
		},
		{
			name: "unreleased resource",
			modify: func(r *Receipt) {
				r.Resources[0].Released = false
			},
		},
		{
			name: "resource missing ID",
			modify: func(r *Receipt) {
				r.Resources[0].ID = ""
			},
		},
		{
			name: "decision missing ID",
			modify: func(r *Receipt) {
				r.Decisions[0].ID = ""
			},
		},
		{
			name: "decision missing Kind",
			modify: func(r *Receipt) {
				r.Decisions[0].Kind = ""
			},
		},
		{
			name: "decision missing Actual",
			modify: func(r *Receipt) {
				r.Decisions[0].Actual = ""
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := valid()
			tc.modify(&r)
			_, err := Summarize(r)
			if err == nil {
				t.Fatalf("expected error for case %q, got nil", tc.name)
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("expected ErrInvalid for case %q, got %v", tc.name, err)
			}
		})
	}
}
