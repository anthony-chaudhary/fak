package model

import (
	"os"
	"testing"
)

func v41PanelTestEnv(t *testing.T, key, value string, present bool) {
	t.Helper()
	t.Setenv(key, value)
	if !present {
		if err := os.Unsetenv(key); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41PrefillPanelIndependentOptIn(t *testing.T) {
	m := v41PinnedSlateFixture(t, false)
	b := newV41ExpertPanelRecorder()
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	inputs := make([][]float32, 2)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for lane := range inputs[row] {
			inputs[row][lane] = float32((row+lane)%13-6) / 16
		}
	}
	for _, tc := range []struct {
		name, legacy, panel                  string
		legacyPresent, panelPresent, enabled bool
	}{
		{name: "absent_default"},
		{name: "absent_ignores_decode_on", legacy: "1", legacyPresent: true},
		{name: "absent_inherits_off", legacy: "0", legacyPresent: true},
		{name: "panel_only", panel: "1", panelPresent: true, enabled: true},
		{name: "panel_with_decode_off", legacy: "0", panel: "1", legacyPresent: true, panelPresent: true, enabled: true},
		{name: "explicit_off_overrides_on", legacy: "1", panel: "0", legacyPresent: true, panelPresent: true},
		{name: "empty_overrides_on", legacy: "1", legacyPresent: true, panelPresent: true},
		{name: "invalid_overrides_on", legacy: "1", panel: "true", legacyPresent: true, panelPresent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v41PanelTestEnv(t, "FAK_V41_ROUTED_ID", tc.legacy, tc.legacyPresent)
			v41PanelTestEnv(t, "FAK_V41_PREFILL_EXPERT_PANEL", tc.panel, tc.panelPresent)
			before := b.idCalls["ids"]
			rows, outcome, cause := s.v41RoutedExpertPanel(0, "ffn.experts.0", inputs, []float32{0.25, 0.5}, s.v41State().gatherOps.bf16PublishFunc())
			if cause != nil {
				t.Fatal(cause)
			}
			if tc.enabled {
				if outcome != v41RoutedExpertHandled || len(rows) != 2 || b.idCalls["ids"] != before+1 {
					t.Fatalf("enabled panel outcome=%v rows=%d ids=%d", outcome, len(rows), b.idCalls["ids"]-before)
				}
			} else if outcome != v41RoutedExpertDeclined || len(rows) != 0 || b.idCalls["ids"] != before {
				t.Fatalf("disabled panel outcome=%v rows=%d ids=%d", outcome, len(rows), b.idCalls["ids"]-before)
			}
		})
	}
}

// fak-test:runtime slow est=30s lane=default
func TestV41PrefillPanelOnlyPreservesDecode(t *testing.T) {
	v41PanelTestEnv(t, "FAK_V41_ROUTED_ID", "", false)
	t.Setenv("FAK_V41_PREFILL_EXPERT_PANEL", "1")
	panelModel, rowModel := v41PinnedSlateFixture(t, false), v41PinnedSlateFixture(t, false)
	b := newV41ExpertPanelRecorder()
	panelSession, err := panelModel.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer panelSession.Close()
	rowSession, err := rowModel.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
	if err != nil {
		t.Fatal(err)
	}
	defer rowSession.Close()
	prompt := []int{0, 1, 2, 3, 4, 5, 6, 7}
	got, want := panelSession.Prefill(prompt), rowSession.Prefill(prompt)
	if !v41SameBits(got, want) {
		t.Fatal("panel-only prefill changed logits")
	}
	if b.panels == 0 || b.idCalls["ids"] == 0 {
		t.Fatal("panel-only Prefill never dispatched a panel")
	}
	if len(b.pending) != 0 || len(b.multiPending) != 0 {
		t.Fatal("prefill left deferred records unread")
	}
	b.idCalls = map[string]int{}
	got, want = panelSession.Step(argmaxF32(got)), rowSession.Step(argmaxF32(want))
	if !v41SameBits(got, want) {
		t.Fatal("panel-only continuation changed logits")
	}
	for operation, calls := range b.idCalls {
		if calls != 0 {
			t.Fatalf("panel-only switch enabled decode ID operation %s (%d calls)", operation, calls)
		}
	}
}
