package localappux

import (
	"testing"
)

func BenchmarkLocalAppUX(b *testing.B) {
	views := []View{
		{
			State:        StateFirstRun,
			Mode:         ModeAutomatic,
			AssetBytes:   6 << 30,
			FreeBytes:    20 << 30,
			PendingTasks: []string{"draft", "summarize"},
		},
		{
			State:        StatePartial,
			Mode:         ModePreferLocal,
			ReadyTasks:   []string{"summarize"},
			PendingTasks: []string{"draft"},
		},
		{
			State:      StatePressure,
			Mode:       ModeAutomatic,
			AssetBytes: 6 << 30,
			FreeBytes:  2 << 30,
		},
		{
			State:       StateHandoffAsk,
			Mode:        ModeAutomatic,
			LocalData:   "local buffer text",
			Destination: "provider",
		},
		{
			State: StateReady,
			Mode:  ModeAutomatic,
		},
	}

	diag := Diagnostic{
		AppVersion: "1.2",
		State:      StateHelperRestart,
		Mode:       ModeLocalOnly,
		Engine:     "fak-native",
		ErrorCode:  "HELPER_EXIT",
		Paths:      []string{"/Users/alice/private"},
		Prompt:     "sensitive query",
		Token:      "secret-token",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, v := range views {
			_ = Render(v)
		}
		_, _ = PreviewDiagnostic(diag)
	}
}
