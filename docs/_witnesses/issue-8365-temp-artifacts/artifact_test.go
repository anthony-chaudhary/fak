package issue8365witness

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type witness struct {
	Schema        string `json:"schema"`
	FailingBefore struct {
		ExitCode              int  `json:"exit_code"`
		CommittedDispatchCase bool `json:"committed_dispatch_case"`
	} `json:"failing_before"`
	WindowsApply struct {
		ExitCode int `json:"exit_code"`
		Receipt  struct {
			Schema     string `json:"schema"`
			Inspection string `json:"inspection"`
			Items      []struct {
				Reason string `json:"reason"`
			} `json:"items"`
			Summary struct {
				EligibleCount int   `json:"eligible_count"`
				EligibleBytes int64 `json:"eligible_bytes"`
				ReapedCount   int   `json:"reaped_count"`
				ReapedBytes   int64 `json:"reaped_bytes"`
			} `json:"summary"`
		} `json:"receipt"`
	} `json:"windows_apply"`
}

func TestCapturedFailingBeforeAndWindowsApplyReceipt(t *testing.T) {
	raw, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	var got witness
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-issue-8365-witness/1" || got.FailingBefore.ExitCode != 2 || got.FailingBefore.CommittedDispatchCase {
		t.Fatalf("failing-before witness is not discriminating: %+v", got.FailingBefore)
	}
	if got.WindowsApply.ExitCode != 0 || got.WindowsApply.Receipt.Schema != "fak-temp-artifacts/1" || got.WindowsApply.Receipt.Inspection != "complete" {
		t.Fatalf("Windows receipt header is not witnessed: %+v", got.WindowsApply)
	}
	if got.WindowsApply.Receipt.Summary.EligibleCount != 1 || got.WindowsApply.Receipt.Summary.EligibleBytes != 20 || got.WindowsApply.Receipt.Summary.ReapedCount != 1 || got.WindowsApply.Receipt.Summary.ReapedBytes != 20 {
		t.Fatalf("Windows receipt totals drifted: %+v", got.WindowsApply.Receipt.Summary)
	}
	reasons := map[string]bool{}
	for _, item := range got.WindowsApply.Receipt.Items {
		reasons[item.Reason] = true
	}
	for _, reason := range []string{"active_reference", "fresh", "reaped"} {
		if !reasons[reason] {
			t.Fatalf("Windows receipt is missing %q: %+v", reason, reasons)
		}
	}
	if strings.Contains(strings.ToLower(string(raw)), "commandline") || strings.Contains(strings.ToLower(string(raw)), "executablepath") {
		t.Fatal("public receipt contains private process inspection fields")
	}
}
