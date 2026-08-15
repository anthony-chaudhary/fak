package main

import (
	"fmt"
	"strings"
)

func renderDispatchWavePlanRows(b *strings.Builder, plan []dispatchWaveExecutionPlan) {
	for _, row := range plan {
		fmt.Fprintf(b, "    rank=%d wave=%s size=%d target=%s lane=%s lease=%s account=%s slot=%s\n",
			row.Rank, row.WaveID, row.WaveSize, row.Target.ID, row.Target.Lane,
			row.Target.LeaseID, dispatchWavePlanAccountLabel(row.Account),
			dispatchWavePlanSlotLabel(row.Account))
	}
}

func dispatchWavePlanAccountLabel(account map[string]any) string {
	for _, key := range []string{"tag", "account", "dir"} {
		if val := strings.TrimSpace(fmt.Sprint(account[key])); val != "" && val != "<nil>" {
			return val
		}
	}
	return "-"
}

func dispatchWavePlanSlotLabel(account map[string]any) string {
	slot, cap := dispatchMapInt(account, "session_slot"), dispatchMapInt(account, "session_cap")
	if slot <= 0 && cap <= 0 {
		return "-"
	}
	if cap <= 0 {
		return fmt.Sprint(slot)
	}
	return fmt.Sprintf("%d/%d", slot, cap)
}
