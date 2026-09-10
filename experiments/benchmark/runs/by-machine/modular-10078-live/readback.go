//go:build ignore

// Read the diagnostic hardware capture through the current claim gate.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/webbench"
	"os"
)

func main() {
	if len(os.Args) != 2 {
		panic("usage: go run readback.go <receipt.json>")
	}
	report, err := webbench.LoadServingSweepReport(os.Args[1])
	if err != nil {
		panic(err)
	}
	if err = webbench.EvaluateServingSweep(report); err != nil {
		panic(err)
	}
	gateErr := webbench.ValidateServingSweepClaim("fak native serving peak throughput", report)
	var points []map[string]any
	for _, p := range report.Points {
		for _, tr := range p.Tracks {
			points = append(points, map[string]any{"concurrency": p.Concurrency, "status": tr.Status, "reason_code": tr.ReasonCode, "requests": tr.Stats.Requests, "ok": tr.Stats.OK, "failed": tr.Stats.Failed})
		}
	}
	reason := ""
	if gateErr != nil {
		reason = gateErr.Error()
	}
	result := map[string]any{"schema": "fak.modular-software-readback/v1", "claim_gate_refused": gateErr != nil, "claim_gate_reason": reason, "points": points, "track_summaries": report.Tracks}
	b, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(b))
	if gateErr == nil {
		os.Exit(1)
	}
}
