package hooks

import "strings"

func checkParallelFabricNudge(d *StagedDiff) ([]Finding, error) {
	d.NoteCandidates("PARALLEL_FABRIC_NUDGE", len(d.StagedPaths), "staged path(s)")
	var add strings.Builder
	for _, lines := range d.AddedByFile {
		for _, line := range lines {
			add.WriteString(strings.ToLower(line.Text))
			add.WriteByte('\n')
		}
	}
	text := add.String()
	trigger := (strings.Contains(text, "massively parallel") || strings.Contains(text, "parallel agent") || strings.Contains(text, "logical contexts")) && (strings.Contains(text, "workers") || strings.Contains(text, "fan-out"))
	if !trigger || strings.Contains(text, "microcontext") || strings.Contains(text, "micro-context") || strings.Contains(text, "bounded physical") {
		return nil, nil
	}
	return []Finding{{Gate: "PARALLEL_FABRIC_NUDGE", File: "(staged diff)", Detail: "new parallel-agent fan-out does not name the bounded micro-context route; run `go run ./cmd/microcontextdemo -selfcheck -contexts 10000 -workers 64` and document why this surface cannot reuse it", Advisory: true}}, nil
}
