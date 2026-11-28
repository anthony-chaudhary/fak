package main

import (
	"bytes"
	"fmt"
	"os"
)

// DIRECT-PORT of ai-dynamo/aiperf (Apache-2.0), tools/check_agent_files_sync.py:45
// @ab610add4e8876fca223197596c728e17365f381: the agent-instruction files that must
// stay byte-identical (for example AGENTS.md and its tool-specific mirrors) are
// compared byte for byte, and every file that drifts from the first is reported.

// agentFileSyncDrift lists the files whose bytes differ from paths[0].
func agentFileSyncDrift(paths []string) (drifted []string, err error) {
	if len(paths) < 2 {
		return nil, nil
	}
	want, err := os.ReadFile(paths[0])
	if err != nil {
		return nil, fmt.Errorf("agent file sync: read %s: %w", paths[0], err)
	}
	for _, p := range paths[1:] {
		got, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("agent file sync: read %s: %w", p, err)
		}
		if !bytes.Equal(want, got) {
			drifted = append(drifted, p)
		}
	}
	return drifted, nil
}
