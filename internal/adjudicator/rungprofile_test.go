package adjudicator

import "testing"

// TestRiskClassGeneralizesShapePredicates pins riskClass as the single coarse
// classifier the RungProfile keys on: it must agree with writeShaped /
// lowRiskReadShaped on the clear cases, escalate a read-shaped name that resolves a
// write-capable payload, and fail closed (classWrite) on an ambiguous name.
func TestRiskClassGeneralizesShapePredicates(t *testing.T) {
	cases := []struct {
		name string
		tool string
		args map[string]any
		want class
	}{
		// write-shaped names: always classWrite.
		{"write tool", "write_file", nil, classWrite},
		{"edit tool", "edit", map[string]any{"path": "x.go"}, classWrite},
		{"delete tool", "delete_account", nil, classWrite},
		{"exec tool", "Bash", map[string]any{"command": "ls"}, classWrite},

		// read-shaped names with NO write-capable payload: classRead.
		{"plain read", "read_report", nil, classRead},
		{"get", "get_user_details", map[string]any{}, classRead},
		{"search", "search_flights", map[string]any{"from": "SFO"}, classRead},
		{"calculate", "calculate", map[string]any{"expr": "1+1"}, classRead},

		// read-shaped name that resolves a write-capable payload: ESCALATES to write.
		{"read with path target", "read_file", map[string]any{"path": "internal/abi/x.go"}, classWrite},
		{"read with file_path", "get_blob", map[string]any{"file_path": "out.txt"}, classWrite},
		{"read with command", "search_logs", map[string]any{"command": "grep -r x ."}, classWrite},
		{"read with cmd", "list_dir", map[string]any{"cmd": "rm -rf x"}, classWrite},

		// ambiguous names (neither clearly read nor write): fail closed to classWrite.
		{"ambiguous", "transfer_to_human_agents", nil, classWrite},
		{"empty", "", nil, classWrite},
		{"unknown verb", "frobnicate", map[string]any{}, classWrite},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := riskClass(c.tool, c.args); got != c.want {
				t.Fatalf("riskClass(%q, %v) = %d, want %d", c.tool, c.args, got, c.want)
			}
		})
	}
}

// TestRiskClassReadShapedTargetEscalationMatchesTargetPath proves the escalation
// trigger is exactly targetPath: a read-shaped tool whose path arg the self-modify
// rung would read must classify as write, so eliding that rung for the read class
// never hides a path-bearing call from it.
func TestRiskClassReadShapedTargetEscalationMatchesTargetPath(t *testing.T) {
	for _, key := range []string{"path", "file_path", "filePath", "filepath", "file", "target", "filename", "dir"} {
		args := map[string]any{key: "some/where.txt"}
		if got := riskClass("read_thing", args); got != classWrite {
			t.Fatalf("read-shaped tool with path arg %q: got %d, want classWrite", key, got)
		}
	}
}
