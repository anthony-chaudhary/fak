package main

import "testing"

// deriveWorkKey folds a launch turn into the authoritative work identity used to dedup a
// crashed session against a live one. These pin the recognition order (goal > issue > loop)
// and the normalization, so two sessions launched on the same work produce the same key.
func TestDeriveWorkKey(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		{
			name: "dispatch-loop lane",
			text: "<command-name>/dos-dispatch-loop</command-name> <command-args>--lane claude</command-args>",
			want: "loop:--lane claude",
		},
		{
			name: "slash loop lane",
			text: "/loop /dos-dispatch --lane ci then keep going",
			want: "loop:--lane ci",
		},
		{
			// The real transcript form: the command tags are newline-separated, so the loop
			// marker and the --lane flag sit on different lines (the bug that made a crashed
			// dispatch-loop fail to dedup against its live twin).
			name: "dispatch-loop with newline-separated command tags",
			text: "<command-message>dos-dispatch-loop</command-message>\n<command-name>/dos-dispatch-loop</command-name>\n<command-args>--lane claude</command-args>",
			want: "loop:--lane claude",
		},
		{
			// A bare --lane mention in prose, with NO loop marker, must not be read as a loop
			// identity (it is just discussion, not a launch contract).
			name: "bare --lane without a loop marker does not key",
			text: "you could pick --lane docs for this if you wanted",
			want: "",
		},
		{
			name: "resolve github issue",
			text: "your goal: resolve GitHub issue #1538 (turn-tax adaptive planner) with the smallest change",
			want: "issue:#1538",
		},
		{
			name: "issue outranks a lane mention",
			text: "resolve GitHub issue #1538 — you may run dos-dispatch-loop --lane docs while you work",
			want: "issue:#1538",
		},
		{
			name: "goal outranks everything",
			text: "<command-name>/goal</command-name> <command-message>goal</command-message> <command-args>audit LLM cache folder for ideas, resolve issue #99, --lane ci</command-args>",
			want: "goal:audit llm cache folder for ideas, resolve issue #99, --lane ci",
		},
		{
			name: "goal is whitespace-normalized and lowercased",
			text: "<command-name>/goal</command-name> <command-args>Fix   the\nBADGES  on readme</command-args>",
			want: "goal:fix the badges on readme",
		},
		{
			name: "no work signal",
			text: "just a normal assistant reply about the weather",
			want: "",
		},
		{
			name: "empty goal args do not key",
			text: "<command-name>/goal</command-name> <command-args>   </command-args>",
			want: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := deriveWorkKey(c.text); got != c.want {
				t.Fatalf("deriveWorkKey(%q) = %q, want %q", c.text, got, c.want)
			}
		})
	}
}

// A long goal is truncated to a bounded key so the map stays small and two long goals that
// agree on their first 80 chars still collide (which is the intent: same launch objective).
func TestDeriveWorkKeyGoalTruncation(t *testing.T) {
	long := "<command-name>/goal</command-name> <command-args>" +
		"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaTAIL" +
		"</command-args>"
	got := deriveWorkKey(long)
	if len(got) != len("goal:")+80 {
		t.Fatalf("key len = %d, want %d (goal: + 80)", len(got), len("goal:")+80)
	}
}
