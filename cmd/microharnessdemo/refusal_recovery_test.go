package main

import (
	"context"
	"strings"
	"testing"
)

func TestMicroharnessRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name string
		task task
		want []string
	}{
		{name: "identity", task: task{}, want: []string{"id", "goal", "required"}},
		{name: "depth", task: task{ID: "safe", Goal: "g", Depth: 3, MaxTurns: 1, Class: bandOneTurn}, want: []string{"depth", "2"}},
		{name: "turn budget", task: task{ID: "safe", Goal: "g", Depth: 1, MaxTurns: 4, Class: bandBoundedCorrection}, want: []string{"turn", "3"}},
		{name: "class", task: task{ID: "safe", Goal: "g", Depth: 1, MaxTurns: 1, Class: "unknown"}, want: []string{"decision class", "unsupported"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			admitted, _, err := admitTask(tc.task)
			if err == nil || admitted {
				t.Fatal("admission accepted refusal input")
			}
			for _, want := range tc.want {
				if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(want)) {
					t.Fatalf("error %q omits recovery text %q", err, want)
				}
			}
		})
	}
	t.Run("cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := run(ctx)
		if err == nil || !strings.Contains(err.Error(), "spawn architecture") || !strings.Contains(strings.ToLower(err.Error()), "canceled") {
			t.Fatalf("run error=%v", err)
		}
	})
}
