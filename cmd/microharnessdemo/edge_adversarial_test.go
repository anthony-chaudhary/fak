package main

import (
	"context"
	"strings"
	"testing"
)

func TestMicroharnessEdgeAndAdversarialInputs(t *testing.T) {
	cases := []struct {
		name string
		task task
		want string
	}{
		{name: "empty", task: task{}, want: "id"},
		{name: "oversized id", task: task{ID: strings.Repeat("x", 1<<20), Goal: "g", Depth: 1, TurnBudget: 1, DecisionClass: "one_turn"}, want: "id"},
		{name: "hostile id", task: task{ID: "../../escape", Goal: "g", Depth: 1, TurnBudget: 1, DecisionClass: "one_turn"}, want: "id"},
		{name: "negative depth", task: task{ID: "safe", Goal: "g", Depth: -1, TurnBudget: 1, DecisionClass: "one_turn"}, want: "depth"},
		{name: "depth overflow", task: task{ID: "safe", Goal: "g", Depth: 3, TurnBudget: 1, DecisionClass: "one_turn"}, want: "depth"},
		{name: "turn overflow", task: task{ID: "safe", Goal: "g", Depth: 1, TurnBudget: 4, DecisionClass: "bounded_correction"}, want: "turn"},
		{name: "hostile class", task: task{ID: "safe", Goal: "g", Depth: 1, TurnBudget: 1, DecisionClass: "root_only;delegate"}, want: "decision"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			admitted, _, err := admitTask(tc.task)
			if err == nil || admitted || !strings.Contains(strings.ToLower(err.Error()), tc.want) {
				t.Fatalf("admitted=%v err=%v want %q", admitted, err, tc.want)
			}
		})
	}
	t.Run("canceled run", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := run(ctx)
		if err == nil || !strings.Contains(strings.ToLower(err.Error()), "cancel") {
			t.Fatalf("run error=%v", err)
		}
	})
}
