package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/categorybaseline"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestMaturityBaselinePromotionIsWitnessGatedAndImmediatelyEnforced(t *testing.T) {
	root := t.TempDir()
	old := maturityBaselineRunWitness
	defer func() { maturityBaselineRunWitness = old }()
	maturityBaselineRunWitness = func(string, string, []string, io.Writer, io.Writer) error { return errors.New("red") }
	var out, errOut bytes.Buffer
	args := []string{"promote", "--workspace", root, "--category", "serving", "--layers", "medium-model,l2-cache,l3-cache", "--completed", "medium-model", "--next", "l2-cache", "--witness", "fak", "--witness-arg", "serve-selfcheck"}
	if code := runMaturityBaseline(&out, &errOut, args); code != 1 {
		t.Fatalf("failed witness code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(categorybaseline.DefaultPath))); !os.IsNotExist(err) {
		t.Fatalf("failed witness mutated registry: %v", err)
	}

	maturityBaselineRunWitness = func(_ string, name string, args []string, _, _ io.Writer) error {
		if name != "fak" || len(args) != 1 || args[0] != "serve-selfcheck" {
			t.Fatalf("witness %q %v", name, args)
		}
		return nil
	}
	out.Reset()
	errOut.Reset()
	if code := runMaturityBaseline(&out, &errOut, args); code != 0 {
		t.Fatalf("promotion code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	r := categorybaseline.Load(root)
	if len(r.Categories) != 1 || r.Categories[0].NextLayer != "l2-cache" {
		t.Fatalf("registry=%+v", r)
	}
	payload := dispatchtick.RouterPayload{Issues: []dispatchtick.IssueRoute{{Number: 10, Lane: "model", Category: "serving", Layer: "medium-model", ExpectedSteps: 1}}, Lanes: map[string]dispatchtick.RouterLaneGroup{"model": {Issues: []int{10}, Count: 1, StepBudget: 1}}, Counts: dispatchtick.RouterCounts{Routed: 1, RoutedStepBudget: 1}}
	got := holdCompletedCategoryBaselines(root, payload)
	if len(got.Issues) != 0 || len(got.SkippedHumanBlocked) != 1 || got.SkippedHumanBlocked[0].Reason != reasonCategoryBaselineComplete {
		t.Fatalf("promoted enforcement=%+v", got)
	}

	out.Reset()
	errOut.Reset()
	if code := runMaturityBaseline(&out, &errOut, []string{"list", "--workspace", root}); code != 0 || !strings.Contains(out.String(), "medium-model complete -> l2-cache next") {
		t.Fatalf("list code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
}
