package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

func TestTaskSampleJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTask(&stdout, &stderr, []string{
		"sample",
		"--json",
		"--task", "task_demo",
		"--title", "Demo task",
		"--step", "step_demo",
		"--concept", "verify",
		"--done", "2",
		"--total", "4",
		"--unit", "phase",
	})
	if code != 0 {
		t.Fatalf("runTask code=%d stderr=%s", code, stderr.String())
	}
	var snap taskmgr.Snapshot
	if err := json.Unmarshal(stdout.Bytes(), &snap); err != nil {
		t.Fatalf("unmarshal snapshot: %v\n%s", err, stdout.String())
	}
	if snap.Schema != taskmgr.SchemaSnapshot {
		t.Fatalf("schema = %q, want %q", snap.Schema, taskmgr.SchemaSnapshot)
	}
	if len(snap.Tasks) != 1 || snap.Tasks[0].TaskID != "task_demo" {
		t.Fatalf("tasks = %+v, want task_demo", snap.Tasks)
	}
	task := snap.Tasks[0]
	if task.Progress.Done != 2 || task.Progress.Total != 4 || task.Progress.Unit != "phase" {
		t.Fatalf("task progress = %+v", task.Progress)
	}
	if task.LivenessClass != taskmgr.LivenessLive {
		t.Fatalf("task liveness = %s, want live", task.LivenessClass)
	}
	if len(task.Steps) != 1 || task.Steps[0].Concept != "verify" {
		t.Fatalf("steps = %+v, want one verify step", task.Steps)
	}
	if task.Steps[0].LivenessClass != taskmgr.LivenessLive {
		t.Fatalf("step liveness = %s, want live", task.Steps[0].LivenessClass)
	}
}

func TestTaskSampleHumanOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTask(&stdout, &stderr, []string{"sample", "--task", "task_human", "--concept", "observe"})
	if code != 0 {
		t.Fatalf("runTask code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"process pid=", "task task_human", "liveness=idle", "step snapshot", "concept=observe"} {
		if !strings.Contains(out, want) {
			t.Fatalf("human output missing %q:\n%s", want, out)
		}
	}
}

func TestTaskSampleRejectsNegativeProgress(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTask(&stdout, &stderr, []string{"sample", "--done", "-1"})
	if code != 2 {
		t.Fatalf("runTask code=%d, want 2 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "non-negative") {
		t.Fatalf("stderr = %q, want non-negative error", stderr.String())
	}
}
