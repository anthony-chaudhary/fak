package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func TestUltracodeChildAccessCompilation(t *testing.T) {
	parent, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	roles := []orchestration.Role{
		{ID: "observer", Purpose: "inspect", TaskID: "task-access", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
		{ID: "writer", Purpose: "implement", TaskID: "task-access", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, Lane: "cmd", WriteTree: "cmd/fak/access/**", Tools: []string{"Read", "Write"}}},
		{ID: "colliding-writer", Purpose: "implement same region", TaskID: "task-access", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, Lane: "cmd", WriteTree: "cmd/fak/access/**", Tools: []string{"Read", "Write"}}},
	}
	resolution := orchestration.Resolution{Resolved: orchestration.WorkflowPlan{
		Profile:   orchestration.ProfileUltracode,
		TaskID:    "task-access",
		WorkClass: orchestration.WorkGrind,
		Roles:     roles,
		Budget:    orchestration.Budget{MaxWorkers: 4, MaxTokens: 4096},
		SOLRoute:  orchestration.SOLRoute{Model: "gpt-5.6-sol", Mode: orchestration.SOLUltra, ReasoningEffort: "high"},
	}}

	oldSnapshot := orchestrationChildAccessSnapshotLoader
	orchestrationChildAccessSnapshotLoader = func(string) (orchestrationChildAccessSnapshot, error) {
		return orchestrationChildAccessSnapshot{
			Parent: parent,
			Taxonomy: laneadmit.Taxonomy{
				Loaded:    true,
				Trees:     map[string][]string{"cmd": {"cmd/**"}},
				Exclusive: map[string]bool{},
			},
		}, nil
	}
	t.Cleanup(func() { orchestrationChildAccessSnapshotLoader = oldSnapshot })

	oldLauncher := orchestrationWorkerLauncher
	var launched []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launched = append(launched, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 100 + len(launched), Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	home := externalOrchestrationTestHome(t)
	receipt, err := launchCodexOrchestrationWorkers(home, "session-access", "ultracode", "native", "compile access", resolution)
	if err == nil || !strings.Contains(err.Error(), laneadmit.ReasonCollisionRisk) {
		t.Fatalf("colliding writer error = %v, want %s", err, laneadmit.ReasonCollisionRisk)
	}
	if receipt.Status != "partial" || len(launched) != 2 {
		t.Fatalf("receipt=%+v launched=%d, want observer+writer only", receipt, len(launched))
	}
	observer, writer := launched[0], launched[1]
	if observer.Role.ID != "observer" || !observer.Access.Admission.ReadOnly {
		t.Fatalf("observer access = %+v", observer.Access)
	}
	if writer.Role.ID != "writer" || writer.Access.Admission.ReadOnly || writer.Access.Admission.Lane != "cmd" {
		t.Fatalf("writer access = %+v", writer.Access)
	}
	if observer.Access.PolicyPath == "" || writer.Access.PolicyPath == "" {
		t.Fatalf("compiled policy paths missing: observer=%q writer=%q", observer.Access.PolicyPath, writer.Access.PolicyPath)
	}
	assertChildAccessVerdict(t, observer.Access.Policy, "Read", `{"file_path":"README.md"}`, abi.VerdictAllow)
	assertChildAccessVerdict(t, observer.Access.Policy, "Write", `{"file_path":"cmd/fak/access/nope.go","content":"x"}`, abi.VerdictDeny)
	assertChildAccessVerdict(t, observer.Access.Policy, "Bash", `{"command":"echo x > cmd/fak/access/nope.go"}`, abi.VerdictDeny)
	assertChildAccessVerdict(t, writer.Access.Policy, "Write", `{"file_path":"cmd/fak/access/ok.go","content":"x"}`, abi.VerdictAllow)
	assertChildAccessVerdict(t, writer.Access.Policy, "Write", `{"file_path":"internal/model/nope.go","content":"x"}`, abi.VerdictDeny)

	observerArgs := strings.Join(orchestrationWorkerArgs(observer, filepath.Join(home, "observer.audit.jsonl")), " ")
	writerArgs := strings.Join(orchestrationWorkerArgs(writer, filepath.Join(home, "writer.audit.jsonl")), " ")
	if !strings.Contains(observerArgs, "--policy "+observer.Access.PolicyPath) || strings.Contains(observerArgs, "--lease") {
		t.Fatalf("observer args do not carry policy-only read access: %s", observerArgs)
	}
	for _, want := range []string{"--policy " + writer.Access.PolicyPath, "--lease mode=enforce,lane=cmd,tree=cmd/fak/access/**"} {
		if !strings.Contains(writerArgs, want) {
			t.Fatalf("writer args missing %q: %s", want, writerArgs)
		}
	}
	for _, req := range launched {
		if _, err := os.Stat(req.Access.PolicyPath); err != nil {
			t.Fatalf("policy %s was not persisted before launch: %v", req.Access.PolicyPath, err)
		}
	}

	compiled, err := compileOrchestrationChildAccess(roles[1], parent, laneadmit.Request{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if compiled.Admission.ReadOnly {
		t.Fatal("caller-supplied read-only hint upgraded an effect child to shared admission")
	}
	for _, access := range []orchestration.ChildAccess{{}, {Mode: "unknown"}} {
		role := orchestration.Role{ID: "bad", Access: access}
		if _, err := compileOrchestrationChildAccess(role, parent, laneadmit.Request{}); err == nil {
			t.Fatalf("access %+v was accepted", access)
		}
	}
	shellObserver := orchestration.Role{ID: "shell-observer", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve, Tools: []string{"Bash"}}}
	if _, err := compileOrchestrationChildAccess(shellObserver, parent, laneadmit.Request{}); err == nil {
		t.Fatal("observer acquired a shell whose non-CLI commands are not structurally read-only")
	}
}

func TestUltracodeChildAccessLowersTypedWorkerEffectIntoLaunch(t *testing.T) {
	parent, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	maxWorkers := 2
	resolution, err := orchestration.Resolve(
		orchestration.OrchestrationProfile{Name: orchestration.ProfileUltracode, MaxWorkers: &maxWorkers},
		orchestration.TaskSpec{
			Schema: "fak-orchestration-task/1",
			ID:     "typed-worker-access",
			WorkerAccess: []orchestration.WorkerAccessSpec{{
				RoleID: "worker-1",
				Access: orchestration.ChildAccess{
					Mode:     orchestration.ChildAccessEffect,
					ReadSet:  []string{"internal/orchestration"},
					WriteSet: []string{"internal/orchestration"},
					Tools:    []string{"Read", "Write"},
				},
			}},
		},
		orchestration.HarnessCapabilities{
			Concurrency: orchestration.SupportNative, TaskMessaging: orchestration.SupportNative,
			Cancellation: orchestration.SupportNative, Leases: orchestration.SupportNative,
			IndependentWitness: orchestration.SupportNative, ClaudeSpeed: orchestration.SupportNative,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resolution.Resolved.Roles[0].ID != "lead" || resolution.Resolved.Roles[0].Access.Mode != orchestration.ChildAccessObserve {
		t.Fatalf("lead access = %+v", resolution.Resolved.Roles[0].Access)
	}

	oldSnapshot := orchestrationChildAccessSnapshotLoader
	orchestrationChildAccessSnapshotLoader = func(string) (orchestrationChildAccessSnapshot, error) {
		return orchestrationChildAccessSnapshot{
			Parent: parent,
			Taxonomy: laneadmit.Taxonomy{
				Loaded:    true,
				Trees:     map[string][]string{"orchestration": {"internal/orchestration/**"}},
				Exclusive: map[string]bool{},
			},
		}, nil
	}
	t.Cleanup(func() { orchestrationChildAccessSnapshotLoader = oldSnapshot })

	oldLauncher := orchestrationWorkerLauncher
	var launched []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launched = append(launched, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 9901, Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	home := externalOrchestrationTestHome(t)
	receipt, err := launchCodexOrchestrationWorkers(home, "session-typed-access", "ultracode", "native", "implement issue 9901", resolution)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "launched" || len(launched) != 1 {
		t.Fatalf("receipt=%+v launched=%d", receipt, len(launched))
	}
	if len(receipt.Workers) != 1 || receipt.Workers[0].AccessMode != string(orchestration.ChildAccessEffect) ||
		receipt.Workers[0].ReadOnly || receipt.Workers[0].WriteTree != "internal/orchestration/**" {
		t.Fatalf("launch receipt lost typed effect: %+v", receipt.Workers)
	}
	worker := launched[0]
	if worker.Role.ID != "worker-1" || worker.Access.Mode != orchestration.ChildAccessEffect ||
		worker.Access.Admission.ReadOnly || worker.Access.Admission.Lane != "orchestration" ||
		len(worker.Access.Admission.Tree) != 1 || worker.Access.Admission.Tree[0] != "internal/orchestration/**" {
		t.Fatalf("lowered worker access = %+v", worker.Access)
	}
	mainBefore, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	assertChildAccessVerdict(t, worker.Access.Policy, "Write", `{"file_path":"internal/orchestration/new.go","content":"x"}`, abi.VerdictAllow)
	assertChildAccessVerdict(t, worker.Access.Policy, "Write", `{"file_path":"cmd/fak/main.go","content":"x"}`, abi.VerdictDeny)
	mainAfter, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	if string(mainBefore) != string(mainAfter) {
		t.Fatal("denied out-of-scope write changed cmd/fak/main.go")
	}

	args := strings.Join(orchestrationWorkerArgs(worker, filepath.Join(home, "worker.audit.jsonl")), " ")
	if !strings.Contains(args, "--lease mode=enforce,lane=orchestration,tree=internal/orchestration/**") {
		t.Fatalf("worker args lack exact enforced lease: %s", args)
	}
	prompt := orchestrationWorkerPrompt(worker)
	for _, want := range []string{"compiled effect envelope", "lane orchestration", "write tree internal/orchestration/**", "tools Read,Write"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("effect prompt missing %q: %s", want, prompt)
		}
	}
}

func assertChildAccessVerdict(t *testing.T, runtime policy.Runtime, tool, args string, want abi.VerdictKind) {
	t.Helper()
	call := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
	got := adjudicator.New(runtime.Adjudicator).Adjudicate(context.Background(), call)
	if got.Kind != want {
		t.Fatalf("%s %s verdict = %v (%s), want %v; allow=%v", tool, args, got.Kind, abi.ReasonName(got.Reason), want, runtime.Adjudicator.Allow)
	}
}
