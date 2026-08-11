package sessiondiag

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

var updateInventoryWitness = flag.Bool("update-sessiondiag-witness", false, "regenerate the mixed sessiondiag incident render and JSON witnesses")

func TestReconcileInventoryMixedIncidentWitness(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	report := ReconcileInventory(mixedIncidentFixture(now), now)
	if report.Counts.Active != 5 {
		t.Fatalf("active=%d, want 5", report.Counts.Active)
	}
	if report.Counts.ProcessTrees != 8 {
		t.Fatalf("process trees=%d, want 8", report.Counts.ProcessTrees)
	}
	if got := report.Counts.ByKind[KindGuardedTUI]; got != 3 {
		t.Fatalf("guarded TUI count=%d, want 3", got)
	}
	if got := report.Counts.ByKind[KindHeadlessExec]; got != 3 { // two live workers plus one receipt-only historical launch
		t.Fatalf("headless count=%d, want 3", got)
	}
	if got := report.Counts.ByKind[KindResumeWrapper]; got != 3 {
		t.Fatalf("resume count=%d, want 3", got)
	}
	if got := report.Counts.ByHealth[HealthFailedButLocked]; got != 3 {
		t.Fatalf("failed-but-locked=%d, want 3", got)
	}
	if got := report.Counts.ByHealth[HealthStaleLock]; got != 2 {
		t.Fatalf("stale locks=%d, want 2", got)
	}
	if got := report.Counts.ByHealth[HealthReceiptOnly]; got != 1 {
		t.Fatalf("receipt-only=%d, want 1", got)
	}
	for _, session := range report.Sessions {
		if session.Health == HealthActive && len(session.ProcessTrees) == 0 {
			t.Fatalf("ACTIVE without process tree: %+v", session)
		}
		if session.GuardReceipt != nil && session.GuardReceipt.ProvesLiveness {
			t.Fatalf("launch receipt claims liveness: %+v", session)
		}
		if session.WriterLock != nil && session.WriterLock.ProvesLiveness {
			t.Fatalf("writer lock claims liveness: %+v", session)
		}
		for _, tree := range session.ProcessTrees {
			if tree.ProvesLiveness {
				t.Fatalf("process tree claims liveness: %+v", tree)
			}
		}
	}

	jsonBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	jsonBytes = append(jsonBytes, '\n')
	var render bytes.Buffer
	RenderInventory(&render, report)
	assertOrUpdateWitness(t, "mixed-incident.json", jsonBytes)
	assertOrUpdateWitness(t, "mixed-incident.txt", render.Bytes())
}

func TestReconcileInventoryDoesNotTreatPresenceAsLiveness(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	receiptID := "51000001-0000-4000-8000-000000000001"
	lockID := "51000002-0000-4000-8000-000000000002"
	processID := "51000003-0000-4000-8000-000000000003"
	in := InventoryInput{
		Threads: []ThreadEvidence{
			{ThreadID: receiptID, Source: "exec", CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Hour)},
			{ThreadID: lockID, Source: "cli", CreatedAt: now.Add(-time.Hour), UpdatedAt: now.Add(-time.Hour)},
			{ThreadID: processID, Source: "exec", CreatedAt: now.Add(-time.Second), UpdatedAt: now.Add(-time.Hour)},
		},
		GuardReceipts: []GuardReceiptEvidence{{ThreadID: receiptID, RecordedAt: now.Add(-time.Minute)}},
		WriterLocks:   []WriterLockEvidence{{ThreadID: lockID, ModifiedAt: now.Add(-time.Minute)}},
		Processes: []ProcessEvidence{
			{PID: 10, Name: "codex.exe", CommandLine: "codex exec -", StartedAt: now.Add(-time.Second)},
		},
		Window:     2 * time.Hour,
		StaleAfter: 10 * time.Minute,
	}
	report := ReconcileInventory(in, now)
	if report.Counts.Active != 0 {
		t.Fatalf("single-signal rows became active: %+v", report.Counts)
	}
	got := map[string]string{}
	for _, session := range report.Sessions {
		if session.Thread != nil {
			got[session.Thread.ID] = session.Health
		}
	}
	if got[receiptID] != HealthReceiptOnly || got[lockID] != HealthUnknown || got[processID] != HealthOrphanProcess {
		t.Fatalf("health=%v", got)
	}
}

func TestProcessTreePIDReuseAndMissingCommandLine(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	threadID := "52000001-0000-4000-8000-000000000001"
	in := InventoryInput{
		Threads: []ThreadEvidence{{
			ThreadID:  threadID,
			Source:    "exec",
			CreatedAt: now.Add(-2 * time.Minute),
			UpdatedAt: now.Add(-time.Minute),
		}},
		WriterLocks: []WriterLockEvidence{{ThreadID: threadID, ModifiedAt: now.Add(-time.Minute)}},
		Processes: []ProcessEvidence{
			// PID 20 was recycled after child 21 started. The stale PPID must not make
			// the newer process the wrapper of the older Codex process.
			{PID: 20, Name: "cmd.exe", CommandLine: "cmd /c codex", StartedAt: now.Add(-time.Minute)},
			{PID: 21, ParentPID: 20, Name: "codex.exe", CommandLine: "", StartedAt: now.Add(-2 * time.Minute)},
		},
		Window:     time.Hour,
		StaleAfter: 10 * time.Minute,
	}
	report := ReconcileInventory(in, now)
	if len(report.Sessions) != 1 || len(report.Sessions[0].ProcessTrees) != 1 {
		t.Fatalf("sessions=%+v", report.Sessions)
	}
	tree := report.Sessions[0].ProcessTrees[0]
	if tree.RootPID != 21 || len(tree.Nodes) != 1 {
		t.Fatalf("PID reuse linked stale parent: %+v", tree)
	}
	if !containsString(tree.Reasons, ReasonPIDReuse) || !containsString(tree.Reasons, ReasonMissingCommand) {
		t.Fatalf("missing typed process reasons: %+v", tree.Reasons)
	}
}

func TestProcessStartJoinPreservesSimultaneousWorkerOrder(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	first := "53000001-0000-4000-8000-000000000001"
	second := "53000002-0000-4000-8000-000000000002"
	in := InventoryInput{
		Threads: []ThreadEvidence{
			{ThreadID: first, Source: "exec", CreatedAt: now.Add(94 * time.Millisecond), UpdatedAt: now},
			{ThreadID: second, Source: "exec", CreatedAt: now.Add(114 * time.Millisecond), UpdatedAt: now},
		},
		WriterLocks: []WriterLockEvidence{
			{ThreadID: first, ModifiedAt: now},
			{ThreadID: second, ModifiedAt: now},
		},
		Processes: []ProcessEvidence{
			{PID: 31, Name: "codex.exe", CommandLine: "codex exec -", StartedAt: now},
			{PID: 32, Name: "codex.exe", CommandLine: "codex exec -", StartedAt: now.Add(11 * time.Millisecond)},
		},
		Window: time.Hour,
	}
	report := ReconcileInventory(in, now.Add(time.Second))
	byPID := map[int]string{}
	for _, session := range report.Sessions {
		if session.Thread != nil && len(session.ProcessTrees) == 1 {
			byPID[session.ProcessTrees[0].CodexPID] = session.Thread.ID
		}
	}
	if byPID[31] != first || byPID[32] != second {
		t.Fatalf("simultaneous workers crossed: %v", byPID)
	}
}

func TestGatewayProcessKindIsTypedButNotAlive(t *testing.T) {
	now := time.Date(2026, 8, 11, 18, 30, 0, 0, time.UTC)
	report := ReconcileInventory(InventoryInput{
		Processes: []ProcessEvidence{{PID: 40, Name: "fak.exe", CommandLine: "fak serve --stdio", StartedAt: now}},
		Window:    time.Hour,
	}, now)
	if len(report.Sessions) != 1 || report.Sessions[0].Kind != KindGatewayServed || report.Sessions[0].Health != HealthOrphanProcess {
		t.Fatalf("gateway row=%+v", report.Sessions)
	}
}

func mixedIncidentFixture(now time.Time) InventoryInput {
	threads := []ThreadEvidence{}
	turns := []TurnEvidence{}
	locks := []WriterLockEvidence{}
	receipts := []GuardReceiptEvidence{}
	processes := []ProcessEvidence{}
	pid := 100

	addGuardedTUI := func(id, turnStatus string, offset time.Duration) {
		start := now.Add(offset)
		threads = append(threads, ThreadEvidence{ThreadID: id, Source: "cli", ThreadSource: "user", CreatedAt: start.Add(120 * time.Millisecond), UpdatedAt: now.Add(-10 * time.Second)})
		locks = append(locks, WriterLockEvidence{ThreadID: id, ModifiedAt: now.Add(-20 * time.Second)})
		receipts = append(receipts, GuardReceiptEvidence{ThreadID: id, RecordedAt: start.Add(12 * time.Second)})
		turn := TurnEvidence{ThreadID: id, TurnID: strings.Replace(id, "100000", "a00000", 1), Status: turnStatus, StartedAt: now.Add(-time.Minute), RolloutOrdinal: 3}
		if strings.EqualFold(turnStatus, "completed") {
			turn.CompletedAt = now.Add(-20 * time.Second)
			turn.DurationMS = 40000
		}
		turns = append(turns, turn)
		wrapper, guard, cmd, node, codex := pid, pid+1, pid+2, pid+3, pid+4
		pid += 10
		processes = append(processes,
			ProcessEvidence{PID: wrapper, Name: "pwsh.exe", CommandLine: "pwsh -Command fak guard codex", StartedAt: start.Add(-2 * time.Second)},
			ProcessEvidence{PID: guard, ParentPID: wrapper, Name: "fak.exe", CommandLine: "fak guard codex", StartedAt: start.Add(-time.Second)},
			ProcessEvidence{PID: cmd, ParentPID: guard, Name: "cmd.exe", CommandLine: "cmd /c codex", StartedAt: start.Add(-500 * time.Millisecond)},
			ProcessEvidence{PID: node, ParentPID: cmd, Name: "node.exe", CommandLine: "node codex.js", StartedAt: start.Add(-200 * time.Millisecond)},
			ProcessEvidence{PID: codex, ParentPID: node, Name: "codex.exe", CommandLine: "codex -c model_provider=fak", StartedAt: start},
		)
	}
	addGuardedTUI("10000001-0000-4000-8000-000000000001", "inProgress", -8*time.Minute)
	addGuardedTUI("10000002-0000-4000-8000-000000000002", "completed", -7*time.Minute)
	addGuardedTUI("10000003-0000-4000-8000-000000000003", "completed", -6*time.Minute)

	addExec := func(id string, offset time.Duration, commandLine bool) {
		start := now.Add(offset)
		threads = append(threads, ThreadEvidence{ThreadID: id, Source: "exec", ThreadSource: "user", CreatedAt: start.Add(100 * time.Millisecond), UpdatedAt: now.Add(-5 * time.Second)})
		locks = append(locks, WriterLockEvidence{ThreadID: id, ModifiedAt: now.Add(-15 * time.Second)})
		cmd, node, codex := pid, pid+1, pid+2
		pid += 10
		codexCommand := "codex exec --json -"
		if !commandLine {
			codexCommand = ""
		}
		processes = append(processes,
			ProcessEvidence{PID: cmd, Name: "cmd.exe", CommandLine: "cmd /c codex exec", StartedAt: start.Add(-100 * time.Millisecond)},
			ProcessEvidence{PID: node, ParentPID: cmd, Name: "node.exe", CommandLine: "node codex.js", StartedAt: start.Add(-50 * time.Millisecond)},
			ProcessEvidence{PID: codex, ParentPID: node, Name: "codex.exe", CommandLine: codexCommand, StartedAt: start},
		)
	}
	addExec("20000001-0000-4000-8000-000000000001", -5*time.Minute, true)
	addExec("20000002-0000-4000-8000-000000000002", -4*time.Minute, false)

	addFailedResume := func(id string, offset time.Duration) {
		start := now.Add(offset)
		threads = append(threads, ThreadEvidence{ThreadID: id, Source: "cli", ThreadSource: "user", CreatedAt: now.Add(-2 * time.Hour), UpdatedAt: now.Add(-time.Minute)})
		turns = append(turns, TurnEvidence{
			ThreadID: id, TurnID: strings.Replace(id, "300000", "b00000", 1), Status: "failed",
			ErrorJSON: `{"codexErrorInfo":"other","message":"{\"error\":{\"type\":\"invalid_request_error\",\"code\":\"invalid_id_prefix\"}}"}`,
			StartedAt: start, CompletedAt: start.Add(5 * time.Second), DurationMS: 5000, RolloutOrdinal: 100,
		})
		locks = append(locks, WriterLockEvidence{ThreadID: id, ModifiedAt: now.Add(-17 * time.Minute)})
		receipts = append(receipts, GuardReceiptEvidence{ThreadID: id, RecordedAt: now.Add(-2 * time.Hour)})
		cmd, node, codex := pid, pid+1, pid+2
		pid += 10
		processes = append(processes,
			ProcessEvidence{PID: cmd, Name: "cmd.exe", CommandLine: "cmd /c codex exec resume " + id, StartedAt: start.Add(-100 * time.Millisecond)},
			ProcessEvidence{PID: node, ParentPID: cmd, Name: "node.exe", CommandLine: "node codex.js exec resume " + id, StartedAt: start.Add(-50 * time.Millisecond)},
			ProcessEvidence{PID: codex, ParentPID: node, Name: "codex.exe", CommandLine: "codex exec resume " + id, StartedAt: start},
		)
	}
	addFailedResume("30000001-0000-4000-8000-000000000001", -3*time.Minute)
	addFailedResume("30000002-0000-4000-8000-000000000002", -2*time.Minute)
	addFailedResume("30000003-0000-4000-8000-000000000003", -time.Minute)

	staleThread := "60000001-0000-4000-8000-000000000001"
	threads = append(threads, ThreadEvidence{ThreadID: staleThread, Source: "cli", CreatedAt: now.Add(-3 * time.Hour), UpdatedAt: now.Add(-time.Hour)})
	turns = append(turns, TurnEvidence{ThreadID: staleThread, TurnID: "61000001-0000-4000-8000-000000000001", Status: "completed", CompletedAt: now.Add(-time.Hour), RolloutOrdinal: 1})
	locks = append(locks, WriterLockEvidence{ThreadID: staleThread, ModifiedAt: now.Add(-time.Hour)})
	locks = append(locks, WriterLockEvidence{ThreadID: "60000002-0000-4000-8000-000000000002", ModifiedAt: now.Add(-2 * time.Hour)})

	receiptOnly := "70000001-0000-4000-8000-000000000001"
	threads = append(threads, ThreadEvidence{ThreadID: receiptOnly, Source: "exec", CreatedAt: now.Add(-30 * time.Minute), UpdatedAt: now.Add(-30 * time.Minute)})
	receipts = append(receipts, GuardReceiptEvidence{ThreadID: receiptOnly, RecordedAt: now.Add(-29 * time.Minute)})

	child := "80000001-0000-4000-8000-000000000001"
	threads = append(threads, ThreadEvidence{ThreadID: child, Source: "exec", ThreadSource: "subagent", CreatedAt: now.Add(-9 * time.Minute), UpdatedAt: now.Add(-5 * time.Minute)})
	turns = append(turns, TurnEvidence{ThreadID: child, TurnID: "81000001-0000-4000-8000-000000000001", Status: "completed", CompletedAt: now.Add(-5 * time.Minute), RolloutOrdinal: 1})

	return InventoryInput{
		Threads:       threads,
		Turns:         turns,
		WriterLocks:   locks,
		GuardReceipts: receipts,
		Processes:     processes,
		Registrations: []sessionregistry.Record{
			{Schema: sessionregistry.Schema, RegistrationID: "reg-guard-parent", RootRegistrationID: "reg-guard-parent", RootOutcome: "trace child work", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-parent", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex", PID: 104, ProcessStartedAt: now.Add(-8 * time.Minute)}, State: sessionregistry.StateActive, CreatedAt: now.Add(-9 * time.Minute)},
			{Schema: sessionregistry.Schema, RegistrationID: "reg-headless-child", ParentRegistrationID: "reg-guard-parent", ParentAttemptID: "attempt-parent", RootRegistrationID: "reg-guard-parent", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-child", LaunchKind: "headless_worker", Identity: sessionregistry.Identity{Runtime: "codex", PID: 132, ProcessStartedAt: now.Add(-5 * time.Minute)}, State: sessionregistry.StateActive, CreatedAt: now.Add(-5 * time.Minute)},
			{Schema: sessionregistry.Schema, RegistrationID: "reg-nested", ParentRegistrationID: "reg-headless-child", ParentAttemptID: "attempt-child", RootRegistrationID: "reg-guard-parent", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-nested", LaunchKind: "subagent", Identity: sessionregistry.Identity{Runtime: "claude"}, State: sessionregistry.StateCompleted, CreatedAt: now.Add(-4 * time.Minute), TerminalAt: now.Add(-time.Minute), WitnessRef: "commit:mixed"},
			{Schema: sessionregistry.Schema, RegistrationID: "reg-resume", ParentRegistrationID: "reg-guard-parent", ParentAttemptID: "attempt-parent", RootRegistrationID: "reg-guard-parent", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "attempt-resume", ResumeOfAttemptID: "attempt-child", LaunchKind: "resume_wrapper", Identity: sessionregistry.Identity{Runtime: "codex"}, State: sessionregistry.StateLost, CreatedAt: now.Add(-3 * time.Minute), TerminalAt: now.Add(-2 * time.Minute), Reason: "parent_crash"},
			{Schema: sessionregistry.Schema, RegistrationID: "reg-stale", RootRegistrationID: "reg-stale", RootIssue: "6458", AttemptID: "attempt-stale", LaunchKind: "headless_worker", Identity: sessionregistry.Identity{Runtime: "codex", PID: 999, ProcessStartedAt: now.Add(-time.Hour)}, State: sessionregistry.StateActive, CreatedAt: now.Add(-time.Hour)},
		},
		SpawnEdges: []SpawnEdgeEvidence{
			{ParentThreadID: "10000001-0000-4000-8000-000000000001", ChildThreadID: child, Status: "closed"},
			{ParentThreadID: "90000001-0000-4000-8000-000000000001", ChildThreadID: "90000002-0000-4000-8000-000000000002", Status: "open"},
		},
		Window:             time.Hour,
		StaleAfter:         10 * time.Minute,
		ProcessMatchWindow: 3 * time.Second,
	}
}

func assertOrUpdateWitness(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *updateInventoryWitness {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read witness %s: %v (run go test ./internal/sessiondiag -run TestReconcileInventoryMixedIncidentWitness -update-sessiondiag-witness)", path, err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s drifted\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestInventoryJoinsRegistrationLineageAndDetectsUnregisteredProcess(t *testing.T) {
	now := time.Date(2026, 8, 11, 20, 0, 0, 0, time.UTC)
	in := InventoryInput{Window: time.Hour, StaleAfter: time.Minute, Processes: []ProcessEvidence{{PID: 10, Name: "codex.exe", StartedAt: now.Add(-time.Minute), CommandLine: "codex"}, {PID: 11, Name: "claude.exe", StartedAt: now.Add(-time.Minute), CommandLine: "claude -p"}}, Registrations: []sessionregistry.Record{{Schema: sessionregistry.Schema, RegistrationID: "root", RootRegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "a", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex", PID: 10, ProcessStartedAt: now.Add(-time.Minute)}, State: sessionregistry.StateActive, CreatedAt: now.Add(-2 * time.Minute)}, {Schema: sessionregistry.Schema, RegistrationID: "child", ParentRegistrationID: "root", ParentAttemptID: "a", RootRegistrationID: "root", RootIssue: "6458", TaskID: "issue-6458", AttemptID: "b", LaunchKind: "subagent", Identity: sessionregistry.Identity{Runtime: "claude"}, State: sessionregistry.StateCompleted, CreatedAt: now.Add(-time.Minute), TerminalAt: now, WitnessRef: "commit:x"}}}
	got := ReconcileInventory(in, now)
	if got.Counts.Registrations.Active != 1 || got.Counts.Registrations.Terminal != 1 || got.Counts.Registrations.UnregisteredObserved != 1 {
		t.Fatalf("counts=%+v", got.Counts.Registrations)
	}
	if len(got.Registrations) != 2 || !got.Registrations[0].ProcessMatched || got.Registrations[1].Health != "TERMINAL" {
		t.Fatalf("registrations=%+v", got.Registrations)
	}
	if len(got.UnregisteredObserved) != 1 || got.UnregisteredObserved[0].Process.PID != 11 {
		t.Fatalf("unregistered=%+v", got.UnregisteredObserved)
	}
	var out bytes.Buffer
	RenderInventory(&out, got)
	for _, want := range []string{"REGISTRATIONS total=2", "root", "child", "UNREGISTERED_OBSERVED"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, out.String())
		}
	}
}

func TestInventoryPIDReuseRequiresStartIdentity(t *testing.T) {
	now := time.Now().UTC()
	in := InventoryInput{Window: time.Hour, StaleAfter: time.Minute, Processes: []ProcessEvidence{{PID: 7, Name: "codex.exe", StartedAt: now}}, Registrations: []sessionregistry.Record{{Schema: sessionregistry.Schema, RegistrationID: "old", RootRegistrationID: "old", AttemptID: "a", LaunchKind: "guarded_tui", Identity: sessionregistry.Identity{Runtime: "codex", PID: 7, ProcessStartedAt: now.Add(-time.Hour)}, State: sessionregistry.StateActive, CreatedAt: now.Add(-time.Hour)}}}
	got := ReconcileInventory(in, now)
	if got.Registrations[0].ProcessMatched || len(got.UnregisteredObserved) != 1 {
		t.Fatalf("pid reuse incorrectly matched: %+v %+v", got.Registrations, got.UnregisteredObserved)
	}
}
