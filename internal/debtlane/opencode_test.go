package debtlane

import (
	"strings"
	"testing"
)

func TestFormatDebtOpencodePrompt(t *testing.T) {
	lane := DebtLane{
		Lane:           "gateway",
		UnitOfWork:     "internal/gateway",
		Criticality:    CriticalityCore,
		Maturity:       6.0,
		TargetMaturity: 10.0,
		MaturityRung:   "integrated",
		TotalDebt:      12.0,
		NextAction:     "author benchmark and dogfood in runtime proof",
		Evidence: Evidence{
			HasCode:           true,
			HasTests:          true,
			Integrated:        true,
			Dogfooded:         false,
			Benchmarked:       false,
			ModularityDeficit: true,
			ExcessComments:    true,
		},
	}

	opts := OpencodeChatOptions{
		Variant:   "high",
		PerfFocus: true,
	}

	prompt := FormatDebtOpencodePrompt(lane, opts)

	// Invariants check
	if !strings.Contains(prompt, "Maturity Debt Lane: gateway (internal/gateway)") {
		t.Errorf("prompt missing lane title: %s", prompt)
	}
	if !strings.Contains(prompt, "Boundary Paths: internal/gateway/...") {
		t.Errorf("prompt missing boundary paths: %s", prompt)
	}
	if !strings.Contains(prompt, "go test -v ./internal/gateway/...") {
		t.Errorf("prompt missing test command: %s", prompt)
	}
	if !strings.Contains(prompt, "[PERFORMANCE BENCHMARK]") {
		t.Errorf("prompt missing performance benchmark deliverable: %s", prompt)
	}
	if !strings.Contains(prompt, "[RUNTIME PROOF / DOGFOOD]") {
		t.Errorf("prompt missing runtime proof deliverable: %s", prompt)
	}
	if !strings.Contains(prompt, "[MODULARITY DEBT]") {
		t.Errorf("prompt missing modularity debt deliverable: %s", prompt)
	}
	if !strings.Contains(prompt, "[COMMENT HYGIENE]") {
		t.Errorf("prompt missing comment hygiene deliverable: %s", prompt)
	}
	if !strings.Contains(prompt, "Leaf Worker Direct Execution") {
		t.Errorf("prompt missing leaf worker invariant: %s", prompt)
	}
	if !strings.Contains(prompt, "prohibiting nested") && !strings.Contains(prompt, "Prohibit calling the 'task' tool") {
		t.Errorf("prompt missing task tool prohibition: %s", prompt)
	}
	if !strings.Contains(prompt, "Autonomous Safe Git Landing") {
		t.Errorf("prompt missing safe git landing: %s", prompt)
	}
	if !strings.Contains(prompt, "fak sync push") {
		t.Errorf("prompt missing sync push: %s", prompt)
	}
	if !strings.Contains(prompt, "<!-- fak:progress") {
		t.Errorf("prompt missing progress protocol: %s", prompt)
	}
}

func TestBuildDebtOpencodeChat(t *testing.T) {
	lane := DebtLane{
		Lane:        "compute",
		UnitOfWork:  "internal/compute",
		Criticality: CriticalityCore,
		Maturity:    5.0,
	}

	opts := OpencodeChatOptions{
		Model:       "deepseek/deepseek-r1",
		Agent:       "worker",
		Variant:     "high",
		AutoApprove: true,
	}

	chat := BuildDebtOpencodeChat(lane, opts)
	if chat.Lane != "compute" {
		t.Errorf("expected lane compute, got %s", chat.Lane)
	}
	if !strings.Contains(chat.SessionTitle, "Debt: compute") {
		t.Errorf("unexpected session title: %s", chat.SessionTitle)
	}

	cmdStr := strings.Join(chat.Command, " ")
	if !strings.Contains(cmdStr, "opencode run") {
		t.Errorf("expected opencode run in command: %s", cmdStr)
	}
	if !strings.Contains(cmdStr, "--variant high") {
		t.Errorf("expected --variant high in command: %s", cmdStr)
	}
	if !strings.Contains(cmdStr, "--auto") {
		t.Errorf("expected auto-approve flag in command: %s", cmdStr)
	}
	if strings.Contains(cmdStr, "--dangerously-skip-permissions") {
		t.Errorf("command must use OpenCode's supported --auto flag: %s", cmdStr)
	}
	if !strings.Contains(cmdStr, "-m deepseek/deepseek-r1") {
		t.Errorf("expected model override in command: %s", cmdStr)
	}

	// Interactive check
	interactiveOpts := OpencodeChatOptions{Interactive: true}
	iChat := BuildDebtOpencodeChat(lane, interactiveOpts)
	iCmdStr := strings.Join(iChat.Command, " ")
	if !strings.Contains(iCmdStr, "opencode run -i") {
		t.Errorf("expected opencode run -i in interactive mode: %s", iCmdStr)
	}
}

func TestFormatDebtOpencodePromptDoesNotDemandRedundantBenchmark(t *testing.T) {
	lane := DebtLane{
		Lane:        "engine",
		UnitOfWork:  "internal/engine",
		Criticality: CriticalityCore,
		Evidence: Evidence{
			HasCode:     true,
			HasTests:    true,
			Integrated:  true,
			Benchmarked: true,
			Dogfooded:   true,
		},
	}

	prompt := FormatDebtOpencodePrompt(lane, OpencodeChatOptions{PerfFocus: true})
	if strings.Contains(prompt, "[PERFORMANCE BENCHMARK]") {
		t.Fatalf("already-benchmarked lane received redundant benchmark work: %s", prompt)
	}
	if strings.Contains(prompt, "[RUNTIME PROOF / DOGFOOD]") {
		t.Fatalf("already-dogfooded lane received redundant runtime-proof work: %s", prompt)
	}
}

func TestPerfFocusWavePrioritization(t *testing.T) {
	report := Report{
		Workspace: "/test/workspace",
		ProductionGrade: ProductionGrade{
			DenominatorPoints: 100.0,
			RealizedPoints:    50.0,
			GradePercent:      50.0,
			GradeLetter:       "F",
		},
		Lanes: []DebtLane{
			{
				Lane:           "peripheral_tool",
				UnitOfWork:     "internal/tool",
				Criticality:    CriticalityPeripheral,
				Weight:         1.0,
				Maturity:       1.0,
				TargetMaturity: 4.0,
				MaturityGap:    3.0,
				TotalDebt:      20.0, // High debt, but peripheral
				Evidence: Evidence{
					HasCode:  true,
					HasTests: true,
				},
			},
			{
				Lane:           "core_perf_engine",
				UnitOfWork:     "internal/engine",
				Criticality:    CriticalityCore,
				Weight:         3.0,
				Maturity:       5.0,
				TargetMaturity: 10.0,
				MaturityGap:    5.0,
				TotalDebt:      15.0, // Lower total debt, but core unbenchmarked performance hazard!
				Evidence: Evidence{
					HasCode:     true,
					HasTests:    true,
					Integrated:  true,
					Benchmarked: false, // unbenchmarked!
				},
			},
		},
	}

	// Without PerfFocus, peripheral_tool with higher total debt (20.0 vs 15.0) ranks first
	planStandard := PlanWaves(report, WavePlanOptions{
		WaveSize:  1,
		PerfFocus: false,
	})
	if len(planStandard.Waves) < 2 {
		t.Fatalf("expected at least 2 waves, got %d", len(planStandard.Waves))
	}
	if planStandard.Waves[0].Lanes[0].Lane != "peripheral_tool" {
		t.Errorf("expected standard wave 1 to prioritize higher total debt peripheral_tool, got %s",
			planStandard.Waves[0].Lanes[0].Lane)
	}

	// With PerfFocus, core_perf_engine with unbenchmarked core debt is prioritized ahead
	planPerf := PlanWaves(report, WavePlanOptions{
		WaveSize:         1,
		PerfFocus:        true,
		OpencodeCommands: true,
	})
	if len(planPerf.Waves) < 2 {
		t.Fatalf("expected at least 2 waves, got %d", len(planPerf.Waves))
	}
	if planPerf.Waves[0].Lanes[0].Lane != "core_perf_engine" {
		t.Errorf("expected perf-focus wave 1 to prioritize core_perf_engine, got %s",
			planPerf.Waves[0].Lanes[0].Lane)
	}

	// Verify OpenCode chats were attached
	if len(planPerf.Waves[0].OpencodeChats) != 1 {
		t.Fatalf("expected opencode chats to be attached to wave 0")
	}
	if len(planPerf.OpencodeCommands) != 2 {
		t.Fatalf("expected 2 opencode commands on plan, got %d", len(planPerf.OpencodeCommands))
	}
	if len(planPerf.Waves[0].Lanes[0].OpencodeCommand) == 0 {
		t.Errorf("expected lane OpencodeCommand to be populated")
	}
}

func TestPerformanceInterestSurcharges(t *testing.T) {
	bounds := DefaultBoundsAndLimits(CriticalityCore)

	// Core lane integrated but unbenchmarked and undogfooded
	evPerfHazard := Evidence{
		HasCode:         true,
		HasTests:        true,
		Integrated:      true,
		Benchmarked:     false,
		Dogfooded:       false,
		DependentsCount: 1,
	}

	interest := CalculateInterest(CriticalityCore, bounds, evPerfHazard, 2.0)
	drivers := strings.Join(interest.Drivers, "; ")

	if !strings.Contains(drivers, "unbenchmarked_core_perf_hazard") {
		t.Errorf("expected driver to contain unbenchmarked_core_perf_hazard, got: %s", drivers)
	}
	if !strings.Contains(drivers, "unproven_runtime_core_perf_hazard") {
		t.Errorf("expected driver to contain unproven_runtime_core_perf_hazard, got: %s", drivers)
	}

	// Verify 3x harsher health evaluation
	lane := DebtLane{
		Lane:        "core_runtime",
		Criticality: CriticalityCore,
		Maturity:    8.0,
		Evidence:    evPerfHazard,
		Interest:    interest,
	}
	health := EvaluateLaneHealth(lane)
	// Missing benchmark & dogfooding on Core must deduct 0.15 each and status cannot be healthy
	if health.Status == HealthHealthy {
		t.Errorf("core lane missing benchmarks/dogfooding cannot be healthy, got status: %s", health.Status)
	}
	if health.Score > 0.70 {
		t.Errorf("expected score to reflect 3x harsher deductions (<= 0.70), got %.2f", health.Score)
	}
}

func TestFormatShellCommand(t *testing.T) {
	if got := FormatShellCommand(nil); got != "" {
		t.Errorf("expected empty string for nil args, got %q", got)
	}
	if got := FormatShellCommand([]string{}); got != "" {
		t.Errorf("expected empty string for empty args, got %q", got)
	}

	args := []string{
		"opencode",
		"run",
		"--title",
		"Debt: gateway (internal/gateway)",
		"--variant",
		"high",
		"-m",
		"deepseek/deepseek-r1",
		"Maturity Debt Lane: gateway\n\nExecution Invariants:\n- go test",
		`commit with "quote" inside`,
	}

	cmdStr := FormatShellCommand(args)

	// Clean single-word args should not be quoted
	if !strings.Contains(cmdStr, "opencode run --title ") {
		t.Errorf("expected unquoted clean flags, got: %s", cmdStr)
	}
	// Spaces and parentheses should be wrapped in quotes
	if !strings.Contains(cmdStr, `"Debt: gateway (internal/gateway)"`) {
		t.Errorf("expected quoted title with parens, got: %s", cmdStr)
	}
	// Newlines should be wrapped in quotes
	if !strings.Contains(cmdStr, "\"Maturity Debt Lane: gateway\n\nExecution Invariants:\n- go test\"") {
		t.Errorf("expected quoted multi-line prompt, got: %s", cmdStr)
	}
	// Internal double quotes should be escaped
	if !strings.Contains(cmdStr, `"commit with \"quote\" inside"`) {
		t.Errorf("expected escaped quotes, got: %s", cmdStr)
	}
}
