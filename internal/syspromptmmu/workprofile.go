package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// WorkProfileEnvVar selects an optional implementation-policy overlay for the owned agent loop.
// It is deliberately separate from StyleEnvVar: work policy and response shape compose but
// neither selection implies the other.
const WorkProfileEnvVar = "FAK_WORK_PROFILE"

const (
	WorkProfileStandard             = "standard"
	WorkProfileDefault              = "ponytail:medium"
	WorkProfilePonytailNativeLow    = "ponytail:native:low"
	WorkProfilePonytailNativeMed    = "ponytail:native:medium"
	WorkProfilePonytailNativeHigh   = "ponytail:native:high"
	WorkProfilePonytailHeadlessLow  = "ponytail:headless:low"
	WorkProfilePonytailHeadlessMed  = "ponytail:headless:medium"
	WorkProfilePonytailHeadlessHigh = "ponytail:headless:high"
)

// AutonomousActionBiasDirective defines the prompt contract prohibiting conversational pauses in unattended/headless mode (#11519).
const AutonomousActionBiasDirective = "Never stop to ask interactive questions; resolve ambiguities by minimal reversible action and checkable artifacts."

// AutonomousActionBiasFragment aliases AutonomousActionBiasDirective.
const AutonomousActionBiasFragment = AutonomousActionBiasDirective

// TestBreadthCalibrationDirective instructs agents to focus on atomic reproduction tests and prevent token exhaustion (#11520).
const TestBreadthCalibrationDirective = "Calibrate test breadth: focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, and prevent token exhaustion."

// TestBreadthCalibrationFragment aliases TestBreadthCalibrationDirective.
const TestBreadthCalibrationFragment = TestBreadthCalibrationDirective

// AmbiguityResolutionDirective instructs agents to prioritize minimal reversible action over freeze (#11521).
const AmbiguityResolutionDirective = "When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation."

// AmbiguityResolutionFragment aliases AmbiguityResolutionDirective.
const AmbiguityResolutionFragment = AmbiguityResolutionDirective

// DivideAndConquerDirective instructs agents to decompose substantive tasks into atomic units or isolated subagents.
const DivideAndConquerDirective = "Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results."

// DivideAndConquerFragment aliases DivideAndConquerDirective.
const DivideAndConquerFragment = DivideAndConquerDirective

// AutonomousGitSyncPushDirective instructs agents to carry work through to verified completion by default.
const AutonomousGitSyncPushDirective = "Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted."

// AutonomousGitSyncPushFragment aliases AutonomousGitSyncPushDirective.
const AutonomousGitSyncPushFragment = AutonomousGitSyncPushDirective

// WorkProfileReadout is the reproducible record of a selected implementation-policy overlay.
type WorkProfileReadout struct {
	Profile        string
	Family         string
	Implementation string
	Intensity      string
	Known          bool
	Applied        bool
	Segment        string
	Witness        string
}

var workProfileFragments = map[string]string{
	WorkProfilePonytailNativeLow: `Work profile: Ponytail-inspired, native, low intensity.
Before adding machinery, briefly check whether existing code, configuration, or deletion solves the task more simply. Prefer the smallest correct implementation.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, and prevent token exhaustion.
This profile never overrides explicit requirements, repository instructions, policy, security, compatibility, migrations, tests, diagnostics, or evidence.`,
	WorkProfilePonytailNativeMed: `Work profile: Ponytail-inspired, native, medium intensity.
Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task. When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Calibrate test breadth: focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, and prevent token exhaustion.
Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results.
Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted.
Do not optimize for fewer lines alone. Preserve explicit requirements, repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, and evidence.`,
	WorkProfilePonytailNativeHigh: `Work profile: Ponytail-inspired, native, high intensity.
Actively resist accidental complexity. State the required outcome, test the simplest viable route, and prefer deletion or reuse over addition. Use configuration before code, project primitives before dependencies, and standard library before new abstractions. Add machinery only when the simpler rung cannot meet the requirement; stop once the smallest complete solution is witnessed.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Calibrate test breadth strictly: focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, avoid speculative matrix coverage, and prevent token exhaustion.
Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results.
Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted.
Aggressiveness applies only to avoidable complexity. Never narrow requested scope or weaken repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, uncertainty reporting, evidence, or proof.`,
	WorkProfilePonytailHeadlessLow: `Work profile: Ponytail-inspired, headless, low intensity.
Operate unattended with autonomous action bias: Never stop to ask interactive questions; resolve ambiguities by minimal reversible action and checkable artifacts.
Before adding machinery, briefly check whether existing code, configuration, or deletion solves the task more simply. Prefer the smallest correct implementation.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, and prevent token exhaustion.
Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results.
Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted.
This profile never overrides explicit requirements, repository instructions, policy, security, compatibility, migrations, tests, diagnostics, or evidence.`,
	WorkProfilePonytailHeadlessMed: `Work profile: Ponytail-inspired, headless, medium intensity.
Operate unattended with autonomous action bias: Never stop to ask interactive questions; resolve ambiguities by minimal reversible action and checkable artifacts.
Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task. When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Calibrate test breadth: focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, and prevent token exhaustion.
Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results.
Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted.
Do not optimize for fewer lines alone. Preserve explicit requirements, repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, and evidence.`,
	WorkProfilePonytailHeadlessHigh: `Work profile: Ponytail-inspired, headless, high intensity.
Operate unattended with autonomous action bias: Never stop to ask interactive questions; resolve ambiguities by minimal reversible action and checkable artifacts.
Actively resist accidental complexity. State the required outcome, test the simplest viable route, and prefer deletion or reuse over addition. Use configuration before code, project primitives before dependencies, and standard library before new abstractions. Add machinery only when the simpler rung cannot meet the requirement; stop once the smallest complete solution is witnessed.
When encountering ambiguity, prioritize minimal reversible action and test verification over freezing or speculation. Calibrate test breadth strictly: focus on single atomic reproduction tests, prohibit sprawling 20-test suites when 1 witness suffices, avoid speculative matrix coverage, and prevent token exhaustion.
Divide and conquer: decompose substantive or multi-component tasks into atomic single-concern units or isolated subagents; keep coordinator context clean, isolate file sets, and integrate only witnessed results.
Ship green work autonomously: synchronize safely with trunk before and during work, verify green with on-device tests, stage-and-commit by explicit path, and push verified commits unprompted.
Aggressiveness applies only to avoidable complexity. Never narrow requested scope or weaken repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, uncertainty reporting, evidence, or proof.`,
}

// WorkProfileNames returns the closed set accepted by DescribeWorkProfile.
func WorkProfileNames() []string {
	return []string{
		WorkProfileStandard, "off",
		"ponytail:low", "ponytail:medium", "ponytail:high",
		WorkProfilePonytailNativeLow, WorkProfilePonytailNativeMed, WorkProfilePonytailNativeHigh,
		"ponytail:headless", "ponytail:headless:low", "ponytail:headless:med",
		WorkProfilePonytailHeadlessLow, WorkProfilePonytailHeadlessMed, WorkProfilePonytailHeadlessHigh,
	}
}

// DescribeWorkProfile resolves a work-policy name. Unknown and upstream-original spellings
// fail safe to standard/off and Known=false; fak never substitutes native bytes for original.
func DescribeWorkProfile(name string) WorkProfileReadout {
	raw := strings.ToLower(strings.TrimSpace(name))
	if raw == "" || raw == "standard" || raw == "off" {
		return WorkProfileReadout{Profile: WorkProfileStandard, Family: "standard", Implementation: "native", Intensity: "off", Known: true}
	}
	if raw == "ponytail:headless" || raw == "ponytail:headless:med" {
		raw = WorkProfilePonytailHeadlessMed
	} else if strings.HasPrefix(raw, "ponytail:") && strings.Count(raw, ":") == 1 {
		raw = "ponytail:native:" + strings.TrimPrefix(raw, "ponytail:")
	}
	fragment, ok := workProfileFragments[raw]
	if !ok {
		return WorkProfileReadout{Profile: WorkProfileStandard, Family: "standard", Implementation: "native", Intensity: "off"}
	}
	parts := strings.Split(raw, ":")
	segment := "<fak:work-profile>\n" + fragment + "\n</fak:work-profile>"
	sum := sha256.Sum256([]byte(segment))
	return WorkProfileReadout{
		Profile: raw, Family: parts[0], Implementation: parts[1], Intensity: parts[2],
		Known: true, Applied: true, Segment: segment, Witness: "sha256:" + hex.EncodeToString(sum[:]),
	}
}

// WorkProfileFromEnv reads the operator selection through an injected lookup.
func WorkProfileFromEnv(getenv func(string) string) WorkProfileReadout {
	selected := ""
	if getenv != nil {
		selected = strings.TrimSpace(getenv(WorkProfileEnvVar))
	}
	if selected == "" {
		selected = WorkProfileDefault
	}
	return DescribeWorkProfile(selected)
}
