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

// AutonomousActionBiasDirective defines the autonomous action-bias prompt contract
// prohibiting conversational pauses in unattended mode.
const AutonomousActionBiasDirective = "Autonomous action-bias: unattended mode is active. Never pause to ask clarifying or conversational questions when running headless, and instead prioritize minimal reversible action and execution."

// HeadlessActionBiasDirective is an alias for AutonomousActionBiasDirective.
const HeadlessActionBiasDirective = AutonomousActionBiasDirective

// AutonomousActionBiasContract is an alias for AutonomousActionBiasDirective.
const AutonomousActionBiasContract = AutonomousActionBiasDirective

// HeadlessActionBiasContract is an alias for AutonomousActionBiasDirective.
const HeadlessActionBiasContract = AutonomousActionBiasDirective

// AmbiguityResolutionDirective defines the prompt-level ambiguity resolution rule
// prioritizing minimal reversible action over hesitation, endless debate, or freeze.
const AmbiguityResolutionDirective = "Ambiguity resolution: when faced with ambiguity, choose the smallest reversible action, inspect/verify, rather than stalling, debating trade-offs endlessly, or halting."

// AmbiguityResolutionRule is an alias for AmbiguityResolutionDirective.
const AmbiguityResolutionRule = AmbiguityResolutionDirective

// AmbiguityResolutionContract is an alias for AmbiguityResolutionDirective.
const AmbiguityResolutionContract = AmbiguityResolutionDirective

// TestBreadthCalibrationDirective defines the prompt-level test breadth calibration directive
// instructing reasoning models to focus test authoring on concise, high-signal, atomic reproduction tests.
const TestBreadthCalibrationDirective = "Test breadth calibration: focus test authoring on concise, high-signal, atomic reproduction tests (1-3 focused assertions proving the defect or contract) rather than generating sprawling, redundant, over-engineered test matrices that burn output token limits or corrupt the working tree."

// TestBreadthCalibrationRule is an alias for TestBreadthCalibrationDirective.
const TestBreadthCalibrationRule = TestBreadthCalibrationDirective

// TestBreadthCalibrationContract is an alias for TestBreadthCalibrationDirective.
const TestBreadthCalibrationContract = TestBreadthCalibrationDirective

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
This profile never overrides explicit requirements, repository instructions, policy, security, compatibility, migrations, tests, diagnostics, or evidence.`,
	WorkProfilePonytailNativeMed: `Work profile: Ponytail-inspired, native, medium intensity.
Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task. When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.
Do not optimize for fewer lines alone. Preserve explicit requirements, repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, and evidence.`,
	WorkProfilePonytailNativeHigh: `Work profile: Ponytail-inspired, native, high intensity.
Actively resist accidental complexity. State the required outcome, test the simplest viable route, and prefer deletion or reuse over addition. Use configuration before code, project primitives before dependencies, and standard library before new abstractions. Add machinery only when the simpler rung cannot meet the requirement; stop once the smallest complete solution is witnessed.
Aggressiveness applies only to avoidable complexity. Never narrow requested scope or weaken repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, uncertainty reporting, evidence, or proof.`,
	WorkProfilePonytailHeadlessLow: "Work profile: Ponytail-inspired, headless, low intensity.\n" +
		AutonomousActionBiasDirective + "\n" +
		AmbiguityResolutionDirective + "\n" +
		TestBreadthCalibrationDirective + "\n" +
		"Before adding machinery, briefly check whether existing code, configuration, or deletion solves the task more simply. Prefer the smallest correct implementation.\n" +
		"This profile never overrides explicit requirements, repository instructions, policy, security, compatibility, migrations, tests, diagnostics, or evidence.",
	WorkProfilePonytailHeadlessMed: "Work profile: Ponytail-inspired, headless, medium intensity.\n" +
		AutonomousActionBiasDirective + "\n" +
		AmbiguityResolutionDirective + "\n" +
		TestBreadthCalibrationDirective + "\n" +
		"Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task. When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.\n" +
		"Do not optimize for fewer lines alone. Preserve explicit requirements, repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, and evidence.",
	WorkProfilePonytailHeadlessHigh: "Work profile: Ponytail-inspired, headless, high intensity.\n" +
		AutonomousActionBiasDirective + "\n" +
		AmbiguityResolutionDirective + "\n" +
		TestBreadthCalibrationDirective + "\n" +
		"Actively resist accidental complexity. State the required outcome, test the simplest viable route, and prefer deletion or reuse over addition. Use configuration before code, project primitives before dependencies, and standard library before new abstractions. Add machinery only when the simpler rung cannot meet the requirement; stop once the smallest complete solution is witnessed.\n" +
		"Aggressiveness applies only to avoidable complexity. Never narrow requested scope or weaken repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, uncertainty reporting, evidence, or proof.",
}

// WorkProfileNames returns the closed set accepted by DescribeWorkProfile.
func WorkProfileNames() []string {
	return []string{
		WorkProfileStandard, "off",
		"ponytail:low", "ponytail:medium", "ponytail:high",
		WorkProfilePonytailNativeLow, WorkProfilePonytailNativeMed, WorkProfilePonytailNativeHigh,
		"headless", "ponytail:headless",
		"headless:low", "headless:medium", "headless:high",
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
	if raw == "headless" || raw == "ponytail:headless" {
		raw = WorkProfilePonytailHeadlessMed
	} else if strings.HasPrefix(raw, "headless:") && strings.Count(raw, ":") == 1 {
		raw = "ponytail:headless:" + strings.TrimPrefix(raw, "headless:")
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
