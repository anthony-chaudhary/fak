package harnesskit

import (
	"reflect"
	"testing"
)

func TestCompatibilityNegotiationIsDeterministicAndAcceptsNMinusOne(t *testing.T) {
	builder := BuilderContract{ContractVersion: ContractVersion, Requirements: []CapabilityRequirement{
		{Name: "tools.invoke", MinRevision: 1, MaxRevision: 2, Status: StatusStable},
		{Name: "events.trace", MinRevision: 1, MaxRevision: 1, Optional: true},
	}}
	host := RuntimeContract{ContractVersion: ContractVersion, Capabilities: []CapabilityOffer{
		{Name: "unrelated", Revision: 9, Status: StatusExperimental},
		{Name: "tools.invoke", Revision: 2, Status: StatusStable},
	}}
	first := NegotiateCompatibility(builder, host)
	second := NegotiateCompatibility(
		BuilderContract{ContractVersion: ContractVersion, Requirements: []CapabilityRequirement{builder.Requirements[1], builder.Requirements[0]}},
		RuntimeContract{ContractVersion: ContractVersion, Capabilities: []CapabilityOffer{host.Capabilities[1], host.Capabilities[0]}},
	)
	firstJSON, _ := first.JSON()
	secondJSON, _ := second.JSON()
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("negotiation depends on declaration order:\n%s\n%s", firstJSON, secondJSON)
	}
	if !first.Compatible {
		t.Fatalf("N-1 range was refused: %s", first.Error())
	}
	if len(first.Outcomes) != 2 || first.Outcomes[0].Name != "events.trace" || first.Outcomes[0].Compatible || first.Outcomes[0].Required || first.Outcomes[0].Reason != ReasonCapabilityAbsent {
		t.Fatalf("optional absence was not preserved: %+v", first.Outcomes)
	}
}

func TestCompatibilityRefusesInvalidDuplicateAndStrictRequirements(t *testing.T) {
	tests := []struct {
		name    string
		builder BuilderContract
		host    RuntimeContract
		want    CompatibilityReason
	}{
		{name: "invalid range", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 2, MaxRevision: 1}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 2, Status: StatusStable}), want: ReasonInvalidRequirement},
		{name: "invalid optional range", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 2, MaxRevision: 1, Optional: true}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 2, Status: StatusStable}), want: ReasonInvalidRequirement},
		{name: "invalid revision", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 0, Status: StatusStable}), want: ReasonInvalidOffer},
		{name: "invalid unrelated offer", builder: contractWith(), host: hostWith(CapabilityOffer{Name: "unused", Revision: 0, Status: StatusStable}), want: ReasonInvalidOffer},
		{name: "duplicate requirement", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}, CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 1, Status: StatusStable}), want: ReasonDuplicateRequirement},
		{name: "duplicate offer", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 1, Status: StatusStable}, CapabilityOffer{Name: "tools", Revision: 2, Status: StatusStable}), want: ReasonDuplicateOffer},
		{name: "required absent", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}), host: hostWith(), want: ReasonCapabilityAbsent},
		{name: "too new", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 3, Status: StatusStable}), want: ReasonRevisionAboveMax},
		{name: "status", builder: contractWith(CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 2, Status: StatusStable}), host: hostWith(CapabilityOffer{Name: "tools", Revision: 2, Status: StatusExperimental}), want: ReasonStatusMismatch},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			report := NegotiateCompatibility(tc.builder, tc.host)
			if report.Compatible {
				t.Fatalf("incompatible declaration accepted: %+v", report)
			}
			found := false
			for _, outcome := range report.Outcomes {
				found = found || outcome.Reason == tc.want
			}
			for _, issue := range report.Issues {
				found = found || issue.Reason == tc.want
			}
			if !found {
				t.Fatalf("reason = %+v, want %s", report.Outcomes, tc.want)
			}
		})
	}
}

func TestPublicCompatibilityContractNamesCanonicalFormats(t *testing.T) {
	contract := PublicCompatibilityContract()
	if contract.ReportSchema != CompatibilityReportSchema || contract.DiffSchema != ContractDiffSchema || contract.PlanSchema != UpgradePlanSchema {
		t.Fatalf("machine schemas = %+v", contract)
	}
	if !reflect.DeepEqual(contract.Statuses, []CapabilityStatus{StatusStable, StatusExperimental, StatusDeprecated}) || contract.Absence == "" || contract.Planning == "" {
		t.Fatalf("compatibility vocabulary = %+v", contract)
	}
}

func TestDeprecatedCapabilityRequiresUsableReplacement(t *testing.T) {
	builder := contractWith(CapabilityRequirement{Name: "tools.v1", MinRevision: 1, MaxRevision: 1})
	invalid := hostWith(CapabilityOffer{Name: "tools.v1", Revision: 1, Status: StatusDeprecated, Deprecation: &Deprecation{Replacement: "tools.v2", RemovalHorizon: "2027-01"}})
	if got := NegotiateCompatibility(builder, invalid); got.Compatible || got.Outcomes[0].Reason != ReasonInvalidDeprecation {
		t.Fatalf("unavailable replacement accepted: %+v", got)
	}
	valid := hostWith(
		CapabilityOffer{Name: "tools.v1", Revision: 1, Status: StatusDeprecated, Deprecation: &Deprecation{Replacement: "tools.v2", RemovalHorizon: "2027-01"}},
		CapabilityOffer{Name: "tools.v2", Revision: 1, Status: StatusStable},
	)
	if got := NegotiateCompatibility(builder, valid); !got.Compatible {
		t.Fatalf("validated deprecation refused: %s", got.Error())
	}
}

func TestDiffContractsClassifiesSemanticChanges(t *testing.T) {
	before := hostWith(
		CapabilityOffer{Name: "add-status", Revision: 1, Status: StatusExperimental},
		CapabilityOffer{Name: "behavior", Revision: 1, Status: StatusStable, Semantics: "old"},
		CapabilityOffer{Name: "removed", Revision: 1, Status: StatusStable},
	)
	after := hostWith(
		CapabilityOffer{Name: "add-status", Revision: 1, Status: StatusStable},
		CapabilityOffer{Name: "added", Revision: 1, Status: StatusExperimental},
		CapabilityOffer{Name: "behavior", Revision: 2, Status: StatusStable, Semantics: "new"},
	)
	diff := DiffContracts(before, after)
	classes := map[ChangeClass]bool{}
	for _, change := range diff.Changes {
		classes[change.Class] = true
	}
	if !reflect.DeepEqual(classes, map[ChangeClass]bool{ChangeAdditive: true, ChangeBehavioral: true, ChangeBreaking: true}) {
		t.Fatalf("semantic classes = %+v; changes=%+v", classes, diff.Changes)
	}
	first, _ := diff.JSON()
	second, _ := DiffContracts(RuntimeContract{ContractVersion: before.ContractVersion, Capabilities: reverseOffers(before.Capabilities)}, RuntimeContract{ContractVersion: after.ContractVersion, Capabilities: reverseOffers(after.Capabilities)}).JSON()
	if string(first) != string(second) {
		t.Fatalf("diff is not deterministic:\n%s\n%s", first, second)
	}
}

func TestUpgradePlanBlocksBreakageAndSurfacesOptionalAndDeprecation(t *testing.T) {
	builder := contractWith(
		CapabilityRequirement{Name: "required", MinRevision: 1, MaxRevision: 2},
		CapabilityRequirement{Name: "optional", MinRevision: 1, MaxRevision: 1, Optional: true},
	)
	current := hostWith(CapabilityOffer{Name: "required", Revision: 1, Status: StatusStable})
	target := hostWith(
		CapabilityOffer{Name: "legacy", Revision: 1, Status: StatusDeprecated, Deprecation: &Deprecation{Replacement: "replacement", RemovalHorizon: "2027-01"}},
		CapabilityOffer{Name: "replacement", Revision: 1, Status: StatusStable},
	)
	plan := PlanUpgrade(builder, current, target)
	if plan.Allowed {
		t.Fatalf("breaking target was allowed: %+v", plan)
	}
	want := map[UpgradeStepCode]bool{StepBlockBreaking: false, StepBlockRequired: false, StepOptionalGap: false, StepReplaceLegacy: false}
	for _, step := range plan.Steps {
		if _, ok := want[step.Code]; ok {
			want[step.Code] = true
		}
	}
	for code, found := range want {
		if !found {
			t.Errorf("plan missing %s: %+v", code, plan.Steps)
		}
	}
	first, _ := plan.JSON()
	second, _ := PlanUpgrade(builder, current, target).JSON()
	if string(first) != string(second) {
		t.Fatal("upgrade plan is not deterministic")
	}
}

func TestContractMismatchIsNotGuessedCompatible(t *testing.T) {
	report := NegotiateCompatibility(BuilderContract{ContractVersion: "v1"}, RuntimeContract{ContractVersion: "v2"})
	if report.Compatible || report.ContractReason != ReasonContractMismatch {
		t.Fatalf("contract mismatch accepted: %+v", report)
	}
}

func TestInvalidDuplicateReportsRemainDeterministic(t *testing.T) {
	first := NegotiateCompatibility(
		contractWith(
			CapabilityRequirement{Name: "tools", MinRevision: 2, MaxRevision: 2},
			CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 1, Optional: true},
		),
		hostWith(
			CapabilityOffer{Name: "tools", Revision: 2, Status: StatusStable},
			CapabilityOffer{Name: "tools", Revision: 1, Status: StatusExperimental},
		),
	)
	second := NegotiateCompatibility(
		contractWith(
			CapabilityRequirement{Name: "tools", MinRevision: 1, MaxRevision: 1, Optional: true},
			CapabilityRequirement{Name: "tools", MinRevision: 2, MaxRevision: 2},
		),
		hostWith(
			CapabilityOffer{Name: "tools", Revision: 1, Status: StatusExperimental},
			CapabilityOffer{Name: "tools", Revision: 2, Status: StatusStable},
		),
	)
	a, _ := first.JSON()
	b, _ := second.JSON()
	if string(a) != string(b) {
		t.Fatalf("invalid canonical input produced order-dependent reports:\n%s\n%s", a, b)
	}
}

func contractWith(requirements ...CapabilityRequirement) BuilderContract {
	return BuilderContract{ContractVersion: ContractVersion, Requirements: requirements}
}

func hostWith(offers ...CapabilityOffer) RuntimeContract {
	return RuntimeContract{ContractVersion: ContractVersion, Capabilities: offers}
}

func reverseOffers(input []CapabilityOffer) []CapabilityOffer {
	out := append([]CapabilityOffer(nil), input...)
	for left, right := 0, len(out)-1; left < right; left, right = left+1, right-1 {
		out[left], out[right] = out[right], out[left]
	}
	return out
}
