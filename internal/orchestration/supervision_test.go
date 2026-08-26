package orchestration

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/supervisionpolicy"
)

func TestUltracodeRolesCarryIndependentSupervision(t *testing.T) {
	resolution, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, TaskSpec{ID: "restart-contract", WorkClass: WorkRigor}, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	if len(resolution.Resolved.Roles) < 2 {
		t.Fatalf("roles=%d", len(resolution.Resolved.Roles))
	}
	for _, role := range resolution.Resolved.Roles {
		spec := role.Supervision
		if spec.Owner != "orchestration" || spec.FaultDomain == "" || spec.Strategy != supervisionpolicy.StrategyOneForOne || spec.Restart != supervisionpolicy.RestartTransient || spec.Escalation != supervisionpolicy.EscalateChild {
			t.Fatalf("role %q supervision = %+v", role.ID, spec)
		}
	}
}
