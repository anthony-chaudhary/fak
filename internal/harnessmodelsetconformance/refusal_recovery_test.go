package harnessmodelsetconformance_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

func TestTwoRoleModelSetConformanceRefusalsNameRecovery(t *testing.T) {
	t.Run("invalid inputs retain typed diagnostics", func(t *testing.T) {
		_, err := modelsetresolve.Resolve(harnessmodelset.Intent{}, modelinventory.Inventory{}, conformanceTime)
		var inputErr *modelsetresolve.InputError
		if !errors.As(err, &inputErr) || len(inputErr.IntentDiagnostics) == 0 || len(inputErr.InventoryDiagnostics) == 0 {
			t.Fatalf("Resolve error = %#v, want typed intent and inventory diagnostics", err)
		}
		if !strings.Contains(err.Error(), "intent diagnostics=") || !strings.Contains(err.Error(), "inventory diagnostics=") {
			t.Fatalf("Resolve error does not name recovery evidence: %q", err)
		}
	})

	t.Run("required role refusal names role and detailed rejection", func(t *testing.T) {
		intent := readIntent(t)
		inventory, _ := normalize(t, successObservations())
		inventory.Candidates = inventory.Candidates[:1]
		resolution, err := modelsetresolve.Resolve(intent, inventory, conformanceTime)
		var required *modelsetresolve.RequiredRolesError
		if !errors.As(err, &required) || len(required.RoleIDs) == 0 {
			t.Fatalf("Resolve error = %#v, want typed required-role refusal", err)
		}
		if !strings.Contains(err.Error(), required.RoleIDs[0]) || len(resolution.Rejections) == 0 {
			t.Fatalf("refusal omits role or recovery detail: err=%q resolution=%+v", err, resolution)
		}
		for _, rejection := range resolution.Rejections {
			if rejection.Constraint == "" || rejection.Remediation == "" {
				t.Fatalf("rejection lacks constraint/remediation: %+v", rejection)
			}
		}
	})

	t.Run("receipt parse refusal names structural repair", func(t *testing.T) {
		_, err := modelsetreceipt.ParseJSON([]byte(`{"schema":"wrong"}`))
		var validation *modelsetreceipt.ValidationError
		if !errors.As(err, &validation) || len(validation.Problems) == 0 {
			t.Fatalf("ParseJSON error = %#v, want typed validation problems", err)
		}
		if !strings.Contains(err.Error(), "schema") || !strings.Contains(err.Error(), "required") {
			t.Fatalf("receipt refusal does not name repair: %q", err)
		}
	})
}
