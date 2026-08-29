package selfupdate

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func TestClassifyCheck(t *testing.T) {
	roleDrift := func(role selfinstall.Role) selfinstall.AuditPartition {
		a := selfinstall.Audit{Divergent: []selfinstall.Role{role}}
		return a.Partition()
	}
	cases := []struct {
		name      string
		freshness binstamp.Freshness
		audit     selfinstall.AuditPartition
		want      CheckPosture
	}{
		{"current", binstamp.Fresh, selfinstall.AuditPartition{}, CheckPosture{StatusCurrent, "fak version"}},
		{"stale target", binstamp.Stale, selfinstall.AuditPartition{}, CheckPosture{StatusStale, "fak self-update"}},
		{"repairable worker drift", binstamp.Fresh, roleDrift(selfinstall.RoleWorker), CheckPosture{StatusDivergent, "fak self-update"}},
		{"audit-only gate drift", binstamp.Fresh, roleDrift(selfinstall.RoleGate), CheckPosture{StatusAttention, "fak self-update --check"}},
		{"dirty convergeable copy", binstamp.Fresh, selfinstall.Audit{Dirty: []selfinstall.Role{selfinstall.RolePath}}.Partition(), CheckPosture{StatusDivergent, "fak self-update"}},
		{"unattested audit-only copy", binstamp.Fresh, selfinstall.Audit{Unattested: []selfinstall.Role{selfinstall.RoleGate}}.Partition(), CheckPosture{StatusAttention, "fak self-update --check"}},
		{"stale target wins mixed drift", binstamp.Stale, selfinstall.Audit{Divergent: []selfinstall.Role{selfinstall.RoleGate, selfinstall.RoleWorker}}.Partition(), CheckPosture{StatusStale, "fak self-update"}},
		{"repairable drift wins mixed scopes", binstamp.Fresh, selfinstall.Audit{Divergent: []selfinstall.Role{selfinstall.RoleGate, selfinstall.RoleWorker}}.Partition(), CheckPosture{StatusDivergent, "fak self-update"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyCheck(tc.freshness, tc.audit); got != tc.want {
				t.Fatalf("ClassifyCheck() = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestClassifyInstallSeparatesCompletionFromAuditAttention(t *testing.T) {
	auditOnly := selfinstall.Audit{Dirty: []selfinstall.Role{selfinstall.RoleGate}}.Partition()
	if got := ClassifyInstall(auditOnly); !got.Completed || !got.AuditOnlyAttention {
		t.Fatalf("audit-only drift posture = %+v, want completed with attention", got)
	}
	repairable := selfinstall.Audit{Unattested: []selfinstall.Role{selfinstall.RolePath}}.Partition()
	if got := ClassifyInstall(repairable); got.Completed || got.AuditOnlyAttention {
		t.Fatalf("repairable drift posture = %+v, want incomplete without audit-only attention", got)
	}
}
