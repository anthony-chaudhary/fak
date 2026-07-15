package main

import (
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardaudit"
	"github.com/anthony-chaudhary/fak/internal/logvault"
)

// pruneGuardAuditTick mirrors guard journals into the hash-chained vault, reads
// that mirror back through the independent witness API, and only then applies
// bounded retention. Dispatch remains fail-open: vault contention or damage
// must retain evidence, not prevent a worker from launching.
func pruneGuardAuditTick(workspace string, now time.Time) int {
	vaultDir := os.Getenv("FAK_LOG_VAULT")
	if vaultDir == "" {
		vaultDir = filepath.Join(filepath.Dir(workspace), "fak-log-vault")
	}
	sourceRoot := filepath.Join(workspace, ".dispatch-runs")
	v := &logvault.Vault{
		Dir: vaultDir,
		Sources: []logvault.Source{{
			ID:       "dispatch-runs",
			Root:     sourceRoot,
			Includes: []string{"guard-audit/"},
		}},
	}
	if _, err := v.Capture(); err != nil {
		return 0
	}
	witnessed, err := v.WitnessedFiles("dispatch-runs")
	if err != nil {
		return 0
	}
	rep, err := guardaudit.Plan(workspace, vaultDir, now, guardaudit.DefaultMaxAge, guardaudit.DefaultMaxFiles, witnessed)
	if err != nil {
		return 0
	}
	if err := guardaudit.Apply(&rep); err != nil {
		return rep.GuardAuditPruned
	}
	return rep.GuardAuditPruned
}
