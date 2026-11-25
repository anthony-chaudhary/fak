package fleetaccounts

import (
	"path/filepath"
	"strings"
)

// WorkerCheck answers fleet_accounts.is_worker — "does this config-dir basename classify as a
// worker?" — for many basenames against one read of the config-home registry. Unlike
// Discover it needs no account dir on disk, so it also screens throttle/auth entries carried
// for a dir that has since been removed.
type WorkerCheck struct {
	home       string
	configHome string
	pol        Policy
	idx        accountsRegistryIndex
}

// NewWorkerCheck loads the config-home registry once for repeated IsWorker calls.
func NewWorkerCheck(home, configHome string, pol Policy) WorkerCheck {
	return WorkerCheck{home: home, configHome: configHome, pol: pol, idx: indexAccountsRegistry(loadAccountsRegistry(home))}
}

// IsWorker reports whether account is offered as a worker: not .DELETED, not tombstoned in
// the config-home registry, not excluded by policy, and inside any include_only allowlist.
func (w WorkerCheck) IsWorker(account string) bool {
	if strings.Contains(strings.ToLower(account), ".deleted") {
		return false
	}
	tag := AccountTag(account)
	product := AccountProduct(account)
	root := w.home
	if product == "opencode" {
		root = w.configHome
	}
	acctDir := filepath.Join(root, account)
	var id Identity
	if product == "claude" {
		id = ReadAccountIdentity(acctDir)
	}
	if accountsRegistryExclusion(acctDir, product, account, tag, id, w.idx) != "" {
		return false
	}
	if excludedMatch(tag, account, w.pol.Exclude) != "" {
		return false
	}
	var includeOnly []string
	for _, t := range w.pol.IncludeOnly {
		if t != "" {
			includeOnly = append(includeOnly, t)
		}
	}
	if len(includeOnly) == 0 {
		return true
	}
	for _, t := range includeOnly {
		if strings.Contains(strings.ToLower(tag), strings.ToLower(t)) {
			return true
		}
	}
	return false
}
