package watchdoghealth

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// health_triage.go applies the decenter-the-human doctrine at the watchdog-status seam.
// The digest's one NeedsAttention bit (health.go) trips whenever ANY monitor sits at or
// above the attention floor — DOWN, UNKNOWN, or GAVE_UP — and `fak watchdog status --check`
// exits 1 on it, so all three page an operator identically. But those three are not one job:
//
//   - DOWN is installed-but-dead with no heal recorded yet. There is a real autonomous
//     actor here — the autoheal restarts a dead monitor on its next tick with no person in
//     the loop — so DOWN is the fleet's to clear (TAKE_OBVIOUS), not a page. This running
//     actor is precisely why watchdoghealth earns the fold that fleet (no redeploy verb)
//     does not.
//   - UNKNOWN is a failed probe: liveness could not even be read. The state is knowable —
//     re-probe and look — but not known-bad, so it is a fresh-context evaluation, not a
//     reflexive page.
//   - GAVE_UP is the terminal class: the autoheal's restart budget is spent and the machine
//     has stopped trying. What remains is genuinely a person's call — keep intervening by
//     hand or accept the monitor down — a priority/authority decision the fleet cannot make
//     for itself. This is the irreducible HUMAN_RESIDUAL, and it is where the fold keeps the
//     page.
//
// PartitionAttention folds each attention-floor monitor's own remedy through
// internal/choicetriage and splits them: only a HUMAN_RESIDUAL disposition (GAVE_UP, or a
// monitor whose persisted heal-reason names an auth wall) waits on a person; DOWN and
// UNKNOWN are the fleet's to clear. It is the watchdog analogue of the fleet-pane and
// resume-stopped folds, and it soaks behind FAK_WATCHDOG_TRIAGE_GATE (read at the CLI): the
// default `--check` and readout are byte-for-byte unchanged until enforce, so the split is
// observable before it changes what an operator is paged for.

// attentionRemedy maps an attention-floor status to the concrete remedy the triage fold
// reasons about, phrased as a choicetriage.Signal Action. DOWN names the autoheal's runnable
// restart (a fleet action -> TakeObvious). UNKNOWN carries no remedy in hand — the probe
// failed — so it folds to a fresh-context re-probe rather than a page. GAVE_UP names the
// person's authority: with the restart budget spent, whether to keep intervening is a
// priority call a person holds (-> HumanResidual). A status below the attention floor has no
// entry here and never reaches the fold.
var attentionRemedy = map[Status]choicetriage.Signal{
	StatusDown: {
		Source:   "watchdog",
		Question: "a default monitor is down",
		Detail:   "installed but not alive, no restart recorded yet",
		Action:   "`fak watchdog heal` restarts it — the autoheal picks it up on the next tick",
	},
	StatusUnknown: {
		Source:   "watchdog",
		Question: "a default monitor's liveness is unknown",
		Detail:   "the probe itself failed; liveness could not be read — re-probe to learn the real state",
	},
	StatusGaveUp: {
		Severity: "decision",
		Source:   "watchdog",
		Question: "a default monitor has exhausted automatic recovery",
		Detail:   "the autoheal's restart budget is spent; whether to keep intervening — or accept it down — is a priority call a person holds",
		Action:   "decide whether to keep manually recovering the monitor, or accept it down",
	},
}

// AttentionItem is one attention-floor monitor's identity plus the concrete next move the
// triage fold named for it. Order within a bucket follows the digest's monitor order.
type AttentionItem struct {
	ID      string `json:"id"`
	Status  Status `json:"status"`
	Resolve string `json:"resolve"`
}

// triageMonitor folds one monitor's attention-floor status through the shared page-vs-act
// gate. The Signal is the status's canonical remedy, with the monitor's persisted heal-reason
// woven into the Detail so a GAVE_UP (or any) monitor whose real cause is an auth wall is
// surfaced to a person even where the status alone would not — the same conservative
// fail-toward-paging posture the sibling folds take.
func triageMonitor(h Health) choicetriage.Verdict {
	sig := attentionRemedy[h.Status]
	if r := strings.TrimSpace(h.LastReason); r != "" {
		sig.Detail = strings.TrimSpace(sig.Detail + " (last heal reason: " + r + ")")
	}
	return choicetriage.Triage(sig)
}

// PartitionAttention splits the digest's attention-floor monitors into the ones that
// genuinely wait on a person (GAVE_UP / an auth-walled monitor) and the ones the fleet
// clears itself (a DOWN monitor the autoheal restarts, an UNKNOWN monitor to re-probe). Pure
// and total: only monitors at or above the attention floor appear, healthy / not-installed /
// self-healing monitors never do, and an all-clear digest yields two empty buckets.
func PartitionAttention(d Digest) (needHuman, fleetClears []AttentionItem) {
	for _, h := range d.Monitors {
		if !h.NeedsAttention() {
			continue
		}
		v := triageMonitor(h)
		it := AttentionItem{ID: h.ID, Status: h.Status, Resolve: v.Resolve}
		if v.NeedsHuman {
			needHuman = append(needHuman, it)
		} else {
			fleetClears = append(fleetClears, it)
		}
	}
	return needHuman, fleetClears
}

// NeedsHumanAttention reports whether any monitor folds to a genuine human decision — the
// enforce-mode replacement for Digest.NeedsAttention as the `--check` exit condition. Under
// enforce, `fak watchdog status --check` exits 1 only on this (a GAVE_UP or auth-walled
// monitor), leaving DOWN (the autoheal's) and UNKNOWN (a re-probe) off the page.
func NeedsHumanAttention(d Digest) bool {
	needHuman, _ := PartitionAttention(d)
	return len(needHuman) > 0
}

// WatchdogTriageEnforced reports whether the decenter split is active for the given mode
// string. Only "enforce" (case-insensitive) flips the render and the --check condition;
// "warn", "" and anything else leave the digest's single NeedsAttention gate unchanged so
// the change can soak. Mirrors the enforce/warn switch every other seam reads.
func WatchdogTriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// AttentionTriageLine renders the decenter split as one extra readout line: the monitors
// that genuinely wait on a person (needs-you) vs the ones the fleet clears itself
// (fleet-clears), each with its per-status breakdown. It returns "" when nothing sits at the
// attention floor (no split to show), so the caller appends nothing. Rendered only under
// FAK_WATCHDOG_TRIAGE_GATE=enforce.
func AttentionTriageLine(d Digest) string {
	needHuman, fleetClears := PartitionAttention(d)
	if len(needHuman) == 0 && len(fleetClears) == 0 {
		return ""
	}
	return "attention-triage: needs-you=" + strconv.Itoa(len(needHuman)) + fmtItemStatuses(needHuman) +
		" fleet-clears=" + strconv.Itoa(len(fleetClears)) + fmtItemStatuses(fleetClears)
}

// fmtItemStatuses renders " (id=STATUS,...)" for a non-empty bucket, or "" when empty, so the
// readout names which monitors landed on each side and why.
func fmtItemStatuses(items []AttentionItem) string {
	if len(items) == 0 {
		return ""
	}
	parts := make([]string, 0, len(items))
	for _, it := range items {
		parts = append(parts, it.ID+"="+string(it.Status))
	}
	return " (" + strings.Join(parts, ",") + ")"
}

// TriageSelfcheck is the deterministic, no-I/O proof of the watchdog fold: a DOWN monitor is
// the autoheal's to restart, an UNKNOWN monitor is a fresh-context re-probe, a GAVE_UP
// monitor genuinely waits on a person, an auth-walled monitor is surfaced to a person even
// where its status alone would not, and healthy / not-installed / healing monitors appear in
// neither bucket. It is the witness the CLI surfaces as `fak watchdog selfcheck`.
func TriageSelfcheck() error {
	d := Fold([]Monitor{
		{ID: "mon-healthy", Installed: true, Alive: true},
		{ID: "mon-noinstall"},
		{ID: "mon-healing", Installed: true, Attempts: 1, MaxAttempts: 5},
		{ID: "mon-down", Installed: true},
		{ID: "mon-unknown", ProbeErr: true},
		{ID: "mon-gaveup", Installed: true, Attempts: 5, MaxAttempts: 5},
	})
	needHuman, fleetClears := PartitionAttention(d)

	if len(needHuman) != 1 || needHuman[0].ID != "mon-gaveup" {
		return fmt.Errorf("only a GAVE_UP monitor must wait on a person, got %v", needHuman)
	}
	if got := itemIDs(fleetClears); got != "mon-down,mon-unknown" {
		return fmt.Errorf("a DOWN monitor (autoheal) and an UNKNOWN monitor (re-probe) must be the fleet's to clear, got %q", got)
	}
	// An auth-walled DOWN monitor is surfaced to a person even though its status alone
	// (DOWN) would be the fleet's — the conservative fail-toward-paging rung.
	authWalled := Fold([]Monitor{{ID: "mon-auth", Installed: true, LastReason: "login required"}})
	nh, _ := PartitionAttention(authWalled)
	if len(nh) != 1 || nh[0].ID != "mon-auth" {
		return fmt.Errorf("a monitor whose heal-reason names an auth wall must wait on a person, got %v", nh)
	}
	// An all-healthy digest splits into nothing (no page, no fleet churn).
	if line := AttentionTriageLine(Fold([]Monitor{{ID: "ok", Installed: true, Alive: true}})); line != "" {
		return fmt.Errorf("an all-healthy digest must not surface a triage split, got %q", line)
	}
	if !NeedsHumanAttention(d) {
		return fmt.Errorf("NeedsHumanAttention must be true when a GAVE_UP monitor is present")
	}
	if !WatchdogTriageEnforced("enforce") || WatchdogTriageEnforced("") || WatchdogTriageEnforced("warn") {
		return fmt.Errorf("WatchdogTriageEnforced must flip only on \"enforce\"")
	}
	return nil
}

// itemIDs joins a bucket's monitor ids sorted, for a stable selfcheck compare independent of
// the digest's monitor order.
func itemIDs(items []AttentionItem) string {
	ids := make([]string, 0, len(items))
	for _, it := range items {
		ids = append(ids, it.ID)
	}
	sort.Strings(ids)
	return strings.Join(ids, ",")
}
