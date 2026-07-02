package dogfoodscore

// The CHAIN axis scores the outsider-path half of the dogfood loop: the
// recent-feature packet (tools/recent_feature_dogfood.py) probes fak the way an
// outside human or agent would, writes its findings under
// .fak/recent-feature-dogfood/<stamp>/report.json, and `fak dogfood-issues`
// bridges the report's ACTION findings into deduped tracker issues, leaving an
// issues-sync.json receipt beside the report. The law this axis enforces:
// friction must be found and filed BY THE LOOP, not stumbled into by an
// outsider. A stale packet or an unbridged report is chain debt, and because
// this card feeds the control-pane ratchet and the `improve-loops` /`tend`
// super loops, that debt lands on the worst-first walk instead of waiting for a
// human to remember.
//
// Once packet evidence exists on a host, the chain never passes on weaker
// evidence: a report without a receipt FAILS the bridge rung (unverified is not
// verified-clean). A host with NO packet evidence at all (a fresh clone, the CI
// ratchet runner) is honestly UNSCORED instead: both rungs report their failure
// with the producing command but stay SOFT, and Build excludes the axis from the
// composite — hard-failing every clean checkout would red the shared control-pane
// ratchet with unretirable debt rather than drive the loop (the fresh-clone
// pressure surface is the dogfood.yml daily gate+bridge, which runs the packet
// itself). The receipt's name/schema/fields are owned by internal/dogfoodissues;
// this package reads them by their stable JSON shape (pinned by paired tests in
// both packages) so the two stay decoupled at the import graph.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

const (
	// DefaultChainWindowHours is how fresh the newest packet report must be for
	// chain_packet_fresh to hold. CI runs the packet daily; a week of silence
	// means the outsider-path loop has gone dark on this host.
	DefaultChainWindowHours = 168
	chainEvidenceRel        = ".fak/recent-feature-dogfood"
	chainReceiptName        = "issues-sync.json"
)

// ChainEvidence is the folded outsider-path evidence on this host.
type ChainEvidence struct {
	EvidenceDir     string  `json:"evidence_dir"`
	Reports         int     `json:"reports"`
	NewestReport    string  `json:"newest_report,omitempty"`
	NewestAgeHours  float64 `json:"newest_age_hours,omitempty"`
	ReceiptPresent  bool    `json:"receipt_present"`
	ReceiptFresh    bool    `json:"receipt_fresh"`
	ReceiptMode     string  `json:"receipt_mode,omitempty"`
	PendingCreates  int     `json:"pending_creates"`
	SyncedOK        int     `json:"synced_ok"`
	SyncedFailed    int     `json:"synced_failed"`
	ReceiptParseErr string  `json:"receipt_parse_err,omitempty"`
}

// scanChain folds the packet evidence dir: newest report by mtime, and the
// bridge receipt (if any) sitting beside it.
func scanChain(root string, now time.Time) ChainEvidence {
	dir := filepath.Join(root, filepath.FromSlash(chainEvidenceRel))
	ce := ChainEvidence{EvidenceDir: dir}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ce
	}
	var newest string
	var newestMtime time.Time
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		p := filepath.Join(dir, e.Name(), "report.json")
		info, err := os.Stat(p)
		if err != nil {
			continue
		}
		ce.Reports++
		if newest == "" || info.ModTime().After(newestMtime) {
			newest, newestMtime = p, info.ModTime()
		}
	}
	if newest == "" {
		return ce
	}
	ce.NewestReport = newest
	ce.NewestAgeHours = now.Sub(newestMtime).Hours()
	if ce.NewestAgeHours < 0 {
		ce.NewestAgeHours = 0
	}

	receiptPath := filepath.Join(filepath.Dir(newest), chainReceiptName)
	rinfo, err := os.Stat(receiptPath)
	if err != nil {
		return ce
	}
	ce.ReceiptPresent = true
	ce.ReceiptFresh = !rinfo.ModTime().Before(newestMtime)
	raw := readFile(receiptPath)
	var rec map[string]any
	if err := json.Unmarshal(raw, &rec); err != nil {
		ce.ReceiptParseErr = err.Error()
		return ce
	}
	ce.ReceiptMode = anyStr(rec["mode"])
	ce.PendingCreates = anyInt(rec["planned_creates"])
	ce.SyncedOK = anyInt(rec["synced_ok"])
	ce.SyncedFailed = anyInt(rec["synced_failed"])
	return ce
}

// chainResults grades the two chain rungs. With evidence present both are HARD:
// the packet loop going dark, or a report whose findings never reached the
// tracker, is exactly the "outsider stumbles into it first" failure this axis
// exists to prevent. With no evidence at all the rungs stay SOFT (see the
// package comment: honestly unscored, not falsely red).
func chainResults(ce ChainEvidence, windowHours int) []KPIResult {
	hard := ce.Reports > 0
	fresh := ce.Reports > 0 && ce.NewestAgeHours <= float64(windowHours)
	freshDetail := ""
	switch {
	case ce.Reports == 0:
		freshDetail = "no packet evidence under " + chainEvidenceRel + " on this host — run `make dogfood-recent` so friction is found by the loop, not by an outsider"
	case !fresh:
		freshDetail = "newest packet report is " + itoa(int(ce.NewestAgeHours)) + "h old (> " + itoa(windowHours) + "h window) — re-run `make dogfood-recent`"
	default:
		freshDetail = "newest packet report is " + itoa(int(ce.NewestAgeHours)) + "h old (within the " + itoa(windowHours) + "h window): " + filepath.Base(filepath.Dir(ce.NewestReport))
	}

	bridged := false
	bridgeDetail := ""
	switch {
	case ce.Reports == 0:
		bridgeDetail = "no packet report to bridge — run `make dogfood-recent`, then `fak dogfood-issues --live`"
	case !ce.ReceiptPresent:
		bridgeDetail = "the newest report has no bridge receipt (" + chainReceiptName + ") — its ACTION findings never reached the tracker; run `fak dogfood-issues --live` (files deduped issues) or `--fetch-existing` (verifies they are already tracked)"
	case ce.ReceiptParseErr != "":
		bridgeDetail = "bridge receipt is unreadable (" + ce.ReceiptParseErr + ") — re-run `fak dogfood-issues --live`"
	case !ce.ReceiptFresh:
		bridgeDetail = "bridge receipt predates the newest report — re-run `fak dogfood-issues --live`"
	case ce.ReceiptMode == "live" && ce.SyncedFailed == 0:
		bridged = true
		bridgeDetail = itoa(ce.SyncedOK) + " action(s) filed/updated as deduped tracker issues (receipt mode live)"
	case ce.ReceiptMode == "live":
		bridgeDetail = itoa(ce.SyncedFailed) + " gh sync failure(s) in the bridge receipt — re-run `fak dogfood-issues --live`"
	case ce.ReceiptMode == "fetch-existing" && ce.PendingCreates == 0:
		bridged = true
		bridgeDetail = "all extracted actions are already tracked (receipt mode fetch-existing, 0 pending creates)"
	case ce.ReceiptMode == "fetch-existing":
		bridgeDetail = itoa(ce.PendingCreates) + " unfiled action(s) in the bridge receipt — run `fak dogfood-issues --live`"
	default:
		bridgeDetail = "unrecognized bridge receipt mode " + strconv.Quote(ce.ReceiptMode) + " — re-run `fak dogfood-issues --live`"
	}

	return []KPIResult{
		result("chain_packet_fresh", "chain", hard, 2,
			"the recent-feature dogfood packet has run recently on this host", fresh, freshDetail),
		result("chain_actions_bridged", "chain", hard, 2,
			"the newest packet report's ACTION findings are bridged to tracker issues", bridged, bridgeDetail),
	}
}
