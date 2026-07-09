package logvault

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// observe_metrics.go — the /metrics surface for vault observability (#2455): the
// three fak_logvault_* gauges, rendered from the observe.go footprint fold plus a
// Verify pass, so a scrape shows last-capture age, footprint, and verify
// mismatches where operators already look. This is the WITNESSED sibling of the
// `fak logvault du` verb (same Footprint fold) with integrity layered on top.

// RenderLogvaultGauges renders the three fak_logvault_* gauges in Prometheus/
// OpenMetrics text form from already-computed observability values. It is PURE
// (no I/O, no clock): callers supply the footprint fold (NewestCaptureUnixNano /
// TotalBytes), the verify outcome, and the scrape clock. Per the conflation law
// each gauge's HELP text declares the value WITNESSED — fak computed it from its
// own hash-chained manifest and a mirror re-hash, never a self-reported counter —
// so a scrape can never mistake it for an unverified live number.
func RenderLogvaultGauges(nowUnixNano, lastCaptureUnixNano, vaultBytes int64, verifyMismatches int, chainBroken bool) string {
	var b strings.Builder

	// -1 is the honest "never captured" sentinel: a vault with no completed
	// capture has no age, and reporting 0 (fresh) would invert the alert.
	ageSec := -1.0
	if lastCaptureUnixNano > 0 {
		ageSec = float64(nowUnixNano-lastCaptureUnixNano) / 1e9
		if ageSec < 0 {
			ageSec = 0 // clock skew: never report a negative age
		}
	}
	b.WriteString("# HELP fak_logvault_last_capture_age_seconds Seconds since the newest successful logvault capture. WITNESSED: newest content-bearing row in fak's own hash-chained vault manifest (-1 = the vault has never captured).\n")
	b.WriteString("# TYPE fak_logvault_last_capture_age_seconds gauge\n")
	b.WriteString("fak_logvault_last_capture_age_seconds ")
	b.WriteString(strconv.FormatFloat(ageSec, 'g', -1, 64))
	b.WriteByte('\n')

	b.WriteString("# HELP fak_logvault_vault_bytes Current captured mirror footprint in bytes. WITNESSED: sum of the captured content sizes recorded in fak's own vault manifest (excludes superseded .history/ versions).\n")
	b.WriteString("# TYPE fak_logvault_vault_bytes gauge\n")
	fmt.Fprintf(&b, "fak_logvault_vault_bytes %d\n", vaultBytes)

	// A broken chain means the whole vault is unverifiable: report at least one
	// mismatch so a scrape never reads 0 (intact) on a vault that failed to verify.
	mismatches := verifyMismatches
	if chainBroken && mismatches < 1 {
		mismatches = 1
	}
	b.WriteString("# HELP fak_logvault_verify_mismatches Mirror/anchor problems from the most recent logvault verify pass (0 = intact). WITNESSED: fak re-hashed its own mirrors against the manifest; a broken manifest chain renders as >=1.\n")
	b.WriteString("# TYPE fak_logvault_verify_mismatches gauge\n")
	fmt.Fprintf(&b, "fak_logvault_verify_mismatches %d\n", mismatches)

	return b.String()
}

// MetricsText is the /metrics provider core: it reads the manifest, folds the
// footprint (Footprint), runs a bounded Verify (verifySample mirrors re-hashed,
// 0 = all — the same knob `fak logvault verify -sample` uses), and renders the
// three #2455 gauges. nowUnixNano anchors the last-capture age; pass the
// scrape-time clock.
//
// A missing or empty vault renders the family with a -1 "never captured" age and
// zero footprint/mismatches — the valid-empty posture, never an error. A broken
// manifest chain renders as chainBroken (>=1 mismatch), NOT an error: the whole
// point is to make "this vault is NOT intact" scrapeable. Only a manifest that
// exists but cannot be read returns an error.
func (v *Vault) MetricsText(verifySample int, nowUnixNano int64) (string, error) {
	rows, err := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if err != nil {
		return "", err
	}
	fps := Footprint(rows)
	mismatches, chainBroken := 0, false
	if _, _, problems, verr := v.Verify(verifySample); verr != nil {
		chainBroken = true
	} else {
		mismatches = len(problems)
	}
	return RenderLogvaultGauges(nowUnixNano, NewestCaptureUnixNano(fps), TotalBytes(fps), mismatches, chainBroken), nil
}
