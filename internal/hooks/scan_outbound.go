package hooks

import "strings"

// scan_outbound.go — the outbound-payload leak fence (#2262, epic #2259). The PUBLIC_LEAK
// and SECRET_SHAPE gates guard what enters the REPO; an outbound Slack message leaves the
// box entirely, so it passes the same two scans before send. This exists because of the
// nightrun-ledger lesson: an internal hostname in a note tripped PUBLIC_LEAK only because
// a commit happened to carry it — a chat post has no commit, so the fence runs at the wire.

// ScanOutboundText applies the PUBLIC_LEAK needle/regex scan and the SECRET_SHAPE shape
// scan to an arbitrary outbound payload. Unlike ScanMessageNeedles it scans EVERY line —
// an outbound body has no git comment lines, scissors blocks, or identity trailers to
// exempt. root locates the optional gitignored private-needle sidecar ("" skips the
// sidecar; the base needles and shapes still apply). Findings carry the 1-based payload
// line number and no File (there is no file). An empty result means the payload may leave.
func ScanOutboundText(payload, root string) []Finding {
	needles := needlesWithSidecar(root)
	var findings []Finding
	for i, line := range strings.Split(payload, "\n") {
		findings = append(findings, publicLeakLineFindings(line, i+1, "", needles)...)
		for _, hit := range scanShapes(line) {
			findings = append(findings, Finding{Gate: "SECRET_SHAPE", Line: i + 1, Detail: "[" + hit.shape + "] " + hit.text})
		}
	}
	return findings
}
