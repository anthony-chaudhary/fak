package main

import (
	"os"
	"strings"
)

// Print-time color for the live `fak info` visual pane. It follows the SAME discipline the guard
// exit summary already ships (guardColorizeSummary in guard_format_layout.go): the pure block
// builders (renderGuardInfoVisualBlock / renderGuardInfoInteractiveBlock) stay byte-clean so a
// piped/redirected pane or a --json sink keeps plain text and every width/layout test can keep
// asserting on the monochrome rows. Color is layered on HERE, at the moment the frame is written to
// the terminal, and only to a real TTY that has not opted out via NO_COLOR.
//
// The one invariant that makes this safe: the block handed to colorizeGuardInfoBlock is ALREADY
// width-capped — renderGuardInfoVisualBlock trims every row to the pane width with takeCellsTUI
// before returning. We only ever wrap a WHOLE row in an SGR pair (open … reset); the escapes are
// zero display cells and we never re-truncate, so the terminal draws exactly the same columns it
// did without color, and the reset on every colored row means a hue can never bleed into the next.

// guardInfoColorEnabled decides whether the live info pane may emit SGR color this run: only to a
// real terminal, and never when NO_COLOR is set — the community-standard opt-out the guard restart
// audit and `fak tui --color` already honor. A piped/redirected pane (isTTY=false) always stays
// byte-clean so a captured log keeps its plain text.
func guardInfoColorEnabled(isTTY bool) bool {
	return isTTY && os.Getenv("NO_COLOR") == ""
}

// colorizeGuardInfoBlock layers TTY color onto an already-assembled, already-width-capped info
// block. It splits the block into rows and, for each row whose structural role it recognizes,
// wraps the WHOLE row in an SGR pair. A row it does not recognize passes through untouched, so the
// function is safe on any block shape (the plain stacked-panels overview, the interactive tabbed
// block, or the 1-row tiny fallback) and is idempotent per row. color=false (non-TTY / NO_COLOR)
// returns the block verbatim.
func colorizeGuardInfoBlock(block string, color bool) string {
	if !color || block == "" {
		return block
	}
	rows := strings.Split(block, "\n")
	for i, r := range rows {
		if sgr := guardInfoRowSGR(r); sgr != "" {
			rows[i] = sgr + r + tuiSGRReset
		}
	}
	return strings.Join(rows, "\n")
}

// guardInfoRowSGR maps one block row to the SGR escape that colors it, or "" to leave it plain. It
// keys off the row's own STRUCTURE — the section-rule dash prefix and the panels' fixed gutter
// labels plus the plain-words verdict tokens fak itself emits ("nothing blocked", "saving money") —
// never a general parse of the variable data, so the mapping stays robust as counts and paths
// change. The palette matches the guard exit summary's grammar (cyan-bold headings, dim chrome) and
// adds the live pane's two at-a-glance signals: a red incident and a green/yellow safety verdict.
func guardInfoRowSGR(row string) string {
	// Section rules ("── trends ─────") are the pane's headings, like the exit summary's banners.
	if strings.HasPrefix(row, "── ") {
		return tuiSGRCyanBold
	}
	// The panels sit one space in under their gutter label; trim it so we can match the label.
	body := strings.TrimLeft(row, " ")
	switch {
	case strings.HasPrefix(body, "incident "):
		// The incident panel only renders when something upstream is wrong — its mere presence is
		// the signal, so it is always alarm-red with no need to inspect the detail.
		return tuiSGRRedBold
	case strings.HasPrefix(body, "safety "):
		// "nothing blocked" is the clean-session all-clear (green); any other safety text means fak
		// blocked / fixed / set aside a call this session — worth a warm highlight, not an alarm.
		if strings.Contains(row, "nothing blocked") {
			return tuiSGRGreen
		}
		return tuiSGRYellowBold
	case strings.HasPrefix(body, "cache "):
		// Green only once re-use has actually paid for itself ("saving money"); "not saving yet" /
		// "no cache yet" stay neutral so the pane never shows a premature green.
		if strings.Contains(row, "saving money") {
			return tuiSGRGreen
		}
		return ""
	case strings.HasPrefix(body, "why "):
		// The forensic deny-reason breakdown is a demoted annotation under the safety count — dim it
		// so it recedes, matching how the exit summary dims its "↳ …" continuation notes.
		return tuiSGRDim
	case strings.HasPrefix(body, "ablation "):
		// The live cache-ablation framing line is a demoted annotation, like the "why" note above.
		return tuiSGRDim
	case strings.HasPrefix(body, "provider prompt-cache"):
		// A provider-owned cache-ablation mechanism row — cyan, matching the ablate pane's provider
		// hue. Matched on the full label (not a bare "provider ") so no other pane row false-colors.
		return tuiSGRCyanBold
	case strings.HasPrefix(body, "fak compaction shed"),
		strings.HasPrefix(body, "fak KV-prefix reuse"),
		strings.HasPrefix(body, "fak vDSO"):
		// A fak-authored cache-ablation mechanism row — green, matching the ablate pane's fak hue.
		return tuiSGRGreen
	}
	return ""
}
