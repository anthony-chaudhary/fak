package metrics

// device_spine_text.go — the one-shot text-table renderer for the
// device-telemetry spine (issue #3237, epic #3236): the "TUI as a thin
// renderer" seam in its minimal, non-interactive form. ZML's TUI is a thin
// reader over the shared snapshot; the fanout names "a `fak status` one-shot
// pretty-print" that shares it. RenderText is that one-shot: a stateless
// reader driven by the SAME metricTable as Prom/CSV/JSON, over the identical
// snapshot object — no new collection path.
//
// Generation: gen/next. Promotion evidence: wire RenderText into a `fak
// status`/`fak metrics` verb and grow the interactive overview-grid ↔
// drill-down TUI (ring-buffer sparklines) on top of the same snapshot.
// Demotion evidence: if operators only ever consume Prom/JSON, the text
// surface is dead weight — retire it. Invalidating assumption: one row per
// device fits a terminal line; a fleet wide enough to overflow needs the
// responsive column-wrap layout the interactive TUI owns.

import (
	"bytes"
	"strconv"
	"strings"
)

// RenderText renders the snapshot as an aligned plain-text table: identity
// columns (backend, device, origin) then one column per normalized metric in
// metricTable order, one row per device. The null-on-error contract survives
// to this surface too: an unread metric renders as "-", never a zero. The
// origin column shows "local" or the federated peer name, so a federated
// pane-of-glass snapshot is readable at a glance. A nil snapshot renders the
// header row alone, matching RenderCSV's self-describing-schema contract.
func RenderText(snapshot []DeviceMetrics) []byte {
	header := make([]string, 0, 3+len(metricTable))
	header = append(header, "BACKEND", "DEVICE", "ORIGIN")
	for _, m := range metricTable {
		header = append(header, strings.ToUpper(m.Key))
	}

	rows := make([][]string, 0, len(snapshot)+1)
	rows = append(rows, header)
	for _, d := range snapshot {
		row := make([]string, 0, len(header))
		row = append(row, d.Backend, d.DeviceID, textOrigin(d))
		for _, m := range metricTable {
			if v, ok := m.get(d); ok {
				row = append(row, strconv.FormatFloat(v, 'f', -1, 64))
			} else {
				row = append(row, "-")
			}
		}
		rows = append(rows, row)
	}

	widths := make([]int, len(header))
	for _, row := range rows {
		for i, cell := range row {
			if len(cell) > widths[i] {
				widths[i] = len(cell)
			}
		}
	}

	var buf bytes.Buffer
	for _, row := range rows {
		for i, cell := range row {
			if i > 0 {
				buf.WriteString("  ")
			}
			buf.WriteString(cell)
			if i < len(row)-1 {
				buf.WriteString(strings.Repeat(" ", widths[i]-len(cell)))
			}
		}
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

// textOrigin renders the federation origin cell: "local" for a locally
// collected row, the peer name for a federated one (falling back to "remote"
// when a peer name is absent).
func textOrigin(d DeviceMetrics) string {
	if !d.Remote {
		return "local"
	}
	if d.Peer != "" {
		return d.Peer
	}
	return "remote"
}
