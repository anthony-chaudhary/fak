package metrics

// device_spine_csv.go — the CSV renderer for the device-telemetry spine (issue
// #3237, epic #3236). It completes the Prom/CSV/JSON renderer triad the spine
// commit (830cda28d) opened with RenderProm + RenderJSON: CSV is the third
// stateless reader over the one shared snapshot, driven by the SAME metricTable,
// mirroring ZML's reflection-driven csv/csv.zig. No new collection path — CSV,
// JSON, and Prometheus all read the identical DeviceMetrics snapshot.

import (
	"bytes"
	"encoding/csv"
	"strconv"
)

// csvIdentityColumns are the fixed leading columns of the CSV table: device
// identity plus the federation origin. Metric columns follow, one per
// metricTable row, so the CSV schema is driven by the same shared table that
// drives JSON and Prometheus — add a metric to metricTable and the CSV surface
// grows its column with the others.
var csvIdentityColumns = []string{"backend", "device", "remote", "peer"}

// RenderCSV renders the snapshot as CSV: a header row of identity columns
// followed by one column per normalized metric (in metricTable order), then one
// row per device. It is a stateless reader over the same snapshot RenderJSON and
// RenderProm consume — the reflection-driven CSV surface from ZML's csv.zig,
// completing the Prom/CSV/JSON renderer triad. The null-on-error contract
// survives to the wire: an unread (nil) metric renders as an empty cell, never a
// zero, so a spreadsheet reader can tell "unsupported/unread" apart from a
// measured 0. A nil snapshot renders the schema header row alone (still
// self-describing), not an error. Values use non-scientific 'f' formatting so
// large byte counts stay spreadsheet-friendly.
func RenderCSV(snapshot []DeviceMetrics) ([]byte, error) {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)

	header := make([]string, 0, len(csvIdentityColumns)+len(metricTable))
	header = append(header, csvIdentityColumns...)
	for _, m := range metricTable {
		header = append(header, m.Key)
	}
	if err := w.Write(header); err != nil {
		return nil, err
	}

	for _, d := range snapshot {
		row := make([]string, 0, len(header))
		row = append(row, d.Backend, d.DeviceID, strconv.FormatBool(d.Remote), d.Peer)
		for _, m := range metricTable {
			if v, ok := m.get(d); ok {
				row = append(row, strconv.FormatFloat(v, 'f', -1, 64))
			} else {
				row = append(row, "")
			}
		}
		if err := w.Write(row); err != nil {
			return nil, err
		}
	}

	w.Flush()
	if err := w.Error(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
