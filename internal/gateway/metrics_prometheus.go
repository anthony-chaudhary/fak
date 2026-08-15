package gateway

import (
	"fmt"
	"strconv"
	"strings"
)

func writeHelpType(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n", name, help)
	fmt.Fprintf(b, "# TYPE %s %s\n", name, typ)
}

func writeCounter(b *strings.Builder, name, help string, n int64) {
	writeHelpType(b, name, help, "counter")
	fmt.Fprintf(b, "%s %d\n", name, n)
}

// writeKeyedFamily renders one LABEL-KEYED metric family: the HELP/TYPE header once, then
// one sample per row under the label set that family derives from the row. Header and
// samples are emitted by the same call because that is what keeps the members of a family
// label-identical — a consumer joining plan_bytes_total against plan_observations_total
// needs the label sets to match exactly, and a family whose header and rows are written
// apart is where they drift. value returns the sample as any so a uint64 counter is
// rendered by %d unsigned, exactly as an untyped literal loop would.
func writeKeyedFamily[R any](b *strings.Builder, name, help, typ string, rows []R, labels func(R) string, value func(R) any) {
	writeHelpType(b, name, help, typ)
	for _, row := range rows {
		fmt.Fprintf(b, "%s{%s} %d\n", name, labels(row), value(row))
	}
}

func writeHistogram(b *strings.Builder, name, baseLabels string, s latencySnapshot) {
	// A label-less histogram (baseLabels=="") must not emit a leading comma inside the
	// bucket braces, and renders _sum/_count with no brace set at all.
	lead := baseLabels
	if lead != "" {
		lead += ","
	}
	for i, le := range gatewayLatencyBuckets {
		fmt.Fprintf(b, "%s_bucket{%sle=\"%s\"} %d\n", name, lead, promQuote(promFloat(le)), s.buckets[i])
	}
	fmt.Fprintf(b, "%s_bucket{%sle=\"+Inf\"} %d\n", name, lead, s.count)
	if baseLabels == "" {
		fmt.Fprintf(b, "%s_sum %s\n", name, promFloat(s.sum))
		fmt.Fprintf(b, "%s_count %d\n", name, s.count)
	} else {
		fmt.Fprintf(b, "%s_sum{%s} %s\n", name, baseLabels, promFloat(s.sum))
		fmt.Fprintf(b, "%s_count{%s} %d\n", name, baseLabels, s.count)
	}
}

func promFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func promQuote(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}
