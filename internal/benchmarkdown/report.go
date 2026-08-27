// Package benchmarkdown owns the byte-level layout shared by benchmark adapter
// reports. Benchmark packages project their domain rows into this narrow shape;
// this package owns only headings, metadata order, tables, and section spacing.
package benchmarkdown

import (
	"strconv"
	"strings"
)

// Metadata is the common report preamble emitted by benchmark adapters.
type Metadata struct {
	GeneratedAt              string
	Benchmark                string
	Model                    string
	EvidenceClass            string
	TaskCount                int
	OfficialHarnessRequired  bool
	OfficialHarnessAvailable bool
	OfficialHarnessReason    string
	ResultClaimAllowed       bool
	ClaimBoundary            string
}

// Table is a fully projected Markdown table. Rows exclude their trailing newline;
// the renderer adds it so callers cannot drift on section spacing.
type Table struct {
	Header    string
	Separator string
	Rows      []string
}

// AdapterReport is the shared layout contract for benchmark adapter reports.
// Row contents remain benchmark-owned and arrive already formatted.
type AdapterReport struct {
	Title                 string
	Metadata              Metadata
	Summary               Table
	Tasks                 Table
	PromotionRequirements []string
}

// RenderAdapterReport renders the common adapter-report layout deterministically.
func RenderAdapterReport(report AdapterReport) string {
	var b strings.Builder
	b.WriteString("# ")
	b.WriteString(report.Title)
	b.WriteString("\n\n")

	writeCodeMetadata(&b, "Generated", report.Metadata.GeneratedAt)
	writeCodeMetadata(&b, "Benchmark", report.Metadata.Benchmark)
	if report.Metadata.Model != "" {
		writeCodeMetadata(&b, "Model", report.Metadata.Model)
	}
	if report.Metadata.EvidenceClass != "" {
		writeCodeMetadata(&b, "Evidence class", report.Metadata.EvidenceClass)
	}
	writeCodeMetadata(&b, "Tasks", strconv.Itoa(report.Metadata.TaskCount))
	b.WriteString("- Official harness: required=")
	b.WriteString(strconv.FormatBool(report.Metadata.OfficialHarnessRequired))
	b.WriteString(" available=")
	b.WriteString(strconv.FormatBool(report.Metadata.OfficialHarnessAvailable))
	if report.Metadata.OfficialHarnessReason != "" {
		b.WriteString(" (")
		b.WriteString(report.Metadata.OfficialHarnessReason)
		b.WriteByte(')')
	}
	b.WriteByte('\n')
	writeCodeMetadata(&b, "Result claim allowed", strconv.FormatBool(report.Metadata.ResultClaimAllowed))
	writePlainMetadata(&b, "Boundary", report.Metadata.ClaimBoundary)
	b.WriteByte('\n')

	writeTable(&b, report.Summary)
	b.WriteString("\n## Tasks\n\n")
	writeTable(&b, report.Tasks)
	if len(report.PromotionRequirements) > 0 {
		b.WriteString("\n## Promotion Requirements\n\n")
		for _, requirement := range report.PromotionRequirements {
			b.WriteString("- ")
			b.WriteString(requirement)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func writeCodeMetadata(b *strings.Builder, label, value string) {
	b.WriteString("- ")
	b.WriteString(label)
	b.WriteString(": `")
	b.WriteString(value)
	b.WriteString("`\n")
}

func writePlainMetadata(b *strings.Builder, label, value string) {
	b.WriteString("- ")
	b.WriteString(label)
	b.WriteString(": ")
	b.WriteString(value)
	b.WriteByte('\n')
}

func writeTable(b *strings.Builder, table Table) {
	b.WriteString(table.Header)
	b.WriteByte('\n')
	b.WriteString(table.Separator)
	b.WriteByte('\n')
	for _, row := range table.Rows {
		b.WriteString(row)
		b.WriteByte('\n')
	}
}
