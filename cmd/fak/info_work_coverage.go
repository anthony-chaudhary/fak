package main

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/workaccount"
)

func runInfoWorkCoverage(stdout, stderr io.Writer, asJSON bool) int {
	report, err := workaccount.BuildReport(workaccount.Registry())
	if err != nil {
		fmt.Fprintf(stderr, "fak info --work-coverage: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak info --work-coverage")
	}
	fmt.Fprintf(stdout, "WORK ACCOUNTING COVERAGE · %d declared\n", len(report.Mechanisms))
	statuses := []workaccount.Status{workaccount.Accounted, workaccount.Overlapping, workaccount.Unavailable, workaccount.Excluded}
	for _, status := range statuses {
		fmt.Fprintf(stdout, "  %-22s %d\n", status, report.Counts[status])
	}
	for _, row := range report.Mechanisms {
		detail := row.SourceID
		if detail == "" {
			detail = row.Reason
		}
		fmt.Fprintf(stdout, "  %-24s %-22s %s\n", row.ID, row.Status, detail)
	}
	return 0
}

func workCoverageUnavailableSources() []string {
	var ids []string
	for _, row := range workaccount.Registry() {
		if row.Status == workaccount.Unavailable {
			ids = append(ids, row.ID)
		}
	}
	sort.Strings(ids)
	return ids
}

func workCoverageUnavailableText() string {
	ids := workCoverageUnavailableSources()
	if len(ids) == 0 {
		return ""
	}
	return "coverage unavailable: " + strings.Join(ids, ", ")
}

func marshalWorkCoverageForTest() ([]byte, error) {
	r, err := workaccount.BuildReport(workaccount.Registry())
	if err != nil {
		return nil, err
	}
	return json.Marshal(r)
}
