package main

// Optional headroom-bench digest -- summarizes the context-headroom benchmark
// that sizes the compression gate.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

func loadCachevalueHeadroomBenchStatus(path string) (cachevalueHeadroomBenchDigest, []cachevalueStatusRow) {
	digest := cachevalueHeadroomBenchDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{headroomBenchUnavailableRow(path, err.Error())}
	}
	var rep headroom.BenchReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{headroomBenchUnavailableRow(path, "decode: "+err.Error())}
	}
	status := headroomBenchStatus(rep)
	owner, dependency, fidelity, evidence := headroomBenchReportAttribution(rep)
	digest.Status = status
	digest.Compressor = rep.Compressor
	digest.Owner = owner
	digest.Dependency = dependency
	digest.Fidelity = fidelity
	digest.Evidence = evidence
	digest.Samples = len(rep.Samples)
	digest.OrigTotal = rep.OrigTotal
	digest.NewTotal = rep.NewTotal
	digest.SavedRatio = rep.Saved
	digest.Reason = rep.Reason
	return digest, rowsFromHeadroomBenchReport(rep, path)
}

func headroomBenchUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "context_compression",
		Component:     "headroom_bench_report",
		Owner:         "unknown",
		Dependency:    "local_json_report",
		Fidelity:      "recoverable",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "headroom_bench_artifact",
		SessionImpact: "headroom compression proof could not be folded, so compressor behavior is not evidenced in this rollup",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak headroom bench --via native --json",
	}
}

func rowsFromHeadroomBenchReport(rep headroom.BenchReport, path string) []cachevalueStatusRow {
	owner, dependency, fidelity, evidence := headroomBenchReportAttribution(rep)
	rows := []cachevalueStatusRow{{
		Plane:         "context_compression",
		Component:     "headroom_bench_report",
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        headroomBenchStatus(rep),
		FailureDomain: headroomBenchFailureDomain(rep),
		SessionImpact: "headroom bench proves realized compressor savings on a representative or supplied corpus; use it to separate compressor value from provider cache behavior",
		Reason:        headroomBenchReportReason(rep, path),
		NextAction:    headroomBenchNextAction(rep),
	}}
	for _, sample := range rep.Samples {
		rows = append(rows, rowFromHeadroomBenchSample(rep.Compressor, sample))
	}
	return rows
}

func rowFromHeadroomBenchSample(compressor string, sample headroom.BenchSample) cachevalueStatusRow {
	owner, dependency, fidelity, evidence := headroomBenchAttribution(compressor)
	status := headroomBenchSampleStatus(sample)
	return cachevalueStatusRow{
		Plane:         "context_compression",
		Component:     "headroom_bench_sample:" + nonEmpty(sample.Name, "unnamed"),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        status,
		FailureDomain: headroomBenchOwnerDomain(owner, compressor),
		SessionImpact: "per-sample compressor evidence; no_effect on one sample can be normal, aggregate status decides whether the compressor proof worked",
		Reason:        headroomBenchSampleReason(sample),
	}
}

func headroomBenchAttribution(compressor string) (owner, dependency, fidelity, evidence string) {
	switch strings.ToLower(strings.TrimSpace(compressor)) {
	case headroom.HeadroomName:
		return "external", "external_http_sidecar", "recoverable", "observed"
	case headroom.NoopName:
		return "fak", "none", "no-op", "configured"
	case headroom.NativeName:
		return "fak", "in_process", "recoverable", "witnessed"
	default:
		return "unknown", "registered_plugin", "unknown", "configured"
	}
}

func headroomBenchReportAttribution(rep headroom.BenchReport) (owner, dependency, fidelity, evidence string) {
	owner, dependency, fidelity, evidence = headroomBenchAttribution(rep.Compressor)
	if strings.TrimSpace(rep.Owner) != "" {
		owner = rep.Owner
	}
	if strings.TrimSpace(rep.Dependency) != "" {
		dependency = rep.Dependency
	}
	if strings.TrimSpace(rep.Fidelity) != "" {
		fidelity = rep.Fidelity
	}
	if strings.TrimSpace(rep.Evidence) != "" {
		evidence = rep.Evidence
	}
	return owner, dependency, fidelity, evidence
}

func headroomBenchStatus(rep headroom.BenchReport) string {
	if strings.TrimSpace(rep.Status) != "" {
		return rep.Status
	}
	switch {
	case rep.Compressor == "" || len(rep.Samples) == 0 || rep.OrigTotal <= 0:
		return "missing"
	case strings.EqualFold(rep.Compressor, headroom.NoopName):
		return "no-op"
	case rep.Saved <= 0:
		return "no_saving"
	default:
		return "measured"
	}
}

func headroomBenchFailureDomain(rep headroom.BenchReport) string {
	owner, _, _, _ := headroomBenchReportAttribution(rep)
	switch headroomBenchStatus(rep) {
	case "unavailable":
		return headroomBenchOwnerDomain(owner, rep.Compressor) + "_unavailable"
	case "error":
		return headroomBenchOwnerDomain(owner, rep.Compressor) + "_error"
	}
	if rep.Saved > 0 || strings.EqualFold(rep.Compressor, headroom.NoopName) {
		return headroomBenchOwnerDomain(owner, rep.Compressor)
	}
	return headroomBenchOwnerDomain(owner, rep.Compressor) + "_or_corpus"
}

func headroomBenchOwnerDomain(owner, compressor string) string {
	switch owner {
	case "external":
		return "external:" + nonEmpty(compressor, "headroom")
	case "fak":
		return "fak"
	default:
		return "unknown"
	}
}

func headroomBenchNextAction(rep headroom.BenchReport) string {
	switch headroomBenchStatus(rep) {
	case "missing":
		return "rerun fak headroom bench --via native --json"
	case "unavailable":
		return "start headroom proxy or select FAK_COMPRESSOR=native/noop"
	case "error":
		return "inspect the compressor error and rerun fak headroom bench"
	case "no_saving":
		return "try fak headroom bench --via native on a representative captured tool-output corpus"
	default:
		return ""
	}
}

func headroomBenchReportReason(rep headroom.BenchReport, path string) string {
	reason := strings.TrimSpace(rep.Reason)
	if reason == "" {
		reason = fmt.Sprintf("compressor=%s samples=%d orig=%d new=%d saved=%.2f%%", rep.Compressor, len(rep.Samples), rep.OrigTotal, rep.NewTotal, rep.Saved*100)
	}
	return fmt.Sprintf("%s source=%s", reason, path)
}

func headroomBenchSampleStatus(sample headroom.BenchSample) string {
	if strings.TrimSpace(sample.Status) != "" {
		return sample.Status
	}
	if sample.Saved > 0 {
		return "saved"
	}
	return "no_effect"
}

func headroomBenchSampleReason(sample headroom.BenchSample) string {
	reason := strings.TrimSpace(sample.Reason)
	if reason == "" {
		reason = "sample compressor outcome"
	}
	return fmt.Sprintf("kind=%s codec=%s orig=%d new=%d saved=%.2f%% detail=%s", sample.Kind, sample.Codec, sample.OrigLen, sample.NewLen, sample.Saved*100, reason)
}
