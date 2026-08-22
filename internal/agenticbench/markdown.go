package agenticbench

import (
	"fmt"
	"strings"
)

func RenderMarkdown(r *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Agentic Benchmark Epic #868 Rollup\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", r.GeneratedAt)
	fmt.Fprintf(&b, "- Status: `%s`\n", r.Status)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", r.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Children parsed: `%d/%d`\n", r.Summary.ChildrenParsed, r.Summary.ChildrenTotal)
	fmt.Fprintf(&b, "- Result-claim artifacts: `%d`\n", r.Summary.ResultClaimArtifacts)
	fmt.Fprintf(&b, "- Result packets: `%d` passed / `%d` total\n", r.Summary.ResultPacketsPassed, r.Summary.ResultPacketsTotal)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", r.ClaimBoundary)

	fmt.Fprintf(&b, "## Children\n\n")
	fmt.Fprintf(&b, "| Issue | Packet | Artifact | Gate | Status | Detail |\n")
	fmt.Fprintf(&b, "|---:|---|---|---|---|---|\n")
	for _, child := range r.Children {
		fmt.Fprintf(&b, "| #%d | `%s` | `%s` | `%s` | `%s` | %s |\n",
			child.Issue, child.Packet, child.Artifact, child.Gate, child.Status, mdCell(child.Detail))
	}

	fmt.Fprintf(&b, "\n## Result Packet Intake\n\n")
	fmt.Fprintf(&b, "- Directory: `%s/*.json`\n", DefaultResultPacketDir)
	fmt.Fprintf(&b, "- Schema: `%s`\n", ResultPacketSchema)
	fmt.Fprintf(&b, "- Required gates: `benchmark_native`, `same_task_ids`, `same_model`, `same_budget`, `official_grader.available`, raw/fak arms, checked-in artifacts, and metric categories `%s`.\n",
		strings.Join(requiredMetricCategories, "`, `"))

	if len(r.ResultPackets) > 0 {
		fmt.Fprintf(&b, "\n### Result Packets\n\n")
		fmt.Fprintf(&b, "| Path | Issue | Gate | Status | Detail |\n")
		fmt.Fprintf(&b, "|---|---:|---|---|---|\n")
		for _, packet := range r.ResultPackets {
			fmt.Fprintf(&b, "| `%s` | #%d | `%s` | `%s` | %s |\n",
				packet.Path, packet.Issue, packet.Gate, packet.Status, mdCell(packet.Detail))
		}

		fmt.Fprintf(&b, "\n### Latency Phases\n\n")
		fmt.Fprintf(&b, "Gateway request timing is nested inside agent execution and is never added to the harness total.\n\n")
		fmt.Fprintf(&b, "| Packet | Arm | Queue wait | Agent execution | Evaluation | Harness total | Gateway observations |\n")
		fmt.Fprintf(&b, "|---|---|---|---|---|---|---|\n")
		for _, packet := range r.ResultPackets {
			for _, arm := range packet.Latency {
				fmt.Fprintf(&b, "| `%s` | `%s` | %s | %s | %s | %s | %s |\n",
					packet.Path,
					arm.Name,
					mdCell(formatLatencyMeasurement(arm.QueueWait)),
					mdCell(formatLatencyMeasurement(arm.AgentExecution)),
					mdCell(formatLatencyMeasurement(arm.Evaluation)),
					mdCell(formatLatencyMeasurement(arm.Total)),
					mdCell(formatGatewayObservations(arm.GatewayRequests)))
			}
		}
	}

	fmt.Fprintf(&b, "\n## Acceptance Gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|---:|---|\n")
	for _, gate := range r.Acceptance {
		fmt.Fprintf(&b, "| `%s` | %t | %s |\n", gate.Name, gate.OK, mdCell(gate.Detail))
	}
	return b.String()
}

func formatLatencyMeasurement(measurement LatencyMeasurement) string {
	if measurement.Duration == nil {
		if measurement.UnknownReason == "" {
			return "missing"
		}
		return fmt.Sprintf("unknown (%s; source: %s)", measurement.UnknownReason, measurement.Source)
	}
	return fmt.Sprintf("%g %s (source: %s)", *measurement.Duration, measurement.Unit, measurement.Source)
}

func formatGatewayObservations(observations []GatewayLatencyObservation) string {
	if len(observations) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(observations))
	for _, observation := range observations {
		parts = append(parts, fmt.Sprintf("%s: %s; nested, non-additive", observation.Name, formatLatencyMeasurement(observation.LatencyMeasurement)))
	}
	return strings.Join(parts, "; ")
}

func mdCell(s string) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", `\|`)
	return s
}
