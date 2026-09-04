// Package egressfloor is the structural network-egress capability floor and destination classifier.
//
// Invariant: egress floor evaluation is fail-closed and default-deny; unverified or link-local destinations are denied unconditionally.
//
// Guard: Classify and EnsureSendable guard against cloud-metadata SSRF exfiltration and unverified outbound platform delivery.
//
// Contract:
//   - Evaluation of network destinations is deterministic, pure, and performs zero runtime DNS or external I/O.
//   - Cloud-metadata endpoints (including 169.254.169.254, fd00:ec2::254, and alternate-radix IP representations) are unconditionally blocked.
//   - Outbound platform deliveries must be vetted by DeliveryPolicy.Adjudicate and carry non-forgeable approval proof before dispatch.
//   - Secrets matching known credential and token patterns are redacted prior to payload transmission or witness generation.
package egressfloor
