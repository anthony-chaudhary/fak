// Self-registration: enroll the shipped ASRSteward in the always-on steward
// population so a defense regression fires the gate at RUNTIME, not only under
// `go test`. This mirrors every other leaf's idiom — ifc, normgate, shipgate all
// self-register their fold-driver in init() and are turned on by exactly one
// blank-import line in internal/registrations (the defconfig); the kernel never
// imports a leaf. Until this file existed, the ASRSteward was reachable only from
// the package tests and cmd/agentdojoredteam, so the dynamic AgentDojo battery
// could not gate a live build the way the static poison.json fixture it replaced
// once did.
//
// HONEST SCOPE. Registration makes agentdojo registration-ready: the moment the
// leaf is blank-imported, NewASRSteward() is in abi.Stewards() and any steward
// sweep (steward.Population.Sweep over the registered set) runs the expanded
// adaptive battery through the full stack and fires with a reproducible winning
// attack if ASR climbs off zero. It does NOT itself install a sweep DRIVER — at
// time of writing nothing in the boot path drives abi.Stewards() on a cadence;
// that driver is a separate, cross-lane unit. So this closes the "the steward
// self-registers nowhere" gap (the leaf is now wired like its peers) without
// over-claiming that the gate is being swept yet.
//
// COST. ASRSteward.Check is model-free, in-process, and deterministic: it scores
// the expanded battery (seeds + generative paraphrases) through a freshly-built
// full stack. It runs once per Sweep, never on the per-tool-call hot path, so
// enrolling it imposes no syscall-path cost — it keeps the kernel O(1) in feature
// count (the always-on registry is read via an atomic snapshot).
package agentdojo

import "github.com/anthony-chaudhary/fak/internal/abi"

func init() {
	// The dynamic, ASR-scored replacement for the static poison.json check. It
	// abstains on a healthy full stack (ASR==0 across the expanded battery) and
	// returns a violation ONLY with an independently-reproducible witness (the
	// winning attack), per the steward discipline.
	abi.RegisterSteward(NewASRSteward())
	abi.RegisterCapability("agentdojo.v1")
}
