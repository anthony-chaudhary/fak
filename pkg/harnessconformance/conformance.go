// Package harnessconformance provides an external black-box compatibility suite
// for harness adapters without importing fak internals.
package harnessconformance

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

const ContractVersion = "harness-conformance/v1"

type Capability string

const (
	Lifecycle       Capability = "lifecycle"
	Streaming       Capability = "streaming"
	Cancellation    Capability = "cancellation"
	Backpressure    Capability = "backpressure"
	Tools           Capability = "tools"
	Approvals       Capability = "approvals"
	StateResume     Capability = "state-resume"
	Policy          Capability = "policy"
	Telemetry       Capability = "telemetry"
	RemovedNegative Capability = "removed-negative"
)

var Required = []Capability{Lifecycle, Streaming, Cancellation, Backpressure, Tools, Approvals, StateResume, Policy, Telemetry, RemovedNegative}

type Outcome string

const (
	Pass Outcome = "pass"
	Fail Outcome = "fail"
	Skip Outcome = "skip"
)

type Probe interface {
	Capabilities() []Capability
	Check(Capability) (Outcome, string)
}
type Check struct {
	Capability Capability `json:"capability"`
	Outcome    Outcome    `json:"outcome"`
	Detail     string     `json:"detail,omitempty"`
}
type Certificate struct {
	Contract      string  `json:"contract"`
	FixtureDigest string  `json:"fixture_digest"`
	Full          bool    `json:"full"`
	Checks        []Check `json:"checks"`
}

func Run(probe Probe) Certificate {
	declared := map[Capability]bool{}
	for _, c := range probe.Capabilities() {
		declared[c] = true
	}
	checks := make([]Check, 0, len(Required))
	full := true
	for _, capability := range Required {
		if !declared[capability] {
			checks = append(checks, Check{capability, Skip, "not declared"})
			full = false
			continue
		}
		outcome, detail := probe.Check(capability)
		if outcome != Pass {
			full = false
		}
		checks = append(checks, Check{capability, outcome, detail})
	}
	return Certificate{Contract: ContractVersion, FixtureDigest: fixtureDigest(), Full: full, Checks: checks}
}
func (c Certificate) JSON() ([]byte, error) { return json.Marshal(c) }
func (c Certificate) Validate() error {
	if c.Contract != ContractVersion {
		return fmt.Errorf("contract %q", c.Contract)
	}
	if c.FixtureDigest != fixtureDigest() {
		return fmt.Errorf("fixture digest mismatch")
	}
	for _, check := range c.Checks {
		if check.Outcome == Fail {
			return fmt.Errorf("%s failed: %s", check.Capability, check.Detail)
		}
		if check.Outcome == Skip {
			return fmt.Errorf("%s skipped: certificate is partial", check.Capability)
		}
	}
	if !c.Full {
		return fmt.Errorf("certificate is partial")
	}
	return nil
}
func fixtureDigest() string {
	names := make([]string, len(Required))
	for i, c := range Required {
		names[i] = string(c)
	}
	sort.Strings(names)
	sum := sha256.Sum256([]byte(ContractVersion + ":" + fmt.Sprint(names)))
	return hex.EncodeToString(sum[:])
}
