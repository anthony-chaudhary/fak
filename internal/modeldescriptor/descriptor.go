// Package modeldescriptor defines declarative model capabilities and onboarding coupling budgets.
package modeldescriptor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"sort"
)

const Schema = "fak.model-capability-descriptor/1"

type Geometry struct {
	Kind            string `json:"kind"`
	Shape           []int  `json:"shape"`
	BytesPerElement int    `json:"bytes_per_element"`
}
type Descriptor struct {
	Schema, ID, Revision, Provenance, Trust                                                                          string
	Aliases                                                                                                          []string
	Topology                                                                                                         []string
	State                                                                                                            []Geometry
	Quantization, Storage, Tokenizer, Tools, Multimodal, Backends, Kernels, Envelopes, Oracles, Readiness, Migration []string
	NativeEngine                                                                                                     string
	Forbidden                                                                                                        [][]string
}
type Candidate struct {
	Descriptor                                                                                   Descriptor
	CoreSwitches, OutsideLeafFiles, ArchitectureBranches, DuplicatedLifecycle, DuplicatedMetrics int
}
type Budget struct{ CoreSwitches, OutsideLeafFiles, ArchitectureBranches, DuplicatedLifecycle, DuplicatedMetrics int }
type Report struct {
	Schema, DescriptorDigest string
	Counts                   Budget
	Budget                   Budget
	Missing                  []string
	WithinBudget             bool
}

var ErrMismatch = errors.New("modeldescriptor: incompatible descriptor")

// BindFakNative records the engine identity required by Validate. Keeping this
// assignment here makes modeldescriptor the single owner of that invariant.
func BindFakNative(d Descriptor) Descriptor {
	d.NativeEngine = "fak-native"
	return d
}

func Digest(d Descriptor) (string, error) {
	d.Aliases = sorted(d.Aliases)
	d.Topology = sorted(d.Topology)
	d.Quantization = sorted(d.Quantization)
	b, e := json.Marshal(d)
	if e != nil {
		return "", e
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:]), nil
}
func sorted(v []string) []string { x := append([]string(nil), v...); sort.Strings(x); return x }
func Validate(d Descriptor) error {
	if d.Schema != Schema || d.ID == "" || d.Revision == "" || d.Trust != "witnessed" || d.NativeEngine != "fak-native" {
		return ErrMismatch
	}
	dims := [][]string{d.Topology, d.Quantization, d.Storage, d.Tokenizer, d.Backends, d.Envelopes, d.Oracles, d.Readiness, d.Migration}
	for _, x := range dims {
		if len(x) == 0 {
			return ErrMismatch
		}
	}
	present := map[string]bool{}
	for _, sets := range [][]string{d.Topology, d.Quantization, d.Storage, d.Tools, d.Multimodal, d.Backends} {
		for _, x := range sets {
			present[x] = true
		}
	}
	for _, combo := range d.Forbidden {
		all := true
		for _, x := range combo {
			all = all && present[x]
		}
		if all {
			return ErrMismatch
		}
	}
	return nil
}
func Check(c Candidate, b Budget) Report {
	digest, _ := Digest(c.Descriptor)
	counts := Budget{c.CoreSwitches, c.OutsideLeafFiles, c.ArchitectureBranches, c.DuplicatedLifecycle, c.DuplicatedMetrics}
	r := Report{Schema: "fak.model-onboarding-report/1", DescriptorDigest: digest, Counts: counts, Budget: b}
	if Validate(c.Descriptor) != nil {
		r.Missing = append(r.Missing, "descriptor_or_compatibility")
	}
	r.WithinBudget = len(r.Missing) == 0 && counts.CoreSwitches <= b.CoreSwitches && counts.OutsideLeafFiles <= b.OutsideLeafFiles && counts.ArchitectureBranches <= b.ArchitectureBranches && counts.DuplicatedLifecycle <= b.DuplicatedLifecycle && counts.DuplicatedMetrics <= b.DuplicatedMetrics
	return r
}
