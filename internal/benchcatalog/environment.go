package benchcatalog

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

const (
	TaskEnvironmentRequirementSchema = "fak.benchcatalog.task-environment.v1"
	ComputeReceiptSchema             = "fak.benchcatalog.compute-receipt.v1"
	EnvironmentAdmissionSchema       = "fak.benchcatalog.environment-admission.v1"
)

// NetworkMode is the network posture a task requires and a provider observed.
type NetworkMode string

const (
	NetworkForbidden  NetworkMode = "forbidden"
	NetworkRestricted NetworkMode = "restricted"
	NetworkOpen       NetworkMode = "open"
)

// LicensePosture separates a task's license policy from a provider's proof.
type LicensePosture string

const (
	LicenseNone              LicensePosture = "none"
	LicenseRequired          LicensePosture = "required"
	LicenseForbidden         LicensePosture = "forbidden"
	LicenseVerified          LicensePosture = "verified"
	LicensePresentUnverified LicensePosture = "present-unverified"
)

type GPURequirement struct {
	Class    string `json:"class"`
	MinCount int    `json:"min_count"`
}

type GPUObservation struct {
	Class string `json:"class"`
	Count int    `json:"count"`
}

type SoftwareIdentity struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type DataIdentity struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

// TaskEnvironmentRequirement is the complete environment contract for one task.
// Empty axes are unknown, not wildcards. Use explicit values such as "none" for
// a known absence so admission cannot silently inherit the local workstation.
type TaskEnvironmentRequirement struct {
	Schema           string             `json:"schema"`
	TaskID           string             `json:"task_id"`
	OS               string             `json:"os"`
	Arch             string             `json:"arch"`
	ImageID          string             `json:"image_id"`
	MinVCPUs         int                `json:"min_vcpus"`
	MinRAMMiB        int                `json:"min_ram_mib"`
	MinDiskGiB       int                `json:"min_disk_gib"`
	GPU              GPURequirement     `json:"gpu"`
	Network          NetworkMode        `json:"network"`
	RequiredSoftware []SoftwareIdentity `json:"required_software"`
	License          LicensePosture     `json:"license"`
	Input            DataIdentity       `json:"input"`
}

// ComputeReceipt is an observed provider/node fact sheet. Callers populate it
// from their existing probe or fleet registry; benchcatalog does not own nodes.
type ComputeReceipt struct {
	Schema   string             `json:"schema"`
	Provider string             `json:"provider"`
	NodeID   string             `json:"node_id"`
	Source   string             `json:"source"`
	ProbedAt time.Time          `json:"probed_at"`
	OS       string             `json:"os"`
	Arch     string             `json:"arch"`
	ImageID  string             `json:"image_id"`
	VCPUs    int                `json:"vcpus"`
	RAMMiB   int                `json:"ram_mib"`
	DiskGiB  int                `json:"disk_gib"`
	GPU      GPUObservation     `json:"gpu"`
	Network  NetworkMode        `json:"network"`
	Software []SoftwareIdentity `json:"software"`
	License  LicensePosture     `json:"license"`
	Input    DataIdentity       `json:"input"`
}

type AdmissionStatus string

const (
	AdmissionAccepted AdmissionStatus = "accepted"
	AdmissionRefused  AdmissionStatus = "refused"
)

type RefusalKind string

const (
	RefusalMissing      RefusalKind = "missing"
	RefusalInsufficient RefusalKind = "insufficient"
	RefusalForbidden    RefusalKind = "forbidden"
	RefusalUnknown      RefusalKind = "unknown"
)

type RefusalCode string

const (
	CodeRequirementUnknown RefusalCode = "BENCH_REQUIREMENT_UNKNOWN"
	CodeObservationUnknown RefusalCode = "BENCH_OBSERVATION_UNKNOWN"
	CodeMissing            RefusalCode = "BENCH_CAPABILITY_MISSING"
	CodeInsufficient       RefusalCode = "BENCH_CAPABILITY_INSUFFICIENT"
	CodeForbidden          RefusalCode = "BENCH_CAPABILITY_FORBIDDEN"
)

type Refusal struct {
	Code     RefusalCode `json:"code"`
	Kind     RefusalKind `json:"kind"`
	Axis     string      `json:"axis"`
	Required string      `json:"required,omitempty"`
	Observed string      `json:"observed,omitempty"`
	Detail   string      `json:"detail"`
	Action   string      `json:"action,omitempty"`
}

// EnvironmentAdmission is safe to bind into an external result packet: an
// acceptance carries both the task contract and observed receipt digests.
type EnvironmentAdmission struct {
	Schema          string          `json:"schema"`
	Status          AdmissionStatus `json:"status"`
	TaskID          string          `json:"task_id"`
	RequirementHash string          `json:"requirement_hash"`
	ReceiptHash     string          `json:"receipt_hash"`
	Refusals        []Refusal       `json:"refusals"`
}

// ActionResolver is the adapter seam for the existing fleet/node registry.
// It may return an exact sanctioned-node or credential action for a refusal.
type ActionResolver func(Refusal) string

func DecodeTaskEnvironmentRequirement(r io.Reader) (TaskEnvironmentRequirement, error) {
	return decodeStrict[TaskEnvironmentRequirement](r, "task environment requirement")
}

func DecodeComputeReceipt(r io.Reader) (ComputeReceipt, error) {
	return decodeStrict[ComputeReceipt](r, "compute receipt")
}

func decodeStrict[T any](r io.Reader, label string) (T, error) {
	var out T
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err != nil {
		return out, fmt.Errorf("benchcatalog: decode %s: %w", label, err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return out, fmt.Errorf("benchcatalog: decode %s: multiple JSON values", label)
		}
		return out, fmt.Errorf("benchcatalog: decode %s: %w", label, err)
	}
	return out, nil
}

func RequirementHash(req TaskEnvironmentRequirement) (string, error) {
	copy := canonicalRequirement(req)
	return hashJSON(copy)
}

func ReceiptHash(receipt ComputeReceipt) (string, error) {
	copy := canonicalReceipt(receipt)
	return hashJSON(copy)
}

func hashJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func canonicalRequirement(req TaskEnvironmentRequirement) TaskEnvironmentRequirement {
	req.Schema = strings.TrimSpace(req.Schema)
	req.TaskID = strings.TrimSpace(req.TaskID)
	req.OS = strings.ToLower(strings.TrimSpace(req.OS))
	req.Arch = strings.ToLower(strings.TrimSpace(req.Arch))
	req.ImageID = strings.TrimSpace(req.ImageID)
	req.GPU.Class = strings.ToLower(strings.TrimSpace(req.GPU.Class))
	req.Network = NetworkMode(strings.ToLower(strings.TrimSpace(string(req.Network))))
	req.License = LicensePosture(strings.ToLower(strings.TrimSpace(string(req.License))))
	req.Input = canonicalData(req.Input)
	req.RequiredSoftware = canonicalSoftware(req.RequiredSoftware)
	return req
}

func canonicalReceipt(receipt ComputeReceipt) ComputeReceipt {
	receipt.Schema = strings.TrimSpace(receipt.Schema)
	receipt.Provider = strings.TrimSpace(receipt.Provider)
	receipt.NodeID = strings.TrimSpace(receipt.NodeID)
	receipt.Source = strings.TrimSpace(receipt.Source)
	receipt.OS = strings.ToLower(strings.TrimSpace(receipt.OS))
	receipt.Arch = strings.ToLower(strings.TrimSpace(receipt.Arch))
	receipt.ImageID = strings.TrimSpace(receipt.ImageID)
	receipt.Network = NetworkMode(strings.ToLower(strings.TrimSpace(string(receipt.Network))))
	receipt.License = LicensePosture(strings.ToLower(strings.TrimSpace(string(receipt.License))))
	receipt.GPU.Class = strings.ToLower(strings.TrimSpace(receipt.GPU.Class))
	receipt.Input = canonicalData(receipt.Input)
	receipt.Software = canonicalSoftware(receipt.Software)
	return receipt
}

func canonicalSoftware(in []SoftwareIdentity) []SoftwareIdentity {
	if in == nil {
		return nil
	}
	out := make([]SoftwareIdentity, len(in))
	copy(out, in)
	for i := range out {
		out[i].Name = strings.ToLower(strings.TrimSpace(out[i].Name))
		out[i].Version = strings.TrimSpace(out[i].Version)
		out[i].Digest = strings.TrimSpace(out[i].Digest)
	}
	sort.Slice(out, func(i, j int) bool {
		return softwareKey(out[i]) < softwareKey(out[j])
	})
	return out
}

func softwareKey(v SoftwareIdentity) string { return v.Name + "\x00" + v.Version + "\x00" + v.Digest }

func canonicalData(in DataIdentity) DataIdentity {
	in.Name = strings.ToLower(strings.TrimSpace(in.Name))
	in.Digest = strings.TrimSpace(in.Digest)
	return in
}

func AdmitEnvironment(req TaskEnvironmentRequirement, receipt ComputeReceipt) EnvironmentAdmission {
	return AdmitEnvironmentWithResolver(req, receipt, nil)
}

// AdmitCatalogEnvironment preserves legacy catalog rows without treating their
// coarse Need value as proof. A nil requirement is a stable fail-closed result;
// list, describe, and legacy run behavior remain available separately.
func AdmitCatalogEnvironment(taskID string, req *TaskEnvironmentRequirement, receipt ComputeReceipt) EnvironmentAdmission {
	if req != nil {
		copy := *req
		if strings.TrimSpace(copy.TaskID) == "" {
			copy.TaskID = taskID
		}
		return AdmitEnvironment(copy, receipt)
	}
	receiptHash, _ := ReceiptHash(receipt)
	return EnvironmentAdmission{
		Schema:      EnvironmentAdmissionSchema,
		Status:      AdmissionRefused,
		TaskID:      strings.TrimSpace(taskID),
		ReceiptHash: receiptHash,
		Refusals: []Refusal{{
			Code: CodeRequirementUnknown, Kind: RefusalUnknown, Axis: "environment",
			Detail: "catalog entry has only a legacy cold-start Need; supply a typed task environment requirement before launch",
		}},
	}
}

func AdmitEnvironmentWithResolver(req TaskEnvironmentRequirement, receipt ComputeReceipt, resolve ActionResolver) EnvironmentAdmission {
	req = canonicalRequirement(req)
	receipt = canonicalReceipt(receipt)
	reqHash, _ := RequirementHash(req)
	receiptHash, _ := ReceiptHash(receipt)
	out := EnvironmentAdmission{
		Schema: EnvironmentAdmissionSchema, TaskID: req.TaskID,
		RequirementHash: reqHash, ReceiptHash: receiptHash,
		Refusals: []Refusal{},
	}

	out.Refusals = append(out.Refusals, unknownRequirements(req)...)
	if len(out.Refusals) == 0 {
		out.Refusals = append(out.Refusals, unknownObservations(receipt)...)
	}
	if len(out.Refusals) == 0 {
		out.Refusals = append(out.Refusals, compareEnvironment(req, receipt)...)
	}
	if resolve != nil {
		for i := range out.Refusals {
			out.Refusals[i].Action = strings.TrimSpace(resolve(out.Refusals[i]))
		}
	}
	if len(out.Refusals) > 0 {
		out.Status = AdmissionRefused
		return out
	}
	out.Status = AdmissionAccepted
	return out
}

func unknownRequirements(req TaskEnvironmentRequirement) []Refusal {
	var out []Refusal
	unknown := func(axis, detail string) {
		out = append(out, Refusal{Code: CodeRequirementUnknown, Kind: RefusalUnknown, Axis: axis, Detail: detail})
	}
	if req.Schema != TaskEnvironmentRequirementSchema {
		unknown("schema", fmt.Sprintf("requirement schema %q is not %q", req.Schema, TaskEnvironmentRequirementSchema))
	}
	if req.TaskID == "" {
		unknown("task_id", "task identity is not declared")
	}
	if req.OS == "" {
		unknown("os", "required OS is not declared")
	}
	if req.Arch == "" {
		unknown("arch", "required architecture is not declared")
	}
	if !immutableIdentity(req.ImageID) {
		unknown("image_id", "required image is missing or mutable; use an immutable image ID/digest")
	}
	if req.MinVCPUs <= 0 {
		unknown("vcpu", "minimum vCPU count is not declared")
	}
	if req.MinRAMMiB <= 0 {
		unknown("ram_mib", "minimum RAM is not declared")
	}
	if req.MinDiskGiB <= 0 {
		unknown("disk_gib", "minimum disk is not declared")
	}
	if req.GPU.MinCount < 0 || (req.GPU.MinCount == 0 && !strings.EqualFold(req.GPU.Class, "none")) || (req.GPU.MinCount > 0 && strings.TrimSpace(req.GPU.Class) == "") {
		unknown("gpu", "GPU requirement must declare class and non-negative count; use class=none,count=0 explicitly")
	}
	if !validNetwork(req.Network) {
		unknown("network", "network requirement is not declared")
	}
	if req.RequiredSoftware == nil {
		unknown("software", "required_software must be an explicit array; [] means none")
	} else {
		for _, software := range req.RequiredSoftware {
			if !validSoftware(software) {
				unknown("software:"+software.Name, "required software must carry name, version, and immutable digest")
			}
		}
	}
	if req.License != LicenseNone && req.License != LicenseRequired && req.License != LicenseForbidden {
		unknown("license", "license policy must be none, required, or forbidden")
	}
	if !validDataIdentity(req.Input) {
		unknown("input_data", "input data must carry an explicit name and immutable digest; use name=none,digest=none for no input")
	}
	return out
}

func unknownObservations(receipt ComputeReceipt) []Refusal {
	var out []Refusal
	unknown := func(axis, detail string) {
		out = append(out, Refusal{Code: CodeObservationUnknown, Kind: RefusalUnknown, Axis: axis, Detail: detail})
	}
	if receipt.Schema != ComputeReceiptSchema {
		unknown("schema", fmt.Sprintf("receipt schema %q is not %q", receipt.Schema, ComputeReceiptSchema))
	}
	if receipt.Provider == "" {
		unknown("provider", "provider identity was not observed")
	}
	if receipt.NodeID == "" {
		unknown("node_id", "compute node identity was not observed")
	}
	if receipt.Source == "" {
		unknown("source", "receipt source/probe identity was not observed")
	}
	if receipt.ProbedAt.IsZero() {
		unknown("probed_at", "receipt probe time was not observed")
	}
	if receipt.OS == "" {
		unknown("os", "provider OS was not observed")
	}
	if receipt.Arch == "" {
		unknown("arch", "provider architecture was not observed")
	}
	if !immutableIdentity(receipt.ImageID) {
		unknown("image_id", "provider image is missing or mutable; record an immutable image ID/digest")
	}
	if receipt.VCPUs <= 0 {
		unknown("vcpu", "provider vCPU count was not observed")
	}
	if receipt.RAMMiB <= 0 {
		unknown("ram_mib", "provider RAM was not observed")
	}
	if receipt.DiskGiB <= 0 {
		unknown("disk_gib", "provider disk was not observed")
	}
	if receipt.GPU.Count < 0 || (receipt.GPU.Count == 0 && !strings.EqualFold(receipt.GPU.Class, "none")) || (receipt.GPU.Count > 0 && strings.TrimSpace(receipt.GPU.Class) == "") {
		unknown("gpu", "provider GPU class/count was not observed; use class=none,count=0 explicitly")
	}
	if !validNetwork(receipt.Network) {
		unknown("network", "provider network enforcement was not observed")
	}
	if receipt.Software == nil {
		unknown("software", "software inventory must be an explicit array; [] means none observed")
	} else {
		for _, software := range receipt.Software {
			if !validSoftware(software) {
				unknown("software:"+software.Name, "observed software must carry name, version, and immutable digest")
			}
		}
	}
	if receipt.License != LicenseNone && receipt.License != LicenseVerified && receipt.License != LicensePresentUnverified {
		unknown("license", "license observation must be none, verified, or present-unverified")
	}
	if !validDataIdentity(receipt.Input) {
		unknown("input_data", "observed input data must carry an explicit name and immutable digest")
	}
	return out
}

func compareEnvironment(req TaskEnvironmentRequirement, receipt ComputeReceipt) []Refusal {
	var out []Refusal
	missing := func(axis, required, observed string) {
		out = append(out, Refusal{Code: CodeMissing, Kind: RefusalMissing, Axis: axis, Required: required, Observed: observed, Detail: "required capability is absent or incompatible"})
	}
	insufficient := func(axis string, required, observed int) {
		out = append(out, Refusal{Code: CodeInsufficient, Kind: RefusalInsufficient, Axis: axis, Required: fmt.Sprint(required), Observed: fmt.Sprint(observed), Detail: "observed capacity is below the task minimum"})
	}
	forbidden := func(axis, required, observed string) {
		out = append(out, Refusal{Code: CodeForbidden, Kind: RefusalForbidden, Axis: axis, Required: required, Observed: observed, Detail: "provider exposes a capability the task forbids"})
	}

	if !strings.EqualFold(req.OS, receipt.OS) {
		missing("os", req.OS, receipt.OS)
	}
	if !strings.EqualFold(req.Arch, receipt.Arch) {
		missing("arch", req.Arch, receipt.Arch)
	}
	if req.ImageID != receipt.ImageID {
		missing("image_id", req.ImageID, receipt.ImageID)
	}
	if receipt.VCPUs < req.MinVCPUs {
		insufficient("vcpu", req.MinVCPUs, receipt.VCPUs)
	}
	if receipt.RAMMiB < req.MinRAMMiB {
		insufficient("ram_mib", req.MinRAMMiB, receipt.RAMMiB)
	}
	if receipt.DiskGiB < req.MinDiskGiB {
		insufficient("disk_gib", req.MinDiskGiB, receipt.DiskGiB)
	}
	if req.GPU.MinCount > 0 && receipt.GPU.Count > 0 && !strings.EqualFold(req.GPU.Class, receipt.GPU.Class) {
		missing("gpu_class", req.GPU.Class, receipt.GPU.Class)
	}
	if receipt.GPU.Count < req.GPU.MinCount {
		insufficient("gpu_count", req.GPU.MinCount, receipt.GPU.Count)
	}
	if req.Network != receipt.Network {
		if receipt.Network == NetworkOpen || req.Network == NetworkForbidden {
			forbidden("network", string(req.Network), string(receipt.Network))
		} else {
			missing("network", string(req.Network), string(receipt.Network))
		}
	}
	for _, want := range req.RequiredSoftware {
		got, ok := findSoftware(receipt.Software, want.Name)
		if !ok || got.Version != want.Version || got.Digest != want.Digest {
			observed := "absent"
			if ok {
				observed = got.Version + "@" + got.Digest
			}
			missing("software:"+want.Name, want.Version+"@"+want.Digest, observed)
		}
	}
	switch req.License {
	case LicenseRequired:
		switch receipt.License {
		case LicenseNone:
			missing("license", string(LicenseVerified), string(receipt.License))
		case LicensePresentUnverified:
			out = append(out, Refusal{Code: CodeObservationUnknown, Kind: RefusalUnknown, Axis: "license", Required: string(LicenseVerified), Observed: string(receipt.License), Detail: "license presence was observed but entitlement was not verified"})
		}
	case LicenseForbidden:
		if receipt.License != LicenseNone {
			forbidden("license", string(LicenseNone), string(receipt.License))
		}
	}
	if req.Input.Name != receipt.Input.Name || req.Input.Digest != receipt.Input.Digest {
		missing("input_data", req.Input.Name+"@"+req.Input.Digest, receipt.Input.Name+"@"+receipt.Input.Digest)
	}
	return out
}

func findSoftware(all []SoftwareIdentity, name string) (SoftwareIdentity, bool) {
	for _, software := range all {
		if strings.EqualFold(software.Name, name) {
			return software, true
		}
	}
	return SoftwareIdentity{}, false
}

func validNetwork(network NetworkMode) bool {
	return network == NetworkForbidden || network == NetworkRestricted || network == NetworkOpen
}

func validSoftware(software SoftwareIdentity) bool {
	return strings.TrimSpace(software.Name) != "" && strings.TrimSpace(software.Version) != "" && immutableIdentity(software.Digest) && software.Digest != "none"
}

func validDataIdentity(input DataIdentity) bool {
	if input.Name == "none" || input.Digest == "none" {
		return input.Name == "none" && input.Digest == "none"
	}
	return input.Name != "" && immutableIdentity(input.Digest)
}

func immutableIdentity(value string) bool {
	value = strings.TrimSpace(value)
	if value == "none" {
		return true
	}
	if strings.HasPrefix(value, "sha256:") && len(value) == len("sha256:")+64 {
		_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
		return err == nil
	}
	if strings.Contains(value, "@sha256:") {
		parts := strings.Split(value, "@sha256:")
		if len(parts) == 2 && parts[0] != "" && len(parts[1]) == 64 {
			_, err := hex.DecodeString(parts[1])
			return err == nil
		}
	}
	for _, prefix := range []string{"ami-", "snapshot-", "image-id:"} {
		if strings.HasPrefix(value, prefix) && len(value) > len(prefix) {
			return true
		}
	}
	return false
}
