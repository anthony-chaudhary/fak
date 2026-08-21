package benchcatalog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAdmitEnvironmentExactAndSuperset(t *testing.T) {
	req := fixtureRequirement(t)
	receipt := fixtureReceipt(t, "receipt-pass.json")

	got := AdmitEnvironment(req, receipt)
	if got.Status != AdmissionAccepted || len(got.Refusals) != 0 {
		t.Fatalf("admission = %+v, want accepted", got)
	}
	if !strings.HasPrefix(got.RequirementHash, "sha256:") || !strings.HasPrefix(got.ReceiptHash, "sha256:") {
		t.Fatalf("accepted hashes = requirement %q receipt %q, want sha256 bindings", got.RequirementHash, got.ReceiptHash)
	}
}

func TestAdmitEnvironmentRefusalMatrix(t *testing.T) {
	req := fixtureRequirement(t)
	base := fixtureReceipt(t, "receipt-pass.json")

	tests := []struct {
		name string
		axis string
		kind RefusalKind
		edit func(*ComputeReceipt)
	}{
		{name: "os", axis: "os", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.OS = "windows" }},
		{name: "arch unknown", axis: "arch", kind: RefusalUnknown, edit: func(r *ComputeReceipt) { r.Arch = "" }},
		{name: "immutable image", axis: "image_id", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.ImageID = digest("b") }},
		{name: "vcpu", axis: "vcpu", kind: RefusalInsufficient, edit: func(r *ComputeReceipt) { r.VCPUs = req.MinVCPUs - 1 }},
		{name: "ram", axis: "ram_mib", kind: RefusalInsufficient, edit: func(r *ComputeReceipt) { r.RAMMiB = req.MinRAMMiB - 1 }},
		{name: "disk", axis: "disk_gib", kind: RefusalInsufficient, edit: func(r *ComputeReceipt) { r.DiskGiB = req.MinDiskGiB - 1 }},
		{name: "gpu class", axis: "gpu_class", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.GPU.Class = "cpu" }},
		{name: "gpu count", axis: "gpu_count", kind: RefusalInsufficient, edit: func(r *ComputeReceipt) {
			r.GPU.Count = req.GPU.MinCount - 1
			r.GPU.Class = "none"
		}},
		{name: "network forbidden", axis: "network", kind: RefusalForbidden, edit: func(r *ComputeReceipt) { r.Network = NetworkOpen }},
		{name: "software", axis: "software:libreoffice", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.Software = []SoftwareIdentity{} }},
		{name: "license", axis: "license", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.License = LicenseNone }},
		{name: "input data", axis: "input_data", kind: RefusalMissing, edit: func(r *ComputeReceipt) { r.Input.Digest = digest("c") }},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt := cloneReceipt(base)
			tc.edit(&receipt)
			got := AdmitEnvironment(req, receipt)
			if got.Status != AdmissionRefused || len(got.Refusals) != 1 {
				t.Fatalf("admission = %+v, want one refusal", got)
			}
			ref := got.Refusals[0]
			if ref.Axis != tc.axis || ref.Kind != tc.kind {
				t.Fatalf("refusal = %+v, want axis=%q kind=%q", ref, tc.axis, tc.kind)
			}
		})
	}
}

func TestAdmitEnvironmentFailsClosedOnUnknownRequirement(t *testing.T) {
	req := fixtureRequirement(t)
	req.OS = ""
	got := AdmitEnvironment(req, fixtureReceipt(t, "receipt-pass.json"))
	if got.Status != AdmissionRefused || len(got.Refusals) != 1 {
		t.Fatalf("admission = %+v, want one refusal", got)
	}
	if ref := got.Refusals[0]; ref.Code != CodeRequirementUnknown || ref.Axis != "os" {
		t.Fatalf("refusal = %+v, want requirement-unknown os", ref)
	}
}

func TestAdmitEnvironmentCarriesResolvedSanctionedAction(t *testing.T) {
	req := fixtureRequirement(t)
	receipt := fixtureReceipt(t, "receipt-missing.json")
	got := AdmitEnvironmentWithResolver(req, receipt, func(ref Refusal) string {
		return "dispatch to sanctioned-l4 for " + ref.Axis
	})
	if got.Status != AdmissionRefused || len(got.Refusals) != 1 {
		t.Fatalf("admission = %+v, want one refusal", got)
	}
	if action := got.Refusals[0].Action; action != "dispatch to sanctioned-l4 for software:libreoffice" {
		t.Fatalf("action = %q, want fleet-resolved sanctioned action", action)
	}
}

func TestEnvironmentOfflineFixturesCoverTypedOutcomes(t *testing.T) {
	req := fixtureRequirement(t)
	tests := []struct {
		file   string
		status AdmissionStatus
		kind   RefusalKind
	}{
		{file: "receipt-pass.json", status: AdmissionAccepted},
		{file: "receipt-missing.json", status: AdmissionRefused, kind: RefusalMissing},
		{file: "receipt-unknown.json", status: AdmissionRefused, kind: RefusalUnknown},
		{file: "receipt-forbidden.json", status: AdmissionRefused, kind: RefusalForbidden},
	}
	for _, tc := range tests {
		t.Run(tc.file, func(t *testing.T) {
			got := AdmitEnvironment(req, fixtureReceipt(t, tc.file))
			if got.Status != tc.status {
				t.Fatalf("status = %q, want %q (%+v)", got.Status, tc.status, got)
			}
			if tc.status == AdmissionRefused && (len(got.Refusals) != 1 || got.Refusals[0].Kind != tc.kind) {
				t.Fatalf("refusals = %+v, want one %q", got.Refusals, tc.kind)
			}
		})
	}
}

func TestReceiptHashCanonicalizesSoftwareOrder(t *testing.T) {
	receipt := fixtureReceipt(t, "receipt-pass.json")
	receipt.Software = append(receipt.Software,
		SoftwareIdentity{Name: "python", Version: "3.13.5", Digest: digest("d")})
	reversed := cloneReceipt(receipt)
	reversed.Software[0], reversed.Software[1] = reversed.Software[1], reversed.Software[0]

	a, err := ReceiptHash(receipt)
	if err != nil {
		t.Fatal(err)
	}
	b, err := ReceiptHash(reversed)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("receipt hash depends on software order: %s != %s", a, b)
	}
}

func fixtureRequirement(t *testing.T) TaskEnvironmentRequirement {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "environment-admission", "requirement.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	req, err := DecodeTaskEnvironmentRequirement(f)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func fixtureReceipt(t *testing.T, name string) ComputeReceipt {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "environment-admission", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	receipt, err := DecodeComputeReceipt(f)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.ProbedAt.IsZero() || receipt.ProbedAt.After(time.Now().Add(24*time.Hour)) {
		t.Fatalf("fixture probe time is invalid: %s", receipt.ProbedAt)
	}
	return receipt
}

func cloneReceipt(in ComputeReceipt) ComputeReceipt {
	out := in
	if in.Software != nil {
		out.Software = make([]SoftwareIdentity, len(in.Software))
		copy(out.Software, in.Software)
	}
	return out
}

func digest(ch string) string { return "sha256:" + strings.Repeat(ch, 64) }
