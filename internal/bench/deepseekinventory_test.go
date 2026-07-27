package bench

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestDeepSeekInventoryProvenancePinned pins the immutable identity of the official
// artifact (#4788 DoD item 1). A moving tag or a drifted size would silently change
// which artifact every downstream feasibility claim is about, so the revision and the
// byte size are pinned literally, not recomputed.
func TestDeepSeekInventoryProvenancePinned(t *testing.T) {
	inv := DeepSeekV4ProInventory()

	if got, want := inv.ModelID, "deepseek-ai/DeepSeek-V4-Pro"; got != want {
		t.Errorf("model_id = %q, want %q", got, want)
	}
	if got, want := inv.Revision, "b5968e9190ef611bbf34a7229255be88a0e937c1"; got != want {
		t.Errorf("revision = %q, want %q (an immutable commit, never a moving tag)", got, want)
	}
	if len(inv.Revision) != 40 {
		t.Errorf("revision %q is not a full 40-hex immutable commit", inv.Revision)
	}
	if got, want := inv.TotalSizeBytes, int64(864704792696); got != want {
		t.Errorf("total_size_bytes = %d, want %d (the pinned weight-index total)", got, want)
	}
	if inv.Gated {
		t.Error("gated = true, want false (the pinned revision is public/ungated)")
	}
	if inv.License == "" {
		t.Error("license is empty; an artifact with no recorded license is not provisionable")
	}
}

// TestDeepSeekInventoryRefusesEveryWitnessedNode is the #4788 failure-class proof: the
// artifact is larger than the aggregate HBM of every sanctioned node witnessed, so
// single-node placement is refused everywhere — a TYPED infeasibility result, which the
// DoD accepts in place of a runnable plan.
//
// The refusal is DERIVED from two witnessed numbers rather than asserted, and the
// positive controls below prove the derivation discriminates. Without them this test
// would pass against an admit() that refused unconditionally.
func TestDeepSeekInventoryRefusesEveryWitnessedNode(t *testing.T) {
	inv := DeepSeekV4ProInventory()

	if len(inv.Admission) == 0 {
		t.Fatal("no admission results; the inventory must decide every witnessed node")
	}
	if len(inv.Admission) != len(inv.WitnessedNodes) {
		t.Fatalf("admission results = %d, witnessed nodes = %d; every node must get a verdict",
			len(inv.Admission), len(inv.WitnessedNodes))
	}

	for _, r := range inv.Admission {
		if r.Admissible {
			t.Errorf("node %s reported admissible; no witnessed node can hold the %d-byte artifact "+
				"(aggregate HBM %d)", r.Node, r.WeightBytes, r.AggregateHBMBytes)
		}
		if r.Verdict == AdmissionFits {
			t.Errorf("node %s verdict = %s, want a refusal", r.Node, r.Verdict)
		}
		if r.Why == "" {
			t.Errorf("node %s refused with no reason; a refusal must say why to be actionable", r.Node)
		}
	}

	if got, want := inv.AdmissionSummary, AdmissionNotSingleNode; got != want {
		t.Errorf("admission_summary = %s, want %s", got, want)
	}
	if inv.Admissible() {
		t.Error("Admissible() = true; #4781 would capture a witness with no real placement")
	}
	if got, want := inv.BlockingIssue(), 4801; got != want {
		t.Errorf("BlockingIssue() = %d, want %d (placement owns the reservation)", got, want)
	}

	// The binding constraint on the largest node: even every rank free is short.
	byName := map[string]AdmissionResult{}
	for _, r := range inv.Admission {
		byName[r.Node] = r
	}
	best, ok := byName["node-a"]
	if !ok {
		t.Fatal("the 8x80GB node is not in the witnessed set; it is the largest candidate")
	}
	if got, want := best.Verdict, AdmissionInsufficientHBM; got != want {
		t.Errorf("largest node verdict = %s, want %s — its ceiling is physical, not a reservation",
			got, want)
	}
	if best.AggregateHBMBytes >= inv.TotalSizeBytes {
		t.Errorf("aggregate HBM %d >= artifact %d, but the verdict claims a shortfall",
			best.AggregateHBMBytes, inv.TotalSizeBytes)
	}
	if got, want := best.ShortfallBytes, inv.TotalSizeBytes-best.AggregateHBMBytes; got != want {
		t.Errorf("shortfall = %d, want %d (what a quantization/multi-node plan must close)", got, want)
	}
	if best.ShortfallBytes <= 0 {
		t.Errorf("shortfall = %d, want > 0 on a refused node", best.ShortfallBytes)
	}

	// The GPU-less node refuses on device absence, not on arithmetic.
	if cpu, ok := byName["node-d"]; ok {
		if got, want := cpu.Verdict, AdmissionNoGPU; got != want {
			t.Errorf("GPU-less node verdict = %s, want %s", got, want)
		}
	}
}

// TestDeepSeekAdmitDiscriminates is the positive control that makes the refusal above a
// witness rather than a tautology: the same admitNode() that refuses every real node
// ADMITS a hypothetical node with enough free HBM, and distinguishes a
// reservation-solvable shortage from a physical one. If admitNode() ever degrades to
// "always refuse", this fails.
func TestDeepSeekAdmitDiscriminates(t *testing.T) {
	const weight = int64(864704792696) // the pinned artifact size

	fits := NodeCapacity{
		Name: "hypothetical-fits", GPUModel: "test", GPUCount: 16, FreeGPUCount: 16,
		HBMBytesPerGPU: 80 * bytesPerGiB, // 1280 GiB aggregate, all free
	}
	if r := admitNode(fits, weight); !r.Admissible || r.Verdict != AdmissionFits {
		t.Errorf("a node with 1280 GiB free was refused (verdict=%s, admissible=%v); "+
			"the refusal must derive from capacity, not be unconditional", r.Verdict, r.Admissible)
	} else if r.ShortfallBytes != 0 {
		t.Errorf("shortfall = %d on a fitting node, want 0", r.ShortfallBytes)
	}

	// Aggregate holds it, free ranks do not => reservation-solvable, no eviction implied.
	reserve := NodeCapacity{
		Name: "hypothetical-reserve", GPUModel: "test", GPUCount: 16, FreeGPUCount: 4,
		HBMBytesPerGPU: 80 * bytesPerGiB, // 1280 GiB aggregate, 320 GiB free
	}
	r := admitNode(reserve, weight)
	if r.Admissible {
		t.Error("a node whose free ranks cannot hold the artifact was admitted")
	}
	if got, want := r.Verdict, AdmissionNeedsReservation; got != want {
		t.Errorf("verdict = %s, want %s — this shortage is a reservation, not a physical ceiling",
			got, want)
	}
	if r.ShortfallBytes != 0 {
		t.Errorf("shortfall = %d, want 0: aggregate holds the artifact, so no plan must close a gap",
			r.ShortfallBytes)
	}

	// A boundary: aggregate exactly equal to the artifact fits (no headroom modelled here,
	// which is why the leaf's verdict is necessary-not-sufficient for a real serve).
	exact := NodeCapacity{
		Name: "hypothetical-exact", GPUModel: "test", GPUCount: 1, FreeGPUCount: 1,
		HBMBytesPerGPU: weight,
	}
	if r := admitNode(exact, weight); !r.Admissible {
		t.Errorf("aggregate == artifact was refused (verdict=%s); the comparison must not be strict",
			r.Verdict)
	}
	// One byte short flips it.
	short := NodeCapacity{
		Name: "hypothetical-short", GPUModel: "test", GPUCount: 1, FreeGPUCount: 1,
		HBMBytesPerGPU: weight - 1,
	}
	if r := admitNode(short, weight); r.Admissible || r.ShortfallBytes != 1 {
		t.Errorf("one byte short: admissible=%v shortfall=%d, want false/1", r.Admissible, r.ShortfallBytes)
	}
}

// TestDeepSeekInventoryIsNotARuntimeWitness fences the conflation #4788 names as its top
// confusion risk: this artifact records provenance and feasibility, never a runtime
// result. While no deterministic inference has run, RuntimeWitnessed must stay false so
// #4781 cannot read a provisioning record as a throughput witness.
func TestDeepSeekInventoryIsNotARuntimeWitness(t *testing.T) {
	inv := DeepSeekV4ProInventory()
	if inv.RuntimeWitnessed {
		t.Error("runtime_witnessed = true, but no weights have been transferred or loaded; " +
			"placement is refused and the artifact is transfer-refused pending #4801")
	}
	if inv.RuntimeWitnessed && !inv.Admissible() {
		t.Error("runtime_witnessed = true while placement is refused — impossible by construction")
	}
	// The upstream path documents no CPU/NVMe offload, which is what makes aggregate HBM
	// the binding ceiling rather than a tunable.
	if inv.Runtime.CPUOffload {
		t.Error("runtime.cpu_offload = true; upstream documents no CPU/NVMe offload recipe, " +
			"so the HBM ceiling cannot be sidestepped by offloading")
	}
}

// TestDeepSeekRuntimeMinimaAreComparableVersions keeps the recorded minima machine-usable:
// a bare version a consumer can compare, not a prose constraint it would have to parse.
// Runtime compatibility is separately satisfied on the witnessed nodes — recording it as
// data keeps a future runtime regression distinguishable from today's placement refusal.
func TestDeepSeekRuntimeMinimaAreComparableVersions(t *testing.T) {
	rt := DeepSeekV4ProInventory().Runtime
	for name, v := range map[string]string{
		"torch_min":        rt.TorchMin,
		"transformers_min": rt.TransformersMin,
		"safetensors_min":  rt.SafetensorsMin,
	} {
		if v == "" {
			t.Errorf("%s is empty; an unrecorded minimum cannot be checked before a transfer", name)
			continue
		}
		for _, op := range []string{">", "<", "=", "~", "^", " "} {
			if bytes.Contains([]byte(v), []byte(op)) {
				t.Errorf("%s = %q contains the operator %q; the field name already means "+
					"'minimum', so the value must be a bare comparable version", name, v, op)
			}
		}
	}
	if got, want := rt.TensorParallel, 8; got != want {
		t.Errorf("tensor_parallel = %d, want %d (the documented MP= / --nproc-per-node)", got, want)
	}
	if got, want := rt.Experts, 384; got != want {
		t.Errorf("runtime.experts = %d, want %d (the documented EXPERTS= for conversion)", got, want)
	}
	if !rt.MultiNode {
		t.Error("runtime.multi_node = false; upstream documents a multi-node launch, which is " +
			"the route the placement reservation depends on")
	}
}

// TestDeepSeekInventoryDigestSelfVerifies proves the artifact carries its own integrity
// check, so a consumer that reads it off disk or the wire can tell it was not edited in
// transit without a sidecar.
func TestDeepSeekInventoryDigestSelfVerifies(t *testing.T) {
	inv := DeepSeekV4ProInventory()
	if inv.Digest == "" {
		t.Fatal("digest is empty; the artifact must self-verify")
	}
	if !inv.VerifyDigest() {
		t.Errorf("digest %q does not match the inventory content", inv.Digest)
	}
	// A tampered field must break the digest — otherwise the check is decorative.
	tampered := inv
	tampered.Revision = "0000000000000000000000000000000000000000"
	if tampered.VerifyDigest() {
		t.Error("digest still verified after the revision was altered; it does not cover provenance")
	}
	tampered = inv
	tampered.RuntimeWitnessed = true
	if tampered.VerifyDigest() {
		t.Error("digest still verified after runtime_witnessed was flipped; it does not cover the " +
			"field that separates a provisioning record from a runtime claim")
	}
}

// TestDeepSeekInventoryRoundTrips proves the inventory survives JSON in both directions —
// the property that makes it a machine-readable handoff #4781 can consume rather than a
// Go-only struct.
func TestDeepSeekInventoryRoundTrips(t *testing.T) {
	inv := DeepSeekV4ProInventory()
	b, err := json.Marshal(inv)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got DeepSeekInventory
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !got.VerifyDigest() {
		t.Error("digest did not survive the JSON round trip")
	}
	if got.Schema != DeepSeekInventorySchema {
		t.Errorf("schema = %q, want %q", got.Schema, DeepSeekInventorySchema)
	}
	if got.AdmissionSummary != inv.AdmissionSummary || got.Revision != inv.Revision {
		t.Error("admission summary or revision drifted across the round trip")
	}
}

// TestDeepSeekInventoryScrubbed guards the #4788 scrub requirement mechanically: the
// public artifact carries node-class labels only — no hosts, mounts, channels, or tokens.
// The read-backs behind these numbers came over a private bridge; the numbers are public,
// the route is not.
func TestDeepSeekInventoryScrubbed(t *testing.T) {
	b, err := json.Marshal(DeepSeekV4ProInventory())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	blob := bytes.ToLower(b)
	// Needles for the leak classes the DoD names: an FQDN/host, an absolute private
	// mount, a bridge channel, or a credential. Credential needles are qualified
	// ("auth_token", not a bare "token") so a legitimate field like experts_per_token
	// is not a false hit — a scrub check that cries wolf gets disabled, not fixed.
	for _, needle := range []string{
		".local", ".internal", "http://", "https://", "ssh://", "@",
		"/mnt/", "/projects/", "/home/", "c:\\",
		"auth_token", "access_token", "api_token", "bearer ",
		"password", "secret", "api_key", "apikey", "private_key",
		// The unconditional hardware tells tools/scrub_hardware_names.py names: these
		// strings are only ever the operator's private lab box, and they carry no
		// capacity information the HBM fields do not already state.
		"sxm4", "dgx", "da33",
	} {
		if bytes.Contains(blob, []byte(needle)) {
			t.Errorf("inventory leaks %q; the public artifact carries node-class labels only", needle)
		}
	}
}

// TestDeepSeekInventoryGolden pins the machine-readable artifact #4788 owes #4781 — the
// recorded inventory itself, reviewable as data in the diff. Regenerate with
// UPDATE_GOLDEN=1 only when a provisioning fact genuinely changed (a new node read-back,
// a new revision), never to paper over an unexpected drift.
func TestDeepSeekInventoryGolden(t *testing.T) {
	got, err := json.MarshalIndent(DeepSeekV4ProInventory(), "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	golden := filepath.Join("testdata", "deepseek_v4_pro_inventory.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("inventory drifted from golden %s — a provisioning fact changed; re-run with "+
			"UPDATE_GOLDEN=1 ONLY if intended:\n got: %s\nwant: %s", golden, got, want)
	}
}

// TestDeepSeekInventoryGoldenIsConsumable reads the recorded artifact BACK off disk and
// treats it the way #4781 must: parse it, verify its digest, then RE-DERIVE every
// admission verdict from the capacities the file itself records. The golden test above
// only proves the code still emits these bytes; this one proves the bytes are a
// self-standing, arithmetically coherent record — a consumer reaching the same refusal
// from the file alone. It is the check that would catch a hand-edited verdict, a node
// whose recorded aggregate does not match its own rank count, or a digest left stale
// after a manual tweak.
func TestDeepSeekInventoryGoldenIsConsumable(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v4_pro_inventory.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var inv DeepSeekInventory
	if err := json.Unmarshal(raw, &inv); err != nil {
		t.Fatalf("the recorded artifact does not parse as the schema it claims: %v", err)
	}

	if got, want := inv.Schema, DeepSeekInventorySchema; got != want {
		t.Errorf("recorded schema = %q, want %q", got, want)
	}
	if !inv.VerifyDigest() {
		t.Error("the on-disk artifact fails its own digest; a consumer cannot trust it after transport")
	}
	if got, want := inv.TotalSizeBytes, ArtifactBytes; got != want {
		t.Errorf("recorded artifact = %d bytes but the placement seam reserves for %d; the two "+
			"sibling records would be describing different artifacts", got, want)
	}
	if len(inv.WitnessedNodes) == 0 || len(inv.Admission) != len(inv.WitnessedNodes) {
		t.Fatalf("recorded %d witnessed nodes and %d verdicts; every node must carry exactly one",
			len(inv.WitnessedNodes), len(inv.Admission))
	}

	for i, n := range inv.WitnessedNodes {
		got := inv.Admission[i]
		if got.Node != n.Name {
			t.Errorf("verdict %d is for %q but node %d is %q; the two lists are misaligned",
				i, got.Node, i, n.Name)
			continue
		}
		if n.FreeGPUCount > n.GPUCount {
			t.Errorf("node %s records %d free of %d ranks, which is not physically possible",
				n.Name, n.FreeGPUCount, n.GPUCount)
		}
		if want := int64(n.GPUCount) * n.HBMBytesPerGPU; got.AggregateHBMBytes != want {
			t.Errorf("node %s: recorded aggregate HBM %d, but %d ranks x %d bytes = %d",
				n.Name, got.AggregateHBMBytes, n.GPUCount, n.HBMBytesPerGPU, want)
		}
		if want := int64(n.FreeGPUCount) * n.HBMBytesPerGPU; got.FreeHBMBytes != want {
			t.Errorf("node %s: recorded free HBM %d, but %d free ranks x %d bytes = %d",
				n.Name, got.FreeHBMBytes, n.FreeGPUCount, n.HBMBytesPerGPU, want)
		}
		// The load-bearing check: the recorded verdict must be what the shared derivation
		// produces from the recorded capacity, not prose typed in beside it.
		if want := admitNode(n, inv.TotalSizeBytes); got != want {
			t.Errorf("node %s: recorded verdict does not re-derive from its own capacity\n got: %+v\nwant: %+v",
				n.Name, got, want)
		}
	}

	if got, want := inv.AdmissionSummary, summarizeAdmission(inv.Admission); got != want {
		t.Errorf("recorded summary = %s, but the recorded per-node verdicts roll up to %s", got, want)
	}
	if inv.RuntimeWitnessed {
		t.Error("the recorded artifact claims runtime_witnessed = true; no inference has run and " +
			"placement is still refused, so the file would be asserting a witness it does not carry")
	}
}
