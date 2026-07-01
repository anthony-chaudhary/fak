package compute

// scaffold.go — the `fak backend scaffold <name>` generator (issue #1685 / dev kit, parent
// #1678 the binding-layer epic). Every C-series backend (ROCm #266/C-002, TPU/ANE #261/C-004,
// OpenVINO #257/C-006) hand-wrote the SAME shape: an always-compiled arch/device taxonomy
// (canonical target token + native dtype + which Caps field it keys on, failing closed on an
// unsupported part), a cgo `//go:build <tag>` registration stub into the existing
// compute.Register/Pick seam, a parity-test skeleton mirroring cuda_test.go/vulkan_test.go, and
// a NOTES.md in the house format. This file lifts that shape into a generator so a new vendor
// starts from a green, build-clean, taxonomy-tested scaffold instead of reverse-engineering four
// existing backends.
//
// Scope: this generates the SCAFFOLD, not a working device backend. The emitted
// <name>_backend.go registration stub has TODO op bodies — filling them in and wiring a real
// device library is the vendor work the acceptance bullet in #1685 explicitly leaves out.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// Lane is a build-tag family the generator knows a starting Caps/dtype hint for. "custom" is the
// escape hatch for a vendor whose device does not fit the other four (it emits the same shape
// with a Caps{} that advertises nothing and an F32 native dtype, exactly like the CPU-parity
// floor other backends measure against).
type Lane string

const (
	LaneXLAScaffold      Lane = "xla"
	LaneHIP              Lane = "hip"
	LaneCoreMLScaffold   Lane = "coreml"
	LaneOpenVINOScaffold Lane = "openvino"
	LaneCustom           Lane = "custom"
)

// laneHint is the per-lane starting point the generator seeds into the emitted taxonomy: the
// build tag the registration stub compiles under, the native dtype a device in this lane
// typically runs, and the Caps field name the NOTES.md documents as the primary capability
// (mirroring PrimaryCap() on the three existing taxonomies). It is a STARTING POINT — the vendor
// edits the emitted arch table to match their actual device lineup.
type laneHint struct {
	tag         string // //go:build tag
	nativeDtype Dtype  // seed native dtype for the first table row
	primaryCap  string // Caps field name documented as this lane's load-bearing capability
	backendName string // the string registered via compute.Register/Pick, e.g. "rocm"
}

var laneHints = map[Lane]laneHint{
	LaneXLAScaffold:      {tag: "xla", nativeDtype: BF16, primaryCap: "GraphCompile", backendName: "xla"},
	LaneHIP:              {tag: "hip", nativeDtype: F32, primaryCap: "", backendName: "hip"},
	LaneCoreMLScaffold:   {tag: "coreml", nativeDtype: F16, primaryCap: "FusedFFN", backendName: "coreml"},
	LaneOpenVINOScaffold: {tag: "openvino", nativeDtype: F32, primaryCap: "", backendName: "openvino"},
	LaneCustom:           {tag: "custom", nativeDtype: F32, primaryCap: "", backendName: "custom"},
}

// KnownLanes returns the lane tokens the generator accepts, in the order `--lane` should list
// them, for a usage string / flag validator.
func KnownLanes() []string {
	return []string{string(LaneXLAScaffold), string(LaneHIP), string(LaneCoreMLScaffold), string(LaneOpenVINOScaffold), string(LaneCustom)}
}

// LookupLane resolves a `--lane` flag value to its hint, or (zero, false) if unrecognized — the
// same fail-closed shape the arch taxonomies use for a device string: an unknown lane must not
// silently fall back to a default tag, since that would emit a stub gated on the WRONG build tag.
func LookupLane(s string) (laneHint, bool) {
	h, ok := laneHints[Lane(strings.ToLower(strings.TrimSpace(s)))]
	return h, ok
}

// ScaffoldSpec is the fully-resolved input to Generate: a validated backend name + lane.
type ScaffoldSpec struct {
	Name string // lowercase identifier, e.g. "mychip" — becomes <name>_arch.go, taxonomy type prefix, etc.
	Lane Lane
}

// ScaffoldFile is one emitted file: its path relative to the target directory, and its content.
type ScaffoldFile struct {
	RelPath string
	Content string
}

// ValidateName fails closed on a name that would not produce a valid Go identifier / file stem —
// the same "reject the unsupported part, never guess" discipline the arch normalizers use. Only
// lowercase ASCII letters and digits, starting with a letter, are accepted: the emitted Go type
// names (e.g. MychipArch) and file names (mychip_arch.go) depend on this shape.
func ValidateName(name string) error {
	if name == "" {
		return fmt.Errorf("backend name must not be empty")
	}
	for i, r := range name {
		if unicode.IsUpper(r) {
			return fmt.Errorf("backend name %q must be lowercase (got uppercase %q)", name, r)
		}
		isLetter := r >= 'a' && r <= 'z'
		isDigit := r >= '0' && r <= '9'
		if i == 0 && !isLetter {
			return fmt.Errorf("backend name %q must start with a lowercase letter", name)
		}
		if !isLetter && !isDigit {
			return fmt.Errorf("backend name %q must be [a-z][a-z0-9]* (got %q)", name, r)
		}
	}
	return nil
}

// exportedPrefix titlecases the first rune of name for use in exported Go identifiers
// (mychip -> Mychip), so the taxonomy types don't collide with the lowercase package-private
// table/map names the template also emits.
func exportedPrefix(name string) string {
	if name == "" {
		return name
	}
	r := []rune(name)
	r[0] = unicode.ToUpper(r[0])
	return string(r)
}

// upperName upper-cases name for the NOTES.md title and the C-series doc id
// (mychip -> MYCHIP), matching ROCM-C002-NOTES.md / TPU-C004-NOTES.md / OPENVINO-C006-NOTES.md.
func upperName(name string) string { return strings.ToUpper(name) }

// dtypeIdent returns the Go identifier for a Dtype constant (F32, F16, BF16, ...) as declared in
// compute.go — NOT Dtype.String()'s lowercase display form ("f32"), which is not a valid Go
// expression. The emitted taxonomy's NativeDtype field must reference the real exported constant.
func dtypeIdent(d Dtype) string {
	switch d {
	case F32:
		return "F32"
	case F16:
		return "F16"
	case BF16:
		return "BF16"
	case Q8_0:
		return "Q8_0"
	case I8:
		return "I8"
	case I4:
		return "I4"
	case FP8:
		return "FP8"
	case Q4_K:
		return "Q4_K"
	default:
		return "F32"
	}
}

// Generate renders the four scaffold files for spec without touching disk — the pure function
// WriteScaffold and the golden test both call. It fails closed (returns an error) on an invalid
// name or unknown lane rather than emitting a scaffold that would not build.
func Generate(spec ScaffoldSpec) ([]ScaffoldFile, error) {
	if err := ValidateName(spec.Name); err != nil {
		return nil, err
	}
	hint, ok := LookupLane(string(spec.Lane))
	if !ok {
		return nil, fmt.Errorf("unknown lane %q; known lanes: %s", spec.Lane, strings.Join(KnownLanes(), ", "))
	}
	n := spec.Name
	up := upperName(n)
	exp := exportedPrefix(n)

	files := []ScaffoldFile{
		{RelPath: n + "_arch.go", Content: renderArch(n, exp, hint)},
		{RelPath: n + "_arch_test.go", Content: renderArchTest(n, exp)},
		{RelPath: n + "_backend.go", Content: renderBackendStub(n, exp, hint)},
		{RelPath: up + "-NOTES.md", Content: renderNotes(n, up, exp, hint)},
	}
	return files, nil
}

// WriteScaffold generates spec's files and writes them under dir (created if absent), refusing
// to overwrite a file that already exists there — the same fail-closed stance as the taxonomy
// lookups: a generator that silently clobbers an existing hand-written backend is worse than one
// that refuses. Returns the paths written, in the order of Generate's file list.
func WriteScaffold(dir string, spec ScaffoldSpec) ([]string, error) {
	files, err := Generate(spec)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("scaffold: create dir %s: %w", dir, err)
	}
	var written []string
	for _, f := range files {
		full := filepath.Join(dir, f.RelPath)
		if _, statErr := os.Stat(full); statErr == nil {
			return written, fmt.Errorf("scaffold: refusing to overwrite existing file %s", full)
		}
		if err := os.WriteFile(full, []byte(f.Content), 0o644); err != nil {
			return written, fmt.Errorf("scaffold: write %s: %w", full, err)
		}
		written = append(written, full)
	}
	return written, nil
}

// renderArch is the always-compiled taxonomy half: one supported target row (the vendor's
// device string mapping to a canonical token + native dtype), a fail-closed Lookup/Token
// normalizer, and a KnownXArches enumerator — the exact shape of rocm_arch.go / tpu_arch.go /
// openvino_arch.go, minimized to one seed row the vendor extends.
func renderArch(name, exp string, hint laneHint) string {
	var b strings.Builder
	fmt.Fprintf(&b, "package compute\n\n")
	fmt.Fprintf(&b, "// %s_arch.go — GENERATED by `fak backend scaffold %s --lane %s` (issue #1685).\n", name, name, hint.tag)
	fmt.Fprintf(&b, "// This is the always-compiled, hardware-independent half of the %s backend: the\n", name)
	fmt.Fprintf(&b, "// device/arch taxonomy a %s build needs BEFORE any kernel runs. It has no build tag and\n", strings.ToUpper(hint.tag))
	fmt.Fprintf(&b, "// is unit-witnessed on any host — the same split ROCM-C002-NOTES.md / TPU-C004-NOTES.md /\n")
	fmt.Fprintf(&b, "// OPENVINO-C006-NOTES.md use (ship the exact, host-tractable part; defer the device run).\n")
	fmt.Fprintf(&b, "// EDIT THIS: replace the single seed row in %sArches with your real device lineup, and\n", name)
	fmt.Fprintf(&b, "// extend %sFamily with your actual generations.\n\n", exp)

	fmt.Fprintf(&b, "// %sFamily is a %s device generation fak can target.\n", exp, name)
	fmt.Fprintf(&b, "type %sFamily uint8\n\n", exp)
	fmt.Fprintf(&b, "const (\n")
	fmt.Fprintf(&b, "\t// %sUnknown is the zero value: a device string fak has no target for.\n", exp)
	fmt.Fprintf(&b, "\t%sUnknown %sFamily = iota\n", exp, exp)
	fmt.Fprintf(&b, "\t// %sGen1 is the seed generation row — RENAME to your first real device family.\n", exp)
	fmt.Fprintf(&b, "\t%sGen1\n", exp)
	fmt.Fprintf(&b, ")\n\n")

	fmt.Fprintf(&b, "// String returns the short generation label.\n")
	fmt.Fprintf(&b, "func (f %sFamily) String() string {\n", exp)
	fmt.Fprintf(&b, "\tswitch f {\n")
	fmt.Fprintf(&b, "\tcase %sGen1:\n", exp)
	fmt.Fprintf(&b, "\t\treturn \"gen1\"\n")
	fmt.Fprintf(&b, "\tdefault:\n")
	fmt.Fprintf(&b, "\t\treturn \"unknown\"\n")
	fmt.Fprintf(&b, "\t}\n}\n\n")

	fmt.Fprintf(&b, "// %sArch is one supported %s compile target: its canonical target token, its family,\n", exp, name)
	fmt.Fprintf(&b, "// the device-native compute dtype, and a representative product.\n")
	fmt.Fprintf(&b, "type %sArch struct {\n", exp)
	fmt.Fprintf(&b, "\tTarget      string      // canonical target token, e.g. %q\n", name+"-gen1")
	fmt.Fprintf(&b, "\tFamily      %sFamily // generation\n", exp)
	fmt.Fprintf(&b, "\tNativeDtype Dtype       // device-native compute tier\n")
	fmt.Fprintf(&b, "\tExamples    string      // representative product(s)\n")
	fmt.Fprintf(&b, "\taliases     []string    // device-reported spellings (already normalized) that resolve here\n")
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "// %sArches is the supported-target table, declared once. EDIT THIS: one seed row —\n", name)
	fmt.Fprintf(&b, "// add every real device your backend targets.\n")
	fmt.Fprintf(&b, "var %sArches = []%sArch{\n", name, exp)
	fmt.Fprintf(&b, "\t{Target: %q, Family: %sGen1, NativeDtype: %s, Examples: \"TODO: representative product\", aliases: []string{%q}},\n",
		name+"-gen1", exp, dtypeIdent(hint.nativeDtype), name+"gen1")
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "// %sByKey indexes the table by the canonical target token AND every alias (all\n", name)
	fmt.Fprintf(&b, "// normalized) for O(1) lookup of a noisy device-reported string.\n")
	fmt.Fprintf(&b, "var %sByKey = func() map[string]%sArch {\n", name, exp)
	fmt.Fprintf(&b, "\tm := make(map[string]%sArch, len(%sArches)*2)\n", exp, name)
	fmt.Fprintf(&b, "\tfor _, a := range %sArches {\n", name)
	fmt.Fprintf(&b, "\t\tm[normalize%s(a.Target)] = a\n", exp)
	fmt.Fprintf(&b, "\t\tfor _, al := range a.aliases {\n")
	fmt.Fprintf(&b, "\t\t\tm[normalize%s(al)] = a\n", exp)
	fmt.Fprintf(&b, "\t\t}\n\t}\n\treturn m\n}()\n\n")

	fmt.Fprintf(&b, "// normalize%s canonicalizes a device-reported string to a compact lowercase key: it\n", exp)
	fmt.Fprintf(&b, "// lowercases and strips whitespace/separators. It does not invent a target: an\n")
	fmt.Fprintf(&b, "// unrecognized key is returned for Lookup to reject (fail closed). EDIT THIS if your\n")
	fmt.Fprintf(&b, "// device reports noisier strings (case, suffixes, instance ordinals — see\n")
	fmt.Fprintf(&b, "// rocm_arch.go's normalizeGFX or openvino_arch.go's normalizeOVDevice for patterns).\n")
	fmt.Fprintf(&b, "func normalize%s(s string) string {\n", exp)
	fmt.Fprintf(&b, "\tout := make([]byte, 0, len(s))\n")
	fmt.Fprintf(&b, "\tfor i := 0; i < len(s); i++ {\n")
	fmt.Fprintf(&b, "\t\tc := s[i]\n")
	fmt.Fprintf(&b, "\t\tswitch c {\n\t\tcase ' ', '\\t', '\\n', '\\r', '-', '_', '.':\n\t\t\tcontinue\n\t\t}\n")
	fmt.Fprintf(&b, "\t\tif c >= 'A' && c <= 'Z' {\n\t\t\tc += 'a' - 'A'\n\t\t}\n")
	fmt.Fprintf(&b, "\t\tout = append(out, c)\n\t}\n\treturn string(out)\n}\n\n")

	fmt.Fprintf(&b, "// Lookup%sArch resolves a device-reported string (any case, separators) to its\n", exp)
	fmt.Fprintf(&b, "// supported target, or (zero, false) if fak has no target for it. This is the\n")
	fmt.Fprintf(&b, "// fail-closed admission a build/runtime path uses so an unknown device is never\n")
	fmt.Fprintf(&b, "// silently compiled for the wrong target.\n")
	fmt.Fprintf(&b, "func Lookup%sArch(name string) (%sArch, bool) {\n", exp, exp)
	fmt.Fprintf(&b, "\ta, ok := %sByKey[normalize%s(name)]\n", name, exp)
	fmt.Fprintf(&b, "\treturn a, ok\n}\n\n")

	fmt.Fprintf(&b, "// %sToken returns the canonical target token for a device-reported string, or\n", exp)
	fmt.Fprintf(&b, "// (\"\", false) if unsupported. The deferred build/registration path feeds this to the\n")
	fmt.Fprintf(&b, "// vendor toolchain; a `--backend %s --device <token>` selector accepts the same token.\n", name)
	fmt.Fprintf(&b, "func %sToken(name string) (string, bool) {\n", exp)
	fmt.Fprintf(&b, "\ta, ok := Lookup%sArch(name)\n", exp)
	fmt.Fprintf(&b, "\tif !ok {\n\t\treturn \"\", false\n\t}\n")
	fmt.Fprintf(&b, "\treturn a.Target, true\n}\n\n")

	fmt.Fprintf(&b, "// Known%sArches returns the supported-target table in declared order. A `fak`\n", exp)
	fmt.Fprintf(&b, "// diagnostic or the backend's build script enumerates it to print exactly which\n")
	fmt.Fprintf(&b, "// %s devices this build of fak targets.\n", name)
	fmt.Fprintf(&b, "func Known%sArches() []%sArch {\n", exp, exp)
	fmt.Fprintf(&b, "\tout := make([]%sArch, len(%sArches))\n", exp, name)
	fmt.Fprintf(&b, "\tcopy(out, %sArches)\n", name)
	fmt.Fprintf(&b, "\treturn out\n}\n")

	return b.String()
}

// renderArchTest is the host-tractable witness suite: taxonomy round-trip, normalization, and
// the fail-closed-on-unsupported-input invariant — green on any host immediately after
// generation, mirroring rocm_arch_test.go / tpu_arch_test.go's shape.
func renderArchTest(name, exp string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "package compute\n\n")
	fmt.Fprintf(&b, "import \"testing\"\n\n")
	fmt.Fprintf(&b, "// %s_arch_test.go — GENERATED by `fak backend scaffold %s`. Host-tractable witnesses\n", name, name)
	fmt.Fprintf(&b, "// for the %s device taxonomy: every assertion here is hardware-independent, so this\n", name)
	fmt.Fprintf(&b, "// file is green on any host right after generation, before any device code exists.\n\n")

	fmt.Fprintf(&b, "func Test%sArchLookupKnown(t *testing.T) {\n", exp)
	fmt.Fprintf(&b, "\tfor _, a := range Known%sArches() {\n", exp)
	fmt.Fprintf(&b, "\t\tgot, ok := Lookup%sArch(a.Target)\n", exp)
	fmt.Fprintf(&b, "\t\tif !ok {\n\t\t\tt.Fatalf(\"Lookup%sArch(%%q): not found, want supported\", a.Target)\n\t\t}\n", exp)
	fmt.Fprintf(&b, "\t\tif got.Target != a.Target {\n\t\t\tt.Errorf(\"Target = %%q, want %%q\", got.Target, a.Target)\n\t\t}\n")
	fmt.Fprintf(&b, "\t\tfor _, al := range a.aliases {\n")
	fmt.Fprintf(&b, "\t\t\tif _, ok := Lookup%sArch(al); !ok {\n\t\t\t\tt.Errorf(\"alias %%q for %%q did not resolve\", al, a.Target)\n\t\t\t}\n\t\t}\n", exp)
	fmt.Fprintf(&b, "\t}\n}\n\n")

	fmt.Fprintf(&b, "// Test%sArchNormalizationRejectsUnsupported pins the fail-closed contract: an\n", exp)
	fmt.Fprintf(&b, "// unrecognized device string must never resolve to a target.\n")
	fmt.Fprintf(&b, "func Test%sArchNormalizationRejectsUnsupported(t *testing.T) {\n", exp)
	fmt.Fprintf(&b, "\tfor _, in := range []string{\"\", \"not-a-real-device\", \"unknown-gen99\"} {\n")
	fmt.Fprintf(&b, "\t\tif got, ok := %sToken(in); ok {\n", exp)
	fmt.Fprintf(&b, "\t\t\tt.Errorf(\"%sToken(%%q) = (%%q,true), want unsupported\", in, got)\n\t\t}\n\t}\n}\n\n", exp)

	fmt.Fprintf(&b, "// Test%sArchTokenRoundTrips checks the canonical token normalizes to itself and that\n", exp)
	fmt.Fprintf(&b, "// case noise on a known alias still resolves.\n")
	fmt.Fprintf(&b, "func Test%sArchTokenRoundTrips(t *testing.T) {\n", exp)
	fmt.Fprintf(&b, "\tarches := Known%sArches()\n", exp)
	fmt.Fprintf(&b, "\tif len(arches) == 0 {\n\t\tt.Fatal(\"Known%sArches() returned no rows; the seed row was removed without a replacement\")\n\t}\n", exp)
	fmt.Fprintf(&b, "\tfirst := arches[0]\n")
	fmt.Fprintf(&b, "\tif tok, ok := %sToken(first.Target); !ok || tok != first.Target {\n", exp)
	fmt.Fprintf(&b, "\t\tt.Errorf(\"%sToken(%%q) = (%%q,%%v), want (%%q,true)\", first.Target, tok, ok, first.Target)\n\t}\n}\n", exp)

	return b.String()
}

// renderBackendStub is the cgo `//go:build <tag>` registration half: it registers an Approx
// backend under compute.Register the moment the vendor fills in the TODO op bodies and links a
// real device library. It deliberately does NOT compile by default (the build tag excludes it
// from `go build ./...`), mirroring rocm.go/openvino.go's deferred-integration shape described
// in the C-series NOTES.md files: adding a backend is a registration, never a forward-loop edit.
func renderBackendStub(name, exp string, hint laneHint) string {
	var b strings.Builder
	fmt.Fprintf(&b, "//go:build %s\n\n", hint.tag)
	fmt.Fprintf(&b, "// %s_backend.go — GENERATED by `fak backend scaffold %s --lane %s` (issue #1685).\n", name, name, hint.tag)
	fmt.Fprintf(&b, "// This is the cgo registration stub for the %s backend: it registers an Approx\n", name)
	fmt.Fprintf(&b, "// backend named %q into the EXISTING compute.Register/Pick seam, unchanged — adding a\n", hint.backendName)
	fmt.Fprintf(&b, "// backend is a registration, never a forward-loop edit, exactly as cuda.go/vulkan.go/\n")
	fmt.Fprintf(&b, "// metal.go do. TODO: every op body below panics with \"not implemented\"; replace each\n")
	fmt.Fprintf(&b, "// with a call into your vendor's device library (see cuda.go / vulkan.go for the cgo\n")
	fmt.Fprintf(&b, "// wrapper shape, or metal.go for a from-scratch Go+ObjC example). Until then this file\n")
	fmt.Fprintf(&b, "// only compiles under -tags %s; the default `go build ./...` excludes it entirely.\n\n", hint.tag)
	fmt.Fprintf(&b, "package compute\n\n")

	fmt.Fprintf(&b, "func init() {\n")
	fmt.Fprintf(&b, "\t// TODO: probe for a reachable device here; return without registering if none is\n")
	fmt.Fprintf(&b, "\t// found, so a binary built with -tags %s on a host with no device still falls back\n", hint.tag)
	fmt.Fprintf(&b, "\t// to cpu-ref (the Reference Default), exactly as metal.go's init does when MPS is\n")
	fmt.Fprintf(&b, "\t// unavailable.\n")
	fmt.Fprintf(&b, "\tRegister(&%sBackend{name: %q, tier: \"TODO\"})\n", name, hint.backendName)
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "// %sBackend implements compute.Backend. It is an Approx backend (never Reference —\n", exp)
	fmt.Fprintf(&b, "// RequireReference must stay false for every device backend; see compute.go's\n")
	fmt.Fprintf(&b, "// CorrectnessClass doc).\n")
	fmt.Fprintf(&b, "type %sBackend struct {\n\tname string\n\ttier string\n}\n\n", name)

	fmt.Fprintf(&b, "func (c *%sBackend) Name() string            { return c.name }\n", name)
	fmt.Fprintf(&b, "func (c *%sBackend) Tier() string            { return c.tier }\n", name)
	fmt.Fprintf(&b, "func (c *%sBackend) Class() CorrectnessClass { return Approx }\n", name)
	fmt.Fprintf(&b, "func (c *%sBackend) Caps() Caps              { return Caps{} } // TODO: advertise this device's real capabilities", name)
	if hint.primaryCap != "" {
		fmt.Fprintf(&b, " (likely %s: true)", hint.primaryCap)
	}
	fmt.Fprintf(&b, "\n\n")

	notImpl := func(sig, name2 string) {
		fmt.Fprintf(&b, "func (c *%sBackend) %s { panic(\"compute: %s.%s not implemented (scaffold TODO)\") }\n", name, sig, name, name2)
	}
	notImpl("Upload(t Tensor, as Dtype) Tensor", "Upload")
	notImpl("Host(t Tensor) ([]float32, bool)", "Host")
	notImpl("Read(t Tensor) []float32", "Read")
	notImpl("Free(t Tensor)", "Free")
	notImpl("NewKV(cfg KVConfig) KVStore", "NewKV")
	notImpl("MatMul(w, x Tensor) Tensor", "MatMul")
	notImpl("BatchedMatMul(w, X Tensor, P int) Tensor", "BatchedMatMul")
	notImpl("RMSNorm(x, weight Tensor, eps float32) Tensor", "RMSNorm")
	notImpl("RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor", "RoPE")
	notImpl("SwiGLU(gate, up Tensor) Tensor", "SwiGLU")
	notImpl("AddInPlace(dst, src Tensor)", "AddInPlace")
	notImpl("AddBias(dst, bias Tensor)", "AddBias")
	notImpl("Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor", "Attention")
	notImpl("Argmax(logits Tensor) int", "Argmax")

	return b.String()
}

// renderNotes emits the C-series house-format NOTES.md skeleton (mirroring ROCM-C002-NOTES.md /
// TPU-C004-NOTES.md / OPENVINO-C006-NOTES.md): what shipped (the taxonomy), what's blocked (the
// device run), and the hand-off to the next agent on a node with the real hardware.
func renderNotes(name, up, exp string, hint laneHint) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s backend — scaffold (fak backend scaffold, issue #1685)\n\n", up)
	fmt.Fprintf(&b, "**Status:** GENERATED scaffold. The taxonomy half (`%s_arch.go` + `%s_arch_test.go`) is\n", name, name)
	fmt.Fprintf(&b, "real Go, host-tractable, and unit-witnessed on any host. The registration half\n")
	fmt.Fprintf(&b, "(`%s_backend.go`, `//go:build %s`) is a STUB — every op panics \"not implemented\" until\n", name, hint.tag)
	fmt.Fprintf(&b, "you fill it in against a real device / toolchain. Nothing here has run on real %s\n", name)
	fmt.Fprintf(&b, "silicon; no throughput or correctness number has been measured, estimated, or fabricated.\n\n")

	fmt.Fprintf(&b, "## What this scaffold gives you\n\n")
	fmt.Fprintf(&b, "| Piece | File | Status |\n|---|---|---|\n")
	fmt.Fprintf(&b, "| Device/arch taxonomy | `%s_arch.go` | shipped — extend `%sArches` with your real device lineup |\n", name, name)
	fmt.Fprintf(&b, "| Taxonomy tests | `%s_arch_test.go` | shipped — green now; add cases as you extend the table |\n", name)
	fmt.Fprintf(&b, "| cgo registration stub | `%s_backend.go` | stub — `//go:build %s`, registers %q via compute.Register, every op TODO |\n", name, hint.tag, hint.backendName)
	fmt.Fprintf(&b, "| This notes file | `%s-NOTES.md` | you are here |\n\n", up)

	fmt.Fprintf(&b, "## Next steps (on a node with the real device)\n\n")
	fmt.Fprintf(&b, "1. Extend `%sArches` (`%s_arch.go`) with your actual device generations, replacing the\n", name, name)
	fmt.Fprintf(&b, "   %sGen1 seed row. Add the noisy device-reported spellings as `aliases` so\n", exp)
	fmt.Fprintf(&b, "   `Lookup%sArch` normalizes them (see `rocm_arch.go`'s `normalizeGFX` or\n", exp)
	fmt.Fprintf(&b, "   `openvino_arch.go`'s `normalizeOVDevice` for the noisy-string patterns to copy).\n")
	fmt.Fprintf(&b, "2. Fill in `%s_backend.go`'s op bodies against your vendor's device library — mirror\n", name)
	fmt.Fprintf(&b, "   `cuda.go`/`vulkan.go` for a cgo wrapper over a flat C ABI, or `metal.go` for a\n")
	fmt.Fprintf(&b, "   from-scratch Go+native example. Keep `cpu-ref` the Reference Default: nothing should\n")
	fmt.Fprintf(&b, "   run on the device unless explicitly selected (`FAK_BACKEND=%s` / `--backend %s`).\n", hint.backendName, hint.backendName)
	fmt.Fprintf(&b, "3. Add the op-level + full-forward parity witnesses (`-tags %s`) mirroring\n", hint.tag)
	fmt.Fprintf(&b, "   `cuda_test.go` / `vulkan_test.go`: hold `Argmax` to EXACT, everything else to\n")
	fmt.Fprintf(&b, "   logit-cosine + a small max|Δ| (the `Approx` correctness class — never promote a\n")
	fmt.Fprintf(&b, "   device backend to `Reference`).\n")
	fmt.Fprintf(&b, "4. Benchmark vs the `cpu-ref` baseline (or the nearest existing backend) using the\n")
	fmt.Fprintf(&b, "   existing `WithinTarget(measured, baseline, factor)` (`prefill.go`) — no new gate needed.\n")
	fmt.Fprintf(&b, "5. `go build ./...` (default, no tag) must stay clean, and\n")
	fmt.Fprintf(&b, "   `go test ./internal/compute/ -run %sArch` must stay green — both true immediately\n", exp)
	fmt.Fprintf(&b, "   after generation, before step 1.\n")

	return b.String()
}
