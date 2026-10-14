package computebuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"reflect"
	"strings"
	"unicode"
)

const (
	V41EPOwnerBuildSupplementSchemaV1 = "fak.v41-ep-owner.build-supplement/v1"
	V41EPOwnerBuildProfile            = "linux-amd64-v1-cgo-vulkan-owner/v1"
	V41EPOwnerBuildPinSymbol          = "github.com/anthony-chaudhary/fak/internal/localbench.v41EPOwnerGuardBinarySHA256"
	V41EPOwnerBuildPinPolicy          = "elf64le-x86_64-et_exec-go-string/v1"
	V41EPOwnerBuildSupplementMaxSize  = 64 << 10
	V41EPOwnerBuildImageMaxSize       = 512 << 20
)

// V41EPOwnerBuildConfig contains only independently approved paths and the one
// helper pin. There is no flag, environment, overlay, target, or profile escape
// hatch. Planning neither inspects paths nor proves that OutputPath is fresh.
type V41EPOwnerBuildConfig struct {
	SourceRoot, GoExecutable, GOROOT string
	GOCACHE, GOMODCACHE, TempDir     string
	Home                             string
	PathDirs                         []string
	CC, CXX                          string
	VulkanInclude, VulkanLibrary     string
	OutputPath                       string
	HelperSHA256                     string
}

// V41EPOwnerBuildInvocation is a value description, never execution evidence.
// Environment is the complete, sorted environment, not an inherited overlay.
type V41EPOwnerBuildInvocation struct {
	Directory   string   `json:"directory"`
	Argv        []string `json:"argv"`
	Environment []string `json:"environment"`
	GOOS        string   `json:"goos"`
	GOARCH      string   `json:"goarch"`
	GOAMD64     string   `json:"goamd64"`
}

// V41EPOwnerBuildPlan fixes the helper-pinned production invocation. Supplying
// -X does not establish that the symbol exists or that its value was linked.
type V41EPOwnerBuildPlan struct {
	Profile      string                    `json:"profile"`
	LinkerSymbol string                    `json:"linker_symbol"`
	HelperSHA256 string                    `json:"helper_sha256"`
	Invocation   V41EPOwnerBuildInvocation `json:"invocation"`
}

// PlanV41EPOwnerBuild is pure: no environment discovery, filesystem access,
// subprocess, native load, or authority construction occurs here. A later
// executor must use this environment as a replacement, check approved bytes,
// and create an exclusive fresh output outside the source tree.
func PlanV41EPOwnerBuild(c V41EPOwnerBuildConfig) (V41EPOwnerBuildPlan, error) {
	if !validLowerSHA256(c.HelperSHA256) {
		return V41EPOwnerBuildPlan{}, fmt.Errorf("owner build requires one lowercase SHA-256 helper pin")
	}
	paths := []string{c.SourceRoot, c.GoExecutable, c.GOROOT, c.GOCACHE, c.GOMODCACHE, c.TempDir, c.Home, c.CC, c.CXX, c.VulkanInclude, c.VulkanLibrary, c.OutputPath}
	for _, p := range append(paths, c.PathDirs...) {
		if !v41EPOwnerSafePath(p) {
			return V41EPOwnerBuildPlan{}, fmt.Errorf("owner build requires canonical absolute paths without whitespace, quotes, controls, or separators")
		}
	}
	if c.GoExecutable != c.GOROOT+"/bin/go" || len(c.PathDirs) == 0 {
		return V41EPOwnerBuildPlan{}, fmt.Errorf("owner build requires the approved GOROOT go executable and explicit PATH")
	}
	seen := make(map[string]bool, len(c.PathDirs))
	for _, p := range c.PathDirs {
		if seen[p] {
			return V41EPOwnerBuildPlan{}, fmt.Errorf("owner build PATH contains a duplicate directory")
		}
		seen[p] = true
	}
	if c.OutputPath == c.SourceRoot || strings.HasPrefix(c.OutputPath, c.SourceRoot+"/") {
		return V41EPOwnerBuildPlan{}, fmt.Errorf("owner build output must be outside the source tree")
	}
	pkg := c.SourceRoot + "/internal/compute"
	env := EnvMapToSlice(map[string]string{
		"GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "CGO_ENABLED": "1",
		"GOENV": "off", "GOWORK": "off", "GOTOOLCHAIN": "local", "GOFLAGS": "",
		"GOPROXY": "off", "GOSUMDB": "off", "GOTELEMETRY": "off",
		"GOROOT": c.GOROOT, "GOCACHE": c.GOCACHE, "GOMODCACHE": c.GOMODCACHE,
		"GOTMPDIR": c.TempDir, "TMPDIR": c.TempDir, "HOME": c.Home,
		"PATH": strings.Join(c.PathDirs, ":"), "CC": c.CC, "CXX": c.CXX,
		"CGO_CFLAGS":   "-I" + c.VulkanInclude + " -I" + pkg,
		"CGO_CPPFLAGS": "", "CGO_CXXFLAGS": "-O2 -g", "CGO_FFLAGS": "",
		"CGO_LDFLAGS":      "-L" + pkg + " -lfakvulkan -L" + c.VulkanLibrary + " -lvulkan -lstdc++",
		"FAK_VULKAN_SPIRV": pkg + "/spirv", "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
	})
	return V41EPOwnerBuildPlan{
		Profile: V41EPOwnerBuildProfile, LinkerSymbol: V41EPOwnerBuildPinSymbol, HelperSHA256: c.HelperSHA256,
		Invocation: V41EPOwnerBuildInvocation{
			Directory: c.SourceRoot, GOOS: "linux", GOARCH: "amd64", GOAMD64: "v1",
			Argv:        []string{c.GoExecutable, "build", "-p=1", "-mod=readonly", "-buildvcs=false", "-tags", "vulkan", "-ldflags", "-X " + V41EPOwnerBuildPinSymbol + "=" + c.HelperSHA256, "-o", c.OutputPath, "./cmd/fak"},
			Environment: env,
		},
	}, nil
}

// Reject token-splitting and quoting in paths before embedding them in CGO or
// CC/CXX values. Linux path syntax is fixed even on a non-Linux planning host.
func v41EPOwnerSafePath(p string) bool {
	return p != "/" && path.IsAbs(p) && path.Clean(p) == p &&
		!strings.ContainsAny(p, "\\\"':=") && strings.IndexFunc(p, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsControl(r)
	}) == -1
}

// V41EPOwnerBuildFileIdentity is supplied byte identity, not a file observation.
type V41EPOwnerBuildFileIdentity struct {
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

// V41EPOwnerPreparedV19 identifies genuine prepared-input evidence and its own
// base artifact. It never says that the stock v19 invocation built the owner.
type V41EPOwnerPreparedV19 struct {
	Schema            string                       `json:"schema"`
	ReceiptSHA256     string                       `json:"receipt_sha256"`
	BaseArtifact      V41EPOwnerBuildFileIdentity  `json:"base_artifact"`
	Source            BuildSourceProvenance        `json:"source"`
	ShaderRegistry    VulkanShaderRegistryIdentity `json:"shader_registry"`
	SPIRVBundleSHA256 string                       `json:"spirv_bundle_sha256"`
	NativeArchive     V41EPOwnerBuildFileIdentity  `json:"native_archive"`
	Tools             []BuildToolIdentity          `json:"tools"`
	ToolchainSHA256   string                       `json:"toolchain_sha256"`
}

// V41EPOwnerBuildInputs must be independently approved and observed. OwnerSource
// identifies the complete clean production source containing the handoff, not a
// bare pre-handoff archive with an unrecorded overlay. A digest cannot establish
// that property; the reviewed manifest and its issuer remain orchestration's job.
// Qualification-overlay sources require a separate profile, unsupported here.
type V41EPOwnerBuildInputs struct {
	Scope                       string                      `json:"scope"`
	PreparedV19                 V41EPOwnerPreparedV19       `json:"prepared_v19"`
	OwnerSource                 BuildSourceProvenance       `json:"owner_source"`
	ReviewedInputManifestSHA256 string                      `json:"reviewed_input_manifest_sha256"`
	Helper                      V41EPOwnerBuildFileIdentity `json:"helper"`
}

// V41EPOwnerBuildSession is a declaration of an invocation and its joined exit.
// Parsing it never creates an execution-observed flag or an authenticated issuer.
type V41EPOwnerBuildSession struct {
	ReceiptSHA256 string                      `json:"receipt_sha256"`
	Invocation    V41EPOwnerBuildInvocation   `json:"invocation"`
	Finished      bool                        `json:"finished"`
	Joined        bool                        `json:"joined"`
	ExitCode      int                         `json:"exit_code"`
	Output        V41EPOwnerBuildFileIdentity `json:"output"`
}

// V41EPOwnerLinkedPin is separate from invocation evidence because a linker may
// silently ignore an -X symbol that does not exist. This leaf has no inspector.
type V41EPOwnerLinkedPin struct {
	Policy string                      `json:"policy"`
	Symbol string                      `json:"symbol"`
	Value  string                      `json:"value"`
	Owner  V41EPOwnerBuildFileIdentity `json:"owner"`
}

// V41EPOwnerBuildSupplementV1 has its own envelope, not a ComputeBuildReceipt.
// Its canonical encoding is exactly encoding/json.Marshal on this type, without
// a trailing newline. Nil optional observations must be omitted, not JSON null.
type V41EPOwnerBuildSupplementV1 struct {
	Schema            string                      `json:"schema"`
	Profile           string                      `json:"profile"`
	Inputs            V41EPOwnerBuildInputs       `json:"inputs"`
	Plan              V41EPOwnerBuildPlan         `json:"plan"`
	OwnerOutput       V41EPOwnerBuildFileIdentity `json:"owner_output"`
	BuildSession      *V41EPOwnerBuildSession     `json:"build_session,omitempty"`
	LinkedPin         *V41EPOwnerLinkedPin        `json:"linked_pin,omitempty"`
	UnavailableClaims []string                    `json:"unavailable_claims"`
}

// V41EPOwnerBuildExpected is supplied by independent trusted orchestration,
// never copied from the request or supplement being checked. This API cannot
// authenticate its caller or turn caller-supplied hashes into authority.
type V41EPOwnerBuildExpected struct {
	Config V41EPOwnerBuildConfig
	Inputs V41EPOwnerBuildInputs
}

// V41EPOwnerBuildObserved is a separate supplied-value counterpart. Gathering,
// stabilizing, and authenticating these observations is outside this pure API.
type V41EPOwnerBuildObserved struct {
	Inputs       V41EPOwnerBuildInputs
	OwnerOutput  V41EPOwnerBuildFileIdentity
	BuildSession *V41EPOwnerBuildSession
	LinkedPin    *V41EPOwnerLinkedPin
}

// V41EPOwnerBuildDiagnostic reports equality only. It contains no native
// authority, success receipt, admission token, or execution-observed boolean.
type V41EPOwnerBuildDiagnostic struct {
	SupplementSHA256    string
	IdentityComparison  string
	SessionComparison   string
	LinkedPinComparison string
	UnavailableClaims   []string
}

// V41EPOwnerBuildUnavailableClaims returns a fresh fixed list. Even matching
// separately supplied session and pin values does not authenticate observations.
func V41EPOwnerBuildUnavailableClaims() []string {
	return []string{
		"observation_provenance", "historical_build_causality", "source_completeness",
		"fresh_output", "native_archive_to_binary_linkage", "device_f32_qualification", "native_admission",
	}
}

// CheckV41EPOwnerBuildSupplementIdentity validates bounded canonical JSON and
// compares supplied values. It performs no observation itself, invokes no v19
// verifier or build, and never upgrades a diagnostic match to execution authority.
func CheckV41EPOwnerBuildSupplementIdentity(raw []byte, expected V41EPOwnerBuildExpected, observed V41EPOwnerBuildObserved) (*V41EPOwnerBuildDiagnostic, error) {
	if len(raw) == 0 || len(raw) > V41EPOwnerBuildSupplementMaxSize {
		return nil, fmt.Errorf("owner build supplement exceeds the bounded JSON contract")
	}
	var s V41EPOwnerBuildSupplementV1
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("decode owner build supplement: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return nil, fmt.Errorf("owner build supplement has trailing JSON data")
	}
	canonical, err := json.Marshal(s)
	if err != nil || !bytes.Equal(raw, canonical) {
		return nil, fmt.Errorf("owner build supplement must use exact canonical JSON")
	}
	if s.Schema != V41EPOwnerBuildSupplementSchemaV1 || s.Profile != V41EPOwnerBuildProfile || !reflect.DeepEqual(s.UnavailableClaims, V41EPOwnerBuildUnavailableClaims()) {
		return nil, fmt.Errorf("owner build supplement schema, profile, or unavailable claims mismatch")
	}
	plan, err := PlanV41EPOwnerBuild(expected.Config)
	if err != nil {
		return nil, err
	}
	if err := validateV41EPOwnerBuildInputs(expected.Inputs); err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(s.Inputs, expected.Inputs) || !reflect.DeepEqual(observed.Inputs, expected.Inputs) || s.Inputs.Helper.SHA256 != expected.Config.HelperSHA256 {
		return nil, fmt.Errorf("owner build input identities do not match independently supplied expectations and observations")
	}
	if !reflect.DeepEqual(s.Plan, plan) {
		return nil, fmt.Errorf("owner build plan differs from the fixed approved invocation")
	}
	if !validV41EPOwnerBuildFile(s.OwnerOutput) || s.OwnerOutput != observed.OwnerOutput {
		return nil, fmt.Errorf("owner build output identity mismatch")
	}
	if !reflect.DeepEqual(s.BuildSession, observed.BuildSession) || !reflect.DeepEqual(s.LinkedPin, observed.LinkedPin) {
		return nil, fmt.Errorf("owner build observation declaration lacks an exact independently supplied counterpart")
	}
	result := &V41EPOwnerBuildDiagnostic{
		IdentityComparison: "supplied-values-match", SessionComparison: "unavailable", LinkedPinComparison: "unavailable",
		UnavailableClaims: V41EPOwnerBuildUnavailableClaims(),
	}
	if session := s.BuildSession; session != nil {
		if !validLowerSHA256(session.ReceiptSHA256) || !session.Finished || !session.Joined || session.ExitCode != 0 || session.Output != s.OwnerOutput || !reflect.DeepEqual(session.Invocation, plan.Invocation) {
			return nil, fmt.Errorf("owner build session requires the fixed invocation and a finished joined zero exit with matching output")
		}
		result.SessionComparison = "supplied-values-match"
	}
	if pin := s.LinkedPin; pin != nil {
		if pin.Policy != V41EPOwnerBuildPinPolicy || pin.Symbol != V41EPOwnerBuildPinSymbol || pin.Value != plan.HelperSHA256 || pin.Owner != s.OwnerOutput {
			return nil, fmt.Errorf("owner build linked pin policy, symbol, value, or image mismatch")
		}
		result.LinkedPinComparison = "supplied-values-match"
	}
	digest := sha256.Sum256(raw)
	result.SupplementSHA256 = hex.EncodeToString(digest[:])
	return result, nil
}

func validV41EPOwnerBuildFile(f V41EPOwnerBuildFileIdentity) bool {
	return validLowerSHA256(f.SHA256) && f.SizeBytes > 0 && f.SizeBytes <= V41EPOwnerBuildImageMaxSize
}

func validateV41EPOwnerBuildInputs(in V41EPOwnerBuildInputs) error {
	base := in.PreparedV19
	if in.Scope != "production-clean-source" || base.Schema != VulkanBuildReceiptSchemaV19 || !validLowerSHA256(base.ReceiptSHA256) || !validLowerSHA256(in.ReviewedInputManifestSHA256) {
		return fmt.Errorf("owner build requires production source, a prepared v19 reference, and a reviewed input manifest")
	}
	for _, source := range []BuildSourceProvenance{base.Source, in.OwnerSource} {
		if !source.Clean || !validGitObjectID(source.GitCommit) || !validGitObjectID(source.GitTree) || !validLowerSHA256(source.SourceArchiveSHA256) {
			return fmt.Errorf("owner build source identity is incomplete")
		}
	}
	for _, file := range []V41EPOwnerBuildFileIdentity{base.BaseArtifact, base.NativeArchive, in.Helper} {
		if !validV41EPOwnerBuildFile(file) {
			return fmt.Errorf("owner build input image identity is incomplete or exceeds the size cap")
		}
	}
	registry, err := vulkanRegistryV19.identity()
	if err != nil || base.ShaderRegistry != registry || !validLowerSHA256(base.SPIRVBundleSHA256) {
		return fmt.Errorf("owner build requires the exact v19 current83 prepared bundle")
	}
	roles := []string{"go", "cc", "cxx", "ar", "glslc"}
	if len(base.Tools) != len(roles) {
		return fmt.Errorf("owner build prepared tool identities are incomplete")
	}
	for i, tool := range base.Tools {
		if tool.Role != roles[i] || tool.Executable == "" || tool.Executable == "." || tool.Executable == ".." || strings.ContainsAny(tool.Executable, "/\\") || !validLowerSHA256(tool.SHA256) {
			return fmt.Errorf("owner build prepared tool identity is invalid")
		}
	}
	toolchainSHA, err := hashJSON(struct {
		Tools      []BuildToolIdentity `json:"tools"`
		CXXRuntime string              `json:"cxx_runtime"`
		IsWindows  bool                `json:"is_windows"`
	}{base.Tools, "-lstdc++", false})
	if err != nil || base.ToolchainSHA256 != toolchainSHA {
		return fmt.Errorf("owner build prepared toolchain identity mismatch")
	}
	return nil
}
