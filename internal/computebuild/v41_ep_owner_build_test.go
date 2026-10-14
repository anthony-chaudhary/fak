package computebuild

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
// This is a supplied-value contract test. It builds and observes no executable.
func TestV41EPOwnerBuildContract(t *testing.T) {
	t.Run("fixed plan ignores ambient flags", func(t *testing.T) {
		t.Setenv("GOFLAGS", "-overlay=unapproved.json -ldflags=-X=other.symbol=value")
		s, expected, _ := v41EPOwnerBuildFixture(t)
		plan, err := PlanV41EPOwnerBuild(expected.Config)
		if err != nil {
			t.Fatal(err)
		}
		wantArgv := []string{"/tool/go/bin/go", "build", "-p=1", "-mod=readonly", "-buildvcs=false", "-tags", "vulkan", "-ldflags", "-X " + V41EPOwnerBuildPinSymbol + "=" + strings.Repeat("d", 64), "-o", "/output/owner", "./cmd/fak"}
		if !reflect.DeepEqual(plan.Invocation.Argv, wantArgv) || !reflect.DeepEqual(plan, s.Plan) {
			t.Fatalf("fixed helper-pinned argv differs: %+v", plan)
		}
		env := strings.Join(plan.Invocation.Environment, "\n")
		for _, fixed := range []string{"GOFLAGS=", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "CGO_ENABLED=1", "GOENV=off", "GOWORK=off", "GOTOOLCHAIN=local", "CC=/tool/bin/cc", "CXX=/tool/bin/c++", "LC_ALL=C", "TZ=UTC"} {
			if !strings.Contains("\n"+env+"\n", "\n"+fixed+"\n") {
				t.Fatalf("fixed environment lacks %q", fixed)
			}
		}
		if strings.Contains(env, "unapproved") || strings.Contains(env, "overlay") {
			t.Fatal("ambient settings entered the plan")
		}
		plan.Invocation.Argv[0] = "changed"
		again, err := PlanV41EPOwnerBuild(expected.Config)
		if err != nil || again.Invocation.Argv[0] != "/tool/go/bin/go" {
			t.Fatal("returned plan storage is shared")
		}
	})
	t.Run("reject token splitting and alternate paths", func(t *testing.T) {
		_, expected, _ := v41EPOwnerBuildFixture(t)
		cases := []struct {
			name string
			edit func(*V41EPOwnerBuildConfig)
		}{
			{"helper injection", func(c *V41EPOwnerBuildConfig) { c.HelperSHA256 += " -X other=bad" }},
			{"uppercase helper", func(c *V41EPOwnerBuildConfig) { c.HelperSHA256 = strings.Repeat("D", 64) }},
			{"relative compiler", func(c *V41EPOwnerBuildConfig) { c.CC = "cc" }},
			{"compiler flag", func(c *V41EPOwnerBuildConfig) { c.CC = "/tool/cc -DOTHER" }},
			{"compiler quote", func(c *V41EPOwnerBuildConfig) { c.CXX = "/tool/c++\"" }},
			{"include newline", func(c *V41EPOwnerBuildConfig) { c.VulkanInclude = "/sdk/include\n-DOTHER" }},
			{"library tab", func(c *V41EPOwnerBuildConfig) { c.VulkanLibrary = "/sdk/lib\t-Lother" }},
			{"source quote", func(c *V41EPOwnerBuildConfig) { c.SourceRoot = "/source'" }},
			{"unclean source", func(c *V41EPOwnerBuildConfig) { c.SourceRoot = "/source/../other" }},
			{"output inside source", func(c *V41EPOwnerBuildConfig) { c.OutputPath = "/source/owner" }},
			{"go outside goroot", func(c *V41EPOwnerBuildConfig) { c.GoExecutable = "/other/go" }},
			{"missing path", func(c *V41EPOwnerBuildConfig) { c.PathDirs = nil }},
			{"duplicate path", func(c *V41EPOwnerBuildConfig) { c.PathDirs = []string{"/tool/bin", "/tool/bin"} }},
			{"path separator", func(c *V41EPOwnerBuildConfig) { c.PathDirs = []string{"/tool/bin:/other"} }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				c := expected.Config
				tc.edit(&c)
				if _, err := PlanV41EPOwnerBuild(c); err == nil {
					t.Fatal("unsafe configuration accepted")
				}
			})
		}
	})
	t.Run("diagnostics never acquire observation authority", func(t *testing.T) {
		s, expected, observed := v41EPOwnerBuildFixture(t)
		if s.Inputs.PreparedV19.BaseArtifact == s.OwnerOutput {
			t.Fatal("fixture must retain the distinct stock base artifact")
		}
		result, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed)
		if err != nil || result.IdentityComparison != "supplied-values-match" || result.SessionComparison != "unavailable" || result.LinkedPinComparison != "unavailable" {
			t.Fatalf("identity-only result: %+v, %v", result, err)
		}
		if !reflect.DeepEqual(result.UnavailableClaims, V41EPOwnerBuildUnavailableClaims()) {
			t.Fatal("identity comparison upgraded unavailable claims")
		}
		s.BuildSession = &V41EPOwnerBuildSession{ReceiptSHA256: strings.Repeat("9", 64), Invocation: s.Plan.Invocation, Finished: true, Joined: true, Output: s.OwnerOutput}
		if _, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed); err == nil {
			t.Fatal("parsing a declared build session manufactured an observation")
		}
		observed.BuildSession = s.BuildSession
		result, err = CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed)
		if err != nil || result.SessionComparison != "supplied-values-match" || result.LinkedPinComparison != "unavailable" {
			t.Fatalf("invocation was incorrectly treated as a linked pin: %+v, %v", result, err)
		}
		s.LinkedPin = &V41EPOwnerLinkedPin{Policy: V41EPOwnerBuildPinPolicy, Symbol: V41EPOwnerBuildPinSymbol, Value: s.Inputs.Helper.SHA256, Owner: s.OwnerOutput}
		if _, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed); err == nil {
			t.Fatal("parsing a declared linked pin manufactured an inspection")
		}
		observed.LinkedPin = s.LinkedPin
		result, err = CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed)
		if err != nil || result.LinkedPinComparison != "supplied-values-match" || !reflect.DeepEqual(result.UnavailableClaims, V41EPOwnerBuildUnavailableClaims()) {
			t.Fatalf("pin diagnostic acquired authority: %+v, %v", result, err)
		}
		s.BuildSession.Joined = false
		if _, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed); err == nil {
			t.Fatal("matching supplied but unjoined session accepted")
		}
		s.BuildSession.Joined = true
		s.LinkedPin.Symbol = "other.symbol"
		if _, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed); err == nil {
			t.Fatal("matching supplied but wrong linked symbol accepted")
		}
	})
	t.Run("strict canonical bounded wire", func(t *testing.T) {
		s, expected, observed := v41EPOwnerBuildFixture(t)
		raw := string(v41EPOwnerBuildJSON(t, s))
		cases := map[string]string{
			"unknown":          strings.Replace(raw, "{", `{"unknown":true,`, 1),
			"duplicate":        strings.Replace(raw, "{", `{"schema":"`+s.Schema+`",`, 1),
			"nested duplicate": strings.Replace(raw, `"helper":{`, `"helper":{"size_bytes":64,`, 1),
			"case alias":       strings.Replace(raw, `"schema"`, `"Schema"`, 1),
			"null required":    strings.Replace(raw, `"scope":"production-clean-source"`, `"scope":null`, 1),
			"missing required": strings.Replace(raw, `"scope":"production-clean-source",`, "", 1),
			"null optional":    strings.Replace(raw, `"unavailable_claims":`, `"build_session":null,"unavailable_claims":`, 1),
			"trailing":         raw + `{}`, "newline": raw + "\n", "null root": "null",
			"oversize": strings.Repeat(" ", V41EPOwnerBuildSupplementMaxSize+1),
		}
		for name, bad := range cases {
			t.Run(name, func(t *testing.T) {
				if _, err := CheckV41EPOwnerBuildSupplementIdentity([]byte(bad), expected, observed); err == nil {
					t.Fatal("noncanonical or ambiguous wire accepted")
				}
			})
		}
	})
	t.Run("reject mismatched identities and widened claims", func(t *testing.T) {
		cases := []struct {
			name string
			edit func(*V41EPOwnerBuildSupplementV1, *V41EPOwnerBuildExpected, *V41EPOwnerBuildObserved)
		}{
			{"stock schema", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.Schema = VulkanBuildReceiptSchemaV19
			}},
			{"widened claims", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.UnavailableClaims = nil
			}},
			{"alternate target", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.Plan.Invocation.Argv[len(s.Plan.Invocation.Argv)-1] = "./internal/localbench"
			}},
			{"overlay", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.Plan.Invocation.Argv = append(s.Plan.Invocation.Argv, "-overlay=x.json")
			}},
			{"extra environment", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.Plan.Invocation.Environment = append(s.Plan.Invocation.Environment, "GOFLAGS=-race")
			}},
			{"platform", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.Plan.Invocation.GOARCH = "arm64"
			}},
			{"missing observation", func(_ *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, o *V41EPOwnerBuildObserved) {
				o.Inputs = V41EPOwnerBuildInputs{}
			}},
			{"output substituted", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				s.OwnerOutput = s.Inputs.PreparedV19.BaseArtifact
			}},
			{"helper differs from pin", func(_ *V41EPOwnerBuildSupplementV1, e *V41EPOwnerBuildExpected, _ *V41EPOwnerBuildObserved) {
				e.Config.HelperSHA256 = strings.Repeat("e", 64)
			}},
			{"over cap", func(s *V41EPOwnerBuildSupplementV1, _ *V41EPOwnerBuildExpected, o *V41EPOwnerBuildObserved) {
				s.OwnerOutput.SizeBytes = V41EPOwnerBuildImageMaxSize + 1
				o.OwnerOutput = s.OwnerOutput
			}},
			{"matching dirty source", func(s *V41EPOwnerBuildSupplementV1, e *V41EPOwnerBuildExpected, o *V41EPOwnerBuildObserved) {
				s.Inputs.OwnerSource.Clean = false
				e.Inputs = s.Inputs
				o.Inputs = s.Inputs
			}},
			{"matching overlay scope", func(s *V41EPOwnerBuildSupplementV1, e *V41EPOwnerBuildExpected, o *V41EPOwnerBuildObserved) {
				s.Inputs.Scope = "qualification-overlay"
				e.Inputs = s.Inputs
				o.Inputs = s.Inputs
			}},
			{"matching wrong registry", func(s *V41EPOwnerBuildSupplementV1, e *V41EPOwnerBuildExpected, o *V41EPOwnerBuildObserved) {
				s.Inputs.PreparedV19.ShaderRegistry.ModuleCount = 82
				e.Inputs = s.Inputs
				o.Inputs = s.Inputs
			}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				s, expected, observed := v41EPOwnerBuildFixture(t)
				tc.edit(&s, &expected, &observed)
				if _, err := CheckV41EPOwnerBuildSupplementIdentity(v41EPOwnerBuildJSON(t, s), expected, observed); err == nil {
					t.Fatal("invalid or mismatched contract accepted")
				}
			})
		}
	})
}

func v41EPOwnerBuildJSON(t *testing.T, s V41EPOwnerBuildSupplementV1) []byte {
	t.Helper()
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func v41EPOwnerBuildFixture(t *testing.T) (V41EPOwnerBuildSupplementV1, V41EPOwnerBuildExpected, V41EPOwnerBuildObserved) {
	t.Helper()
	// Deliberately synthetic values: they are neither receipts nor observations
	// of an actual owner build, source checkout, helper, or Vulkan installation.
	c := V41EPOwnerBuildConfig{
		SourceRoot: "/source", GoExecutable: "/tool/go/bin/go", GOROOT: "/tool/go",
		GOCACHE: "/cache/go", GOMODCACHE: "/cache/mod", TempDir: "/temp", Home: "/home/builder",
		PathDirs: []string{"/tool/bin", "/usr/bin"}, CC: "/tool/bin/cc", CXX: "/tool/bin/c++",
		VulkanInclude: "/sdk/include", VulkanLibrary: "/sdk/lib", OutputPath: "/output/owner",
		HelperSHA256: strings.Repeat("d", 64),
	}
	plan, err := PlanV41EPOwnerBuild(c)
	if err != nil {
		t.Fatal(err)
	}
	registry, err := vulkanRegistryV19.identity()
	if err != nil {
		t.Fatal(err)
	}
	tools := []BuildToolIdentity{}
	for _, role := range []string{"go", "cc", "cxx", "ar", "glslc"} {
		tools = append(tools, BuildToolIdentity{Role: role, Executable: role, SHA256: strings.Repeat("a", 64)})
	}
	toolchainSHA, err := hashJSON(struct {
		Tools      []BuildToolIdentity `json:"tools"`
		CXXRuntime string              `json:"cxx_runtime"`
		IsWindows  bool                `json:"is_windows"`
	}{tools, "-lstdc++", false})
	if err != nil {
		t.Fatal(err)
	}
	source := BuildSourceProvenance{GitCommit: strings.Repeat("1", 40), GitTree: strings.Repeat("2", 40), Clean: true, SourceArchiveSHA256: strings.Repeat("3", 64)}
	inputs := V41EPOwnerBuildInputs{
		Scope: "production-clean-source", OwnerSource: source,
		ReviewedInputManifestSHA256: strings.Repeat("4", 64), Helper: V41EPOwnerBuildFileIdentity{SHA256: c.HelperSHA256, SizeBytes: 64},
		PreparedV19: V41EPOwnerPreparedV19{
			Schema: VulkanBuildReceiptSchemaV19, ReceiptSHA256: strings.Repeat("5", 64),
			BaseArtifact: V41EPOwnerBuildFileIdentity{SHA256: strings.Repeat("6", 64), SizeBytes: 128},
			Source:       source, ShaderRegistry: registry, SPIRVBundleSHA256: strings.Repeat("7", 64),
			NativeArchive: V41EPOwnerBuildFileIdentity{SHA256: strings.Repeat("8", 64), SizeBytes: 256},
			Tools:         tools, ToolchainSHA256: toolchainSHA,
		},
	}
	s := V41EPOwnerBuildSupplementV1{
		Schema: V41EPOwnerBuildSupplementSchemaV1, Profile: V41EPOwnerBuildProfile,
		Inputs: inputs, Plan: plan, OwnerOutput: V41EPOwnerBuildFileIdentity{SHA256: strings.Repeat("b", 64), SizeBytes: 512},
		UnavailableClaims: V41EPOwnerBuildUnavailableClaims(),
	}
	return s, V41EPOwnerBuildExpected{Config: c, Inputs: inputs}, V41EPOwnerBuildObserved{Inputs: inputs, OwnerOutput: s.OwnerOutput}
}
