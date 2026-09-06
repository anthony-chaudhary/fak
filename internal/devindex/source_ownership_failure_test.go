package devindex

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ExtractDevOnlySourceOwnership is an alias for ExtractDevSourceOwnership,
// exercising the fail-closed dev extraction contract.
func ExtractDevOnlySourceOwnership(root string) ([]SourceOwnership, error) {
	return ExtractDevSourceOwnership(root)
}

// ValidateRuntimeDevSplit validates the runtime/dev split command ownership and reuse registry,
// returning any refusal or error diagnostics.
func ValidateRuntimeDevSplit(verbs []Verb, inventory []CommandOwnership) []string {
	return ValidateCommandOwnership(verbs, inventory)
}

// ValidateRuntimeDevSplitBaseline validates a runtime/dev split baseline JSON payload,
// enforcing required fields, schema version, commit hash format, and nonzero metrics.
func ValidateRuntimeDevSplitBaseline(data []byte) error {
	var got splitBaseline
	if err := json.Unmarshal(data, &got); err != nil {
		return fmt.Errorf("malformed runtime/dev split baseline JSON: %w", err)
	}
	if got.Schema != "fak-runtime-dev-split-baseline/1" {
		return fmt.Errorf("invalid schema %q, want %q", got.Schema, "fak-runtime-dev-split-baseline/1")
	}
	if len(got.Commit) != 40 {
		return fmt.Errorf("invalid commit SHA %q: must be 40-character hex string", got.Commit)
	}
	if got.GoVersion == "" {
		return fmt.Errorf("missing required go_version in runtime/dev split baseline")
	}
	if got.GOOS == "" {
		return fmt.Errorf("missing required goos in runtime/dev split baseline")
	}
	if got.GOARCH == "" {
		return fmt.Errorf("missing required goarch in runtime/dev split baseline")
	}
	if got.Command != "./cmd/fak" {
		return fmt.Errorf("invalid command %q: must be ./cmd/fak", got.Command)
	}
	if got.PackageCount <= 0 {
		return fmt.Errorf("invalid package_count %d: must be positive", got.PackageCount)
	}
	if got.InternalPackageCount <= 0 {
		return fmt.Errorf("invalid internal_package_count %d: must be positive", got.InternalPackageCount)
	}
	if got.BinarySizeBytes <= 0 {
		return fmt.Errorf("invalid binary_size_bytes %d: must be positive", got.BinarySizeBytes)
	}
	if got.CleanBuildElapsedMS <= 0 {
		return fmt.Errorf("invalid clean_build_elapsed_ms %d: must be positive", got.CleanBuildElapsedMS)
	}
	if got.Provenance == "" {
		return fmt.Errorf("missing required provenance in runtime/dev split baseline")
	}
	return nil
}

func writeDevDispatchFixture(t *testing.T, fakDevFiles, devcmdFiles map[string]string) string {
	t.Helper()
	root := t.TempDir()
	cmdDir := filepath.Join(root, "cmd", "fak-dev")
	if err := os.MkdirAll(cmdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, src := range fakDevFiles {
		if err := os.WriteFile(filepath.Join(cmdDir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	devcmdDir := filepath.Join(root, "internal", "devcmd")
	if err := os.MkdirAll(devcmdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, src := range devcmdFiles {
		if err := os.WriteFile(filepath.Join(devcmdDir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestExtractDevSourceOwnershipErrorsAndRefusals(t *testing.T) {
	tests := []struct {
		name        string
		fakDevFiles map[string]string
		devcmdFiles map[string]string
		wantErr     string
	}{
		{
			name:        "MissingFakDevPackage",
			fakDevFiles: map[string]string{},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "parsed only 0 non-test file(s) under cmd/fak-dev",
		},
		{
			name:        "MissingDevcmdPackage",
			fakDevFiles: map[string]string{"main.go": "package main\nfunc run(){}\n"},
			devcmdFiles: map[string]string{},
			wantErr:     "parsed only 0 non-test file(s) under internal/devcmd",
		},
		{
			name: "MalformedGoSyntaxInFakDev",
			fakDevFiles: map[string]string{
				"main.go": "package main\nfunc run( {\n",
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "cmd/fak-dev/main.go:",
		},
		{
			name: "MissingRunHandler",
			fakDevFiles: map[string]string{
				"main.go": "package main\nfunc otherHandler(){}\n",
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "cmd/fak-dev run handler not found",
		},
		{
			name: "RunHasNoDispatchSwitch",
			fakDevFiles: map[string]string{
				"main.go": "package main\nfunc run(){\nprintln(\"no switch\")\n}\n",
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "cmd/fak-dev run has no dispatch switch",
		},
		{
			name: "NonLiteralCaseExpressionHazardous",
			fakDevFiles: map[string]string{
				"main.go": `package main
var dynamicCmd = "study-tickets"
func run(){
	switch "study-tickets" {
	case dynamicCmd:
		devcmd.CmdA()
	}
}
`,
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "non-literal fak-dev command case is hazardous",
		},
		{
			name: "CaseMixesDevAndNonDevSpellings",
			fakDevFiles: map[string]string{
				"main.go": `package main
func run(){
	switch "study-tickets" {
	case "study-tickets", "serve":
		devcmd.CmdA()
	}
}
`,
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "mixes dev and non-dev spellings",
		},
		{
			name: "UnresolvableHandlerZeroMatches",
			fakDevFiles: map[string]string{
				"main.go": `package main
func run(){
	switch "study-tickets" {
	case "study-tickets":
		unknownHandlerCall()
	}
}
`,
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\n"},
			wantErr:     "expected exactly one resolvable handler, found 0",
		},
		{
			name: "AmbiguousHandlerMultipleUnguardedMatches",
			fakDevFiles: map[string]string{
				"main.go": `package main
func run(){
	switch "study-tickets" {
	case "study-tickets":
		devcmd.CmdA()
		devcmd.CmdB()
	}
}
`,
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\nfunc CmdB(){}\n"},
			wantErr:     "expected exactly one resolvable handler, found 2",
		},
		{
			name: "DuplicateFakDevCommandAcrossClauses",
			fakDevFiles: map[string]string{
				"main.go": `package main
func run(){
	switch "study-tickets" {
	case "study-tickets":
		devcmd.CmdA()
	case "study-tickets":
		devcmd.CmdB()
	}
}
`,
			},
			devcmdFiles: map[string]string{"devcmd.go": "package devcmd\nfunc CmdA(){}\nfunc CmdB(){}\n"},
			wantErr:     `duplicate fak-dev command "study-tickets"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := writeDevDispatchFixture(t, tt.fakDevFiles, devcmdFilesMap(tt.devcmdFiles))

			// Test primary extraction function
			_, err := ExtractDevSourceOwnership(root)
			if err == nil {
				t.Fatalf("ExtractDevSourceOwnership: expected error with substring %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ExtractDevSourceOwnership: error = %q, want substring %q", err.Error(), tt.wantErr)
			}

			// Test alias extraction function
			_, aliasErr := ExtractDevOnlySourceOwnership(root)
			if aliasErr == nil {
				t.Fatalf("ExtractDevOnlySourceOwnership: expected error with substring %q, got nil", tt.wantErr)
			}
			if !strings.Contains(aliasErr.Error(), tt.wantErr) {
				t.Fatalf("ExtractDevOnlySourceOwnership: error = %q, want substring %q", aliasErr.Error(), tt.wantErr)
			}
		})
	}
}

func devcmdFilesMap(in map[string]string) map[string]string {
	if in == nil {
		return map[string]string{}
	}
	return in
}

func TestBuildRemainingExtractionReportRefusalReasonsAndErrors(t *testing.T) {
	t.Run("ReportLevelErrors", func(t *testing.T) {
		tests := []struct {
			name    string
			files   map[string]string
			wantErr string
		}{
			{
				name:    "BelowFileFloorParsedZeroFiles",
				files:   map[string]string{},
				wantErr: "parsed only 0 non-test file(s) under cmd/fak",
			},
			{
				name: "MalformedSyntaxInCmdFak",
				files: map[string]string{
					"main.go": "package main\nfunc main( {\n",
				},
				wantErr: "cmd/fak/main.go:",
			},
			{
				name: "NoRemainingTierDevHandlersFound",
				files: func() map[string]string {
					m := map[string]string{
						"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "help": cmdHelp() } }
func cmdHelp(){}
`,
					}
					for i := 1; i < vsFileFloor; i++ {
						m[fmt.Sprintf("pad_%03d.go", i)] = "package main\n"
					}
					return m
				}(),
				wantErr: "no remaining TierDev runtime handlers found",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				root := writeEdgeExtractionFixture(t, tt.files)
				_, err := BuildRemainingExtractionReport(root, nil)
				if err == nil {
					t.Fatalf("expected error with substring %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %q, want substring %q", err.Error(), tt.wantErr)
				}
			})
		}
	})

	t.Run("CandidateRefusalReasons", func(t *testing.T) {
		tests := []struct {
			name       string
			files      map[string]string
			setupPkg   func(pkg *vsPkg)
			command    string
			wantReason ExtractionReasonCode
		}{
			{
				name: "UnresolvedHandlerRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdMissing() } }
`,
				},
				setupPkg: func(pkg *vsPkg) {
					// Register synthetic handler in funcs without a file declaration
					pkg.funcs["cmdMissing"] = &ast.FuncDecl{
						Name: ast.NewIdent("cmdMissing"),
						Body: &ast.BlockStmt{},
					}
				},
				command:    "config",
				wantReason: ReasonUnresolvedHandler,
			},
			{
				name: "RuntimeOverlapRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig(); case "serve": cmdServe() } }
`,
					"config.go": "package main\nfunc cmdConfig(){ shared() }\n",
					"serve.go":  "package main\nfunc cmdServe(){ shared() }\n",
					"shared.go": "package main\nfunc shared(){}\n",
				},
				command:    "config",
				wantReason: ReasonRuntimeOverlap,
			},
			{
				name: "UnknownTierOverlapRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig(); case "unknown-fixture-verb": cmdUnknown() } }
`,
					"config.go":  "package main\nfunc cmdConfig(){ shared() }\n",
					"unknown.go": "package main\nfunc cmdUnknown(){ shared() }\n",
					"shared.go":  "package main\nfunc shared(){}\n",
				},
				command:    "config",
				wantReason: ReasonUnknownTierOverlap,
			},
			{
				name: "SharedDeclarationRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
func runtimeHelper(){ devHelper() }
`,
					"config.go": "package main\nfunc cmdConfig(){ devHelper() }\nfunc devHelper(){}\n",
				},
				command:    "config",
				wantReason: ReasonSharedDeclaration,
			},
			{
				name: "HazardCgoRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\nimport \"C\"\nfunc cmdConfig(){}\n",
				},
				command:    "config",
				wantReason: ReasonCgo,
			},
			{
				name: "HazardReflectRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\nimport \"reflect\"\nvar _ = reflect.TypeOf(1)\nfunc cmdConfig(){}\n",
				},
				command:    "config",
				wantReason: ReasonReflect,
			},
			{
				name: "HazardInitRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\nfunc init(){}\nfunc cmdConfig(){}\n",
				},
				command:    "config",
				wantReason: ReasonInit,
			},
			{
				name: "HazardLinknameRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\n//go:linkname x y\nfunc cmdConfig(){}\n",
				},
				command:    "config",
				wantReason: ReasonLinkname,
			},
			{
				name: "HazardEmbedRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\nimport _ \"embed\"\n//go:embed x\nvar x string\nfunc cmdConfig(){}\n",
				},
				command:    "config",
				wantReason: ReasonEmbed,
			},
			{
				name: "HazardSelfExecRefusal",
				files: map[string]string{
					"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
`,
					"config.go": "package main\nimport \"os\"\nfunc cmdConfig(){ _, _ = os.Executable() }\n",
				},
				command:    "config",
				wantReason: ReasonSelfExec,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				pkg := loadExtractionFixture(t, tt.files)
				if tt.setupPkg != nil {
					tt.setupPkg(pkg)
				}
				report, _, err := buildRemainingExtractionReport(pkg)
				if err != nil {
					t.Fatalf("buildRemainingExtractionReport unexpected error: %v", err)
				}
				cand := candidateForCommand(report.Excluded, tt.command)
				if cand == nil {
					t.Fatalf("candidate %q not found in Excluded list: %+v", tt.command, report)
				}
				if !hasReason(cand.Reasons, tt.wantReason) {
					t.Fatalf("candidate %q reasons = %+v, want refusal code %q", tt.command, cand.Reasons, tt.wantReason)
				}
				for _, r := range cand.Reasons {
					if r.Code == "" || r.Source == "" {
						t.Fatalf("refusal reason missing required fields: %+v", r)
					}
				}
			})
		}
	})
}

func TestCheckFreshnessAgainstHEADErrorsAndRefusals(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(t *testing.T) string
		wantErrSub string
	}{
		{
			name: "OutsideGitRepository",
			setup: func(t *testing.T) string {
				return t.TempDir()
			},
			wantErrSub: "", // non-nil error
		},
		{
			name: "CatalogRootNotGitTopLevel",
			setup: func(t *testing.T) string {
				return filepath.Join(repoRootForSurface(t), "cmd")
			},
			wantErrSub: "does not match requested catalog root",
		},
		{
			name: "EmptyGitRepositoryNoHeadCommit",
			setup: func(t *testing.T) string {
				dir := t.TempDir()
				cmd := exec.Command("git", "-C", dir, "init")
				windowgate.ConfigureBackgroundCommand(cmd)
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("git init: %v: %s", err, string(out))
				}
				return dir
			},
			wantErrSub: "", // non-nil error (HEAD does not resolve)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := tt.setup(t)
			c := &Catalog{Root: root}
			_, err := c.CheckFreshnessAgainstHEAD()
			if err == nil {
				t.Fatalf("CheckFreshnessAgainstHEAD: expected error, got nil")
			}
			if tt.wantErrSub != "" && !strings.Contains(err.Error(), tt.wantErrSub) {
				t.Fatalf("CheckFreshnessAgainstHEAD: error = %q, want substring %q", err.Error(), tt.wantErrSub)
			}
		})
	}
}

func TestValidateRuntimeDevSplitErrorsAndRefusals(t *testing.T) {
	validRuntimeItem := CommandOwnership{
		Name:              "serve",
		Owner:             OwnerRuntime,
		Rationale:         "runtime serving engine",
		CompatibilityName: "serve",
		DispatchTarget:    "fak",
		DevReuse:          DevReuseNA,
		DevReuseRationale: "runtime server component",
	}

	validDevItem := CommandOwnership{
		Name:              "study-tickets",
		Owner:             OwnerDev,
		Rationale:         "maintainer ticketing utility",
		CompatibilityName: "study-tickets",
		DispatchTarget:    "fak-dev",
		DevReuse:          DevReuseMaintainer,
		DevReuseRationale: "maintainer dev utility",
	}

	tests := []struct {
		name        string
		verbs       []Verb
		inventory   []CommandOwnership
		wantProblem string
	}{
		{
			name:  "UnknownCommandRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				validRuntimeItem,
				{
					Name:              "unknown-cmd",
					Owner:             OwnerRuntime,
					Rationale:         "unknown command",
					CompatibilityName: "unknown-cmd",
					DispatchTarget:    "fak",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "unknown",
				},
			},
			wantProblem: "unknown command: unknown-cmd",
		},
		{
			name:        "MissingCommandRefusal",
			verbs:       []Verb{{Name: "serve", Tier: TierFrontdoor}, {Name: "study-tickets", Tier: TierDev}},
			inventory:   []CommandOwnership{validRuntimeItem},
			wantProblem: "missing command: study-tickets",
		},
		{
			name:        "DuplicateCommandRefusal",
			verbs:       []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory:   []CommandOwnership{validRuntimeItem, validRuntimeItem},
			wantProblem: "duplicate command: serve",
		},
		{
			name:  "InvalidOwnerRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             CommandOwner("untrusted-owner"),
					Rationale:         "runtime serving engine",
					CompatibilityName: "serve",
					DispatchTarget:    "fak",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "runtime component",
				},
			},
			wantProblem: "invalid owner: serve",
		},
		{
			name:  "MissingRationaleRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             OwnerRuntime,
					Rationale:         "",
					CompatibilityName: "serve",
					DispatchTarget:    "fak",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "runtime component",
				},
			},
			wantProblem: "missing rationale: serve",
		},
		{
			name:  "MissingCompatibilityNameRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             OwnerRuntime,
					Rationale:         "runtime serving engine",
					CompatibilityName: "",
					DispatchTarget:    "fak",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "runtime component",
				},
			},
			wantProblem: "missing compatibility name: serve",
		},
		{
			name:  "MissingDispatchTargetRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             OwnerRuntime,
					Rationale:         "runtime serving engine",
					CompatibilityName: "serve",
					DispatchTarget:    "",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "runtime component",
				},
			},
			wantProblem: "missing dispatch target: serve",
		},
		{
			name:  "EmptyDevReuseRationaleRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             OwnerRuntime,
					Rationale:         "runtime serving engine",
					CompatibilityName: "serve",
					DispatchTarget:    "fak",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "",
				},
			},
			wantProblem: "empty dev reuse rationale: serve",
		},
		{
			name:  "UnclassifiedDevReuseActionableRecovery",
			verbs: []Verb{{Name: "study-tickets", Tier: TierDev}},
			inventory: []CommandOwnership{
				{
					Name:              "study-tickets",
					Owner:             OwnerDev,
					Rationale:         "dev command",
					CompatibilityName: "study-tickets",
					DispatchTarget:    "fak-dev",
					DevReuse:          DevReuseUnclassified,
					DevReuseRationale: "unclassified rationale",
				},
			},
			wantProblem: "new dev command \"study-tickets\" is unclassified; add it to exactly one of portableDevPatterns, maintainerDevCommands, or labDevCommands in internal/devindex/devreuse.go",
		},
		{
			name:  "DevCommandWithNotApplicableReuseRefusal",
			verbs: []Verb{{Name: "study-tickets", Tier: TierDev}},
			inventory: []CommandOwnership{
				{
					Name:              "study-tickets",
					Owner:             OwnerDev,
					Rationale:         "dev command",
					CompatibilityName: "study-tickets",
					DispatchTarget:    "fak-dev",
					DevReuse:          DevReuseNA,
					DevReuseRationale: "na rationale",
				},
			},
			wantProblem: "dev command \"study-tickets\" has not-applicable dev reuse",
		},
		{
			name:  "NonDevCommandWithDevReuseRefusal",
			verbs: []Verb{{Name: "serve", Tier: TierFrontdoor}},
			inventory: []CommandOwnership{
				{
					Name:              "serve",
					Owner:             OwnerRuntime,
					Rationale:         "runtime command",
					CompatibilityName: "serve",
					DispatchTarget:    "fak",
					DevReuse:          DevReusePortable,
					DevReuseRationale: "portable rationale",
				},
			},
			wantProblem: `non-dev command "serve" has dev reuse "portable-pattern"`,
		},
		{
			name:  "InvalidDevReuseCodeRefusal",
			verbs: []Verb{{Name: "study-tickets", Tier: TierDev}},
			inventory: []CommandOwnership{
				{
					Name:              "study-tickets",
					Owner:             OwnerDev,
					Rationale:         "dev command",
					CompatibilityName: "study-tickets",
					DispatchTarget:    "fak-dev",
					DevReuse:          DevReuse("bogus-reuse"),
					DevReuseRationale: "invalid code",
				},
			},
			wantProblem: "command \"study-tickets\" has invalid dev reuse \"bogus-reuse\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			problems := ValidateRuntimeDevSplit(tt.verbs, tt.inventory)
			found := false
			for _, p := range problems {
				if strings.Contains(p, tt.wantProblem) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("ValidateRuntimeDevSplit: problems = %+v, want substring %q", problems, tt.wantProblem)
			}
		})
	}

	t.Run("ValidRuntimeDevSplitPasses", func(t *testing.T) {
		problems := ValidateRuntimeDevSplit(
			[]Verb{{Name: "serve", Tier: TierFrontdoor}, {Name: "study-tickets", Tier: TierDev}},
			[]CommandOwnership{validRuntimeItem, validDevItem},
		)
		if len(problems) != 0 {
			t.Fatalf("unexpected problems for valid runtime/dev split: %v", problems)
		}
	})
}

func TestValidateDevReuseRegistryRefusalErrors(t *testing.T) {
	tests := []struct {
		name        string
		portable    map[string]string
		maintainer  []string
		lab         map[string]string
		wantProblem string
	}{
		{
			name:        "EmptyCommandNameRefusal",
			portable:    map[string]string{"": "some rationale"},
			wantProblem: "dev reuse registry contains an empty command name",
		},
		{
			name:        "EmptyRationaleRefusal",
			portable:    map[string]string{"cmd": ""},
			wantProblem: "dev reuse registry has empty rationale: cmd",
		},
		{
			name:        "DuplicateClassificationAcrossRegistriesRefusal",
			portable:    map[string]string{"shared-cmd": "portable rationale"},
			maintainer:  []string{"shared-cmd"},
			wantProblem: `dev reuse registry classifies "shared-cmd" more than once`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			problems := validateDevReuseRegistry(tt.portable, tt.maintainer, tt.lab)
			found := false
			for _, p := range problems {
				if strings.Contains(p, tt.wantProblem) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("validateDevReuseRegistry: problems = %+v, want substring %q", problems, tt.wantProblem)
			}
		})
	}
}

func TestValidateRuntimeDevSplitBaselineRequiredFieldsErrors(t *testing.T) {
	validBaseline := splitBaseline{
		Schema:               "fak-runtime-dev-split-baseline/1",
		Commit:               "0123456789abcdef0123456789abcdef01234567",
		GoVersion:            "go1.26",
		GOOS:                 "windows",
		GOARCH:               "amd64",
		Command:              "./cmd/fak",
		PackageCount:         42,
		InternalPackageCount: 18,
		BinarySizeBytes:      10485760,
		CleanBuildElapsedMS:  3500,
		Provenance:           "provenance-token",
	}

	tests := []struct {
		name    string
		mutate  func(b *splitBaseline)
		rawJSON string
		wantErr string
	}{
		{
			name:    "MalformedJSONError",
			rawJSON: "{not-valid-json",
			wantErr: "malformed runtime/dev split baseline JSON",
		},
		{
			name: "InvalidSchemaError",
			mutate: func(b *splitBaseline) {
				b.Schema = "fak-runtime-dev-split-baseline/999"
			},
			wantErr: "invalid schema",
		},
		{
			name: "ShortCommitSHAError",
			mutate: func(b *splitBaseline) {
				b.Commit = "short-sha"
			},
			wantErr: "invalid commit SHA",
		},
		{
			name: "MissingGoVersionError",
			mutate: func(b *splitBaseline) {
				b.GoVersion = ""
			},
			wantErr: "missing required go_version",
		},
		{
			name: "MissingGOOSError",
			mutate: func(b *splitBaseline) {
				b.GOOS = ""
			},
			wantErr: "missing required goos",
		},
		{
			name: "MissingGOARCHError",
			mutate: func(b *splitBaseline) {
				b.GOARCH = ""
			},
			wantErr: "missing required goarch",
		},
		{
			name: "InvalidCommandError",
			mutate: func(b *splitBaseline) {
				b.Command = "./cmd/fak-dev"
			},
			wantErr: "invalid command",
		},
		{
			name: "NonPositivePackageCountError",
			mutate: func(b *splitBaseline) {
				b.PackageCount = 0
			},
			wantErr: "invalid package_count",
		},
		{
			name: "NonPositiveInternalPackageCountError",
			mutate: func(b *splitBaseline) {
				b.InternalPackageCount = 0
			},
			wantErr: "invalid internal_package_count",
		},
		{
			name: "NonPositiveBinarySizeBytesError",
			mutate: func(b *splitBaseline) {
				b.BinarySizeBytes = 0
			},
			wantErr: "invalid binary_size_bytes",
		},
		{
			name: "NonPositiveCleanBuildElapsedMSError",
			mutate: func(b *splitBaseline) {
				b.CleanBuildElapsedMS = 0
			},
			wantErr: "invalid clean_build_elapsed_ms",
		},
		{
			name: "MissingProvenanceError",
			mutate: func(b *splitBaseline) {
				b.Provenance = ""
			},
			wantErr: "missing required provenance",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var data []byte
			if tt.rawJSON != "" {
				data = []byte(tt.rawJSON)
			} else {
				cp := validBaseline
				tt.mutate(&cp)
				var err error
				data, err = json.Marshal(cp)
				if err != nil {
					t.Fatal(err)
				}
			}

			err := ValidateRuntimeDevSplitBaseline(data)
			if err == nil {
				t.Fatalf("expected error with substring %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}

	t.Run("RealBaselinePassesValidation", func(t *testing.T) {
		root := repoRootForSurface(t)
		path := filepath.Join(root, "docs", "baselines", "fak-runtime-dev-split-windows-amd64.json")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateRuntimeDevSplitBaseline(data); err != nil {
			t.Fatalf("ValidateRuntimeDevSplitBaseline failed on real baseline: %v", err)
		}
	})
}
