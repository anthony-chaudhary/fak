package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesshost"
	"github.com/anthony-chaudhary/fak/internal/harnessinit"
	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

type harnessInitCLIResult struct {
	harnessinit.Result
	ServerBinding string `json:"server_binding,omitempty"`
}

func runHarness(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "web" {
		return runHarnessWeb(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "gallery" {
		return runHarnessGallery(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "release" {
		return runHarnessRelease(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "study" {
		return runHarnessStudy(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "preview" {
		return runHarnessPreview(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "compare" {
		return runHarnessCompare(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "selfcheck" {
		return runHarnessLifecycleSelfcheck(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "inspect" {
		return runHarnessInspect(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "derive" {
		return runHarnessDerive(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "mix" {
		return runHarnessMix(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "override" {
		return runHarnessOverride(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "verify-run" {
		return runHarnessVerifyRun(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "cross-dogfood" {
		return runHarnessCrossDogfood(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "resolve" {
		return runHarnessResolve(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "compose" {
		return runHarnessCompose(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "classify" {
		return runHarnessClassify(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "discover" {
		return runHarnessDiscover(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "select" {
		return runHarnessSelect(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "protocol" {
		return runHarnessProtocol(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "init" {
		fmt.Fprintln(stderr, "usage: fak harness <init|classify|compare|compose|cross-dogfood|derive|discover|gallery|inspect|selfcheck|mix|override|preview|release|resolve|select|study|protocol|verify-run|web>")
		return 2
	}
	fs := flag.NewFlagSet("harness init", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "external product directory")
	module := fs.String("module", "", "Go module path for the product")
	version := fs.String("fak-version", harnessinit.DefaultFAKVersion, "pinned fak module version")
	host := fs.String("host", "", "seed a versioned first-party host component (codex|claude)")
	serverReceipt := fs.String("server-receipt", "", "external ready-server receipt to import immutably")
	serverModel := fs.String("server-model", "", "required model alias for --server-receipt")
	serverProtocol := fs.String("server-protocol", "", "required protocol family for --server-receipt")
	serverProtocolRevision := fs.String("server-protocol-revision", "", "required protocol revision for --server-receipt")
	serverCapabilities := fs.String("server-capabilities", "", "comma-separated required capabilities including chat.completions")
	serverMinimumGeneration := fs.Uint64("server-min-generation", 0, "minimum accepted ready-server generation")
	jsonOut := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	var binding *harnessserver.Binding
	serverOptionsPresent := *serverModel != "" || *serverProtocol != "" || *serverProtocolRevision != "" || *serverCapabilities != "" || *serverMinimumGeneration != 0
	if *serverReceipt == "" && serverOptionsPresent {
		fmt.Fprintln(stderr, "fak harness init: --server-receipt is required with server requirements")
		return 2
	}
	if *serverReceipt != "" {
		imported, err := harnessserver.Import(pathutil.ExpandTilde(*dir), pathutil.ExpandTilde(*serverReceipt), harnessserver.Requirements{
			ModelAlias:           *serverModel,
			ProtocolFamily:       *serverProtocol,
			ProtocolRevision:     *serverProtocolRevision,
			RequiredCapabilities: splitHarnessServerCapabilities(*serverCapabilities),
			MinimumGeneration:    *serverMinimumGeneration,
		})
		if err != nil {
			fmt.Fprintf(stderr, "fak harness init: server receipt: %v\n", err)
			return 1
		}
		binding = &imported
	}
	hostArtifacts, err := harnesshost.Build(*host, harnessinit.ContractVersion)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness init: %v\n", err)
		return 1
	}
	result, err := harnessinit.Init(harnessinit.Options{
		Dir: pathutil.ExpandTilde(*dir), Module: *module, FAKVersion: *version,
		Host: hostArtifacts.Host, HostManifest: hostArtifacts.Manifest, HostLock: hostArtifacts.Lock,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak harness init: %v\n", err)
		return 1
	}
	var bindingPath string
	if binding != nil {
		bindingPath = filepath.Join(result.Directory, harnessserver.BindingFileName)
		writeResult, err := harnessserver.WriteBinding(bindingPath, *binding)
		if err != nil {
			fmt.Fprintf(stderr, "fak harness init: server binding: %v\n", err)
			return 1
		}
		if writeResult.Created {
			result.Created = append(result.Created, harnessserver.BindingFileName)
		}
		if writeResult.Preserved {
			result.Preserved = append(result.Preserved, harnessserver.BindingFileName)
		}
		sort.Strings(result.Created)
		sort.Strings(result.Preserved)
	}
	cliResult := harnessInitCLIResult{Result: result, ServerBinding: bindingPath}
	if *jsonOut {
		if err := json.NewEncoder(stdout).Encode(cliResult); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "created external product at %s\nrun: cd %s && go run ./cmd/product --launch --agent-id local-agent\n", result.Directory, result.Directory)
		if bindingPath != "" {
			fmt.Fprintf(stdout, "server binding: %s\n", bindingPath)
		}
	}
	return 0
}

func splitHarnessServerCapabilities(raw string) []string {
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	capabilities := make([]string, 0, len(parts))
	for _, part := range parts {
		capabilities = append(capabilities, strings.TrimSpace(part))
	}
	return capabilities
}

func verifyHarnessServerBinding(path string) (*harnessserver.Verified, error) {
	if path == "" {
		return nil, nil
	}
	verified, err := harnessserver.VerifyFile(path)
	if err != nil {
		return nil, err
	}
	return &verified, nil
}
