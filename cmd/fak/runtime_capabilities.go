package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/runtimecap"
)

func runRuntimeCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("runtime-capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "runtime-capabilities")
	receiptSchema := fs.String("receipt-schema", runtimecap.Schema, "output schema: fak-runtime-capabilities/1 or fak-execution-mode-receipt/1")
	fixtureMode := fs.String("execution-mode-fixture", "", "deterministic execution-mode fixture; explicitly unwitnessed and valid only with --receipt-schema fak-execution-mode-receipt/1")
	backend := fs.String("backend", "", "require an exact registered backend name; unknown names never fall back")
	preferBackend := fs.String("prefer-backend", "", "prefer a backend; when it is unavailable, only the explicit local_cpu_degraded policy may select cpu-ref")
	fallbackPolicy := fs.String("fallback-policy", runtimecap.FallbackPolicyPinOrRefuse, "pin_or_refuse or local_cpu_degraded")
	cpuEnvelope := fs.String("cpu-envelope", "", "exact CPU fallback envelope id from supported_cpu_envelopes; evaluated before payload load")
	goos := fs.String("goos", "", "diagnostic override for operating system")
	goarch := fs.String("goarch", "", "diagnostic override for architecture")
	hostTotalRAMBytes := fs.Int64("host-total-ram-bytes", -1, "diagnostic host total RAM override for pre-load CPU-envelope admission")
	hostFreeRAMBytes := fs.Int64("host-free-ram-bytes", -1, "diagnostic host free RAM override for pre-load CPU-envelope admission")
	placement := fs.String("placement", runtimecap.PlacementLocalOnly, "local_only, prefer_local, or remote_allowed")
	remoteTarget := fs.String("remote-target", "", "exact named remote target; no connection is opened by this probe")
	authorizedRemoteTarget := fs.String("authorize-remote-target", "", "exact remote target authorized by policy")
	remoteProvider := fs.String("remote-provider", "", "declared remote provider name")
	remoteEngine := fs.String("remote-engine", "", "declared remote execution engine")
	remoteModel := fs.String("remote-model", "", "declared remote model")
	remoteEndpointClass := fs.String("remote-endpoint-class", "", "declared endpoint class, such as managed_api or private_cluster")
	remoteRegion := fs.String("remote-region", "", "declared remote region")
	remoteStateBoundary := fs.String("remote-state-boundary", "", "comma-separated data classes that would cross the boundary")
	remoteEgress := fs.String("remote-egress", "denied", "egress policy state: allowed or denied")
	remoteCredentialName := fs.String("remote-credential-name", "", "credential reference name only; secret values are never accepted or reported")
	remoteCredentialPresent := fs.Bool("remote-credential-present", false, "declare that the named credential is present in the approved secret store")
	remoteTLS := fs.String("remote-tls", "unverified", "TLS state: verified or unverified")
	remoteProxy := fs.String("remote-proxy", "none", "corporate proxy state: none, verified, or unverified")
	remoteReachability := fs.String("remote-reachability", "unknown", "independently observed target state: reachable, unreachable, or unknown")
	remoteTimeoutMS := fs.Int64("remote-timeout-ms", 0, "positive remote request timeout in milliseconds")
	remoteRetryCeiling := fs.Int("remote-retry-ceiling", 0, "bounded retry ceiling; zero disables retries")
	remoteBudgetMicroUSD := fs.Int64("remote-budget-microusd", 0, "positive request budget in millionths of a US dollar")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	if *receiptSchema != runtimecap.Schema && *receiptSchema != runtimecap.ExecutionModeReceiptSchema {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unsupported --receipt-schema %q\n", *receiptSchema)
		return 2
	}
	if strings.TrimSpace(*fixtureMode) != "" && *receiptSchema != runtimecap.ExecutionModeReceiptSchema {
		fmt.Fprintln(stderr, "fak runtime-capabilities: --execution-mode-fixture requires --receipt-schema fak-execution-mode-receipt/1")
		return 2
	}
	if strings.TrimSpace(*fixtureMode) != "" && !runtimecap.IsExecutionMode(*fixtureMode) {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unsupported --execution-mode-fixture %q\n", *fixtureMode)
		return 2
	}
	if strings.TrimSpace(*backend) != "" && strings.TrimSpace(*preferBackend) != "" {
		fmt.Fprintln(stderr, "fak runtime-capabilities: --backend cannot be combined with --prefer-backend")
		return 2
	}
	if *placement != runtimecap.PlacementLocalOnly && *placement != runtimecap.PlacementPreferLocal && *placement != runtimecap.PlacementRemoteAllowed {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unsupported --placement %q\n", *placement)
		return 2
	}
	if *fallbackPolicy != runtimecap.FallbackPolicyPinOrRefuse && *fallbackPolicy != runtimecap.FallbackPolicyLocalCPUDegrade {
		fmt.Fprintf(stderr, "fak runtime-capabilities: unsupported --fallback-policy %q\n", *fallbackPolicy)
		return 2
	}
	opts := runtimecap.Options{
		RequestedBackend:          *backend,
		PreferredBackend:          *preferBackend,
		CPUFallbackPolicy:         *fallbackPolicy,
		CPUEnvelope:               *cpuEnvelope,
		GOOS:                      *goos,
		GOARCH:                    *goarch,
		PlacementMode:             *placement,
		RemoteTarget:              *remoteTarget,
		AuthorizedTarget:          *authorizedRemoteTarget,
		RemoteProvider:            *remoteProvider,
		RemoteEngine:              *remoteEngine,
		RemoteModel:               *remoteModel,
		RemoteEndpointClass:       *remoteEndpointClass,
		RemoteRegion:              *remoteRegion,
		RemoteStateBoundary:       splitNonEmptyCSV(*remoteStateBoundary),
		RemoteEgress:              *remoteEgress,
		RemoteCredentialName:      *remoteCredentialName,
		RemoteCredentialPresent:   *remoteCredentialPresent,
		RemoteTLS:                 *remoteTLS,
		RemoteProxy:               *remoteProxy,
		RemoteReachability:        *remoteReachability,
		RemoteTimeoutMilliseconds: *remoteTimeoutMS,
		RemoteRetryCeiling:        *remoteRetryCeiling,
		RemoteBudgetMicroUSD:      *remoteBudgetMicroUSD,
	}
	if *hostTotalRAMBytes >= 0 || *hostFreeRAMBytes >= 0 {
		opts.HostMemoryOverride = true
		opts.HostMemory.Known = true
		if *hostTotalRAMBytes >= 0 {
			opts.HostMemory.TotalBytes = *hostTotalRAMBytes
		}
		if *hostFreeRAMBytes >= 0 {
			opts.HostMemory.FreeKnown = true
			opts.HostMemory.FreeBytes = *hostFreeRAMBytes
		}
	}
	report := runtimecap.Probe(opts)
	var output any = report
	if *receiptSchema == runtimecap.ExecutionModeReceiptSchema {
		if strings.TrimSpace(*fixtureMode) != "" {
			output = runtimecap.ExecutionModeFixture(*fixtureMode)
		} else {
			output = runtimecap.ExecutionModeReceiptFromReport(report)
		}
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		fmt.Fprintf(stderr, "fak runtime-capabilities: encode: %v\n", err)
		return 1
	}
	return 0
}

func splitNonEmptyCSV(value string) []string {
	var out []string
	for _, item := range strings.Split(value, ",") {
		if item = strings.TrimSpace(item); item != "" {
			out = append(out, item)
		}
	}
	return out
}
