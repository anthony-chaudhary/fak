//go:build linux

package llamacppinterop

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection(t *testing.T) {
	for i, arg := range os.Args {
		if arg == "strix-exchange-share-holder" && len(os.Args) == i+4 {
			if _, err := strconv.Atoi(os.Args[i+1]); err != nil {
				os.Exit(116)
			}
			if os.Args[i+3] == "zombie-leader" {
				runtime.LockOSThread()
				if syscall.Gettid() != syscall.Getpid() {
					os.Exit(122)
				}
				workerReady := make(chan struct{})
				go func() {
					runtime.LockOSThread()
					close(workerReady)
					for {
						stat, err := os.ReadFile("/proc/self/stat")
						closingParen := strings.LastIndexByte(string(stat), ')')
						fields := []string(nil)
						if closingParen >= 0 {
							fields = strings.Fields(string(stat[closingParen+1:]))
						}
						if err == nil && len(fields) > 0 && fields[0] == "Z" {
							if err := os.WriteFile(os.Args[i+2], []byte("shared"), 0o600); err != nil {
								os.Exit(117)
							}
							for {
								time.Sleep(time.Second)
							}
						}
						time.Sleep(time.Millisecond)
					}
				}()
				<-workerReady
				syscall.RawSyscall(syscall.SYS_EXIT, 0, 0, 0)
				os.Exit(123)
			}
			if os.Args[i+3] != "nodump" {
				os.Exit(120)
			}
			if _, _, errno := syscall.RawSyscall(syscall.SYS_PRCTL, uintptr(4), 0, 0); errno != 0 {
				os.Exit(121)
			}
			if err := os.WriteFile(os.Args[i+2], []byte("shared"), 0o600); err != nil {
				os.Exit(117)
			}
			for {
				time.Sleep(time.Second)
			}
		}
	}
	for marker, arg := range os.Args {
		if arg != "strix-exchange-helper" || len(os.Args) != marker+11 {
			continue
		}
		for _, rawFD := range os.Args[marker+1 : marker+4] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(110)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(111)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper lifetime owns the maps
		}
		addressPath, acceptedPath := os.Args[marker+4], os.Args[marker+5]
		closePath, closedPath, sharePath, sharedPath, shareMode := os.Args[marker+6], os.Args[marker+7], os.Args[marker+8], os.Args[marker+9], os.Args[marker+10]
		listener, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
		if err != nil {
			os.Exit(112)
		}
		defer listener.Close() //nolint:errcheck // helper exits with the authority
		if err := os.WriteFile(addressPath, []byte(listener.Addr().String()), 0o600); err != nil {
			os.Exit(113)
		}
		accepted := make([]*net.TCPConn, 0, 2)
		for range 2 {
			connection, err := listener.AcceptTCP()
			if err != nil {
				os.Exit(114)
			}
			accepted = append(accepted, connection)
			defer connection.Close() //nolint:errcheck // helper exits with the authority
		}
		if err := os.WriteFile(acceptedPath, []byte("accepted"), 0o600); err != nil {
			os.Exit(115)
		}
		shared := false
		for {
			if !shared {
				if _, err := os.Stat(sharePath); err == nil {
					listenerFile, err := listener.File()
					if err != nil {
						os.Exit(118)
					}
					command := exec.Command("/proc/self/exe", "-test.run=^TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection$", "--", "strix-exchange-share-holder", "3", sharedPath, shareMode)
					command.Env = append([]string(nil), strixAuthorityEnvironment...)
					command.ExtraFiles = []*os.File{listenerFile}
					command.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
					if err := command.Start(); err != nil {
						os.Exit(119)
					}
					_ = listenerFile.Close()
					shared = true
				}
			}
			if _, err := os.Stat(closePath); err == nil {
				_ = listener.Close()
				_ = os.WriteFile(closedPath, []byte("closed"), 0o600)
				for {
					time.Sleep(time.Second)
				}
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	if os.Getenv("FAK_STRIX_EXCHANGE_PIDNS") != "1" {
		command := exec.Command("unshare", "--user", "--map-current-user", "--pid", "--fork", "--mount-proc", os.Args[0], "-test.run=^TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection$")
		command.Env = append(os.Environ(), "FAK_STRIX_EXCHANGE_PIDNS=1")
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("run exchange authority witness in isolated pid namespace: %v\n%s", err, output)
		}
		return
	}

	dir := t.TempDir()
	write := func(name string, data []byte, mode os.FileMode) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, data, mode); err != nil {
			t.Fatal(err)
		}
		return path
	}
	digest := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return fmt.Sprintf("%x", sha256.Sum256(data))
	}
	server, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	server, err = filepath.EvalSymlinks(server)
	if err != nil {
		t.Fatal(err)
	}
	source := write("source.tar", []byte("exchange source"), 0o600)
	build := write("build.json", []byte(`{"build":"b10588","vulkan":true}`), 0o600)
	modelPath := write("model.gguf", append([]byte("model"), make([]byte, 4091)...), 0o600)
	loader := write("loader.icd", append([]byte{1}, make([]byte, 4095)...), 0o600)
	dependency := write("dependency.so", append([]byte{2}, make([]byte, 4095)...), 0o600)
	dependencies := []StrixComparatorPinnedFile{{Path: dependency, SHA256: digest(dependency)}}
	serverInfo, err := os.Stat(server)
	if err != nil {
		t.Fatal(err)
	}
	maps, err := os.Open("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]struct{}{}
	scanner := bufio.NewScanner(maps)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		path := strings.TrimSuffix(strings.Join(fields[5:], " "), " (deleted)")
		path, err = filepath.EvalSymlinks(path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if os.SameFile(info, serverInfo) {
			continue
		}
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		dependencies = append(dependencies, StrixComparatorPinnedFile{Path: path, SHA256: digest(path)})
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	_ = maps.Close()

	manifest := validStrixComparatorManifest()
	manifest.SourceArchiveSHA256, manifest.BuildManifestSHA256, manifest.ServerBinarySHA256 = digest(source), digest(build), digest(server)
	closedPath, sharePath, sharedPath := filepath.Join(dir, "listener.closed"), filepath.Join(dir, "share"), filepath.Join(dir, "shared")
	addressPath, acceptedPath, closePath := filepath.Join(dir, "address"), filepath.Join(dir, "accepted"), filepath.Join(dir, "close")
	options := StrixComparatorAuthorityOptions{
		Manifest: manifest, SourceArchivePath: source, BuildManifestPath: build, ServerBinaryPath: server, ModelPath: modelPath,
		LoaderICD: []StrixComparatorPinnedFile{{Path: loader, SHA256: digest(loader)}}, Dependencies: dependencies,
		Arguments:       []string{"-test.run=^TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection$", "--", "strix-exchange-helper", "4", "5", "6", addressPath, acceptedPath, closePath, closedPath, sharePath, sharedPath, "nodump"},
		testModelSHA256: digest(modelPath),
	}
	authority, err := OpenStrixComparatorAuthority(options)
	if err != nil {
		t.Fatal(err)
	}
	defer authority.Close() //nolint:errcheck // assertions below verify scoped cleanup
	waitFile := func(path string) string {
		t.Helper()
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			if data, err := os.ReadFile(path); err == nil {
				return string(data)
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Fatalf("timed out waiting for %s", path)
		return ""
	}
	address := waitFile(addressPath)
	dial := func() *net.TCPConn {
		t.Helper()
		connection, err := net.DialTCP("tcp4", nil, mustResolveStrixTCPAddress(t, address))
		if err != nil {
			t.Fatal(err)
		}
		return connection
	}
	first, second := dial(), dial()
	defer first.Close()  //nolint:errcheck // test-owned client
	defer second.Close() //nolint:errcheck // test-owned client
	waitFile(acceptedPath)

	loaderSet, err := strixComparatorCompleteSetDigest("loader/icd", options.LoaderICD)
	if err != nil {
		t.Fatal(err)
	}
	dependencySet, err := strixComparatorCompleteSetDigest("dependency", options.Dependencies)
	if err != nil {
		t.Fatal(err)
	}
	reversedDependencies := slices.Clone(options.Dependencies)
	slices.Reverse(reversedDependencies)
	if reversed, err := strixComparatorCompleteSetDigest("dependency", reversedDependencies); err != nil || reversed != dependencySet {
		t.Fatalf("complete-set digest is order-sensitive: got=%q err=%v want=%q", reversed, err, dependencySet)
	}
	cell := testStrixExchangeCell(t, manifest, loaderSet, dependencySet)
	exchange, err := authority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, first)
	if err != nil || !exchange.Valid() {
		t.Fatalf("bind accepted exchange: capability=%v err=%v", exchange, err)
	}
	if _, err := json.Marshal(exchange); err == nil {
		t.Fatal("exchange capability marshaled")
	}
	lookalike := *exchange
	if lookalike.Valid() {
		t.Fatal("copied exchange capability remained valid")
	}
	if new(StrixComparatorExchangeAuthority).Valid() {
		t.Fatal("constructed exchange capability became valid")
	}

	for name, mutate := range map[string]func(*qwen38quantrun.StrixComparisonCellManifest){
		"altered": func(c *qwen38quantrun.StrixComparisonCellManifest) {
			c.Reference.SourceArchiveSHA256 = fmt.Sprintf("%064x", 901)
		},
		"additional": func(c *qwen38quantrun.StrixComparisonCellManifest) {
			c.Reference.DependencySHA256 = fmt.Sprintf("%064x", 902)
		},
		"role swapped": func(c *qwen38quantrun.StrixComparisonCellManifest) {
			c.Reference.LoaderSHA256, c.Reference.DependencySHA256 = c.Reference.DependencySHA256, c.Reference.LoaderSHA256
		},
	} {
		t.Run(name+" reference", func(t *testing.T) {
			candidate := cell
			mutate(&candidate)
			candidate.Digest = ""
			candidate, err = qwen38quantrun.SealStrixComparisonCellManifest(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if got, err := authority.BindApprovedReferenceAndAcceptedConnection(candidate, candidate.Digest, first); err == nil || got != nil {
				t.Fatalf("mismatched reference result=(%v,%v)", got, err)
			}
		})
	}
	missing := cell
	missing.Reference.LoaderSHA256 = ""
	if got, err := authority.BindApprovedReferenceAndAcceptedConnection(missing, cell.Digest, first); err == nil || got != nil {
		t.Fatalf("missing reference result=(%v,%v)", got, err)
	}
	if _, err := strixComparatorCompleteSetDigest("dependency", []StrixComparatorPinnedFile{options.Dependencies[0], options.Dependencies[0]}); err == nil {
		t.Fatal("duplicate complete-set identity accepted")
	}

	unaccepted := dial()
	defer unaccepted.Close() //nolint:errcheck // test-owned client
	if got, err := authority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, unaccepted); err == nil || got != nil {
		t.Fatalf("unaccepted connection result=(%v,%v)", got, err)
	}
	unrelatedListener, err := net.ListenTCP("tcp4", &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer unrelatedListener.Close() //nolint:errcheck // test-owned listener
	unrelatedClient, err := net.DialTCP("tcp4", nil, unrelatedListener.Addr().(*net.TCPAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer unrelatedClient.Close() //nolint:errcheck // test-owned connection
	unrelatedAccepted, err := unrelatedListener.AcceptTCP()
	if err != nil {
		t.Fatal(err)
	}
	defer unrelatedAccepted.Close() //nolint:errcheck // test-owned connection
	if got, err := authority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, unrelatedClient); err == nil || got != nil {
		t.Fatalf("unrelated listener result=(%v,%v)", got, err)
	}
	platform := exchange.platform.(*linuxStrixComparatorExchangeAuthority)
	platform.connection = second
	if exchange.Valid() {
		t.Fatal("swapped connection retained exchange authority")
	}
	platform.connection = first
	if !exchange.Valid() {
		t.Fatal("restored exact connection did not restore authority")
	}
	secondExchange, err := authority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, second)
	if err != nil || !secondExchange.Valid() {
		t.Fatalf("bind second accepted exchange: capability=%v err=%v", secondExchange, err)
	}
	secondPlatform := secondExchange.platform.(*linuxStrixComparatorExchangeAuthority)
	closedDuringFinalScan := false
	secondPlatform.state.exchangeHook = func(_ int, connection *net.TCPConn) {
		closedDuringFinalScan = true
		_ = connection.Close()
	}
	if secondExchange.Valid() || !closedDuringFinalScan {
		t.Fatal("connection close during final inspection retained exchange authority")
	}
	secondPlatform.state.exchangeHook = nil
	if !exchange.Valid() {
		t.Fatal("final-inspection close test damaged the original exchange authority")
	}
	if err := os.WriteFile(sharePath, []byte("share"), 0o600); err != nil {
		t.Fatal(err)
	}
	waitFile(sharedPath)
	if exchange.Valid() {
		t.Fatal("shared listener retained exchange authority")
	}
	if err := os.WriteFile(closePath, []byte("close"), 0o600); err != nil {
		t.Fatal(err)
	}
	waitFile(closedPath)
	if exchange.Valid() {
		t.Fatal("closed/replaced listener retained exchange authority")
	}
	if err := authority.Close(); err != nil {
		t.Fatal(err)
	}
	if exchange.Valid() {
		t.Fatal("child exit retained exchange authority")
	}

	childExitAddress := filepath.Join(dir, "child-exit.address")
	childExitAccepted := filepath.Join(dir, "child-exit.accepted")
	childExitClose := filepath.Join(dir, "child-exit.close")
	childExitClosed := filepath.Join(dir, "child-exit.closed")
	childExitShare := filepath.Join(dir, "child-exit.share")
	childExitShared := filepath.Join(dir, "child-exit.shared")
	childExitOptions := options
	childExitOptions.Arguments = []string{"-test.run=^TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection$", "--", "strix-exchange-helper", "4", "5", "6", childExitAddress, childExitAccepted, childExitClose, childExitClosed, childExitShare, childExitShared, "nodump"}
	childExitHookCalled := false
	childExitOptions.testDuringExchangeFinalValidation = func(pid int, _ *net.TCPConn) {
		childExitHookCalled = true
		process, findErr := os.FindProcess(pid)
		if findErr != nil {
			t.Error(findErr)
			return
		}
		if killErr := process.Kill(); killErr != nil {
			t.Error(killErr)
			return
		}
		deadline := time.Now().Add(time.Second)
		for time.Now().Before(deadline) {
			if _, statErr := os.Stat("/proc/" + strconv.Itoa(pid)); os.IsNotExist(statErr) {
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Error("child remained present after final exchange scan exit")
	}
	childExitAuthority, err := OpenStrixComparatorAuthority(childExitOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer childExitAuthority.Close() //nolint:errcheck // test-owned cleanup
	childExitTarget := mustResolveStrixTCPAddress(t, waitFile(childExitAddress))
	childExitFirst, err := net.DialTCP("tcp4", nil, childExitTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer childExitFirst.Close() //nolint:errcheck // test-owned client
	childExitSecond, err := net.DialTCP("tcp4", nil, childExitTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer childExitSecond.Close() //nolint:errcheck // test-owned client
	waitFile(childExitAccepted)
	if got, bindErr := childExitAuthority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, childExitFirst); bindErr == nil || got != nil || !childExitHookCalled {
		t.Fatalf("child exit during final inspection result=(%v,%v) hook=%v", got, bindErr, childExitHookCalled)
	}

	zombieAddress := filepath.Join(dir, "zombie.address")
	zombieAccepted := filepath.Join(dir, "zombie.accepted")
	zombieClose := filepath.Join(dir, "zombie.close")
	zombieClosed := filepath.Join(dir, "zombie.closed")
	zombieShare := filepath.Join(dir, "zombie.share")
	zombieShared := filepath.Join(dir, "zombie.shared")
	zombieOptions := options
	zombieOptions.Arguments = []string{"-test.run=^TestStrixComparatorAuthorityBindsApprovedReferenceAndAcceptedConnection$", "--", "strix-exchange-helper", "4", "5", "6", zombieAddress, zombieAccepted, zombieClose, zombieClosed, zombieShare, zombieShared, "zombie-leader"}
	zombieAuthority, err := OpenStrixComparatorAuthority(zombieOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer zombieAuthority.Close() //nolint:errcheck // test-owned cleanup
	zombieTarget := mustResolveStrixTCPAddress(t, waitFile(zombieAddress))
	zombieFirst, err := net.DialTCP("tcp4", nil, zombieTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer zombieFirst.Close() //nolint:errcheck // test-owned client
	zombieSecond, err := net.DialTCP("tcp4", nil, zombieTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer zombieSecond.Close() //nolint:errcheck // test-owned client
	waitFile(zombieAccepted)
	zombieExchange, err := zombieAuthority.BindApprovedReferenceAndAcceptedConnection(cell, cell.Digest, zombieFirst)
	if err != nil || !zombieExchange.Valid() {
		t.Fatalf("bind zombie-listener exchange: capability=%v err=%v", zombieExchange, err)
	}
	if err := os.WriteFile(zombieShare, []byte("share"), 0o600); err != nil {
		t.Fatal(err)
	}
	waitFile(zombieShared)
	if zombieExchange.Valid() {
		t.Fatal("listener held by live task behind zombie leader retained exchange authority")
	}
}

func mustResolveStrixTCPAddress(t *testing.T, value string) *net.TCPAddr {
	t.Helper()
	address, err := net.ResolveTCPAddr("tcp4", value)
	if err != nil {
		t.Fatal(err)
	}
	return address
}

func testStrixExchangeCell(t *testing.T, manifest StrixComparatorManifest, loaderSet, dependencySet string) qwen38quantrun.StrixComparisonCellManifest {
	t.Helper()
	packet, err := qwen38quantrun.FreezePromptPacket(qwen38quantrun.PromptTokenPacket{
		Schema: qwen38quantrun.PromptTokenPacketSchema, PacketID: "strix-exchange-packet", ArtifactSHA256: qwen38quantrun.StrixComparisonArtifactSHA256,
		TokenizerIdentity: qwen38quantrun.GGUFTokenizerIdentity, TokenizerDigest: qwen38quantrun.StrixComparisonTokenizerSHA256, TemplateDigest: qwen38quantrun.StrixComparisonTemplateSHA256,
		PromptTokenIDs: []int{151644, 872, 198, 2610, 525, 264, 25, 13, 151645, 198, 151644, 77091, 198, 151667, 198, 16, 17, 18, 19, 20, 21, 22, 23, 24, 151645, 198},
		ContextBudget:  qwen38quantrun.ContextBudget{ContextTokens: 32768, ContextBudgetBytes: 48 << 30}, GenerationControls: qwen38quantrun.GenerationControls{TopP: 1, TopK: 1, MaxOutputTokens: 128, IgnoreEOS: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	packetBytes, err := qwen38quantrun.ExportPromptPacket(packet)
	if err != nil {
		t.Fatal(err)
	}
	pairs := []string{"pair-01", "pair-02", "pair-03", "pair-04", "pair-05"}
	order := []string{"candidate:pair-01", "reference:pair-01", "reference:pair-02", "candidate:pair-02", "candidate:pair-03", "reference:pair-03", "reference:pair-04", "candidate:pair-04", "candidate:pair-05", "reference:pair-05"}
	names := []string{"physical_ram", "pci_device", "boot_session", "kernel", "mesa", "radv", "vulkan_loader", "vulkan_icd", "power_policy", "clock_policy", "thermal_policy", "throttle_policy", "gpu_lease"}
	observations := make([]qwen38quantrun.StrixComparisonObservation, len(names))
	for i, name := range names {
		value := "observed-" + name
		observations[i] = qwen38quantrun.StrixComparisonObservation{Name: name, Value: value, ValueSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(value)))}
	}
	hash := func(n int) string { return fmt.Sprintf("%064x", n) }
	cell := qwen38quantrun.StrixComparisonCellManifest{
		Schema:    qwen38quantrun.StrixComparisonCellManifestSchema,
		Challenge: qwen38quantrun.StrixComparisonChallenge{SourceRowID: qwen38quantrun.StrixComparisonSourceRowID, SourceDate: qwen38quantrun.StrixComparisonSourceDate, SourceRevision: qwen38quantrun.StrixComparisonSourceRevision, Concurrency: 1, AcceptedOutputTPS: 16.65, ConfidenceRule: "one-sided-95-percent", PairedRatioRule: "paired-ratio-lcb95>1", SamplingAssumption: "iid-approximately-normal-paired-ratios", Warmups: 3, MeasuredPairs: 5, AlternatingOrder: order, PairIDs: pairs},
		Platform:  qwen38quantrun.StrixComparisonPlatform{ApplianceID: "strix1", CPU: "AMD Ryzen AI MAX+ 395", GPU: "Radeon 8060S", GPUArchitecture: "gfx1151", ComputeUnits: 40, UMAClassBytes: 64 << 30, PhysicalRAMBytes: 64 << 30, LeaseIdentity: "lease-strix-exchange", Observations: observations},
		Workload:  qwen38quantrun.StrixComparisonWorkload{Model: "Qwen3.8-27B", Quantization: "Q4_K_M", ArtifactSHA256: qwen38quantrun.StrixComparisonArtifactSHA256, PromptPacketBytes: packetBytes, PromptPacketDigest: packet.PacketDigest, TokenizerSHA256: qwen38quantrun.StrixComparisonTokenizerSHA256, TemplateSHA256: qwen38quantrun.StrixComparisonTemplateSHA256, RenderedPromptSHA256: qwen38quantrun.StrixComparisonRenderedPromptSHA256, PromptTokenIDs: slices.Clone(packet.PromptTokenIDs), ContextTokens: 32768, AcceptedOutputTokens: 128},
		Memory:    qwen38quantrun.StrixComparisonMemoryEnvelope{ContextBudgetBytes: 48 << 30, KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUUMABudgetBytes: 56 << 30, CandidateBudgetBytes: 56 << 30, ReferenceBudgetBytes: 56 << 30, HostSpillPolicy: "forbid", PrimaryCacheState: "cold-no-prefix", MemoryAccountingPolicy: "uma-overlap-not-summed", ResidentModelBytes: 17_106_775_008, CandidatePeakMethod: "authoritative-peak-uma", ReferencePeakMethod: "authoritative-peak-uma"},
		Candidate: qwen38quantrun.StrixComparisonCandidatePin{CampaignClass: "fak-native", Runtime: "native", Owner: "fak", Planner: "inkernel", Backend: "vulkan", SourceRevision: "internal/compute@r1+gabcdef0", SourceArchiveSHA256: hash(20), BuildManifestSHA256: hash(21), ToolchainSHA256: hash(22), ExecutableSHA256: hash(23), ShaderBundleSHA256: hash(24), ModelSHA256: qwen38quantrun.StrixComparisonArtifactSHA256, ForwardPath: "modelbench/raw-decode/vulkan"},
		Reference: qwen38quantrun.StrixComparisonReferencePin{CampaignClass: "llama.cpp-comparator-only", SourceRevision: qwen38quantrun.StrixComparisonLlamaSourceRevision, SourceTreeSHA256: qwen38quantrun.StrixComparisonLlamaTreeSHA256, BuildType: "Release", GGMLVulkan: true, SourceArchiveSHA256: manifest.SourceArchiveSHA256, BuildManifestSHA256: manifest.BuildManifestSHA256, ToolchainSHA256: hash(32), ServerBinarySHA256: manifest.ServerBinarySHA256, BenchBinarySHA256: hash(34), LoaderSHA256: loaderSet, DependencySHA256: dependencySet, RADVDeviceIdentity: "radv-gfx1151-pci-observed", ModelSHA256: qwen38quantrun.StrixComparisonArtifactSHA256, PromptPacketDigest: packet.PacketDigest},
		Capture:   qwen38quantrun.StrixComparisonCapturePlan{CellNonce: "fresh-cell-nonce-exchange", AuthoritySchema: "fak.qwen38.capture-authority.v1", ArmRoles: []string{"candidate", "reference"}, PairIDs: slices.Clone(pairs), AlternatingOrder: slices.Clone(order), TrialCount: 10, MonotonicClockID: "clock-boottime-session-exchange", SessionIdentity: "boot-session-observed", ReplayKey: "replay-key-exchange", ObservationBindings: []string{"trial_native_timing_reconciliation", "accepted_token_ids", "accepted_token_logprobs", "eos_observation", "resource_observation"}},
	}
	cell, err = qwen38quantrun.SealStrixComparisonCellManifest(cell)
	if err != nil {
		t.Fatal(err)
	}
	return cell
}

func TestOpenStrixComparatorAuthorityPinsExecutedFiles(t *testing.T) {
	if len(os.Args) >= 2 && os.Args[len(os.Args)-1] == "strix-authority-exit" {
		return
	}
	if len(os.Args) >= 7 && os.Args[len(os.Args)-6] == "strix-authority-late-map" {
		for _, rawFD := range os.Args[len(os.Args)-5 : len(os.Args)-2] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(100)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(101)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		trigger := os.Args[len(os.Args)-1]
		for {
			if _, err := os.Stat(trigger); err == nil {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		rogue, err := os.Open(os.Args[len(os.Args)-2])
		if err != nil {
			os.Exit(102)
		}
		defer rogue.Close() //nolint:errcheck // helper exits immediately after the test parent closes it
		mapped, err := syscall.Mmap(int(rogue.Fd()), 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
		if err != nil {
			os.Exit(103)
		}
		defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 6 && os.Args[len(os.Args)-5] == "strix-authority-rogue" {
		if wd, err := os.Getwd(); err != nil || wd != "/" || os.Getenv("PATH") != "" || os.Getenv("STRIX_INJECTED") != "" {
			os.Exit(91)
		}
		for _, rawFD := range os.Args[len(os.Args)-4 : len(os.Args)-1] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(92)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(93)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		rogue, err := os.Open(os.Args[len(os.Args)-1])
		if err != nil {
			os.Exit(94)
		}
		defer rogue.Close() //nolint:errcheck // helper exits immediately after the test parent closes it
		mapped, err := syscall.Mmap(int(rogue.Fd()), 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
		if err != nil {
			os.Exit(95)
		}
		defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 4 && os.Args[len(os.Args)-3] == "strix-authority-no-model" {
		for _, rawFD := range os.Args[len(os.Args)-2:] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(96)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(97)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		for {
			time.Sleep(time.Second)
		}
	}
	if len(os.Args) >= 5 && os.Args[len(os.Args)-4] == "strix-authority-helper" {
		if wd, err := os.Getwd(); err != nil || wd != "/" || os.Getenv("PATH") != "" || os.Getenv("STRIX_INJECTED") != "" {
			os.Exit(91)
		}
		for _, rawFD := range os.Args[len(os.Args)-3:] {
			fd, err := strconv.Atoi(rawFD)
			if err != nil {
				os.Exit(92)
			}
			mapped, err := syscall.Mmap(fd, 0, 4096, syscall.PROT_READ, syscall.MAP_PRIVATE)
			if err != nil {
				os.Exit(93)
			}
			defer syscall.Munmap(mapped) //nolint:errcheck // helper exits immediately after the test parent closes it
		}
		if _, err := syscall.Seek(4, 2, 0); err != nil {
			os.Exit(98)
		}
		for {
			if offset, err := syscall.Seek(4, 0, 1); err != nil || offset != 2 {
				os.Exit(99)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

	dir := t.TempDir()
	write := func(name string, data []byte, mode os.FileMode) string {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, data, mode); err != nil {
			t.Fatal(err)
		}
		return path
	}
	hashFile := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		return hex.EncodeToString(sum[:])
	}

	server, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	server, err = filepath.EvalSymlinks(server)
	if err != nil {
		t.Fatal(err)
	}
	source := write("source.tar", []byte("source archive"), 0o600)
	build := write("build.json", []byte(`{"build":"b10588","vulkan":true}`), 0o600)
	model := write("model.gguf", []byte("model"), 0o600)
	loader := write("loader.icd", make([]byte, 4096), 0o600)
	dependency := write("dependency.so", make([]byte, 4096), 0o600)
	serverInfo, err := os.Stat(server)
	if err != nil {
		t.Fatal(err)
	}
	dependencies := []StrixComparatorPinnedFile{{Path: dependency, SHA256: hashFile(dependency)}}
	maps, err := os.Open("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	seenMappedPath := make(map[string]struct{})
	scanner := bufio.NewScanner(maps)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 6 || !strings.HasPrefix(fields[5], "/") {
			continue
		}
		path := strings.TrimSuffix(strings.Join(fields[5:], " "), " (deleted)")
		path, err = filepath.EvalSymlinks(path)
		if err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if os.SameFile(info, serverInfo) {
			continue
		}
		if _, ok := seenMappedPath[path]; ok {
			continue
		}
		seenMappedPath[path] = struct{}{}
		dependencies = append(dependencies, StrixComparatorPinnedFile{Path: path, SHA256: hashFile(path)})
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if err := maps.Close(); err != nil {
		t.Fatal(err)
	}

	manifest := validStrixComparatorManifest()
	manifest.SourceArchiveSHA256 = hashFile(source)
	manifest.BuildManifestSHA256 = hashFile(build)
	manifest.ServerBinarySHA256 = hashFile(server)

	options := StrixComparatorAuthorityOptions{
		Manifest:          manifest,
		SourceArchivePath: source,
		BuildManifestPath: build,
		ServerBinaryPath:  server,
		ModelPath:         model,
		LoaderICD:         []StrixComparatorPinnedFile{{Path: loader, SHA256: hashFile(loader)}},
		Dependencies:      dependencies,
		Arguments:         []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-helper", "4", "5", "6"},
		testModelSHA256:   hashFile(model),
	}

	t.Setenv("PATH", filepath.Join(dir, "attacker-bin"))
	t.Setenv("STRIX_INJECTED", "must-not-cross-authority")
	authority, err := OpenStrixComparatorAuthority(options)
	if err != nil {
		t.Fatalf("open authority: %v", err)
	}
	pid := authority.PID()
	if pid <= 0 || !authority.Valid() {
		t.Fatalf("authority is not live: pid=%d valid=%v", pid, authority.Valid())
	}
	readFDPosition := func(pid, fd int) int64 {
		t.Helper()
		data, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/fdinfo/" + strconv.Itoa(fd))
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(string(data), "\n") {
			if !strings.HasPrefix(line, "pos:") {
				continue
			}
			position, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "pos:")), 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			return position
		}
		t.Fatal("child fdinfo omitted position")
		return -1
	}
	if before := readFDPosition(pid, 4); before != 2 {
		t.Fatalf("inherited model fd position before validation = %d", before)
	}
	if !authority.Valid() {
		t.Fatal("authority invalid during offset-independent revalidation")
	}
	if after := readFDPosition(pid, 4); after != 2 {
		t.Fatalf("retained-handle hashing changed inherited model fd position to %d", after)
	}
	if _, err := json.Marshal(authority); err == nil {
		t.Fatal("opaque authority unexpectedly marshaled")
	}
	constructed := new(StrixComparatorAuthority)
	if constructed.Valid() || !errors.Is(constructed.Close(), ErrInvalidStrixComparatorAuthority) {
		t.Fatal("constructed lookalike retained capability")
	}
	if err := json.Unmarshal([]byte(`{}`), constructed); err == nil {
		t.Fatal("opaque authority unexpectedly unmarshaled")
	}
	lookalike := *authority
	if lookalike.Valid() {
		t.Fatal("copied authority retained capability")
	}
	if err := lookalike.Close(); !errors.Is(err, ErrInvalidStrixComparatorAuthority) {
		t.Fatalf("copied authority close error = %v", err)
	}
	if !authority.Valid() {
		t.Fatal("closing copied lookalike affected minted authority")
	}
	if err := authority.Close(); err != nil {
		t.Fatalf("close authority: %v", err)
	}
	if authority.Valid() {
		t.Fatal("closed authority remained valid")
	}

	t.Run("rejects symlink and special file", func(t *testing.T) {
		symlink := filepath.Join(dir, "source-link")
		if err := os.Symlink(source, symlink); err != nil {
			t.Fatal(err)
		}
		bad := options
		bad.SourceArchivePath = symlink
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("symlink result = (%v, %v)", got, err)
		}
		bad = options
		bad.ModelPath = "/dev/null"
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("special-file result = (%v, %v)", got, err)
		}
		fifo := filepath.Join(dir, "model.fifo")
		if err := syscall.Mkfifo(fifo, 0o600); err != nil {
			t.Fatal(err)
		}
		bad.ModelPath = fifo
		started := time.Now()
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("FIFO result = (%v, %v)", got, err)
		}
		if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
			t.Fatalf("FIFO rejection blocked for %v", elapsed)
		}
	})

	t.Run("rejects digest mismatch", func(t *testing.T) {
		bad := options
		bad.testModelSHA256 = strings.Repeat("a", 64)
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("hash mismatch result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects undeclared mapped file", func(t *testing.T) {
		rogue := write("rogue.so", make([]byte, 4096), 0o600)
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-rogue", "4", "5", "6", rogue}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("undeclared map result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects partial startup", func(t *testing.T) {
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-exit"}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("partial-start result = (%v, %v)", got, err)
		}
	})

	t.Run("requires the pinned model mapping", func(t *testing.T) {
		bad := options
		bad.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-no-model", "5", "6"}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("unmapped model result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects replaced executable path", func(t *testing.T) {
		copyPath := filepath.Join(dir, "server-copy")
		serverBytes, err := os.ReadFile(server)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(copyPath, serverBytes, 0o700); err != nil {
			t.Fatal(err)
		}
		bad := options
		bad.ServerBinaryPath = copyPath
		bad.testAfterOpen = func() error {
			return os.Rename(write("replacement", serverBytes, 0o700), copyPath)
		}
		if got, err := OpenStrixComparatorAuthority(bad); err == nil || got != nil {
			t.Fatalf("replaced executable result = (%v, %v)", got, err)
		}
	})

	t.Run("rejects child exit during final startup verification", func(t *testing.T) {
		candidate := options
		candidate.testDuringStartupValidation = func(pid int) {
			process, err := os.FindProcess(pid)
			if err != nil {
				t.Error(err)
				return
			}
			if err := process.Kill(); err != nil {
				t.Error(err)
			}
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				if _, err := os.Stat("/proc/" + strconv.Itoa(pid)); os.IsNotExist(err) {
					return
				}
				time.Sleep(5 * time.Millisecond)
			}
			t.Error("child remained present after injected final-startup exit")
		}
		if got, err := OpenStrixComparatorAuthority(candidate); err == nil || got != nil {
			t.Fatalf("final-startup exit result = (%v, %v)", got, err)
		}
	})

	t.Run("retains rejected-startup cleanup after signal failure", func(t *testing.T) {
		rogue := write("late-startup-rogue.so", make([]byte, 4096), 0o600)
		trigger := filepath.Join(dir, "late-startup.trigger")
		candidate := options
		candidate.Arguments = []string{"-test.run=^TestOpenStrixComparatorAuthorityPinsExecutedFiles$", "--", "strix-authority-late-map", "4", "5", "6", rogue, trigger}
		childPID := 0
		candidate.testDuringStartupValidation = func(pid int) {
			childPID = pid
			if err := os.WriteFile(trigger, []byte("map-now"), 0o600); err != nil {
				t.Error(err)
				return
			}
			deadline := time.Now().Add(time.Second)
			for time.Now().Before(deadline) {
				maps, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/maps")
				if err == nil && strings.Contains(string(maps), rogue) {
					return
				}
				time.Sleep(5 * time.Millisecond)
			}
			t.Error("child did not install injected late mapping")
		}
		injected := errors.New("injected rejected-startup signal failure")
		signalCalls := 0
		candidate.testPidfdSignal = func(pidfd, signal int) error {
			signalCalls++
			if signalCalls == 1 {
				return injected
			}
			return sendStrixPidfdSignal(pidfd, signal)
		}
		fdsBefore, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Fatal(err)
		}
		got, openErr := OpenStrixComparatorAuthority(candidate)
		if got != nil || openErr == nil {
			t.Fatalf("late-map startup result = (%v, %v)", got, openErr)
		}
		var cleanupErr *StrixComparatorAuthorityCleanupError
		if !errors.As(openErr, &cleanupErr) || !cleanupErr.CleanupPending() || !errors.Is(openErr, ErrStrixComparatorAuthorityCleanupPending) {
			t.Fatalf("startup error did not retain cleanup authority: %T %v", openErr, openErr)
		}
		if childPID <= 0 {
			t.Fatal("startup hook did not capture child PID")
		}
		if err := cleanupErr.RetryCleanup(); err != nil {
			t.Fatalf("retry rejected-startup cleanup: %v", err)
		}
		if cleanupErr.CleanupPending() {
			t.Fatal("cleanup owner remained pending after successful retry")
		}
		if err := cleanupErr.RetryCleanup(); err != nil {
			t.Fatalf("idempotent cleanup retry: %v", err)
		}
		if _, err := os.Stat("/proc/" + strconv.Itoa(childPID)); !os.IsNotExist(err) {
			t.Fatalf("rejected startup child still exists: %v", err)
		}
		fdsAfter, err := os.ReadDir("/proc/self/fd")
		if err != nil {
			t.Fatal(err)
		}
		if len(fdsAfter) != len(fdsBefore) {
			t.Fatalf("rejected startup leaked parent fds: before=%d after=%d", len(fdsBefore), len(fdsAfter))
		}
	})

	t.Run("brackets retained-handle validation with child identity", func(t *testing.T) {
		candidate := options
		candidate.testDuringValidation = func(pid int) {
			process, err := os.FindProcess(pid)
			if err != nil {
				t.Error(err)
				return
			}
			if err := process.Kill(); err != nil {
				t.Error(err)
			}
		}
		minted, err := OpenStrixComparatorAuthority(candidate)
		if err != nil {
			t.Fatalf("mint bracket authority: %v", err)
		}
		if minted.Valid() {
			t.Fatal("authority survived child exit during retained-handle validation")
		}
		if err := minted.Close(); err != nil {
			t.Fatalf("close bracket authority: %v", err)
		}
	})

	t.Run("retains cleanup authority after bounded signal failure", func(t *testing.T) {
		candidate := options
		injected := errors.New("injected pidfd signal failure")
		calls := 0
		candidate.testPidfdSignal = func(pidfd, signal int) error {
			calls++
			if calls == 1 {
				return injected
			}
			return sendStrixPidfdSignal(pidfd, signal)
		}
		minted, err := OpenStrixComparatorAuthority(candidate)
		if err != nil {
			t.Fatalf("mint cleanup authority: %v", err)
		}
		started := time.Now()
		err = minted.Close()
		if !errors.Is(err, ErrStrixComparatorAuthorityCleanupPending) {
			t.Fatalf("first close error = %v", err)
		}
		if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
			t.Fatalf("failed pidfd signal blocked close for %v", elapsed)
		}
		if !minted.Valid() {
			t.Fatal("signal failure discarded the live cleanup capability")
		}
		if err := minted.Close(); err != nil {
			t.Fatalf("retry close: %v", err)
		}
		if minted.Valid() {
			t.Fatal("retried close retained authority")
		}
	})

	t.Run("invalidates post-mint artifact drift", func(t *testing.T) {
		cloneOptions := func() StrixComparatorAuthorityOptions {
			cloned := options
			cloned.Manifest.BuildFlags = append([]string(nil), options.Manifest.BuildFlags...)
			cloned.Manifest.BuildTargets = append([]string(nil), options.Manifest.BuildTargets...)
			cloned.LoaderICD = append([]StrixComparatorPinnedFile(nil), options.LoaderICD...)
			cloned.Dependencies = append([]StrixComparatorPinnedFile(nil), options.Dependencies...)
			cloned.Arguments = append([]string(nil), options.Arguments...)
			return cloned
		}
		copyFile := func(name, sourcePath string, mode os.FileMode) string {
			t.Helper()
			data, err := os.ReadFile(sourcePath)
			if err != nil {
				t.Fatal(err)
			}
			return write(name, data, mode)
		}
		replacePath := func(path, name string, mode os.FileMode) {
			t.Helper()
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Rename(write(name, data, mode), path); err != nil {
				t.Fatal(err)
			}
		}
		cases := []struct {
			name  string
			setup func(*StrixComparatorAuthorityOptions) func()
		}{
			{
				name: "source held bytes",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-source", source, 0o600)
					candidate.SourceArchivePath = path
					candidate.Manifest.SourceArchiveSHA256 = hashFile(path)
					return func() {
						if err := os.WriteFile(path, []byte("mutated source archive"), 0o600); err != nil {
							t.Fatal(err)
						}
					}
				},
			},
			{
				name: "build manifest path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-build", build, 0o600)
					candidate.BuildManifestPath = path
					candidate.Manifest.BuildManifestSHA256 = hashFile(path)
					return func() { replacePath(path, "post-mint-build-replacement", 0o600) }
				},
			},
			{
				name: "model path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-model", model, 0o600)
					candidate.ModelPath = path
					candidate.testModelSHA256 = hashFile(path)
					return func() { replacePath(path, "post-mint-model-replacement", 0o600) }
				},
			},
			{
				name: "dependency path",
				setup: func(candidate *StrixComparatorAuthorityOptions) func() {
					path := copyFile("post-mint-dependency", dependency, 0o600)
					candidate.Dependencies[0] = StrixComparatorPinnedFile{Path: path, SHA256: hashFile(path)}
					return func() { replacePath(path, "post-mint-dependency-replacement", 0o600) }
				},
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				candidate := cloneOptions()
				mutate := tc.setup(&candidate)
				minted, err := OpenStrixComparatorAuthority(candidate)
				if err != nil {
					t.Fatalf("mint authority: %v", err)
				}
				mutate()
				if minted.Valid() {
					t.Fatal("authority remained valid after pinned artifact drift")
				}
				if err := minted.Close(); err != nil {
					t.Fatalf("close invalidated authority: %v", err)
				}
			})
		}
	})
}
