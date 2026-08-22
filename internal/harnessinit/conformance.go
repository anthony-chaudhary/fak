package harnessinit

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesshost"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

const CrossDogfoodSchema = "fak.harness-cross-dogfood/v1alpha1"

// CrossDogfoodMatrix joins the five independently serialized harness artifacts that must
// keep one semantic identity across guard and generated-product changes.
type CrossDogfoodMatrix struct {
	Schema            string            `json:"schema"`
	Platform          string            `json:"platform"`
	Network           string            `json:"network"`
	Hosts             int               `json:"hosts"`
	SubsystemsPerHost int               `json:"subsystems_per_host"`
	DriftRefusals     int               `json:"drift_refusals"`
	Rows              []CrossDogfoodRow `json:"rows"`
}

type CrossDogfoodRow struct {
	Host           string                   `json:"host"`
	Command        string                   `json:"command"`
	Profile        harnessprofile.Binding   `json:"profile"`
	LaunchBinding  harnessprofile.Binding   `json:"launch_binding"`
	GeneratedHost  string                   `json:"generated_host"`
	ComponentGraph []ComponentIdentity      `json:"component_graph"`
	Lock           LockIdentity             `json:"lock"`
	Receipt        harnesskit.LaunchReceipt `json:"receipt"`
	DriftRefusal   string                   `json:"drift_refusal"`
}

type ComponentIdentity struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

type LockIdentity struct {
	ID         string              `json:"id"`
	Components []ComponentIdentity `json:"components"`
}

type crossDogfoodFixture struct {
	host      string
	command   string
	profiles  []harnessprofile.HarnessProfile
	authority string
	source    string
}

// CrossDogfood runs the matrix without a key, model, network, GPU, or installed harness
// CLI. Generated products import the public harnesskit package through a local module
// replacement and execute with GOPROXY=off.
func CrossDogfood(ctx context.Context, repoRoot string) (CrossDogfoodMatrix, error) {
	repoRoot, err := filepath.Abs(repoRoot)
	if err != nil {
		return CrossDogfoodMatrix{}, err
	}
	if err := verifyRepoModule(repoRoot); err != nil {
		return CrossDogfoodMatrix{}, err
	}
	fixtures, err := crossDogfoodFixtures()
	if err != nil {
		return CrossDogfoodMatrix{}, err
	}
	work, err := os.MkdirTemp("", "fak-harness-cross-dogfood-")
	if err != nil {
		return CrossDogfoodMatrix{}, err
	}
	defer os.RemoveAll(work)

	matrix := CrossDogfoodMatrix{
		Schema: CrossDogfoodSchema, Platform: runtime.GOOS + "/" + runtime.GOARCH,
		Network: "disabled:GOPROXY=off", Hosts: len(fixtures), SubsystemsPerHost: 5,
	}
	for _, fixture := range fixtures {
		row, err := crossDogfoodRow(ctx, repoRoot, work, fixture)
		if err != nil {
			return CrossDogfoodMatrix{}, fmt.Errorf("%s: %w", fixture.host, err)
		}
		matrix.Rows = append(matrix.Rows, row)
		if row.DriftRefusal != "" {
			matrix.DriftRefusals++
		}
	}
	return matrix, nil
}

func crossDogfoodFixtures() ([]crossDogfoodFixture, error) {
	builtins := harnessprofile.Builtins()
	third, err := harnessprofile.Resolve([]byte(`{"harnesses":[{"name":"acme","adapter_version":"2.1.0","names":["acme-cli"],"wire":"openai","default_base_url":"https://api.openai.com/v1","repoint":["env"],"credential":{"kind":"env-key","env_key":"ACME_API_KEY"},"identity":"env-key"}]}`))
	if err != nil {
		return nil, err
	}
	return []crossDogfoodFixture{
		{host: "codex", command: "codex", profiles: builtins, authority: "fak:internal/harnessprofile", source: "fak:internal/harnessprofile/builtin/codex@1.0.0"},
		{host: "claude", command: "claude", profiles: builtins, authority: "fak:internal/harnessprofile", source: "fak:internal/harnessprofile/builtin/claude@1.0.0"},
		{host: "acme", command: "acme-cli", profiles: third, authority: "fixture:harnessprofile", source: "fixture:harnessprofile/acme@2.1.0"},
	}, nil
}

func crossDogfoodRow(ctx context.Context, repoRoot, work string, fixture crossDogfoodFixture) (CrossDogfoodRow, error) {
	profile, ok := profileByHost(fixture.profiles, fixture.host)
	if !ok {
		return CrossDogfoodRow{}, fmt.Errorf("resolved profile not found")
	}
	profileBinding, err := harnessprofile.Bind(profile)
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	launchBinding, ok, err := harnessprofile.ResolveBinding(fixture.profiles, fixture.command)
	if err != nil || !ok {
		return CrossDogfoodRow{}, fmt.Errorf("guard binding: recognized=%v: %w", ok, err)
	}
	if err := harnessprofile.VerifyFresh(profile, launchBinding); err != nil {
		return CrossDogfoodRow{}, fmt.Errorf("guard plan disagrees with profile: %w", err)
	}
	artifacts, err := harnesshost.BuildResolved(profileBinding, fixture.authority, fixture.source, ContractVersion)
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	if err := harnesshost.VerifyResolved(launchBinding, artifacts); err != nil {
		return CrossDogfoodRow{}, fmt.Errorf("guard plan disagrees with graph/lock: %w", err)
	}

	productDir := filepath.Join(work, fixture.host)
	result, err := Init(Options{
		Dir: productDir, Module: "example.test/cross-dogfood/" + fixture.host,
		Host: artifacts.Host, HostManifest: artifacts.Manifest, HostLock: artifacts.Lock,
	})
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	if result.Host != profileBinding.Host {
		return CrossDogfoodRow{}, fmt.Errorf("initializer host %q disagrees with profile %q", result.Host, profileBinding.Host)
	}
	if err := addLocalModuleReplacement(productDir, repoRoot); err != nil {
		return CrossDogfoodRow{}, err
	}
	receipt, err := runExternalSelfcheck(ctx, productDir)
	if err != nil {
		return CrossDogfoodRow{}, err
	}

	manifest, err := harnessresolve.Parse([]byte(artifacts.Manifest))
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	lock, err := harnesskit.ParseProductLock([]byte(artifacts.Lock))
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	graphComponents := manifestComponents(manifest.Components)
	lockComponents := lockedComponents(lock.Components)
	if err := verifyMatrixIdentities(profileBinding, graphComponents, lockComponents, receipt); err != nil {
		return CrossDogfoodRow{}, err
	}

	mutated := profile
	mutated.Repoint = append(mutated.Repoint, firstAbsentRepoint(mutated))
	mutatedBinding, err := harnessprofile.Bind(mutated)
	if err != nil {
		return CrossDogfoodRow{}, err
	}
	driftErr := harnesshost.VerifyResolved(mutatedBinding, artifacts)
	if driftErr == nil {
		return CrossDogfoodRow{}, fmt.Errorf("mutated descriptor accepted old graph and lock")
	}

	return CrossDogfoodRow{
		Host: fixture.host, Command: fixture.command, Profile: profileBinding, LaunchBinding: launchBinding,
		GeneratedHost: result.Host, ComponentGraph: graphComponents,
		Lock: LockIdentity{ID: lock.ID, Components: lockComponents}, Receipt: receipt,
		DriftRefusal: driftErr.Error(),
	}, nil
}

func verifyRepoModule(root string) error {
	raw, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return fmt.Errorf("read repository go.mod: %w", err)
	}
	if !bytes.Contains(raw, []byte("module github.com/anthony-chaudhary/fak")) {
		return fmt.Errorf("%s is not the fak module root", root)
	}
	return nil
}

func addLocalModuleReplacement(productDir, repoRoot string) error {
	path := filepath.Join(productDir, "go.mod")
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	replacement := "\nreplace github.com/anthony-chaudhary/fak => " + strconv.Quote(filepath.ToSlash(repoRoot)) + "\n"
	return os.WriteFile(path, append(raw, replacement...), 0o644)
}

func runExternalSelfcheck(ctx context.Context, productDir string) (harnesskit.LaunchReceipt, error) {
	cmd := exec.CommandContext(ctx, "go", "run", "./cmd/product", "--selfcheck")
	cmd.Dir = productDir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOSUMDB=off", "GOWORK=off")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return harnesskit.LaunchReceipt{}, fmt.Errorf("external selfcheck: %w\n%s", err, out)
	}
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		var event struct {
			Type   string `json:"type"`
			Detail string `json:"detail"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil || event.Type != "harness.locked" {
			continue
		}
		var receipt harnesskit.LaunchReceipt
		if err := json.Unmarshal([]byte(event.Detail), &receipt); err != nil {
			return harnesskit.LaunchReceipt{}, err
		}
		return receipt, nil
	}
	if err := scanner.Err(); err != nil {
		return harnesskit.LaunchReceipt{}, err
	}
	return harnesskit.LaunchReceipt{}, fmt.Errorf("external selfcheck emitted no harness.locked receipt")
}

func profileByHost(profiles []harnessprofile.HarnessProfile, host string) (harnessprofile.HarnessProfile, bool) {
	for _, profile := range profiles {
		if profile.Name == host {
			return profile, true
		}
	}
	return harnessprofile.HarnessProfile{}, false
}

func manifestComponents(components []harnessresolve.Component) []ComponentIdentity {
	out := make([]ComponentIdentity, 0, len(components))
	for _, component := range components {
		out = append(out, ComponentIdentity{ID: component.ID, Version: component.Version, Digest: component.Digest})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func lockedComponents(components []harnesskit.LockedComponent) []ComponentIdentity {
	out := make([]ComponentIdentity, 0, len(components))
	for _, component := range components {
		out = append(out, ComponentIdentity{ID: component.ID, Version: component.Version, Digest: component.Digest})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func verifyMatrixIdentities(binding harnessprofile.Binding, graph, lock []ComponentIdentity, receipt harnesskit.LaunchReceipt) error {
	want := []string{"host:" + binding.Host, "wire:" + string(binding.Wire)}
	for _, mechanism := range binding.Repoint {
		want = append(want, "repoint:"+string(mechanism))
	}
	sort.Strings(want)
	if got := componentIDs(graph); !slices.Equal(got, want) {
		return fmt.Errorf("component graph identities %v want %v", got, want)
	}
	if got := componentIDs(lock); !slices.Equal(got, want) {
		return fmt.Errorf("lock identities %v want %v", got, want)
	}
	receiptIDs := make([]string, 0, len(receipt.Components))
	for _, component := range receipt.Components {
		id, _, ok := strings.Cut(component, "@")
		if !ok {
			return fmt.Errorf("malformed receipt component %q", component)
		}
		receiptIDs = append(receiptIDs, id)
	}
	sort.Strings(receiptIDs)
	if !slices.Equal(receiptIDs, want) {
		return fmt.Errorf("receipt identities %v want %v", receiptIDs, want)
	}
	hostReceipt := "host:" + binding.Host + "@" + binding.AdapterVersion + "#" + binding.AdapterDigest
	if !slices.Contains(receipt.Components, hostReceipt) {
		return fmt.Errorf("receipt missing adapter identity %q", hostReceipt)
	}
	if receipt.Schema != harnesskit.LaunchReceiptSchema {
		return fmt.Errorf("receipt schema %q", receipt.Schema)
	}
	return nil
}

func componentIDs(components []ComponentIdentity) []string {
	ids := make([]string, 0, len(components))
	for _, component := range components {
		ids = append(ids, component.ID)
	}
	sort.Strings(ids)
	return ids
}

func firstAbsentRepoint(profile harnessprofile.HarnessProfile) harnessprofile.RepointMechanism {
	for _, mechanism := range []harnessprofile.RepointMechanism{
		harnessprofile.RepointCLIConfig, harnessprofile.RepointSettingsFile, harnessprofile.RepointExtension,
	} {
		if !profile.HasRepoint(mechanism) {
			return mechanism
		}
	}
	return harnessprofile.RepointEnv
}
