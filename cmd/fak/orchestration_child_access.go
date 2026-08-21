package main

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

const orchestrationChildAccessInvalid = "CHILD_ACCESS_INVALID"

type orchestrationCompiledChildAccess struct {
	Mode         orchestration.ChildAccessMode
	Policy       policy.Runtime
	ManifestJSON []byte
	PolicyPath   string
	Admission    laneadmit.Request
}

type orchestrationChildAccessSnapshot struct {
	Parent   policy.Runtime
	Live     []laneadmit.Lease
	Taxonomy laneadmit.Taxonomy
}

var orchestrationChildAccessSnapshotLoader = loadOrchestrationChildAccessSnapshot

func loadOrchestrationChildAccessSnapshot(root string) (orchestrationChildAccessSnapshot, error) {
	parent, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		return orchestrationChildAccessSnapshot{}, fmt.Errorf("parent policy: %w", err)
	}
	tax, err := regionadmit.LoadTaxonomy(root)
	if err != nil {
		return orchestrationChildAccessSnapshot{}, fmt.Errorf("lane taxonomy: %w", err)
	}
	records, _, err := leaseref.NewInDir(root).Live(context.Background(), time.Now())
	if err != nil {
		return orchestrationChildAccessSnapshot{}, fmt.Errorf("live leases: %w", err)
	}
	live := make([]laneadmit.Lease, 0, len(records))
	for _, record := range records {
		lane := laneadmit.LaneOfLeaseID(record.ID)
		if lane == "" {
			lane = regionadmit.LaneOf(record.TreeGlobs, tax)
		}
		live = append(live, laneadmit.Lease{
			ID: record.ID, Lane: lane, Tree: append([]string(nil), record.TreeGlobs...), Holder: record.Holder,
		})
	}
	return orchestrationChildAccessSnapshot{
		Parent: parent,
		Live:   live,
		Taxonomy: laneadmit.Taxonomy{
			Loaded: true, Exclusive: tax.Exclusive, Trees: tax.Trees,
		},
	}, nil
}

func compileOrchestrationChildAccess(role orchestration.Role, parent policy.Runtime, base laneadmit.Request) (orchestrationCompiledChildAccess, error) {
	role.ID = strings.TrimSpace(role.ID)
	if role.ID == "" {
		return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: role id is required", orchestrationChildAccessInvalid)
	}
	mode := orchestration.ChildAccessMode(strings.ToLower(strings.TrimSpace(string(role.Access.Mode))))
	base.Surface = laneadmit.SurfaceDispatch
	base.Holder = role.ID
	base.LeaseID = ""
	base.Lane = ""
	base.Tree = nil

	var manifest policy.Manifest
	var err error
	switch mode {
	case orchestration.ChildAccessObserve:
		if strings.TrimSpace(role.Access.Lane) != "" || strings.TrimSpace(role.Access.WriteTree) != "" {
			return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: observe child %q must have an empty write envelope", orchestrationChildAccessInvalid, role.ID)
		}
		manifest, err = compileObserveChildManifest(parent, role.Access.Tools)
		base.ReadOnly = true
	case orchestration.ChildAccessEffect:
		lane := strings.TrimSpace(role.Access.Lane)
		writeTree, treeErr := normalizeChildWriteTree(role.Access.WriteTree)
		if treeErr != nil {
			return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: effect child %q: %w", orchestrationChildAccessInvalid, role.ID, treeErr)
		}
		if lane == "" || strings.Contains(lane, ",") {
			return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: effect child %q requires one valid lane", orchestrationChildAccessInvalid, role.ID)
		}
		manifest, err = compileEffectChildManifest(parent, role.Access.Tools, writeTree)
		base.Lane = lane
		base.Tree = []string{writeTree}
		base.ReadOnly = false
	default:
		return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: child %q access mode %q is missing or unknown", orchestrationChildAccessInvalid, role.ID, mode)
	}
	if err != nil {
		return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: child %q: %w", orchestrationChildAccessInvalid, role.ID, err)
	}
	runtime, err := manifest.ToRuntime()
	if err != nil {
		return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: child %q policy: %w", orchestrationChildAccessInvalid, role.ID, err)
	}
	if delta := policy.DiffAmendment(parent.Adjudicator, runtime.Adjudicator); delta.Class() == policy.AmendmentWiden {
		return orchestrationCompiledChildAccess{}, fmt.Errorf("%s: child %q policy would widen its parent: %s", orchestrationChildAccessInvalid, role.ID, policy.FormatAmendmentChanges(delta.Widen))
	}
	return orchestrationCompiledChildAccess{
		Mode: mode, Policy: runtime, ManifestJSON: manifest.JSON(), Admission: base,
	}, nil
}

func compileObserveChildManifest(parent policy.Runtime, requested []string) (policy.Manifest, error) {
	readOnly, err := policy.LoadPreset(policy.PresetReadOnly)
	if err != nil {
		return policy.Manifest{}, err
	}
	manifest := policy.FromPolicy(parent.Adjudicator)
	manifest.Posture = ""
	manifest.Complain = nil
	manifest.AdvisoryReasons = nil
	manifest.AllowPrefix = nil

	tools := normalizeChildTools(requested)
	if len(tools) == 0 {
		for tool := range parent.Adjudicator.Allow {
			if manifestAllowsTool(readOnly.Adjudicator, tool) {
				tools = append(tools, tool)
			}
		}
		for _, prefix := range readOnly.Adjudicator.AllowPrefix {
			if parentAllowsPrefix(parent.Adjudicator, prefix) {
				manifest.AllowPrefix = append(manifest.AllowPrefix, prefix)
			}
		}
		tools = normalizeChildTools(tools)
	}
	manifest.Allow = nil
	for _, tool := range tools {
		if !manifestAllowsTool(parent.Adjudicator, tool) {
			return policy.Manifest{}, fmt.Errorf("tool %q is outside the parent policy", tool)
		}
		if !manifestAllowsTool(readOnly.Adjudicator, tool) {
			return policy.Manifest{}, fmt.Errorf("tool %q is not structurally read-only", tool)
		}
		manifest.Allow = append(manifest.Allow, tool)
	}
	return manifest, nil
}

func compileEffectChildManifest(parent policy.Runtime, requested []string, writeTree string) (policy.Manifest, error) {
	readOnly, err := policy.LoadPreset(policy.PresetReadOnly)
	if err != nil {
		return policy.Manifest{}, err
	}
	tools := normalizeChildTools(requested)
	if len(tools) == 0 {
		return policy.Manifest{}, fmt.Errorf("effect access requires a declared tool envelope")
	}
	manifest := policy.FromPolicy(parent.Adjudicator)
	manifest.Posture = ""
	manifest.Complain = nil
	manifest.AdvisoryReasons = nil
	manifest.AllowPrefix = nil
	manifest.Allow = nil
	writes := 0
	for _, tool := range tools {
		if !manifestAllowsTool(parent.Adjudicator, tool) {
			return policy.Manifest{}, fmt.Errorf("tool %q is outside the parent policy", tool)
		}
		manifest.Allow = append(manifest.Allow, tool)
		if arg := childWriteRuleArg(tool); arg != "" {
			writes++
			manifest.ArgRules = append(manifest.ArgRules, policy.ArgRule{Tool: tool, Arg: arg, AllowGlob: writeTree, Reason: "POLICY_BLOCK", Fix: "write inside the child access declaration, or request a separately admitted effect envelope"})
			continue
		}
		if !manifestAllowsTool(readOnly.Adjudicator, tool) {
			return policy.Manifest{}, fmt.Errorf("effect tool %q has no bounded write-argument compiler", tool)
		}
	}
	if writes == 0 {
		return policy.Manifest{}, fmt.Errorf("effect access requires at least one bounded write tool")
	}
	return manifest, nil
}

func normalizeChildWriteTree(raw string) (string, error) {
	tree := filepath.ToSlash(strings.TrimSpace(raw))
	if tree == "" || strings.Contains(tree, ",") {
		return "", fmt.Errorf("one write tree is required")
	}
	clean := strings.TrimSuffix(tree, "/**")
	clean = path.Clean(clean)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || path.IsAbs(tree) || filepath.IsAbs(tree) {
		return "", fmt.Errorf("write tree %q must be a bounded repo-relative region", raw)
	}
	return tree, nil
}

func normalizeChildTools(in []string) []string {
	seen := map[string]bool{}
	for _, tool := range in {
		tool = strings.TrimSpace(tool)
		if tool == "" {
			continue
		}
		seen[tool] = true
	}
	out := make([]string, 0, len(seen))
	for tool := range seen {
		out = append(out, tool)
	}
	sort.Strings(out)
	return out
}

func manifestAllowsTool(p adjudicator.Policy, tool string) bool {
	if _, denied := p.Deny[tool]; denied {
		return false
	}
	if p.Allow[tool] {
		return true
	}
	for _, prefix := range p.AllowPrefix {
		if strings.HasPrefix(tool, prefix) {
			return true
		}
	}
	return false
}

func parentAllowsPrefix(p adjudicator.Policy, childPrefix string) bool {
	childPrefix = strings.TrimSpace(childPrefix)
	for _, parentPrefix := range p.AllowPrefix {
		if strings.HasPrefix(childPrefix, strings.TrimSpace(parentPrefix)) {
			return true
		}
	}
	return false
}

func childWriteRuleArg(tool string) string {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "write", "edit":
		return "file_path"
	case "write_file", "edit_file":
		return "path"
	default:
		return ""
	}
}
