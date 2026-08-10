package toolplugin

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
)

const BuiltinVersion = "1"

type builtinPlugin struct{ profile Profile }

func (p builtinPlugin) Profile() Profile { return p.profile }
func (p builtinPlugin) Apply(_ context.Context, in Input) (Decision, error) {
	switch p.profile.ID {
	case "builtin.require-witness":
		return Decision{Action: ActionRequireWitness, Reason: "PROFILE_REQUIRES_WITNESS"}, nil
	case "builtin.audit":
		return Decision{Action: ActionDefer, Reason: "AUDIT_OBSERVED"}, nil
	default:
		return Decision{Action: ActionDefer}, nil
	}
}

func builtinDigest(id string, stages []Stage) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%s", id, BuiltinVersion)
	for _, s := range stages {
		fmt.Fprintf(h, "\x00%s", s)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}
func BuiltinProfiles() []Profile {
	ps := []Profile{{ID: "builtin.audit", Version: BuiltinVersion, Stage: StageObserve}, {ID: "builtin.require-witness", Version: BuiltinVersion, Stage: StageAdjudicate}}
	for i := range ps {
		ps[i].Digest = builtinDigest(ps[i].ID, []Stage{ps[i].Stage})
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].ID < ps[j].ID })
	return ps
}
func ResolvePinned(id, version, digest string) (Plugin, error) {
	if id == "" || version == "" || digest == "" {
		return nil, fmt.Errorf("PLUGIN_PIN_REQUIRED: id, version, and digest are required")
	}
	for _, p := range BuiltinProfiles() {
		if p.ID != id {
			continue
		}
		if p.Version != version || p.Digest != digest {
			return nil, fmt.Errorf("PLUGIN_PIN_MISMATCH: %s wants %s@%s, registered %s@%s", id, version, digest, p.Version, p.Digest)
		}
		return builtinPlugin{profile: p}, nil
	}
	return nil, fmt.Errorf("PLUGIN_UNKNOWN: %s", id)
}
