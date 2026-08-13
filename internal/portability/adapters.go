package portability

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
)

type genericAdapter struct{ info AdapterInfo }

func (a genericAdapter) Info() AdapterInfo { return a.info }
func (a genericAdapter) Discover(ctx context.Context, s State, scope string) (Discovery, error) {
	rs, e := s.Discover(ctx, scope)
	if e != nil {
		return Discovery{}, errx("STATE_ERROR", a.info.Kind, "discover", "", e.Error())
	}
	out := rs[:0]
	for _, r := range rs {
		if r.Kind == a.info.Kind {
			out = append(out, r)
		}
	}
	return Discovery{Records: out}, nil
}
func (a genericAdapter) Read(ctx context.Context, s State, scope, name string) (Record, error) {
	r, e := s.Read(ctx, scope, name)
	if e != nil {
		return Record{}, errx("NOT_FOUND", a.info.Kind, "read", "name", e.Error())
	}
	if e = a.Validate(ctx, r); e != nil {
		return Record{}, e
	}
	return r, nil
}
func (a genericAdapter) Validate(_ context.Context, r Record) error {
	if e := validateRecord(a.info.Kind, r); e != nil {
		return e
	}
	return ensureNoSecret(a.info.Kind, "validate", r)
}
func (a genericAdapter) Preview(ctx context.Context, s State, scope string, want []Record) (Plan, error) {
	p := Plan{Kind: a.info.Kind}
	sort.Slice(want, func(i, j int) bool { return want[i].Name < want[j].Name })
	for _, r := range want {
		if e := a.Validate(ctx, r); e != nil {
			return Plan{}, e
		}
		old, e := s.Read(ctx, scope, r.Name)
		if e != nil {
			p.Changes = append(p.Changes, Change{ID: r.Name, After: &r})
			continue
		}
		if old.Kind != a.info.Kind {
			return Plan{}, errx("PRECEDENCE_CONFLICT", a.info.Kind, "preview", "name", "another managed kind owns the identity")
		}
		if !equalRecord(old, r) {
			o := old
			p.Changes = append(p.Changes, Change{ID: r.Name, Before: &o, After: &r})
		}
	}
	if a.info.Support == SupportPartial && a.info.Degradation != "" {
		p.Warnings = []string{a.info.Degradation}
	}
	return p, nil
}
func (a genericAdapter) Apply(ctx context.Context, s State, scope string, p Plan) (AdapterReceipt, error) {
	if p.Kind != a.info.Kind {
		return AdapterReceipt{}, errx("KIND_MISMATCH", a.info.Kind, "apply", "kind", "plan kind does not match adapter")
	}
	rec := AdapterReceipt{PlanID: stablePlanID(p)}
	for _, c := range p.Changes {
		if c.After == nil {
			if e := s.Delete(ctx, scope, c.ID); e != nil {
				_ = a.Rollback(ctx, s, scope, rec)
				return AdapterReceipt{}, errx("APPLY_INTERRUPTED", a.info.Kind, "apply", c.ID, e.Error())
			}
		} else if e := s.Write(ctx, scope, *c.After); e != nil {
			_ = a.Rollback(ctx, s, scope, rec)
			return AdapterReceipt{}, errx("APPLY_INTERRUPTED", a.info.Kind, "apply", c.ID, e.Error())
		}
		rec.Applied = append(rec.Applied, c)
	}
	return rec, nil
}
func (a genericAdapter) Rollback(ctx context.Context, s State, scope string, r AdapterReceipt) error {
	for i := len(r.Applied) - 1; i >= 0; i-- {
		c := r.Applied[i]
		if c.Before == nil {
			if e := s.Delete(ctx, scope, c.ID); e != nil {
				return errx("ROLLBACK_FAILED", a.info.Kind, "rollback", c.ID, e.Error())
			}
		} else if e := s.Write(ctx, scope, *c.Before); e != nil {
			return errx("ROLLBACK_FAILED", a.info.Kind, "rollback", c.ID, e.Error())
		}
	}
	return nil
}
func (a genericAdapter) Migrate(ctx context.Context, r Record, to string) (Record, error) {
	if e := a.Validate(ctx, r); e != nil {
		return Record{}, e
	}
	if to == "" {
		return Record{}, errx("INVALID_VERSION", a.info.Kind, "migrate", "version", "target version required")
	}
	r.Version = to
	return r, nil
}
func (a genericAdapter) Diff(_ context.Context, x, y Record) ([]string, error) {
	if x.Kind != a.info.Kind || y.Kind != a.info.Kind {
		return nil, errx("KIND_MISMATCH", a.info.Kind, "diff", "kind", "record kind mismatch")
	}
	var d []string
	if x.Name != y.Name {
		d = append(d, fmtPath("name"))
	}
	if x.Version != y.Version {
		d = append(d, fmtPath("version"))
	}
	if x.Active != y.Active {
		d = append(d, fmtPath("active"))
	}
	if string(x.Data) != string(y.Data) {
		d = append(d, fmtPath("data"))
	}
	if string(x.Unknown) != string(y.Unknown) {
		d = append(d, fmtPath("unknown"))
	}
	return d, nil
}
func (a genericAdapter) Dependencies(_ context.Context, r Record) ([]string, error) {
	var v struct {
		Dependencies []string `json:"dependencies"`
	}
	if e := json.Unmarshal(r.Data, &v); e != nil {
		return nil, errx("MALFORMED_RECORD", a.info.Kind, "dependencies", "data", e.Error())
	}
	sort.Strings(v.Dependencies)
	return v.Dependencies, nil
}
func (a genericAdapter) Identity(_ context.Context, r Record) (string, error) {
	if e := validateRecord(a.info.Kind, r); e != nil {
		return "", e
	}
	return identity(r)
}
func (a genericAdapter) Export(ctx context.Context, r Record) ([]byte, error) {
	if e := a.Validate(ctx, r); e != nil {
		return nil, e
	}
	return canonical(r)
}

type opaqueAdapter struct{ kind string }

func (a opaqueAdapter) Info() AdapterInfo {
	return AdapterInfo{Kind: a.kind, Version: "unknown", Support: SupportInactive, Sensitivity: "unknown", Compatibility: "preserved-inactive", Degradation: "adapter unavailable; bytes preserved but never activated"}
}
func (a opaqueAdapter) Validate(_ context.Context, r Record) error {
	if r.Kind != a.kind {
		return errx("KIND_MISMATCH", a.kind, "validate", "kind", "record kind mismatch")
	}
	if !json.Valid(r.Data) || len(r.Data) > 8<<20 {
		return errx("MALFORMED_RECORD", a.kind, "validate", "data", "invalid or oversized opaque data")
	}
	return ensureNoSecret(a.kind, "validate", r)
}
func (a opaqueAdapter) Export(ctx context.Context, r Record) ([]byte, error) {
	if e := a.Validate(ctx, r); e != nil {
		return nil, e
	}
	r.Active = false
	return canonical(r)
}
func (a opaqueAdapter) Identity(_ context.Context, r Record) (string, error) {
	r.Active = false
	return identity(r)
}
func (a opaqueAdapter) Discover(context.Context, State, string) (Discovery, error) {
	return Discovery{}, unsupported(a.kind, "discover")
}
func (a opaqueAdapter) Read(context.Context, State, string, string) (Record, error) {
	return Record{}, unsupported(a.kind, "read")
}
func (a opaqueAdapter) Preview(context.Context, State, string, []Record) (Plan, error) {
	return Plan{}, unsupported(a.kind, "preview")
}
func (a opaqueAdapter) Apply(context.Context, State, string, Plan) (AdapterReceipt, error) {
	return AdapterReceipt{}, unsupported(a.kind, "apply")
}
func (a opaqueAdapter) Rollback(context.Context, State, string, AdapterReceipt) error {
	return unsupported(a.kind, "rollback")
}
func (a opaqueAdapter) Migrate(context.Context, Record, string) (Record, error) {
	return Record{}, unsupported(a.kind, "migrate")
}
func (a opaqueAdapter) Diff(context.Context, Record, Record) ([]string, error) {
	return nil, unsupported(a.kind, "diff")
}
func (a opaqueAdapter) Dependencies(context.Context, Record) ([]string, error) {
	return nil, unsupported(a.kind, "dependencies")
}

var referenceSpecs = []AdapterInfo{
	{Kind: "skill", Version: "1", Support: SupportFull, Sensitivity: "private", Compatibility: "v1", Capabilities: allCaps()},
	{Kind: "workflow", Version: "1", Support: SupportPartial, Sensitivity: "private", Compatibility: "v1", Degradation: "host-specific triggers remain inactive", Capabilities: allCaps()},
	{Kind: "policy", Version: "1", Support: SupportFull, Sensitivity: "restricted", Compatibility: "v1", Capabilities: allCaps()},
	{Kind: "profile", Version: "1", Support: SupportPartial, Sensitivity: "private", Compatibility: "v1", Degradation: "provider-local preferences omitted", Capabilities: allCaps()},
	{Kind: "tool-binding", Version: "1", Support: SupportPartial, Sensitivity: "restricted", Compatibility: "v1", Degradation: "credential bindings are never exported", Capabilities: allCaps()},
	{Kind: "model-binding", Version: "1", Support: SupportPartial, Sensitivity: "restricted", Compatibility: "v1", Degradation: "provider identity requires local rebinding", Capabilities: allCaps()},
	{Kind: "session", Version: "1", Support: SupportLocalOnly, Sensitivity: "private", Compatibility: "same-host", Degradation: "live process state is local-only", Capabilities: allCaps()},
	{Kind: "checkpoint", Version: "1", Support: SupportLocalOnly, Sensitivity: "private", Compatibility: "same-runtime", Degradation: "runtime handles are not portable", Capabilities: allCaps()},
	{Kind: "receipt", Version: "1", Support: SupportFull, Sensitivity: "internal", Compatibility: "v1", Capabilities: allCaps()},
}

func allCaps() []Capability {
	return []Capability{CapDiscover, CapRead, CapWrite, CapValidate, CapPreview, CapApply, CapRollback, CapMigrate, CapDiff, CapDependencies, CapIdentity, CapExport, CapCompatibility, CapDegradation}
}
func ReferenceRegistry() *AdapterRegistry {
	r := NewAdapterRegistry()
	for _, s := range referenceSpecs {
		_ = r.Register(genericAdapter{info: s})
	}
	return r
}
func Skeleton(kind string) (string, error) {
	if kind == "" {
		return "", errx("INVALID_ADAPTER", "", "skeleton", "kind", "kind required")
	}
	return fmt.Sprintf("// Adapter skeleton for %s\nregistry.Register(New%sAdapter())\n// Declare version, sensitivity, compatibility, degradation, capabilities, then run RunConformance.\n", kind, camel(kind)), nil
}
func camel(s string) string {
	out := ""
	up := true
	for _, r := range s {
		if r == '-' || r == '_' {
			up = true
			continue
		}
		if up && r >= 'a' && r <= 'z' {
			r -= 32
			up = false
		}
		out += string(r)
	}
	return out
}
