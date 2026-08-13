package portability

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func obj(id string, payload string) Object {
	o := Object{ID: id, Kind: "skill", Name: strings.TrimPrefix(id, "skill:"), Active: true, Payload: json.RawMessage(payload)}
	o.Digest = payloadDigest(o.Payload)
	return o
}
func pkg(objects ...Object) Package {
	p := Package{Schema: Schema, Objects: objects}
	p.Digest = packageDigest(p)
	p.ID = "pkg-" + strings.TrimPrefix(p.Digest, "sha256:")[:16]
	return p
}
func field(t *testing.T, p Package, id, key string) any {
	t.Helper()
	for _, o := range p.Objects {
		if o.ID == id {
			var m map[string]any
			if json.Unmarshal(o.Payload, &m) != nil {
				t.Fatal("not json")
			}
			return m[key]
		}
	}
	t.Fatal("missing", id)
	return nil
}

func TestOfflineThreeWayNonOverlapReorderAndConcurrentAddsConverge(t *testing.T) {
	base := pkg(obj("skill:x", `{"a":{"sensitivity":"public","value":1},"b":{"sensitivity":"public","value":2}}`))
	local := pkg(obj("skill:z", `{"v":{"sensitivity":"public","value":"local-add"}}`), obj("skill:x", `{"b":{"sensitivity":"public","value":2},"a":{"sensitivity":"public","value":10}}`))
	remote := pkg(obj("skill:x", `{"a":{"sensitivity":"public","value":1},"b":{"sensitivity":"public","value":20}}`), obj("skill:y", `{"v":{"sensitivity":"public","value":"remote-add"}}`))
	a, err := PreviewMerge(&base, local, remote, ChannelPrivate)
	if err != nil || len(a.Conflicts) > 0 {
		t.Fatalf("%v %#v", err, a.Conflicts)
	}
	b, err := PreviewMerge(&base, remote, local, ChannelPrivate)
	if err != nil || len(b.Conflicts) > 0 {
		t.Fatalf("%v %#v", err, b.Conflicts)
	}
	if a.Result.Digest != b.Result.Digest || field(t, a.Result, "skill:x", "a") != float64(10) || field(t, a.Result, "skill:x", "b") != float64(20) {
		t.Fatalf("non-convergent merge: %s %s %#v", a.Result.Digest, b.Result.Digest, a.Result)
	}
	// Machine order can alter source labels, but never the replayed result.
	if len(a.Result.Objects) != 3 {
		t.Fatalf("objects=%d", len(a.Result.Objects))
	}
}

func TestTypedConflictsAndOpaquePreservation(t *testing.T) {
	cases := []struct {
		name                string
		base, local, remote Object
		kind                ConflictKind
	}{
		{"same-field", obj("skill:x", `{"x":1}`), obj("skill:x", `{"x":2}`), obj("skill:x", `{"x":3}`), ConflictDivergent},
		{"delete-edit", obj("skill:x", `{"x":1}`), Object{}, obj("skill:x", `{"x":2}`), ConflictDeleteEdit},
		{"version", obj("skill:x", `{"version":1}`), obj("skill:x", `{"version":2}`), obj("skill:x", `{"version":3}`), ConflictVersion},
		{"precedence", obj("skill:x", `{"precedence":1}`), obj("skill:x", `{"precedence":2}`), obj("skill:x", `{"precedence":3}`), ConflictPrecedence},
		{"dependency", obj("skill:x", `{"dependencies":["a"]}`), obj("skill:x", `{"dependencies":["b"]}`), obj("skill:x", `{"dependencies":["c"]}`), ConflictDependency},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := pkg(tc.base)
			local := pkg()
			if tc.local.ID != "" {
				local = pkg(tc.local)
			}
			remote := pkg(tc.remote)
			p, _ := PreviewMerge(&base, local, remote, ChannelMachineLocal)
			found := false
			for _, c := range p.Conflicts {
				if c.Kind == tc.kind {
					found = true
				}
			}
			if !found {
				t.Fatalf("want %s got %#v", tc.kind, p.Conflicts)
			}
		})
	}
	raw := json.RawMessage(`"opaque-v1"`)
	b := Object{ID: "adapter:opaque", Kind: "adapter", Name: "opaque", Payload: raw, Digest: payloadDigest(raw)}
	l := b
	l.Payload = json.RawMessage(`"opaque-local"`)
	l.Digest = payloadDigest(l.Payload)
	r := b
	r.Payload = json.RawMessage(`"opaque-remote"`)
	r.Digest = payloadDigest(r.Payload)
	p, _ := PreviewMerge(ptrPkg(pkg(b)), pkg(l), pkg(r), ChannelMachineLocal)
	if len(p.Conflicts) == 0 || !bytes.Equal(p.Result.Objects[0].Payload, l.Payload) || p.Result.Objects[0].Active {
		t.Fatalf("opaque not byte-preserved inactive: %#v", p)
	}
	missing, _ := PreviewMerge(nil, pkg(obj("skill:x", `{"x":1}`)), pkg(obj("skill:x", `{"x":1}`)), ChannelMachineLocal)
	if missing.Conflicts[0].Kind != ConflictBaseMissing {
		t.Fatalf("%#v", missing.Conflicts)
	}
	skew := pkg(obj("skill:x", `{"x":1}`))
	skew.Schema = "future/v9"
	q, _ := PreviewMerge(ptrPkg(pkg()), skew, pkg(), ChannelMachineLocal)
	found := false
	for _, c := range q.Conflicts {
		found = found || c.Kind == ConflictSchema
	}
	if !found {
		t.Fatal("schema skew untyped")
	}
}
func ptrPkg(p Package) *Package { return &p }

func TestMergeReplayCrashSafetyEgressAndRollback(t *testing.T) {
	base := pkg(obj("skill:x", `{"x":{"sensitivity":"public","value":1}}`))
	local := pkg(obj("skill:x", `{"x":{"sensitivity":"public","value":2}}`))
	remote := base
	p, _ := PreviewMerge(&base, local, remote, ChannelPublic)
	if len(p.Conflicts) > 0 {
		t.Fatalf("%#v", p.Conflicts)
	}
	home := t.TempDir()
	s := fixed(home)
	out := filepath.Join(home, "channel", "merged.json")
	for _, boundary := range []int{1, 2, 3, 4} {
		if _, err := s.CommitMerge(p, out, true, boundary); err == nil {
			t.Fatalf("interrupt %d accepted", boundary)
		}
		if err := s.RecoverMerge(); err != nil {
			t.Fatalf("recover %d: %v", boundary, err)
		}
		if _, err := os.Stat(out); !os.IsNotExist(err) {
			t.Fatalf("crash boundary %d retained export", boundary)
		}
		if active, _ := s.Active(); active != "" {
			t.Fatalf("crash boundary %d changed active", boundary)
		}
		if err := s.RecoverMerge(); err != nil {
			t.Fatalf("idempotent recovery %d: %v", boundary, err)
		}
	}
	r, err := s.CommitMerge(p, out, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != "committed" {
		t.Fatalf("%#v", r)
	}
	before, _ := os.ReadFile(out)
	r2, err := s.CommitMerge(p, out, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(out)
	if !bytes.Equal(before, after) || r2.PackageID != r.PackageID {
		t.Fatal("replay changed result")
	}
	if _, err := s.Rollback(r.ID, true); err != nil {
		t.Fatal(err)
	}
	if active, _ := s.Active(); active != "" {
		t.Fatalf("rollback active=%q", active)
	}
	secretBase := pkg()
	secret := pkg(obj("skill:s", `{"token":"ghp_1234567890abcdef"}`))
	deny, _ := PreviewMerge(&secretBase, secret, secret, ChannelPublic)
	if len(deny.Conflicts) == 0 || deny.Result.ID != "" {
		t.Fatal("secret crossed source boundary")
	}
	blob, _ := json.Marshal(deny)
	if bytes.Contains(blob, []byte("ghp_1234567890abcdef")) {
		t.Fatal("secret leaked in plan")
	}
}
