package managedharness_test

import (
	"errors"
	"strings"
	"testing"

	mh "github.com/anthony-chaudhary/fak/pkg/managedharness"
)

func product(id, variant, compat string, caps, layers []string) mh.Product {
	return mh.Product{ID: mh.ProductID(id), Variant: variant, Compatibility: compat, Capabilities: caps, Layers: layers}
}
func release(t *testing.T, p mh.Product, reply string) mh.Bundle {
	t.Helper()
	b, err := mh.BuildRelease(p, map[string]any{"offline_reply": reply}, mh.Provenance{Source: "fixture", Revision: "r1", Builder: "test"})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestLocalLifecycleTwoVariants(t *testing.T) {
	s, err := mh.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	base := []string{"kernel", "offline-work"}
	selfV1 := release(t, product("fak-self", "self", "v1", []string{"private-context", "offline-work"}, append(base, "private-layer")), "self v1")
	publicV1 := release(t, product("public-safe-project", "public", "v1", []string{"offline-work"}, append(base, "public-layer")), "public v1")
	for _, b := range []mh.Bundle{selfV1, publicV1} {
		if _, err := s.Publish(b); err != nil {
			t.Fatal(err)
		}
	}
	health := func(mh.Bundle) error { return nil }
	if r, err := s.Install("self-local", selfV1.Release.ID, health); err != nil || r.Status != "activated" {
		t.Fatalf("self install: %+v %v", r, err)
	}
	if r, err := s.Install("public-local", publicV1.Release.ID, health); err != nil || r.Status != "activated" {
		t.Fatalf("public install: %+v %v", r, err)
	}
	work := func(b mh.Bundle) (string, error) { return string(b.Payload), nil }
	for _, id := range []mh.InstallationID{"self-local", "public-local"} {
		r, err := s.Run(id, work)
		if err != nil || r.Status != "completed" || !strings.Contains(r.Output, "offline_reply") {
			t.Fatalf("work %s: %+v %v", id, r, err)
		}
	}
	pub, _ := s.Run("public-local", work)
	if contains(pub.Capabilities, "private-context") {
		t.Fatalf("private capability leaked: %+v", pub)
	}

	selfV2 := release(t, product("fak-self", "self", "v1", []string{"private-context", "offline-work"}, append(base, "private-layer")), "self v2")
	if _, err := s.Publish(selfV2); err != nil {
		t.Fatal(err)
	}
	updated, err := s.Update("self-local", selfV2.Release.ID, health)
	if err != nil || updated.Status != "activated" || updated.LastKnownGood == "" {
		t.Fatalf("update: %+v %v", updated, err)
	}

	badHealth := release(t, product("fak-self", "self", "v1", []string{"offline-work"}, base), "unhealthy")
	s.Publish(badHealth)
	rolled, err := s.Update("self-local", badHealth.Release.ID, func(mh.Bundle) error { return errors.New("selfcheck failed") })
	if err != nil || rolled.Status != "rolled_back" || rolled.After != updated.After {
		t.Fatalf("rollback: %+v %v", rolled, err)
	}
	state, _ := s.Inspect("self-local")
	if state.Effective != updated.After || state.Desired != selfV2.Release.ID {
		t.Fatalf("failed update mutated state: %+v", state)
	}

	incompatible := release(t, product("fak-self", "self", "v2", []string{"offline-work"}, base), "future")
	s.Publish(incompatible)
	refused, err := s.Update("self-local", incompatible.Release.ID, health)
	if err != nil || refused.Status != "refused" || refused.After != updated.After {
		t.Fatalf("refusal: %+v %v", refused, err)
	}
}

func TestDeterministicReleaseAndSecretRefusal(t *testing.T) {
	p := product("p", "v", "v1", []string{"b", "a", "a"}, []string{"z", "a"})
	a := release(t, p, "ok")
	b := release(t, p, "ok")
	if a.Release.Digest != b.Release.Digest || a.Release.ID != b.Release.ID {
		t.Fatal("release serialization is nondeterministic")
	}
	secret, err := mh.BuildRelease(p, map[string]any{"token": "do-not-package"}, mh.Provenance{Source: "fixture", Revision: "r1"})
	if err != nil {
		t.Fatal(err)
	}
	s, _ := mh.Open(t.TempDir())
	if _, err = s.Publish(secret); err == nil {
		t.Fatal("release containing installation secret accepted")
	}
}

func contains(in []string, w string) bool {
	for _, v := range in {
		if v == w {
			return true
		}
	}
	return false
}
