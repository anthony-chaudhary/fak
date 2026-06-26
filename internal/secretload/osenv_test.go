package secretload

import "testing"

func TestOSEnvLookupPrefixAndEmpty(t *testing.T) {
	env := map[string]string{
		"FAK_API_KEY": "secret-val",
		"FAK_BLANK":   "", // exported but empty -> must be treated as absent
	}
	e := &OSEnv{Prefix: "FAK_", lookup: func(k string) (string, bool) {
		v, ok := env[k]
		return v, ok
	}}

	if v, ok := e.Lookup("API_KEY"); !ok || v != "secret-val" {
		t.Fatalf("prefixed lookup = (%q,%v), want (secret-val,true)", v, ok)
	}
	if _, ok := e.Lookup("BLANK"); ok {
		t.Error("empty env value must report absent, not a hit")
	}
	if _, ok := e.Lookup("UNSET"); ok {
		t.Error("unset key must report absent")
	}
	if e.Name() != "os-env" {
		t.Errorf("Name = %q", e.Name())
	}
}

func TestOSEnvSourceReadsRealEnv(t *testing.T) {
	t.Setenv("FAK_SECRETS_OSENV_PROBE", "present")
	if v, ok := OSEnvSource().Lookup("FAK_SECRETS_OSENV_PROBE"); !ok || v != "present" {
		t.Fatalf("real-env lookup = (%q,%v)", v, ok)
	}
}
