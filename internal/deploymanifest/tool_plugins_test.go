package deploymanifest

import (
	"reflect"
	"testing"
)

func TestToolPluginManifestParsesPinnedProfilesAndLayers(t *testing.T) {
	m, err := Parse([]byte(`[tool_plugins]
plugins = [{ id = "builtin.audit", version = "1", digest = "sha256:abc" }]
[tool_plugins.organization]
require_witness = true
disclosure = "reviewed"
[tool_plugins.user]
wait_mode = "local"
require_witness = false
`))
	if err != nil {
		t.Fatal(err)
	}
	want := []PluginSelection{{ID: "builtin.audit", Version: "1", Digest: "sha256:abc"}}
	if !reflect.DeepEqual(m.ToolPlugins.Plugins, want) {
		t.Fatalf("plugins=%+v", m.ToolPlugins.Plugins)
	}
	if !m.ToolPlugins.Organization.RequireWitness || m.ToolPlugins.Organization.Disclosure != "reviewed" {
		t.Fatalf("org=%+v", m.ToolPlugins.Organization)
	}
	if m.ToolPlugins.User.WaitMode != "local" || m.ToolPlugins.User.RequireWitness {
		t.Fatalf("user=%+v", m.ToolPlugins.User)
	}
}

func TestToolPluginManifestRejectsUnknownInlineField(t *testing.T) {
	_, err := Parse([]byte("[tool_plugins]\nplugins = [{ id = \"builtin.audit\", path = \"evil\" }]\n"))
	if err == nil {
		t.Fatal("expected error")
	}
	le, ok := err.(*LoadError)
	if !ok || le.Reason != ReasonUnknownKey {
		t.Fatalf("err=%v", err)
	}
}
