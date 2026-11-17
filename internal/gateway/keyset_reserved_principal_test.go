package gateway

// keyset_reserved_principal_test.go — #14448: a --key-principal tenant must never
// name a kernel AUTHORITY label. The keyset principal is the tenant ISOLATION
// identity, deliberately distinct from the authority class — but surfaces compare
// the stamped principal against the labels by string (scopeReadSelf treats caller
// == "human" as read-all; classifyPrincipal maps the relay classes), so a tenant
// bound to a reserved name would inherit that label's semantics on every such
// surface. ParseKeyPrincipals refuses the name at boot.

import "testing"

// TestParseKeyPrincipalsReservedPrincipalRefused pins the reservation: every
// kernel authority label (and a capitalised spelling of two of them) is refused,
// while an ordinary org/project name parses and binds.
func TestParseKeyPrincipalsReservedPrincipalRefused(t *testing.T) {
	env := map[string]string{"ACME_KEY": "tenant-secret"}
	lookup := func(name string) string { return env[name] }

	reserved := []string{"human", "self-model", "peer-agent", "timer", "network-tool", "unknown", "Human", "PEER-AGENT"}
	for _, name := range reserved {
		if _, err := ParseKeyPrincipals([]string{name + "=ACME_KEY"}, lookup); err == nil {
			t.Errorf("principal %q must be refused at boot: a tenant key may not name a kernel authority label", name)
		}
	}

	got, err := ParseKeyPrincipals([]string{"acme=ACME_KEY"}, lookup)
	if err != nil {
		t.Fatalf("an ordinary org name must parse: %v", err)
	}
	if got["tenant-secret"] != "acme" {
		t.Fatalf("ordinary principal misbound: %v", got)
	}
}
