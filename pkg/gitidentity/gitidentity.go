// Package gitidentity refuses commits made under a test-fixture git identity
// (MQ-19, docs/tickets/model-class-quirks/TICKET-MQ19-test-git-identity-on-trunk.md).
//
// Test helpers set `user.name fak-test` / `user.email test@fak.local` (and
// test, DoctorTest, orphan-test, Witness, t <t@t>, ...) inside throwaway
// repositories. When that write leaks into a real checkout, every commit made
// there is authored by the fixture: ~1,400 such commits reached fak-private
// trunk after 2026-09-01, none attributable to a run. The classifier here is
// pure; the pre-commit gate (cmd/fak-flow precommit) and the pre-push gate
// (fak-sync release prepush) apply it and never edit anyone's config: choosing
// the canonical identity is an operator decision.
package gitidentity

import (
	"fmt"
	"strings"
)

// Reason is the closed vocabulary of a fixture verdict.
type Reason string

const (
	// ReasonFixtureName: the name is a well-known fixture name.
	ReasonFixtureName Reason = "FIXTURE_NAME"
	// ReasonConfigKeyName: the name is a git config key ("user.email"), the
	// signature of a mis-quoted `git config user.name user.email ...`.
	ReasonConfigKeyName Reason = "CONFIG_KEY_AS_NAME"
	// ReasonFixtureEmail: the email is a well-known fixture address.
	ReasonFixtureEmail Reason = "FIXTURE_EMAIL"
	// ReasonReservedDomain: the email domain is reserved for tests and
	// documentation (RFC 2606/6761: example.{com,net,org}, .test, .example,
	// .invalid, .localhost) or is the fixture-only .local zone.
	ReasonReservedDomain Reason = "RESERVED_EMAIL_DOMAIN"
	// ReasonUnroutableDomain: the email domain cannot be a real mail domain
	// (no dot, or a one-letter top-level label: t@t, a@b.c).
	ReasonUnroutableDomain Reason = "UNROUTABLE_EMAIL_DOMAIN"
)

// Verdict is the classification of one name/email pair. The zero value (Fixture
// false) is a pass.
type Verdict struct {
	Fixture bool   `json:"fixture"`
	Reason  Reason `json:"reason,omitempty"`
	// Field is "name" or "email": the value that matched.
	Field string `json:"field,omitempty"`
	Value string `json:"value,omitempty"`
}

// String renders a refusal as `<field> "<value>" (<REASON>)`, or "ok".
func (v Verdict) String() string {
	if !v.Fixture {
		return "ok"
	}
	return fmt.Sprintf("%s %q (%s)", v.Field, v.Value, v.Reason)
}

// fixtureNames are compared case-insensitively after trimming.
var fixtureNames = map[string]bool{
	"fak-test": true, "fak-flow-tester": true, "test": true, "tester": true,
	"doctortest": true, "orphan-test": true, "witness": true, "t": true, "a": true,
}

var configKeyNames = map[string]bool{"user.name": true, "user.email": true}

var fixtureEmails = map[string]bool{"t@t": true, "a@b.c": true}

// reservedTLDs are top-level labels that never route mail to a person.
var reservedTLDs = map[string]bool{"test": true, "example": true, "invalid": true, "localhost": true, "local": true}

// reservedDomains are second-level domains reserved for documentation.
var reservedDomains = []string{"example.com", "example.net", "example.org"}

// Classify reports whether name/email is a test-fixture identity. Either field
// matching is enough; the name is checked first. An empty field never matches:
// a missing identity is git's own fail-closed case, not this guard's.
func Classify(name, email string) Verdict {
	n := strings.ToLower(strings.TrimSpace(name))
	switch {
	case n == "":
	case configKeyNames[n]:
		return Verdict{Fixture: true, Reason: ReasonConfigKeyName, Field: "name", Value: name}
	case fixtureNames[n]:
		return Verdict{Fixture: true, Reason: ReasonFixtureName, Field: "name", Value: name}
	}
	e := strings.ToLower(strings.TrimSpace(email))
	if e == "" {
		return Verdict{}
	}
	if fixtureEmails[e] {
		return Verdict{Fixture: true, Reason: ReasonFixtureEmail, Field: "email", Value: email}
	}
	at := strings.LastIndexByte(e, '@')
	if at < 0 {
		return Verdict{}
	}
	domain := strings.TrimSuffix(e[at+1:], ".")
	for _, d := range reservedDomains {
		if domain == d || strings.HasSuffix(domain, "."+d) {
			return Verdict{Fixture: true, Reason: ReasonReservedDomain, Field: "email", Value: email}
		}
	}
	dot := strings.LastIndexByte(domain, '.')
	tld := domain[dot+1:]
	if reservedTLDs[tld] {
		return Verdict{Fixture: true, Reason: ReasonReservedDomain, Field: "email", Value: email}
	}
	if dot < 0 || len(tld) < 2 {
		return Verdict{Fixture: true, Reason: ReasonUnroutableDomain, Field: "email", Value: email}
	}
	return Verdict{}
}

// ParseIdent splits a git ident line ("Name <email> 1727400000 -0700", as
// `git var GIT_AUTHOR_IDENT` prints it) into name and email. ok is false when
// the line has no <email>.
func ParseIdent(line string) (name, email string, ok bool) {
	open := strings.IndexByte(line, '<')
	if open < 0 {
		return "", "", false
	}
	end := strings.IndexByte(line[open:], '>')
	if end < 0 {
		return "", "", false
	}
	return strings.TrimSpace(line[:open]), strings.TrimSpace(line[open+1 : open+end]), true
}

// IsForgeRemote reports whether a remote URL names a network host (https://,
// ssh://, git://, or scp-style user@host:path) rather than a local path or
// file:// URL. The guard scopes to checkouts and pushes that can publish to a
// forge; test fixtures commit and push only to local paths, so their fixture
// identities never trip it.
func IsForgeRemote(url string) bool {
	u := strings.TrimSpace(url)
	if u == "" {
		return false
	}
	if scheme, rest, ok := strings.Cut(u, "://"); ok {
		switch strings.ToLower(scheme) {
		case "file":
			return false
		case "http", "https", "ssh", "git", "git+ssh", "ssh+git":
			host, _, _ := strings.Cut(rest, "/")
			return host != ""
		}
		return false
	}
	// scp-like "[user@]host:path": a colon before any slash. A one-letter
	// "host" is a Windows drive (C:/x, C:\x).
	colon := strings.IndexByte(u, ':')
	if colon <= 0 || strings.ContainsAny(u[:colon], `/\`) {
		return false
	}
	host := u[:colon]
	if at := strings.LastIndexByte(host, '@'); at >= 0 {
		host = host[at+1:]
	}
	return len(host) > 1
}

// RemoteURLsConfig is the `git config` argv tail that lists every remote URL;
// feed its output to AnyForgeRemote.
var RemoteURLsConfig = []string{"config", "--get-regexp", `^remote\..*\.(url|pushurl)$`}

// AnyForgeRemote reports whether any "<key> <url>" line of a RemoteURLsConfig
// listing names a forge (IsForgeRemote). The guards enforce only in such a
// checkout: that is where a fixture identity can reach a published trunk.
func AnyForgeRemote(listing string) bool {
	for _, line := range strings.Split(strings.TrimSpace(listing), "\n") {
		if _, url, ok := strings.Cut(strings.TrimSpace(line), " "); ok && IsForgeRemote(url) {
			return true
		}
	}
	return false
}

// Repair is the one-line operator repair for a refused identity: set the
// canonical identity (the operator's choice) at the scope that holds the
// fixture value, which the caller names from `git config --show-origin`.
const Repair = `git config user.name "<canonical name>" && git config user.email "<canonical email>"   (or unset the repo-local fixture: git config --unset user.name; git config --unset user.email)`
