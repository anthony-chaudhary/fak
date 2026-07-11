package promptaudit

import (
	"strings"
	"testing"
)

// baseURLSegment builds a control-metadata segment that carries BOTH the stego
// hostname-marker (a non-ASCII apostrophe in the "Today's date is" carrier — the
// channel the article uses to smuggle a host CLASS) AND the literal gateway host
// spliced in from a custom ANTHROPIC_BASE_URL, attributed to the provider-shim
// producer (the base-URL-classifier path the article warns about).
func baseURLSegment(host string) Segment {
	return Segment{
		Field:  "controlMetadata",
		Source: SourceProviderShim,
		// U+2019 apostrophe marker -> Scan flags ChannelHostname; the literal
		// host is what RedactHostnames must scrub.
		Raw: "Today’s date is 2026-06-30. base=https://" + host + "/v1",
	}
}

// flaggedHostChannel reports whether any finding in the report is on the
// hostname-marker channel.
func flaggedHostChannel(r Report) bool {
	for _, sf := range r.Findings {
		if sf.Finding.Channel == ChannelHostname {
			return true
		}
	}
	return false
}

// TestAuditFlagsHostAndRedactRemovesIt is the acceptance test for #1693: on a
// custom-ANTHROPIC_BASE_URL fixture the existing Audit/Scan flags the hostname
// channel, and RedactHostnames scrubs the literal gateway host out of the same
// control metadata — while an allowlisted host survives verbatim.
func TestAuditFlagsHostAndRedactRemovesIt(t *testing.T) {
	const gwHost = "gw.shadow-classifier.example"
	seg := baseURLSegment(gwHost)

	// 1) The existing audit flags the HOST channel (ChannelHostname) on this
	//    fixture — the host-marker is detected before any redaction.
	r := Audit([]Segment{seg})
	if !flaggedHostChannel(r) {
		t.Fatalf("Audit did not flag the hostname channel on the base-URL fixture; findings=%v", r.Findings)
	}

	// 2) RedactHostnames removes the literal gateway host when it is NOT allowlisted.
	got := RedactHostnames(seg.Raw, nil)
	if strings.Contains(got, gwHost) {
		t.Errorf("RedactHostnames left the gateway host in place: %q", got)
	}
	if !strings.Contains(got, hostRedaction) {
		t.Errorf("RedactHostnames did not insert the redaction placeholder: %q", got)
	}
	// The date token is not a host and must survive untouched.
	if !strings.Contains(got, "2026-06-30") {
		t.Errorf("RedactHostnames wrongly altered the date token: %q", got)
	}

	// 3) An allowlisted host survives verbatim.
	kept := RedactHostnames(seg.Raw, []string{gwHost})
	if !strings.Contains(kept, gwHost) {
		t.Errorf("RedactHostnames redacted an allowlisted host: %q", kept)
	}
	if strings.Contains(kept, hostRedaction) {
		t.Errorf("RedactHostnames inserted a placeholder for an allowlisted host: %q", kept)
	}
}

// TestRedactHostnames is the table matrix: hosts are scrubbed unless the host
// (or its parent domain) is allowlisted; non-host tokens are never touched.
func TestRedactHostnames(t *testing.T) {
	cases := []struct {
		name       string
		in         string
		allow      []string
		wantHas    []string // substrings that MUST remain
		wantGone   []string // substrings that must NOT remain
		wantRedact bool     // whether the placeholder should appear
	}{
		{
			name:       "bare-host-redacted",
			in:         "base=https://api.evil-gw.example/v1",
			allow:      nil,
			wantGone:   []string{"api.evil-gw.example"},
			wantRedact: true,
		},
		{
			name:       "exact-allowlist-kept",
			in:         "base=https://api.anthropic.com/v1",
			allow:      []string{"api.anthropic.com"},
			wantHas:    []string{"api.anthropic.com"},
			wantRedact: false,
		},
		{
			name:       "subdomain-of-allowlisted-domain-kept",
			in:         "host=api.anthropic.com and edge=gw.anthropic.com",
			allow:      []string{"anthropic.com"},
			wantHas:    []string{"api.anthropic.com", "gw.anthropic.com"},
			wantRedact: false,
		},
		{
			name:       "mixed-keep-and-redact",
			in:         "ok=api.anthropic.com bad=leak.internal.example",
			allow:      []string{"anthropic.com"},
			wantHas:    []string{"api.anthropic.com"},
			wantGone:   []string{"leak.internal.example"},
			wantRedact: true,
		},
		{
			name:       "case-insensitive-allowlist",
			in:         "host=API.Anthropic.COM",
			allow:      []string{"anthropic.com"},
			wantHas:    []string{"API.Anthropic.COM"},
			wantRedact: false,
		},
		{
			name:       "date-tokens-are-not-hosts",
			in:         "Today's date is 2026-06-30 and 2026/06/30.",
			allow:      nil,
			wantHas:    []string{"2026-06-30", "2026/06/30"},
			wantRedact: false,
		},
		{
			name:       "no-hosts-unchanged",
			in:         "plain control metadata with no hosts at all",
			allow:      nil,
			wantHas:    []string{"plain control metadata with no hosts at all"},
			wantRedact: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := RedactHostnames(c.in, c.allow)
			for _, s := range c.wantHas {
				if !strings.Contains(got, s) {
					t.Errorf("want %q retained, got %q", s, got)
				}
			}
			for _, s := range c.wantGone {
				if strings.Contains(got, s) {
					t.Errorf("want %q redacted out, got %q", s, got)
				}
			}
			if hasRedact := strings.Contains(got, hostRedaction); hasRedact != c.wantRedact {
				t.Errorf("placeholder present=%v want=%v (out=%q)", hasRedact, c.wantRedact, got)
			}
		})
	}
}
