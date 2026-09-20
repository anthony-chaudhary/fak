package fakclient

import "testing"

func TestLoopbackFallbackURL(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
		ok   bool
	}{
		{
			name: "ipv4 literal with port and path",
			in:   "http://127.0.0.1:8080/healthz",
			want: "http://localhost:8080/healthz",
			ok:   true,
		},
		{
			name: "ipv4 literal with query",
			in:   "http://127.0.0.1:8080/v1/models?foo=bar",
			want: "http://localhost:8080/v1/models?foo=bar",
			ok:   true,
		},
		{
			name: "ipv4 literal preserves fragment",
			in:   "http://127.0.0.1:8080/healthz#frag",
			want: "http://localhost:8080/healthz#frag",
			ok:   true,
		},
		{
			name: "ipv4 literal in full 127 0 0 0/8 range",
			in:   "http://127.5.6.7:9000/",
			want: "http://localhost:9000/",
			ok:   true,
		},
		{
			name: "ipv4 literal no port",
			in:   "http://127.0.0.1/healthz",
			want: "http://localhost/healthz",
			ok:   true,
		},
		{
			name: "ipv6 loopback bracketed with port and path",
			in:   "http://[::1]:8080/v1/models",
			want: "http://localhost:8080/v1/models",
			ok:   true,
		},
		{
			name: "ipv6 loopback bracketed no port",
			in:   "http://[::1]/healthz",
			want: "http://localhost/healthz",
			ok:   true,
		},
		{
			name: "preserves userinfo",
			in:   "http://user:pass@127.0.0.1:8080/healthz",
			want: "http://user:pass@localhost:8080/healthz",
			ok:   true,
		},
		{
			name: "already localhost yields no fallback",
			in:   "http://localhost:8080/healthz",
			want: "",
			ok:   false,
		},
		{
			name: "non-loopback public ip yields no fallback",
			in:   "http://10.0.0.5:8080/healthz",
			want: "",
			ok:   false,
		},
		{
			name: "non-loopback hostname yields no fallback",
			in:   "http://example.com:8080/healthz",
			want: "",
			ok:   false,
		},
		{
			name: "bad url yields no fallback",
			in:   "http://127.0.0.1:8080/%zz",
			want: "",
			ok:   false,
		},
		{
			name: "empty yields no fallback",
			in:   "",
			want: "",
			ok:   false,
		},
		{
			name: "relative url yields no fallback",
			in:   "/healthz",
			want: "",
			ok:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := LoopbackFallbackURL(tt.in)
			if ok != tt.ok {
				t.Fatalf("LoopbackFallbackURL(%q) ok = %v, want %v", tt.in, ok, tt.ok)
			}
			if got != tt.want {
				t.Fatalf("LoopbackFallbackURL(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
