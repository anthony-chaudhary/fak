package hfhub

import (
	"net/http"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestResolveURLPreservesLiteralHubFilename(t *testing.T) {
	t.Parallel()

	const prefix = "/owner/repo/resolve/main/"
	for _, tc := range []struct {
		name, file, escaped string
	}{
		{"reserved and Unicode", "weights/模型 #1?50%.gguf", "weights/%E6%A8%A1%E5%9E%8B%20%231%3F50%25.gguf"},
		{"percent-looking literal", "weights/model%2Fpart.gguf", "weights/model%252Fpart.gguf"},
		{"ordinary nested path", "a/b.gguf", "a/b.gguf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Hub metadata supplies a filename, not a URL-escaped path.
			file, err := selectLoadArtifact([]repoSibling{{RFilename: tc.file}})
			if err != nil || file != tc.file {
				t.Fatalf("selectLoadArtifact = %q, %v; want literal %q", file, err, tc.file)
			}
			ref := Ref{Repo: "owner/repo", Revision: "main", File: file}
			got := ref.ResolveURL("https://example.test/")
			want := "https://example.test" + prefix + tc.escaped
			if got != want {
				t.Fatalf("ResolveURL = %q, want %q", got, want)
			}
			req, err := http.NewRequest(http.MethodGet, got, nil)
			if err != nil {
				t.Fatal(err)
			}
			if req.URL.Path != prefix+tc.file || req.URL.RawQuery != "" || req.URL.Fragment != "" ||
				req.URL.RequestURI() != prefix+tc.escaped {
				t.Fatalf("request URL = %#v; want literal path %q and target %q",
					req.URL, prefix+tc.file, prefix+tc.escaped)
			}
		})
	}
}
