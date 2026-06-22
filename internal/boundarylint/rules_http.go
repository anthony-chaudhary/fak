package boundarylint

import (
	"go/ast"
	"go/token"
)

// MissingHTTPTimeout flags outbound HTTP made through an API that cannot carry a
// timeout: the net/http package-level helpers (http.Get/Post/Head/PostForm) and
// http.DefaultClient both use a client with Timeout==0 and no transport deadlines, so
// a dead or stalled peer hangs the caller forever. This is an external-boundary claim —
// "the network will answer" — with nothing enforcing it.
//
// The fix is a configured client: for normal calls `&http.Client{Timeout: ...}`; for
// large streamed downloads, a client whose Transport sets DialContext / TLSHandshake /
// ResponseHeaderTimeout but leaves Client.Timeout at 0 (so a multi-GB body is not cut
// off mid-stream). The rule does NOT flag &http.Client{} literals — those may carry
// either form of timeout — only the helpers that structurally cannot.
type MissingHTTPTimeout struct{}

func (MissingHTTPTimeout) Code() string { return "MISSING_HTTP_TIMEOUT" }

var noTimeoutHelpers = map[string]bool{"Get": true, "Post": true, "Head": true, "PostForm": true}

func (r MissingHTTPTimeout) Check(fset *token.FileSet, file *ast.File, relPath string) []Finding {
	var out []Finding
	ast.Inspect(file, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || pkg.Name != "http" {
			return true
		}
		switch {
		case sel.Sel.Name == "DefaultClient":
			out = append(out, Finding{
				Code:   r.Code(),
				File:   relPath,
				Line:   fset.Position(sel.Pos()).Line,
				Detail: "http.DefaultClient has no timeout; inject a client with a Timeout (or transport deadlines for downloads)",
			})
		case noTimeoutHelpers[sel.Sel.Name]:
			out = append(out, Finding{
				Code:   r.Code(),
				File:   relPath,
				Line:   fset.Position(sel.Pos()).Line,
				Detail: "http." + sel.Sel.Name + " uses the default client with no timeout; use a configured *http.Client",
			})
		}
		return true
	})
	return out
}
