package demoutil

import (
	"io/fs"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"
)

func TestIndexHandlerBoundaries(t *testing.T) {
	h := IndexHandler(fstest.MapFS{"page.html": {Data: []byte("<h1>demo</h1>")}})
	for _, tc := range []struct {
		path     string
		wantCode int
		wantType string
	}{
		{"/", http.StatusOK, "text/html; charset=utf-8"},
		{"/missing", http.StatusNotFound, "text/plain; charset=utf-8"},
	} {
		r := httptest.NewRequest(http.MethodGet, tc.path, nil)
		w := httptest.NewRecorder()
		h(w, r)
		if w.Code != tc.wantCode {
			t.Fatalf("%s code = %d, want %d", tc.path, w.Code, tc.wantCode)
		}
		if got := w.Header().Get("Content-Type"); got != tc.wantType {
			t.Fatalf("%s type = %q, want %q", tc.path, got, tc.wantType)
		}
	}

	missing := IndexHandler(fstest.MapFS{})
	w := httptest.NewRecorder()
	missing(w, httptest.NewRequest(http.MethodGet, "/", nil))
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("missing page code = %d, want 500", w.Code)
	}
}

var _ fs.FS = fstest.MapFS{}
