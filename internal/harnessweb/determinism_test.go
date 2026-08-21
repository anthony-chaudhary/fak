package harnessweb

import (
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestHomeRenderDeterminism(t *testing.T) {
	render := func() renderSnapshot {
		response := httptest.NewRecorder()
		request := httptest.NewRequest("GET", "/", nil)
		handler(newStore()).ServeHTTP(response, request)
		return renderSnapshot{
			Code:   response.Code,
			Header: response.Header().Clone(),
			Body:   append([]byte(nil), response.Body.Bytes()...),
		}
	}

	first := render()
	second := render()
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same home input produced different renders\nfirst:  %#v\nsecond: %#v", first, second)
	}
}

type renderSnapshot struct {
	Code   int
	Header map[string][]string
	Body   []byte
}
