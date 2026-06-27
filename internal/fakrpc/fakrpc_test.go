package fakrpc

import (
	"bytes"
	"errors"
	"testing"
)

func TestRequestEncodeRoundTrip(t *testing.T) {
	want := Request{
		Nonce:   "bell0042",
		Kind:    KindDevTurn,
		Model:   "glm-5.2",
		Payload: "say pong",
		Params:  map[string]string{"max_tokens": "512"},
	}
	enc, err := want.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := DecodeRequest(enc)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if got.Nonce != want.Nonce || got.Kind != want.Kind || got.Model != want.Model || got.Payload != want.Payload {
		t.Fatalf("round-trip mismatch:\n got=%+v\nwant=%+v", got, want)
	}
	if got.Params["max_tokens"] != "512" {
		t.Fatalf("params lost in round-trip: %+v", got.Params)
	}
}

func TestRequestEncodeIsSingleLine(t *testing.T) {
	// A spooled request must survive a line-oriented bridge, so its JSON must not contain
	// a newline that would split it into two transport frames.
	enc, err := Request{Nonce: "n1", Kind: KindTurn, Payload: "multi\nline\npayload"}.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if bytes.ContainsRune(enc, '\n') {
		t.Fatalf("encoded request contains a literal newline: %q", enc)
	}
}

func TestRequestValidate(t *testing.T) {
	cases := []struct {
		name string
		req  Request
		want error
	}{
		{"ok turn", Request{Nonce: "n1", Kind: KindTurn}, nil},
		{"ok dev-turn", Request{Nonce: "bell-0042.x", Kind: KindDevTurn}, nil},
		{"empty nonce", Request{Nonce: "", Kind: KindTurn}, ErrEmptyNonce},
		{"blank nonce", Request{Nonce: "   ", Kind: KindTurn}, ErrEmptyNonce},
		{"nonce with space", Request{Nonce: "a b", Kind: KindTurn}, ErrNonceNotToken},
		{"nonce with tab", Request{Nonce: "a\tb", Kind: KindTurn}, ErrNonceNotToken},
		// The frame splits the header on the full unicode.IsSpace set, so Validate must too.
		{"nonce with vertical tab", Request{Nonce: "a\vb", Kind: KindTurn}, ErrNonceNotToken},
		{"nonce with form feed", Request{Nonce: "a\fb", Kind: KindTurn}, ErrNonceNotToken},
		{"nonce with nbsp", Request{Nonce: "a b", Kind: KindTurn}, ErrNonceNotToken},
		// '<'/'>' would collide with the frame sentinels.
		{"nonce with sentinel >>>", Request{Nonce: "a>>>b", Kind: KindTurn}, ErrNonceNotToken},
		{"nonce with angle <", Request{Nonce: "a<b", Kind: KindTurn}, ErrNonceNotToken},
		{"unknown kind", Request{Nonce: "n1", Kind: Kind("curl")}, ErrUnknownKind},
		{"empty kind", Request{Nonce: "n1", Kind: Kind("")}, ErrUnknownKind},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.Validate()
			if !errors.Is(err, tc.want) {
				t.Fatalf("Validate() = %v, want errors.Is %v", err, tc.want)
			}
		})
	}
}

func TestValidNonceMatchesFrameRoundTrip(t *testing.T) {
	// The contract ValidNonce promises: any nonce it accepts survives an
	// EncodeFrame→DecodeFrame round-trip with the nonce intact.
	for _, n := range []string{"n1", "bell0042", "a-b.c:d_e", "a=b", "UPPER123"} {
		if err := ValidNonce(n); err != nil {
			t.Fatalf("ValidNonce(%q) unexpectedly rejected: %v", n, err)
		}
		got, err := DecodeFrame(EncodeFrame(n, 0, []byte("body")))
		if err != nil {
			t.Fatalf("DecodeFrame for accepted nonce %q: %v", n, err)
		}
		if got.Nonce != n {
			t.Fatalf("nonce round-trip mismatch: got %q want %q", got.Nonce, n)
		}
	}
}

func TestEncodeRefusesMalformed(t *testing.T) {
	// Encode must not spool a request a worker would only reject later.
	if _, err := (Request{Nonce: "", Kind: KindTurn}).Encode(); !errors.Is(err, ErrEmptyNonce) {
		t.Fatalf("Encode of empty-nonce request = %v, want ErrEmptyNonce", err)
	}
}

func TestDecodeRequestRejectsBadJSON(t *testing.T) {
	if _, err := DecodeRequest([]byte("not json")); err == nil {
		t.Fatal("DecodeRequest of non-JSON should error")
	}
}

func TestDecodeRequestRejectsUnknownKind(t *testing.T) {
	if _, err := DecodeRequest([]byte(`{"nonce":"n1","kind":"curl","payload":"x"}`)); !errors.Is(err, ErrUnknownKind) {
		t.Fatalf("DecodeRequest of unknown kind = %v, want ErrUnknownKind", err)
	}
}
