package quantwatch

import (
	"net/http"
	"testing"
)

func TestDefaultLiveClientHasBoundedWholeRequestTimeout(t *testing.T) {
	client := liveClient(nil)
	if client.Timeout != defaultLiveTimeout {
		t.Fatalf("default client timeout = %v, want %v", client.Timeout, defaultLiveTimeout)
	}
	if client.Timeout <= 0 {
		t.Fatal("default live client must bound requests without caller deadlines")
	}
}

func TestLiveClientPreservesCallerSuppliedClient(t *testing.T) {
	client := &http.Client{}
	if got := liveClient(client); got != client {
		t.Fatal("caller-supplied client was replaced")
	}
}
