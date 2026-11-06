package modelengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
func TestPDWireHandoffErrorIsLogged(t *testing.T) {
	// Keep this test serial because the production diagnostic uses the standard
	// logger. A truncated length prefix fails before an error reply is possible.
	var diagnostic bytes.Buffer
	previous := log.Writer()
	log.SetOutput(&diagnostic)
	defer log.SetOutput(previous)

	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	deadline := time.Now().Add(time.Second)
	if err := server.SetDeadline(deadline); err != nil {
		t.Fatal(err)
	}
	if err := client.SetDeadline(deadline); err != nil {
		t.Fatal(err)
	}
	written := make(chan error, 1)
	go func() {
		_, err := client.Write([]byte{0, 0})
		written <- err
		client.Close()
	}()

	(&PDWireServer{}).handleConn(context.Background(), server)
	if err := <-written; err != nil {
		t.Fatalf("write truncated frame: %v", err)
	}
	for _, want := range []string{"modelengine: P/D wire handoff failed", `peer="pipe"`, `code=runtime`, `class=truncated_frame`} {
		if got := diagnostic.String(); !strings.Contains(got, want) {
			t.Fatalf("handoff diagnostic = %q, want %q", got, want)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireHandoffLogOmitsRequestData(t *testing.T) {
	m := &model.Model{Cfg: SyntheticConfig()}
	s := &PDWireServer{cluster: &NativePDCluster{m: m}}
	for _, tc := range []struct {
		name   string
		header pdWireHeader
		secret string
	}{
		{
			name:   "protocol",
			header: pdWireHeader{Protocol: "private-wire-version-826491"},
			secret: "private-wire-version-826491",
		},
		{
			name: "prompt token",
			header: pdWireHeader{
				Protocol:     PDWireProtocol,
				ConfigSHA256: pdWireConfigDigest(m.Cfg),
				VocabSize:    m.Cfg.VocabSize,
				PromptIDs:    []int{987654321},
			},
			secret: "987654321",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var diagnostic bytes.Buffer
			previous := log.Writer()
			log.SetOutput(&diagnostic)
			defer log.SetOutput(previous)

			server, client := net.Pipe()
			defer server.Close()
			defer client.Close()
			deadline := time.Now().Add(time.Second)
			if err := server.SetDeadline(deadline); err != nil {
				t.Fatal(err)
			}
			if err := client.SetDeadline(deadline); err != nil {
				t.Fatal(err)
			}
			header := mustJSON(t, tc.header)
			acknowledged := make(chan error, 1)
			go func() {
				if err := pdWireWriteFrame(client, header, pdWireMaxHeaderBytes); err != nil {
					acknowledged <- err
					return
				}
				raw, err := pdWireReadFrame(client, pdWireMaxReplyBytes)
				if err != nil {
					acknowledged <- err
					return
				}
				var ack pdWireAck
				if err := json.Unmarshal(raw, &ack); err != nil {
					acknowledged <- err
					return
				}
				if ack.OK || ack.Code != pdWireCodeProtocol || !strings.Contains(ack.Error, tc.secret) {
					acknowledged <- fmt.Errorf("negative ack did not exercise request-derived error: %+v", ack)
					return
				}
				acknowledged <- nil
			}()
			s.handleConn(context.Background(), server)
			if err := <-acknowledged; err != nil {
				t.Fatal(err)
			}
			got := diagnostic.String()
			if strings.Contains(got, tc.secret) {
				t.Fatalf("handoff diagnostic contains request data: %q", got)
			}
			if !strings.Contains(got, "code=protocol class=protocol") {
				t.Fatalf("handoff diagnostic = %q, want protocol failure classification", got)
			}
		})
	}
}
