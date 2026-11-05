package tcp

import (
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: in-memory pipes; no listener, dial or serving goroutine.
func TestConnectionLimitIncludesPendingHandlers(t *testing.T) {
	t.Parallel()
	for _, limit := range []int{2, 0, -1} {
		t.Run(fmt.Sprintf("limit_%d", limit), func(t *testing.T) {
			s := NewServer("", nil, metrics.NewConnRegistry(), time.Unix(0, 0))
			s.SetMaxConns(limit)
			var conns, peers [3]net.Conn
			for i := range conns {
				conns[i], peers[i] = net.Pipe()
				t.Cleanup(func() {
					conns[i].Close()
					peers[i].Close()
				})
			}
			// Reserve the burst before any handler gets a chance to run.
			wantCount := 0
			for i, conn := range conns {
				want := limit <= 0 || i < limit
				if got := s.trackConn(conn); got != want {
					t.Fatalf("admission %d = %t, want %t", i, got, want)
				}
				if want {
					wantCount++
				}
				if len(s.conns) != wantCount || s.connsPeak != wantCount {
					t.Fatalf("pending handlers: tracked=%d peak=%d, want %d", len(s.conns), s.connsPeak, wantCount)
				}
			}

			// A closed pipe drives the real handler's retirement immediately,
			// without a socket, read timer or background serving goroutine.
			conns[0].Close()
			peers[0].Close()
			s.wg.Add(1)
			s.handleConn(conns[0])
			if _, ok := s.conns[conns[0]]; ok || len(s.conns) != wantCount-1 || s.connsPeak != wantCount {
				t.Fatalf("handler retirement: tracked=%d peak=%d, want %d/%d", len(s.conns), s.connsPeak, wantCount-1, wantCount)
			}
			if limit > 0 {
				if !s.trackConn(conns[2]) || len(s.conns) != limit || s.connsPeak != limit {
					t.Fatalf("released slot was not reusable: tracked=%d peak=%d", len(s.conns), s.connsPeak)
				}
			}
		})
	}
}
