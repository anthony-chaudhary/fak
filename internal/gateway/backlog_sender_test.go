package gateway

import "testing"

func TestSendBacklog(t *testing.T) {
	t.Parallel()
	out := make(chan int, 2)
	s := NewBacklogSender(out)
	for i := 1; i <= 5; i++ {
		s.Send(i) // must never block even though the transport holds only 2
	}
	if got := s.Pending(); got != 3 {
		t.Fatalf("pending = %d, want 3", got)
	}
	var got []int
	for len(got) < 5 {
		for len(out) > 0 {
			got = append(got, <-out)
		}
		if s.Pending() == 0 && len(out) == 0 {
			break
		}
		s.Drain()
	}
	for len(out) > 0 {
		got = append(got, <-out)
	}
	for i, v := range got {
		if v != i+1 {
			t.Fatalf("order broken: %v", got)
		}
	}
	if len(got) != 5 {
		t.Fatalf("delivered %v, want 5 messages", got)
	}
	// A new send behind a non-empty backlog must not jump the queue.
	out2 := make(chan int, 1)
	s2 := NewBacklogSender(out2)
	s2.Send(1)
	s2.Send(2) // backlogged
	<-out2
	s2.Send(3) // must queue behind 2
	s2.Drain()
	if a := <-out2; a != 2 {
		t.Fatalf("first after drain = %d, want 2", a)
	}
}
