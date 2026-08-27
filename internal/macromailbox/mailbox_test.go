package macromailbox

import "testing"

func TestKillRestartDeliversAuthenticatedMessageExactlyOnce(t *testing.T) {
	id := Identity{"macro:steward", "local://steward", []byte("secret")}
	m := New(id)
	msg := Message{ID: "m1", To: id.Address, Body: []byte("ship")}
	msg.Auth = Sign(id, msg)
	if r, e := m.Enqueue(msg); e != nil || !r.Applied {
		t.Fatalf("enqueue=%+v %v", r, e)
	}
	restarted := Restore(id, m.Snapshot())
	got, r, e := restarted.Deliver("m1")
	if e != nil || string(got.Body) != "ship" || !r.Applied {
		t.Fatalf("delivery=%+v %+v %v", got, r, e)
	}
	if r, e := restarted.Enqueue(msg); e != nil || r.Applied || r.Action != "deduplicate" {
		t.Fatalf("duplicate=%+v %v", r, e)
	}
	bad := msg
	bad.ID = "m2"
	bad.Auth = "bad"
	if _, e := restarted.Enqueue(bad); e == nil {
		t.Fatal("unauthenticated message accepted")
	}
	restarted.Retire()
	if _, e := restarted.Enqueue(msg); e == nil {
		t.Fatal("retired mailbox accepted message")
	}
}
