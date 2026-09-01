package main

import (
	"context"
	"testing"
)

func TestPrepareReplyUsesBehindTheScenesAgent(t *testing.T) {
	reply, err := prepareReply(context.Background(), harnessAgent{runner: offlineRunner{}}, Ticket{
		Customer: "Sam",
		Topic:    "changing a delivery address",
	})
	if err != nil {
		t.Fatal(err)
	}

	const wantSubject = "Re: changing a delivery address"
	const wantBody = "Hello Sam,\n\nYou can update the address before the package ships.\n\n— Support"
	if reply.Subject != wantSubject || reply.Body != wantBody {
		t.Fatalf("unexpected reply:\nsubject: %q\nbody: %q", reply.Subject, reply.Body)
	}
}
