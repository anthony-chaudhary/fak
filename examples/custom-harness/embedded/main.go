package main

import (
	"context"
	"fmt"
)

// Ticket is ordinary application data. It has no harness-specific fields.
type Ticket struct {
	Customer string
	Topic    string
}

// Reply is the normal result returned by the host workflow.
type Reply struct {
	Subject string
	Body    string
}

// replyAssistant is the only agent-shaped dependency in the host application.
type replyAssistant interface {
	Suggest(context.Context, Ticket) (string, error)
}

func prepareReply(ctx context.Context, assistant replyAssistant, ticket Ticket) (Reply, error) {
	suggestion, err := assistant.Suggest(ctx, ticket)
	if err != nil {
		return Reply{}, fmt.Errorf("prepare reply: %w", err)
	}
	return Reply{
		Subject: "Re: " + ticket.Topic,
		Body:    "Hello " + ticket.Customer + ",\n\n" + suggestion + "\n\n— Support",
	}, nil
}

func main() {
	agent := harnessAgent{runner: offlineRunner{}}
	reply, err := prepareReply(context.Background(), agent, Ticket{
		Customer: "Sam",
		Topic:    "changing a delivery address",
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s\n\n%s\n", reply.Subject, reply.Body)
}
