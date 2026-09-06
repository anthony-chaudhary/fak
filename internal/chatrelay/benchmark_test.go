package chatrelay

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"testing"
)

var (
	benchSinkString string
	benchSinkBool   bool
	benchSinkInt    int
	benchSinkMsgs   []Message
	benchSinkMsg    Message
)

type benchSlackClient struct {
	msgs []Message
}

func (b *benchSlackClient) History(ctx context.Context, channel, oldestTS string, limit int) ([]Message, error) {
	if limit > 0 && len(b.msgs) > limit {
		res := make([]Message, limit)
		copy(res, b.msgs[:limit])
		return res, nil
	}
	res := make([]Message, len(b.msgs))
	copy(res, b.msgs)
	return res, nil
}

func (b *benchSlackClient) Post(ctx context.Context, channel, threadTS, text string) (string, error) {
	return "1719600000.999999", nil
}

type benchModelClient struct {
	reply string
}

func (m *benchModelClient) Complete(ctx context.Context, prompt string) (string, error) {
	return m.reply, nil
}

func BenchmarkTick(b *testing.B) {
	ctx := context.Background()

	b.Run("NoNewMessages", func(b *testing.B) {
		client := &benchSlackClient{
			msgs: []Message{
				{Type: "message", TS: "1000.000100", User: "U1", Text: "already seen"},
			},
		}
		relay := &Relay{
			Slack:   client,
			Model:   &benchModelClient{reply: "ok"},
			Channel: "C1",
			lastTS:  "1000.000100",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			n, err := relay.Tick(ctx)
			if err != nil {
				b.Fatalf("Tick: %v", err)
			}
			benchSinkInt = n
		}
	})

	b.Run("SingleHuman", func(b *testing.B) {
		client := &benchSlackClient{
			msgs: []Message{
				{Type: "message", TS: "1001.000100", User: "U1", Text: "hello bot"},
			},
		}
		relay := &Relay{
			Slack:   client,
			Model:   &benchModelClient{reply: "hello there"},
			Channel: "C1",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			relay.lastTS = ""
			n, err := relay.Tick(ctx)
			if err != nil {
				b.Fatalf("Tick: %v", err)
			}
			benchSinkInt = n
		}
	})

	b.Run("MixedBatch", func(b *testing.B) {
		msgs := make([]Message, 10)
		for i := 0; i < 10; i++ {
			ts := fmt.Sprintf("100%d.000100", i)
			if i%3 == 0 {
				msgs[i] = Message{Type: "message", TS: ts, BotID: "B1", Text: "bot text"}
			} else if i%3 == 1 {
				msgs[i] = Message{Type: "message", Subtype: "channel_join", TS: ts, Text: "joined"}
			} else {
				msgs[i] = Message{Type: "message", TS: ts, User: "U1", Text: "user query"}
			}
		}
		client := &benchSlackClient{msgs: msgs}
		relay := &Relay{
			Slack:   client,
			Model:   &benchModelClient{reply: "answer"},
			Channel: "C1",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			relay.lastTS = ""
			n, err := relay.Tick(ctx)
			if err != nil {
				b.Fatalf("Tick: %v", err)
			}
			benchSinkInt = n
		}
	})

	b.Run("MentionGated", func(b *testing.B) {
		msgs := []Message{
			{Type: "message", TS: "1001.000100", User: "U1", Text: "random chat"},
			{Type: "message", TS: "1002.000100", User: "U2", Text: "<@U07BOT> help me"},
			{Type: "message", TS: "1003.000100", User: "U3", Text: "another chat"},
		}
		client := &benchSlackClient{msgs: msgs}
		relay := &Relay{
			Slack:   client,
			Model:   &benchModelClient{reply: "here to help"},
			Channel: "C1",
			Mention: "<@U07BOT>",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			relay.lastTS = ""
			n, err := relay.Tick(ctx)
			if err != nil {
				b.Fatalf("Tick: %v", err)
			}
			benchSinkInt = n
		}
	})

	b.Run("HTTPWire", func(b *testing.B) {
		hub := newFakeHub()
		hub.addHistory(map[string]any{
			"type": "message",
			"ts":   "1001.000100",
			"user": "U_HUMAN",
			"text": "hello GLM via wire",
		})
		srv := httptest.NewServer(hub.handler())
		defer srv.Close()

		relay := &Relay{
			Slack:   &HTTPSlack{Token: "xoxb-test", APIBase: srv.URL + "/", HTTP: srv.Client()},
			Model:   &HTTPModel{Endpoint: srv.URL, Model: "glm-5.2", HTTP: srv.Client()},
			Channel: "C_BENCH",
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			relay.lastTS = ""
			n, err := relay.Tick(ctx)
			if err != nil {
				b.Fatalf("Tick: %v", err)
			}
			benchSinkInt = n
		}
	})
}

func BenchmarkDefangMentions(b *testing.B) {
	benchmarks := []struct {
		name string
		in   string
	}{
		{"PlainText", "This is an ordinary message with no mentions or special tokens."},
		{"UserBare", "<@U07BOT>"},
		{"UserLabeled", "<@U07BOT|fakbot>"},
		{"SubteamBare", "<!subteam^S12345>"},
		{"SubteamLabeled", "<!subteam^S12345|core-devs>"},
		{"BroadcastHere", "<!here>"},
		{"BroadcastChannel", "<!channel>"},
		{"BroadcastEveryone", "<!everyone|everyone>"},
		{"Compound", "Hello <@U07BOT|fakbot>, please notify <!subteam^S12345|core-devs> and <!here> that <@U99999> is ready."},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			in := bm.in
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkString = defangMentions(in)
			}
		})
	}
}

func BenchmarkPrime(b *testing.B) {
	ctx := context.Background()

	sizes := []int{1, 10, 50, 100}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			msgs := make([]Message, size)
			for i := 0; i < size; i++ {
				msgs[i] = Message{
					Type: "message",
					TS:   fmt.Sprintf("1000.%06d", i),
					User: "U1",
					Text: "test message",
				}
			}
			client := &benchSlackClient{msgs: msgs}
			relay := &Relay{
				Slack:        client,
				Channel:      "C_PRIME",
				HistoryLimit: size,
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				relay.lastTS = ""
				if err := relay.Prime(ctx); err != nil {
					b.Fatalf("Prime: %v", err)
				}
			}
		})
	}
}

func BenchmarkTSAfter(b *testing.B) {
	benchmarks := []struct {
		name string
		a, b string
	}{
		{"NumericAscending", "1001.000100", "1000.000100"},
		{"NumericEqual", "1000.000100", "1000.000100"},
		{"NumericDescending", "1000.000100", "1001.000100"},
		{"MicroWidth", "1699999999.000200", "1699999999.000100"},
		{"EmptySecond", "1000.000100", ""},
		{"LexicalFallback", "abc", "def"},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			a, bStr := bm.a, bm.b
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkBool = TSAfter(a, bStr)
			}
		})
	}
}

func BenchmarkPollHistory(b *testing.B) {
	ctx := context.Background()
	sizes := []int{10, 50, 100}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			msgs := make([]Message, size)
			for i := 0; i < size; i++ {
				msgs[i] = Message{
					Type: "message",
					TS:   fmt.Sprintf("1000.%06d", size-i),
					User: "U1",
					Text: "chat history",
				}
			}
			client := &benchSlackClient{msgs: msgs}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out, err := PollHistory(ctx, client, "C_POLL", "", size)
				if err != nil {
					b.Fatalf("PollHistory: %v", err)
				}
				benchSinkMsgs = out
			}
		})
	}
}

func BenchmarkPromptFor(b *testing.B) {
	relayDirect := &Relay{}
	relayMention := &Relay{Mention: "<@U07BOT>"}
	relaySelf := &Relay{BotUserID: "U_SELF"}

	b.Run("HumanNoMention", func(b *testing.B) {
		m := Message{Type: "message", TS: "1001.000100", User: "U1", Text: "how does fak work?"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relayDirect.promptFor(m)
		}
	})

	b.Run("BotIDFiltered", func(b *testing.B) {
		m := Message{Type: "message", TS: "1001.000100", BotID: "B123", Text: "automated reply"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relayDirect.promptFor(m)
		}
	})

	b.Run("SubtypeFiltered", func(b *testing.B) {
		m := Message{Type: "message", Subtype: "message_changed", TS: "1001.000100", Text: "edited text"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relayDirect.promptFor(m)
		}
	})

	b.Run("SelfUserFiltered", func(b *testing.B) {
		m := Message{Type: "message", TS: "1001.000100", User: "U_SELF", Text: "my own message"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relaySelf.promptFor(m)
		}
	})

	b.Run("MentionMatched", func(b *testing.B) {
		m := Message{Type: "message", TS: "1001.000100", User: "U1", Text: "<@U07BOT> explain benchmarks"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relayMention.promptFor(m)
		}
	})

	b.Run("MentionUnmatched", func(b *testing.B) {
		m := Message{Type: "message", TS: "1001.000100", User: "U1", Text: "talking to someone else"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString, benchSinkBool = relayMention.promptFor(m)
		}
	})
}

func BenchmarkMessageUnmarshalJSON(b *testing.B) {
	raw := []byte(`{"type":"message","ts":"1719600000.000100","thread_ts":"1719600000.000050","user":"U12345","text":"hello bot from slack"}`)
	var m Message
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := json.Unmarshal(raw, &m); err != nil {
			b.Fatalf("Unmarshal: %v", err)
		}
		benchSinkMsg = m
	}
}
