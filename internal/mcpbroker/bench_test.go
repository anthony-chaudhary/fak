package mcpbroker

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

func BenchmarkBroker_RouteCall_Echo(b *testing.B) {
	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterServer(ServerConfig{
		ID:   "srv-bench",
		Name: "Benchmark Server",
	})
	_ = broker.RegisterTool(ToolRegistration{
		Name:     "echo_tool",
		ServerID: "srv-bench",
	})

	req := CallRequest{
		Tool:      "echo_tool",
		Arguments: json.RawMessage(`{"query":"test"}`),
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		resp, err := broker.RouteCall(ctx, req)
		if err != nil || resp.IsError || resp.Filtered {
			b.Fatalf("RouteCall failed: %v, resp=%+v", err, resp)
		}
	}
}

func BenchmarkBroker_RouteCall_CustomHandler(b *testing.B) {
	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterServer(ServerConfig{
		ID:   "srv-handler",
		Name: "Handler Server",
	})
	_ = broker.RegisterTool(ToolRegistration{
		Name:     "custom_tool",
		ServerID: "srv-handler",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return &CallResponse{
				Tool:    req.Tool,
				Content: json.RawMessage(`{"status":"success"}`),
			}, nil
		},
	})

	req := CallRequest{
		Tool:      "custom_tool",
		SessionID: "sess-123",
		Arguments: json.RawMessage(`{"op":"run"}`),
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		resp, err := broker.RouteCall(ctx, req)
		if err != nil || resp.IsError {
			b.Fatalf("RouteCall failed: %v", err)
		}
	}
}

func BenchmarkBroker_RouteCall_SecurityFilter(b *testing.B) {
	broker := NewBroker(WithGlobalSecurityFilter(func(ctx context.Context, req CallRequest, reg ToolRegistration) (bool, string) {
		if req.Tool == "blocked_tool" {
			return false, "policy blocked"
		}
		return true, ""
	}))
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "allowed_tool",
	})

	req := CallRequest{
		Tool:      "allowed_tool",
		Arguments: json.RawMessage(`{}`),
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		resp, err := broker.RouteCall(ctx, req)
		if err != nil || resp.Filtered {
			b.Fatalf("RouteCall failed: %v, resp=%+v", err, resp)
		}
	}
}

func BenchmarkBroker_ListTools(b *testing.B) {
	broker := NewBroker()
	defer broker.Close()

	for i := 0; i < 32; i++ {
		_ = broker.RegisterTool(ToolRegistration{
			Name:        fmt.Sprintf("tool_%02d", i),
			Description: "Benchmark tool registration",
		})
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		tools := broker.ListTools()
		if len(tools) != 32 {
			b.Fatalf("unexpected tools count: %d", len(tools))
		}
	}
}
