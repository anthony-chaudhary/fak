package main

import "github.com/anthony-chaudhary/fak/internal/agent"

const nativeAgentReceiptSchema = "fak.agent.native.v1"

type nativeAgentReceipt struct {
	Schema  string           `json:"schema"`
	Task    string           `json:"task"`
	Model   string           `json:"model"`
	Metrics agent.ArmMetrics `json:"metrics"`
}

func newNativeAgentReceipt(task, model string, metrics agent.ArmMetrics) nativeAgentReceipt {
	return nativeAgentReceipt{
		Schema:  nativeAgentReceiptSchema,
		Task:    task,
		Model:   model,
		Metrics: metrics,
	}
}
