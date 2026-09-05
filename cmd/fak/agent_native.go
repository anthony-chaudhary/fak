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

const rawAgentReceiptSchema = "agent.raw-receipt.v1"

type rawAgentReceipt struct {
	Schema        string           `json:"schema"`
	Mode          string           `json:"mode"`
	FakMediated   bool             `json:"fak_mediated"`
	Adjudications int              `json:"adjudications"`
	Task          string           `json:"task"`
	Model         string           `json:"model"`
	Metrics       agent.ArmMetrics `json:"metrics"`
}

func newRawAgentReceipt(task, model string, metrics agent.ArmMetrics) rawAgentReceipt {
	return rawAgentReceipt{
		Schema:        rawAgentReceiptSchema,
		Mode:          "raw",
		FakMediated:   false,
		Adjudications: 0,
		Task:          task,
		Model:         model,
		Metrics:       metrics,
	}
}
