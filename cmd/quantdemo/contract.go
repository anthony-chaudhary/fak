package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	ContractID = "fak-quantdemo/1"

	ResultCompatible           = "COMPATIBLE"
	ReasonKnownCombination     = "KNOWN_COMBINATION"
	ResultAbstain              = "ABSTAIN"
	ReasonUnknownFormat        = "UNKNOWN_FORMAT"
	ResultRuntimeHandoff       = "DELEGATE"
	ReasonRuntimeNotPinned     = "RUNTIME_NOT_PINNED"
	ResultRefuse               = "REFUSE"
	ReasonCombinationNotListed = "COMBINATION_NOT_LISTED"

	FormatGGUFV3 = "gguf@3"
	QuantQ4KM    = "q4_k_m"
	RuntimePin   = "llama.cpp@b6500+ga7a98e0fffed"

	ModelName     = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
	ModelSHA256   = "2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d"
	ModelBytes    = int64(105454432)
	ModelLicense  = "Apache-2.0"
	ModelRevision = "bartowski/SmolLM2-135M-Instruct-GGUF@09816acd5d99df7be770d85ea30822623dab342c"
	ModelURL      = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q4_K_M.gguf"

	RuntimeLicense       = "MIT"
	RuntimeLicenseSHA256 = "e562a2ddfaf8280537795ac5ecd34e3012b6582a147ef69ba6a6a5c08c84757d"
)

type decision struct {
	Contract string `json:"contract"`
	Result   string `json:"result"`
	Reason   string `json:"reason"`
	Format   string `json:"format"`
	Quant    string `json:"quantization"`
	Runtime  string `json:"runtime"`
}

func adjudicate(format, quant, runtime string) decision {
	d := decision{Contract: ContractID, Format: format, Quant: quant, Runtime: runtime}
	if format != FormatGGUFV3 {
		d.Result, d.Reason = ResultAbstain, ReasonUnknownFormat
		return d
	}
	if runtime != RuntimePin {
		d.Result, d.Reason = ResultRuntimeHandoff, ReasonRuntimeNotPinned
		return d
	}
	if quant != QuantQ4KM {
		d.Result, d.Reason = ResultRefuse, ReasonCombinationNotListed
		return d
	}
	d.Result, d.Reason = ResultCompatible, ReasonKnownCombination
	return d
}

func inspectGGUF(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var header [8]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return "", fmt.Errorf("read GGUF header: %w", err)
	}
	if string(header[:4]) != "GGUF" {
		return "", fmt.Errorf("%s: magic %q", ReasonUnknownFormat, string(header[:4]))
	}
	version := binary.LittleEndian.Uint32(header[4:])
	if version != 3 {
		return "", fmt.Errorf("%s: gguf version %d", ReasonUnknownFormat, version)
	}
	return FormatGGUFV3, nil
}

func validPin(s string) bool { return strings.TrimSpace(s) == RuntimePin }
