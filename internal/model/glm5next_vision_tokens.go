package model

import (
	"strings"
)

// EstimateGLM5NextImageTokenBudget calculates the exact number of tokens an image will consume
// based on width and height. Each 448x448 tile produces 256 tokens, plus 2 boundary tokens:
// <|begin_of_image|> + (tiles * 256) <|image_pad|> + <|end_of_image|>.
func EstimateGLM5NextImageTokenBudget(imgW, imgH int) int {
	if imgW <= 0 || imgH <= 0 {
		return 0
	}
	const tileSize = 448
	tilesW := (imgW + tileSize - 1) / tileSize
	tilesH := (imgH + tileSize - 1) / tileSize
	totalTiles := tilesW * tilesH
	if totalTiles == 0 {
		totalTiles = 1
	}
	return totalTiles*256 + 2
}

// SpliceGLM5NextImageTokens replaces occurrences of placeholder (e.g. "<image>" or "[image]")
// in the prompt with the exact GLM-5.3-Flash token sequence:
// <|begin_of_image|> + 256x <|image_pad|> + <|end_of_image|>.
func SpliceGLM5NextImageTokens(prompt, placeholder string, numTiles int) string {
	if numTiles <= 0 {
		numTiles = 1
	}
	var sb strings.Builder
	sb.WriteString(GLM5NextTokenBeginImage)
	padCount := numTiles * 256
	for i := 0; i < padCount; i++ {
		sb.WriteString(GLM5NextTokenImagePad)
	}
	sb.WriteString(GLM5NextTokenEndImage)
	replacement := sb.String()

	if placeholder == "" {
		placeholder = "<image>"
	}
	return strings.ReplaceAll(prompt, placeholder, replacement)
}
