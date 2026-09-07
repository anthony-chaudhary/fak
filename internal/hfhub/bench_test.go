package hfhub

import (
	"testing"
)

func BenchmarkParseURI(b *testing.B) {
	uris := []string{
		"hf://mradermacher/Qwen2.5-1.5B-GGUF/model.Q8_0.gguf",
		"hf://meta-llama/Llama-3.1-8B@main/orig/consolidated.00.pth",
		"hf://Qwen/Qwen2.5-1.5B-Instruct@v1.0/config.json",
		"hf://owner/repo/a/b/c.gguf",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		uri := uris[i%len(uris)]
		ref, err := ParseURI(uri)
		if err != nil || ref.Repo == "" {
			b.Fatal(err)
		}
	}
}

func BenchmarkResolveURL(b *testing.B) {
	ref := Ref{
		Repo:     "Qwen/Qwen2.5-1.5B-Instruct",
		Revision: "main",
		File:     "config.json",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		u := ref.ResolveURL(DefaultBaseURL)
		if u == "" {
			b.Fatal("empty resolved url")
		}
	}
}

func BenchmarkIsURI(b *testing.B) {
	candidates := []string{
		"hf://mradermacher/Qwen2.5-1.5B-GGUF/model.Q8_0.gguf",
		"https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
		"file:///local/path/model.gguf",
		"hf://owner/repo",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = IsURI(candidates[i%len(candidates)])
	}
}
