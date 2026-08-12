package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

type Config struct {
	Width        int     `json:"width"`
	Height       int     `json:"height"`
	FPS          int     `json:"fps"`
	MP4          string  `json:"mp4"`
	GIF          string  `json:"gif"`
	Poster       string  `json:"poster"`
	ContactSheet string  `json:"contactSheet"`
	Audit        string  `json:"audit"`
	AppendMP4    string  `json:"appendMP4,omitempty"`
	CompositeMP4 string  `json:"compositeMP4,omitempty"`
	CompositeGIF string  `json:"compositeGIF,omitempty"`
	Scenes       []Scene `json:"scenes"`
}

type Scene struct {
	Kind     string   `json:"kind"`
	Secs     float64  `json:"secs"`
	Eyebrow  string   `json:"eyebrow,omitempty"`
	Title    string   `json:"title,omitempty"`
	Subtitle string   `json:"subtitle,omitempty"`
	Action   string   `json:"action,omitempty"`
	Detail   string   `json:"detail,omitempty"`
	Verdict  string   `json:"verdict,omitempty"`
	Command  string   `json:"command,omitempty"`
	Items    []string `json:"items,omitempty"`
}

func main() {
	cfgPath := flag.String("config", "", "trailer JSON")
	verify := flag.Bool("verify", false, "validate story and readability without rendering")
	all := flag.Bool("all", false, "render frames, MP4, GIF, poster, contact sheet, and audit")
	ffmpeg := flag.String("ffmpeg", os.Getenv("VIDEOGEN_FFMPEG"), "ffmpeg executable")
	flag.Parse()
	if *cfgPath == "" {
		fmt.Fprintln(os.Stderr, "trailer: -config is required")
		os.Exit(2)
	}
	raw, err := os.ReadFile(*cfgPath)
	if err != nil {
		fail(err)
	}
	var cfg Config
	if err = json.Unmarshal(raw, &cfg); err != nil {
		fail(err)
	}
	dir := filepath.Dir(*cfgPath)
	cfg.anchor(dir)
	report, err := audit(cfg)
	if err != nil {
		fail(err)
	}
	if *verify {
		writeAudit(cfg.Audit, report)
		fmt.Printf("trailer: verify OK — %.1fs, %d scenes, min type %dpx at %dpx wide\n", report.Duration, len(cfg.Scenes), report.MinType, cfg.Width)
		return
	}
	if !*all {
		fail(fmt.Errorf("choose -verify or -all"))
	}
	if *ffmpeg == "" {
		fail(fmt.Errorf("-ffmpeg is required for -all"))
	}
	if err := renderAll(cfg, *ffmpeg, report); err != nil {
		fail(err)
	}
}

func (c *Config) anchor(dir string) {
	for _, p := range []*string{&c.MP4, &c.GIF, &c.Poster, &c.ContactSheet, &c.Audit, &c.AppendMP4, &c.CompositeMP4, &c.CompositeGIF} {
		if *p != "" && !filepath.IsAbs(*p) {
			*p = filepath.Join(dir, *p)
		}
	}
}
func fail(err error) { fmt.Fprintln(os.Stderr, "trailer:", err); os.Exit(1) }
