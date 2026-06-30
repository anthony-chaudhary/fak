#!/usr/bin/env python3
"""Tests for sweep_config — the YAML/JSON sweep-profile loader/saver.

These exercise the PURE (de)serialization surface (load_profile / save_profile /
get_profile_path / list_profiles) against a tmp dir, so the round-trip invariant
— save_profile then load_profile recovers every field, including nested models
and price hints — is proven without touching a real sweep. The defaults a
half-specified YAML falls back to are pinned too, since the loader's `.get(...,
default)` calls are the contract a profile author relies on."""
import json
import tempfile
import unittest
from pathlib import Path

import sweep_config as sc


def sample_profile():
    return sc.SweepProfile(
        name="nightly",
        description="the nightly sweep",
        models=[
            sc.ModelConfig(
                name="zai/glm-4.7-flash",
                provider="zai",
                base_url="https://api.example/v1",
                api_key_env="ZAI_KEY",
                price_hint=sc.PriceHint(input=0.5, output=1.5, source="docs"),
                enabled=True,
            ),
            sc.ModelConfig(name="local/qwen", provider="local",
                           local_shim="shims/qwen.sh", enabled=False),
        ],
        workload=sc.WorkloadConfig(max_turns=20, trials=3, timeout_s=900,
                                   transcript_path="t/x.jsonl"),
        output_dir="out/here",
        skip_api=True,
        tags=["a", "b"],
        public=False,
    )


class RoundTrip(unittest.TestCase):
    def test_save_then_load_recovers_every_field(self):
        p = sample_profile()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nightly.yaml"
            sc.save_profile(p, path)
            got = sc.load_profile(path)

        self.assertEqual(got.name, p.name)
        self.assertEqual(got.description, p.description)
        self.assertEqual(got.output_dir, p.output_dir)
        self.assertEqual(got.skip_api, p.skip_api)
        self.assertEqual(got.tags, p.tags)
        self.assertEqual(got.public, p.public)
        # workload
        self.assertEqual(got.workload.max_turns, 20)
        self.assertEqual(got.workload.trials, 3)
        self.assertEqual(got.workload.timeout_s, 900)
        self.assertEqual(got.workload.transcript_path, "t/x.jsonl")
        # models, in order
        self.assertEqual([m.name for m in got.models],
                         ["zai/glm-4.7-flash", "local/qwen"])
        m0, m1 = got.models
        self.assertEqual(m0.provider, "zai")
        self.assertEqual(m0.base_url, "https://api.example/v1")
        self.assertEqual(m0.api_key_env, "ZAI_KEY")
        self.assertTrue(m0.enabled)
        self.assertIsNotNone(m0.price_hint)
        self.assertEqual(m0.price_hint.input, 0.5)
        self.assertEqual(m0.price_hint.output, 1.5)
        self.assertEqual(m0.price_hint.source, "docs")
        self.assertFalse(m1.enabled)
        self.assertEqual(m1.local_shim, "shims/qwen.sh")
        # the second model carried no price hint, so it must come back as None
        self.assertIsNone(m1.price_hint)


class Defaults(unittest.TestCase):
    def test_minimal_json_profile_uses_documented_defaults(self):
        # A profile with only a name must load with every documented default.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bare.json"
            path.write_text(json.dumps({"name": "bare"}), encoding="utf-8")
            got = sc.load_profile(path)
        self.assertEqual(got.name, "bare")
        self.assertEqual(got.description, "")
        self.assertEqual(got.models, [])
        self.assertEqual(got.output_dir, "fak/experiments/agent-live/sweep")
        self.assertFalse(got.skip_api)
        self.assertTrue(got.public)
        # workload defaults
        self.assertEqual(got.workload.max_turns, 12)
        self.assertEqual(got.workload.trials, 1)
        self.assertEqual(got.workload.timeout_s, 600)
        self.assertIsNone(got.workload.transcript_path)

    def test_model_provider_defaults_to_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            path.write_text(json.dumps({"name": "p", "models": [{"name": "x"}]}),
                            encoding="utf-8")
            got = sc.load_profile(path)
        self.assertEqual(len(got.models), 1)
        self.assertEqual(got.models[0].provider, "unknown")
        self.assertTrue(got.models[0].enabled)
        self.assertIsNone(got.models[0].price_hint)


class ProfilePaths(unittest.TestCase):
    def test_get_profile_path_prefers_existing_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "p.yml").write_text("name: p\n", encoding="utf-8")
            # only .yml exists -> returns it
            self.assertEqual(sc.get_profile_path("p", dd), dd / "p.yml")
            # now a .yaml exists too -> .yaml wins (checked first)
            (dd / "p.yaml").write_text("name: p\n", encoding="utf-8")
            self.assertEqual(sc.get_profile_path("p", dd), dd / "p.yaml")

    def test_get_profile_path_defaults_to_yaml_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            self.assertEqual(sc.get_profile_path("nope", dd), dd / "nope.yaml")

    def test_list_profiles_loads_all_in_dir(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            sc.save_profile(sc.SweepProfile(name="one"), dd / "one.yaml")
            sc.save_profile(sc.SweepProfile(name="two"), dd / "two.yaml")
            names = sorted(p.name for p in sc.list_profiles(dd))
            self.assertEqual(names, ["one", "two"])


if __name__ == "__main__":
    unittest.main()
