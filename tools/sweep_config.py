#!/usr/bin/env python3
"""
sweep_config.py - Sweep configuration loader/saver for benchmark sweeps.

Extracts sweep configuration from monolithic scripts into reusable YAML/JSON profiles.
Enables multi-model sweeps without script edits and supports public/private separation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
import json


@dataclass
class PriceHint:
    """Price hint for API models (per million tokens)."""
    input: float
    output: float
    source: str = "manual"


@dataclass
class ModelConfig:
    """Model configuration for sweep."""
    name: str  # Model identifier (e.g., "zai/glm-4.7-flash")
    provider: str = "unknown"
    base_url: Optional[str] = None  # API base URL (for API models)
    api_key_env: Optional[str] = None  # Environment variable for API key
    local_shim: Optional[str] = None  # Path to local shim script (for local models)
    price_hint: Optional[PriceHint] = None
    enabled: bool = True


@dataclass
class WorkloadConfig:
    """Workload parameters for sweep."""
    max_turns: int = 12
    trials: int = 1
    timeout_s: int = 600
    # Optional: specific transcript/workload file
    transcript_path: Optional[str] = None


@dataclass
class SweepProfile:
    """Sweep profile configuration."""
    name: str
    description: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    output_dir: str = "fak/experiments/agent-live/sweep"
    # Execution flags
    skip_api: bool = False
    skip_offline: bool = False
    skip_local_shim: bool = False
    fail_fast: bool = False
    # Metadata
    tags: List[str] = field(default_factory=list)
    public: bool = True  # False for private/internal-only profiles


@dataclass
class SweepResult:
    """Result from a single sweep run."""
    profile_name: str
    model_name: str
    trial: int
    success: bool
    duration_ms: float
    tokens_total: int
    tokens_per_sec: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_profile(path: Path) -> SweepProfile:
    """Load a sweep profile from YAML or JSON file."""
    with open(path, 'r') as f:
        if path.suffix in ['.yml', '.yaml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Parse models
    models = []
    for m in data.get('models', []):
        price_hint = None
        if 'price_hint' in m:
            ph = m['price_hint']
            price_hint = PriceHint(
                input=ph.get('input', 0.0),
                output=ph.get('output', 0.0),
                source=ph.get('source', 'manual')
            )

        models.append(ModelConfig(
            name=m['name'],
            provider=m.get('provider', 'unknown'),
            base_url=m.get('base_url'),
            api_key_env=m.get('api_key_env'),
            local_shim=m.get('local_shim'),
            price_hint=price_hint,
            enabled=m.get('enabled', True)
        ))

    # Parse workload
    wl_data = data.get('workload', {})
    workload = WorkloadConfig(
        max_turns=wl_data.get('max_turns', 12),
        trials=wl_data.get('trials', 1),
        timeout_s=wl_data.get('timeout_s', 600),
        transcript_path=wl_data.get('transcript_path')
    )

    return SweepProfile(
        name=data['name'],
        description=data.get('description', ''),
        models=models,
        workload=workload,
        output_dir=data.get('output_dir', 'fak/experiments/agent-live/sweep'),
        skip_api=data.get('skip_api', False),
        skip_offline=data.get('skip_offline', False),
        skip_local_shim=data.get('skip_local_shim', False),
        fail_fast=data.get('fail_fast', False),
        tags=data.get('tags', []),
        public=data.get('public', True)
    )


def save_profile(profile: SweepProfile, path: Path) -> None:
    """Save a sweep profile to YAML file."""
    data = {
        'name': profile.name,
        'description': profile.description,
        'models': [],
        'workload': {
            'max_turns': profile.workload.max_turns,
            'trials': profile.workload.trials,
            'timeout_s': profile.workload.timeout_s,
        },
        'output_dir': profile.output_dir,
        'skip_api': profile.skip_api,
        'skip_offline': profile.skip_offline,
        'skip_local_shim': profile.skip_local_shim,
        'fail_fast': profile.fail_fast,
        'tags': profile.tags,
        'public': profile.public
    }

    if profile.workload.transcript_path:
        data['workload']['transcript_path'] = profile.workload.transcript_path

    for m in profile.models:
        model_data = {
            'name': m.name,
            'provider': m.provider,
            'enabled': m.enabled
        }
        if m.base_url:
            model_data['base_url'] = m.base_url
        if m.api_key_env:
            model_data['api_key_env'] = m.api_key_env
        if m.local_shim:
            model_data['local_shim'] = m.local_shim
        if m.price_hint:
            model_data['price_hint'] = {
                'input': m.price_hint.input,
                'output': m.price_hint.output,
                'source': m.price_hint.source
            }
        data['models'].append(model_data)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def list_profiles(directory: Path = None) -> List[SweepProfile]:
    """List all sweep profiles in a directory."""
    if directory is None:
        directory = Path(__file__).parent / 'sweep_profiles'
    elif isinstance(directory, str):
        directory = Path(directory)
    profiles = []
    for path in directory.glob('*.yaml'):
        try:
            profiles.append(load_profile(path))
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    for path in directory.glob('*.yml'):
        try:
            profiles.append(load_profile(path))
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    return profiles


def get_profile_path(name: str, directory: Path = None) -> Path:
    """Get the path to a profile by name."""
    if directory is None:
        directory = Path(__file__).parent / 'sweep_profiles'

    # Try with .yaml, then .yml
    for ext in ['.yaml', '.yml']:
        path = directory / f"{name}{ext}"
        if path.exists():
            return path
    return directory / f"{name}.yaml"


# CLI convenience
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        profile = load_profile(path)
        print(f"Profile: {profile.name}")
        print(f"Models: {len(profile.models)}")
        for m in profile.models:
            print(f"  - {m.name} (enabled={m.enabled})")
