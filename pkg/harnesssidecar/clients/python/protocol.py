# Generated from schema/protocol.schema.json. Do not edit.
from dataclasses import dataclass
import json, struct
PROTOCOL = "fak.harness-sidecar/v1"
@dataclass(frozen=True)
class Identity: name: str; version: str; digest: str
@dataclass(frozen=True)
class Limits: max_frame: int; max_inflight: int; cancel_grace: int
def encode_frame(frame: dict, max_frame: int) -> bytes:
    body = json.dumps(frame, separators=(",", ":")).encode()
    if not body or len(body) > max_frame: raise ValueError("frame exceeds negotiated bound")
    return struct.pack(">I", len(body)) + body
