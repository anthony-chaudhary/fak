"""Run a tool as __main__ once per argument with every datetime.now()/utcnow() pinned to one instant.

usage: frozen_now.py <iso8601-instant> <script> <arg> [<arg>...]

Each <arg> is one run of <script> with sys.argv = [script, arg], all in this one interpreter.
stdout carries each run's output followed by a NUL, then the codec name sys.stdout encodes
with, so a caller can decode the runs exactly as they were written.

The Go port takes its clock as an argument; pinning the oracle's clock the same way lets
both classify one fixture at exactly the same instant in every mode. The Go fixture pass
takes no live-driver census either, so the oracle's process-table census reports none
(that walk also costs seconds per run).
"""
import codecs
import datetime as _dt
import os
import runpy
import sys
import types

_FROZEN = _dt.datetime.fromisoformat(sys.argv[1]).astimezone(_dt.timezone.utc)


class _FrozenDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return _FROZEN.astimezone().replace(tzinfo=None)
        return _FROZEN.astimezone(tz)

    @classmethod
    def utcnow(cls):
        return _FROZEN.replace(tzinfo=None)


_dt.datetime = _FrozenDateTime

_no_census = types.ModuleType("resume_sweep")
_no_census.live_resume_sids = lambda: []
sys.modules["resume_sweep"] = _no_census

script = os.path.abspath(sys.argv[2])
runs = sys.argv[3:]
sys.path.insert(0, os.path.dirname(script))
for arg in runs:
    sys.argv = [script, arg]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    sys.stdout.write("\0")
    sys.stdout.flush()
sys.stdout.write(codecs.lookup(sys.stdout.encoding).name)
sys.stdout.flush()
