from pathlib import Path


def test_work_discovery_tasks_are_migrated_to_s4u():
    text = Path(__file__).with_name("migrate_fleet_tasks_to_s4u.ps1").read_text(encoding="utf-8-sig")
    for task in ("FleetScoutLoop", "FleetStaleWorkGarden"):
        assert f"'{task}'" in text
    assert "-LogonType S4U" in text
