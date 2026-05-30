from main_service.logging_config import format_log_event


def test_format_log_event_keeps_event_name_and_adds_marker() -> None:
    line = format_log_event("processing_view_complete", {"view_id": 123})

    assert line.startswith("✅ processing_view_complete ")
    assert '"view_id": 123' in line
