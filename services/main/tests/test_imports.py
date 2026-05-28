def test_pipeline_manager_imports() -> None:
    import main_service.pipeline_manager  # noqa: F401


def test_pano_retrieval_imports() -> None:
    import main_service.pano_retrieval  # noqa: F401


def test_main_does_not_touch_database(monkeypatch) -> None:
    import main_service.__main__ as main_module

    def fail_if_called() -> None:
        raise AssertionError("main() should not initialize the database")

    monkeypatch.setattr(main_module, "initialize_engine", fail_if_called, raising=False)

    main_module.main()
