"""Tests for the version and API stability module."""

import warnings

import pytest

from sonata.version import (
    SONATA_VERSION,
    VERSION_INFO,
    deprecated,
    module_api,
    public_api,
    schema_versions,
    version_string,
)


class TestVersionInfo:
    def test_version_string(self):
        assert SONATA_VERSION == "0.10.0"

    def test_version_info_tuple(self):
        assert VERSION_INFO == (0, 10, 0)
        assert len(VERSION_INFO) == 3

    def test_version_string_function(self):
        assert version_string() == "0.10.0"

    def test_version_string_with_label(self):
        assert version_string(include_label=True) == "Sonata 0.10.0"

    def test_version_consistency(self):
        assert ".".join(str(v) for v in VERSION_INFO) == SONATA_VERSION


class TestDeprecated:
    def test_emits_warning(self):
        @deprecated("old function", since="0.5", replacement="new_func")
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 42
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_func is deprecated" in str(w[0].message)

    def test_preserves_return_value(self):
        @deprecated("use something else")
        def compute(x, y):
            return x + y

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert compute(3, 4) == 7

    def test_warning_message_components(self):
        @deprecated("reason text", since="0.7", replacement="better_func")
        def my_func():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            my_func()
            msg = str(w[0].message)
            assert "my_func is deprecated" in msg
            assert "since v0.7" in msg
            assert "reason text" in msg
            assert "better_func" in msg

    def test_deprecated_attribute(self):
        @deprecated("old")
        def func():
            pass

        assert func.__deprecated__ is True
        assert hasattr(func, "__deprecated_message__")

    def test_preserves_function_name(self):
        @deprecated("old")
        def original_name():
            """Original docstring."""
            pass

        assert original_name.__name__ == "original_name"
        assert original_name.__doc__ == "Original docstring."


class TestSchemaVersions:
    def test_returns_dict(self):
        versions = schema_versions()
        assert isinstance(versions, dict)

    def test_contains_all_schemas(self):
        versions = schema_versions()
        expected_keys = {
            "score_schema",
            "fingerprint_version",
            "eligibility_result_schema",
            "plan_handle_schema",
            "runtime_contract",
            "cache_schema",
        }
        assert set(versions.keys()) == expected_keys

    def test_all_positive_integers(self):
        for key, value in schema_versions().items():
            assert isinstance(value, int), f"{key} is not an int"
            assert value >= 1, f"{key} is not positive"


class TestPublicApi:
    def test_returns_sorted_list(self):
        api = public_api()
        assert isinstance(api, list)
        assert api == sorted(api)

    def test_contains_core_symbols(self):
        api = public_api()
        for symbol in ["Score", "Task", "Dependency", "ShapeAssumption", "RuntimeTarget"]:
            assert symbol in api, f"missing core symbol: {symbol}"

    def test_contains_serialization_symbols(self):
        api = public_api()
        for symbol in ["score_to_dict", "score_from_dict", "score_fingerprint"]:
            assert symbol in api, f"missing serialization symbol: {symbol}"

    def test_contains_cache_symbols(self):
        api = public_api()
        for symbol in ["ScoreCache", "CacheEntry", "cached_score"]:
            assert symbol in api, f"missing cache symbol: {symbol}"

    def test_contains_adapter_symbols(self):
        api = public_api()
        for symbol in ["AdapterDescriptor", "AdapterRegistry", "default_registry"]:
            assert symbol in api, f"missing adapter symbol: {symbol}"

    def test_contains_version_symbols(self):
        api = public_api()
        for symbol in ["SONATA_VERSION", "VERSION_INFO", "schema_versions"]:
            assert symbol in api, f"missing version symbol: {symbol}"

    def test_no_private_symbols(self):
        api = public_api()
        for name in api:
            assert not name.startswith("_"), f"private symbol in public API: {name}"


class TestModuleApi:
    def test_returns_dict(self):
        api = module_api()
        assert isinstance(api, dict)

    def test_grouped_by_module(self):
        api = module_api()
        assert "score" in api
        assert "Score" in api["score"]

    def test_all_public_symbols_accounted(self):
        flat = set()
        for symbols in module_api().values():
            flat.update(symbols)
        assert flat == set(public_api())


class TestDeprecatedField:
    def test_descriptor_emits_warning(self):
        from sonata.version import DeprecatedField

        class Cfg:
            old = DeprecatedField(default=99, message="gone", since="0.8", replacement="new")

        obj = Cfg()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = obj.old
            assert val == 99
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Cfg.old is deprecated" in str(w[0].message)
            assert "since v0.8" in str(w[0].message)
            assert "gone" in str(w[0].message)
            assert "new" in str(w[0].message)

    def test_descriptor_set_and_get(self):
        from sonata.version import DeprecatedField

        class Cfg:
            old = DeprecatedField(default=0)

        obj = Cfg()
        obj.old = 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert obj.old == 42
            assert len(w) == 1

    def test_class_level_access_returns_descriptor(self):
        from sonata.version import DeprecatedField

        class Cfg:
            old = DeprecatedField(default=7)

        assert isinstance(Cfg.__dict__["old"], DeprecatedField)

    def test_score_runtime_target_deprecation(self):
        from sonata.score import Score, RuntimeTarget

        score = Score(name="t", runtime_target=RuntimeTarget(), tasks=())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rt = score.runtime_target
            assert isinstance(rt, RuntimeTarget)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Score.runtime_target is deprecated" in str(w[0].message)

    def test_score_runtime_target_no_duplicate_warning(self):
        from sonata.score import Score, RuntimeTarget

        score = Score(name="t", runtime_target=RuntimeTarget(), tasks=())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = score.runtime_target
            _ = score.runtime_target
            assert len(w) == 1
