"""Tests for the Score cache module."""

import json
import tempfile
from pathlib import Path

import pytest

from sonata.cache import (
    CACHE_SCHEMA_VERSION,
    CacheEntry,
    ScoreCache,
    cached_score,
)
from sonata.plan_handle import PlanHandle
from sonata.score import Dependency, RuntimeTarget, Score, ShapeAssumption, Task
from sonata.serialization import score_fingerprint


def _make_score(name: str = "test_graph") -> Score:
    return Score(
        name=name,
        runtime_target=RuntimeTarget(),
        tasks=(
            Task(task_id=0, func_id=0, core_type="aic", name="func_a"),
            Task(task_id=1, func_id=1, core_type="aiv", name="func_b"),
        ),
        dependencies=(Dependency(producer=0, consumer=1),),
        shape_assumptions=(ShapeAssumption(symbol="N", dims=(128, 64)),),
    )


def _make_score_b() -> Score:
    return Score(
        name="other_graph",
        runtime_target=RuntimeTarget(),
        tasks=(Task(task_id=0, func_id=0, core_type="aic", name="func_c"),),
    )


class TestCacheEntry:
    def test_construct(self):
        entry = CacheEntry(fingerprint="abc123", score_payload={"name": "x"})
        assert entry.fingerprint == "abc123"
        assert entry.score_payload == {"name": "x"}
        assert entry.plan_handle_payload is None

    def test_frozen(self):
        entry = CacheEntry(fingerprint="fp", score_payload={})
        with pytest.raises(AttributeError):
            entry.fingerprint = "other"


class TestScoreCacheStoreLookup:
    def test_store_returns_fingerprint(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        assert fp == score_fingerprint(score)

    def test_store_and_lookup(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        payload = cache.lookup(fp)
        assert payload is not None
        assert payload["name"] == "test_graph"

    def test_lookup_miss(self):
        cache = ScoreCache()
        result = cache.lookup("nonexistent_fingerprint")
        assert result is None

    def test_store_overwrites_same_fingerprint(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        fp2 = cache.store(score)
        assert fp == fp2
        assert cache.entry_count() == 1

    def test_store_different_scores(self):
        cache = ScoreCache()
        cache.store(_make_score())
        cache.store(_make_score_b())
        assert cache.entry_count() == 2

    def test_contains(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        assert cache.contains(fp)
        assert not cache.contains("missing")

    def test_store_with_explicit_fingerprint(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score, fingerprint="custom_fp")
        assert fp == "custom_fp"
        assert cache.contains("custom_fp")


class TestScoreCachePlanHandle:
    def test_store_plan_handle(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        plan = PlanHandle.from_score(score)
        result_fp = cache.store_plan_handle(plan)
        assert result_fp == fp

    def test_lookup_plan_handle(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        plan = PlanHandle.from_score(score)
        cache.store_plan_handle(plan)
        payload = cache.lookup_plan_handle(fp)
        assert payload is not None
        assert payload["score_fingerprint"] == fp

    def test_lookup_plan_handle_miss_no_score(self):
        cache = ScoreCache()
        assert cache.lookup_plan_handle("missing") is None

    def test_lookup_plan_handle_miss_no_plan(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        assert cache.lookup_plan_handle(fp) is None

    def test_store_plan_handle_no_score_raises(self):
        cache = ScoreCache()
        score = _make_score()
        plan = PlanHandle.from_score(score)
        with pytest.raises(KeyError):
            cache.store_plan_handle(plan, fingerprint="nonexistent")

    def test_plan_handle_preserved_on_score_overwrite(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        plan = PlanHandle.from_score(score)
        cache.store_plan_handle(plan)
        cache.store(score)
        assert cache.lookup_plan_handle(fp) is not None


class TestScoreCacheInvalidate:
    def test_invalidate_single(self):
        cache = ScoreCache()
        fp = cache.store(_make_score())
        removed = cache.invalidate(fp)
        assert removed == 1
        assert cache.entry_count() == 0

    def test_invalidate_batch(self):
        cache = ScoreCache()
        fp1 = cache.store(_make_score())
        fp2 = cache.store(_make_score_b())
        removed = cache.invalidate(fp1, fp2)
        assert removed == 2
        assert cache.entry_count() == 0

    def test_invalidate_missing(self):
        cache = ScoreCache()
        removed = cache.invalidate("nope")
        assert removed == 0

    def test_invalidate_all(self):
        cache = ScoreCache()
        cache.store(_make_score())
        cache.store(_make_score_b())
        removed = cache.invalidate_all()
        assert removed == 2
        assert cache.entry_count() == 0


class TestScoreCacheStats:
    def test_initial_stats(self):
        cache = ScoreCache()
        s = cache.stats()
        assert s["entry_count"] == 0
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate_pct"] == 0

    def test_hit_and_miss_tracking(self):
        cache = ScoreCache()
        fp = cache.store(_make_score())
        cache.lookup(fp)
        cache.lookup("miss")
        cache.lookup(fp)
        s = cache.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["total_lookups"] == 3

    def test_hit_rate(self):
        cache = ScoreCache()
        fp = cache.store(_make_score())
        cache.lookup(fp)
        cache.lookup("miss")
        s = cache.stats()
        assert s["hit_rate_pct"] == 50


class TestScoreCachePersistence:
    def test_to_dict_roundtrip(self):
        cache = ScoreCache()
        cache.store(_make_score())
        cache.store(_make_score_b())
        data = cache.to_dict()
        assert data["cache_schema_version"] == CACHE_SCHEMA_VERSION
        assert len(data["entries"]) == 2

        restored = ScoreCache.from_dict(data)
        assert restored.entry_count() == 2

    def test_save_and_load(self, tmp_path):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        plan = PlanHandle.from_score(score)
        cache.store_plan_handle(plan)

        path = cache.save(tmp_path / "cache.json")
        assert path.exists()

        restored = ScoreCache.load(path)
        assert restored.entry_count() == 1
        assert restored.contains(fp)
        assert restored.lookup_plan_handle(fp) is not None

    def test_save_creates_parent_dirs(self, tmp_path):
        cache = ScoreCache()
        path = cache.save(tmp_path / "sub" / "dir" / "cache.json")
        assert path.exists()

    def test_json_valid(self, tmp_path):
        cache = ScoreCache()
        cache.store(_make_score())
        path = cache.save(tmp_path / "cache.json")
        data = json.loads(path.read_text())
        assert "entries" in data


class TestCachedScore:
    def test_miss_builds_and_stores(self):
        cache = ScoreCache()
        calls = 0

        def builder():
            nonlocal calls
            calls += 1
            return _make_score()

        score, fp, was_cached = cached_score(cache, builder)
        assert not was_cached
        assert calls == 1
        assert cache.contains(fp)
        assert score.name == "test_graph"

    def test_hit_skips_builder(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        calls = 0

        def builder():
            nonlocal calls
            calls += 1
            return _make_score_b()

        result, result_fp, was_cached = cached_score(cache, builder, fingerprint_hint=fp)
        assert was_cached
        assert calls == 0
        assert result_fp == fp
        assert result.name == "test_graph"

    def test_no_hint_always_builds(self):
        cache = ScoreCache()
        score = _make_score()
        cache.store(score)

        result, fp, was_cached = cached_score(cache, lambda: _make_score())
        assert not was_cached
        assert result.name == "test_graph"

    def test_schema_mismatch_treated_as_miss(self):
        cache = ScoreCache()
        score = _make_score()
        fp = cache.store(score)
        entry = cache._entries[fp]
        cache._entries[fp] = CacheEntry(
            fingerprint=entry.fingerprint,
            score_payload=entry.score_payload,
            schema_version=999,
        )
        assert not cache.contains(fp)
        assert cache.lookup(fp) is None
