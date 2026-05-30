from sonata.directions import (
    IGNORED_DIRECTIONS,
    MEMORY_DIRECTIONS,
    READ_DIRECTIONS,
    WRITE_DIRECTIONS,
    normalize_direction,
)


def test_normalize_direction_canonicalizes_separators() -> None:
    assert normalize_direction("Output Existing") == "outputexisting"
    assert normalize_direction("output_existing") == "outputexisting"
    assert normalize_direction("output-existing") == "outputexisting"


def test_direction_sets_share_memory_semantics() -> None:
    assert READ_DIRECTIONS == frozenset({"input", "inout"})
    assert WRITE_DIRECTIONS == frozenset({"output", "outputexisting", "inout"})
    assert MEMORY_DIRECTIONS == READ_DIRECTIONS | WRITE_DIRECTIONS
    assert IGNORED_DIRECTIONS == frozenset({"scalar", "nodep"})
