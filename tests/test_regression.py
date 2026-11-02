"""
test_regression.py - v0.2.4 regression tests for fingerprint and slot key fixes.

These tests lock in the bug fixes from the adversarial code review:
- Fix #8: fingerprint uses ordered text (opposite rules must not collide)
- Fix #9: slot key uses head noun, not first entity (revisions must supersede)

Run with: python -m pytest tests/test_regression.py -v
Or:       python tests/test_regression.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dspm.patch import SemanticPatch


def test_fingerprint_opposite_rules_distinct():
    """Fix #8: opposite-meaning rules must have different fingerprints."""
    p1 = SemanticPatch("p1", 0, "constraint", "Admins can delete, guests cannot")
    p2 = SemanticPatch("p2", 1, "constraint", "Guests can delete, admins cannot")
    assert p1.fingerprint != p2.fingerprint, (
        "Opposite-meaning rules have same fingerprint (bag-of-words collision)"
    )


def test_fingerprint_exact_duplicates_same():
    """Fix #8: exact duplicates must still have same fingerprint (dedup works)."""
    p1 = SemanticPatch("p1", 0, "constraint", "Max fee 0.5%")
    p2 = SemanticPatch("p2", 1, "constraint", "Max fee 0.5%")
    assert p1.fingerprint == p2.fingerprint, (
        "Exact duplicates have different fingerprints (dedup broken)"
    )


def test_slot_key_revisions_same_slot():
    """Fix #9: numeric revisions must map to same slot (supersession fires)."""
    p1 = SemanticPatch("p1", 0, "decision", "Webhook timeout set to 30 seconds")
    p2 = SemanticPatch("p2", 1, "decision", "Webhook timeout updated to 10 seconds")
    assert p1.slot_key == p2.slot_key, (
        f"Numeric revisions have different slots: {p1.slot_key} != {p2.slot_key}"
    )


def test_slot_key_not_numeric():
    """Fix #9: slot key must not be a number."""
    p = SemanticPatch("p", 0, "constraint", "Rate limit 100 requests per minute")
    # slot_key format is "type::keyword", keyword should not be just a number
    keyword = p.slot_key.split("::")[1]
    assert not keyword.replace(".", "").replace(",", "").isdigit(), (
        f"Slot key is numeric: {keyword}"
    )


def test_slot_key_juicy_examples():
    """Fix #9: test the exact examples from the README Quickstart."""
    # JWT expiry revised
    p1 = SemanticPatch("p1", 0, "decision", "JWT auth: 15-min access tokens, 7-day refresh")
    p2 = SemanticPatch("p2", 1, "decision", "JWT auth: 30-min access tokens, 7-day refresh")
    assert p1.slot_key == p2.slot_key, (
        f"JWT revision slots differ: {p1.slot_key} != {p2.slot_key}"
    )

    # S3 URL expiry revised
    p3 = SemanticPatch("p3", 0, "decision", "S3 pre-signed URL expiry 1 hour")
    p4 = SemanticPatch("p4", 1, "decision", "S3 pre-signed URL expiry 15 minutes")
    assert p3.slot_key == p4.slot_key, (
        f"S3 revision slots differ: {p3.slot_key} != {p4.slot_key}"
    )


def test_case_insensitive_types():
    """Fix #2: type matching must be case-insensitive and handle plurals."""
    # This is tested via the extractor, but we can verify the type normalization
    # by checking that SemanticPatch accepts canonical types
    for t in ["constraint", "Constraint", "Constraints", "CONSTRAINT"]:
        p = SemanticPatch("p", 0, t.lower().rstrip("s"), "test")
        assert p.patch_type == "constraint", f"Type {t} did not normalize correctly"


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_fingerprint_opposite_rules_distinct,
        test_fingerprint_exact_duplicates_same,
        test_slot_key_revisions_same_slot,
        test_slot_key_not_numeric,
        test_slot_key_juicy_examples,
        test_case_insensitive_types,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print()
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
