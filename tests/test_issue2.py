from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def test_issue2_stale_con_superseded_by_ttl_revision():
    """Issue #2: 'Cache TTL must be 60 seconds' [CON] must be superseded by
    'TTL is now 120 seconds' [DEC] even though slot heads differ (cache/ttl)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Cache TTL must be 60 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'TTL is now 120 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"stale 60s TTL survived: {crits}"
    assert '120' in crits[0]
    assert '(was' in crits[0] and '60' in crits[0], crits[0]


def test_issue2_unrelated_seconds_constraint_not_superseded():
    """Guard: a different quantity that merely shares the unit must stay."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Retry delay must be 5 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'TTL is now 120 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, f"unrelated seconds constraint was removed: {crits}"


def test_issue2_cross_type_pair_does_not_shrink_same_type_overlap():
    """Issue #2: an earlier unmatched cross-type pair must not change the
    word sets used for later same-type comparisons (new_words shadowing)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Maximum fee is 0.5%', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'TTL changed to 120 seconds', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'Time-to-live (TTL) setting updated to 120 seconds as specified.', []))
    crits = [p.payload for p in m.critical_patches]
    ttl = [c for c in crits if 'ttl' in c.lower()]
    assert len(ttl) == 1, f"duplicate TTL decisions survived: {crits}"
    assert any('fee' in c.lower() for c in crits), crits
