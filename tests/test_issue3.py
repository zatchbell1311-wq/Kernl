from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def _add(memory, patch_id, turn, patch_type, payload):
    memory._merge_patch(SemanticPatch(patch_id, turn, patch_type, payload, []))


def test_issue3_same_type_weekday_revision_bakes_replaced_day():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "decision", "Deadline set to Friday")
    _add(memory, "p1_0", 1, "decision", "Deadline moved to Monday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 1, criticals
    assert "Monday" in criticals[0]
    assert "(was Friday)" in criticals[0], criticals[0]


def test_issue3_cross_type_weekday_revision_bakes_replaced_day():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "constraint", "Deadline set to Friday")
    _add(memory, "p1_0", 1, "decision", "Deadline moved to Monday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 1, criticals
    assert "Monday" in criticals[0]
    assert "(was Friday)" in criticals[0], criticals[0]


def test_issue3_same_weekday_restatement_collapses_without_baking():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "decision", "Deadline set to Monday")
    _add(memory, "p1_0", 1, "decision", "Deadline updated to Monday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 1, criticals
    assert "(was" not in criticals[0]


def test_issue3_cross_type_same_weekday_restatement_collapses():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "constraint", "Deadline set to Monday")
    _add(memory, "p1_0", 1, "decision", "Deadline updated to Monday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 1, criticals
    assert "(was" not in criticals[0]


def test_issue3_revision_that_names_both_days_does_not_duplicate_history():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "decision", "Deadline set to Friday")
    _add(memory, "p1_0", 1, "decision",
         "Deadline moved from Friday to Monday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 1, criticals
    assert "Friday" in criticals[0] and "Monday" in criticals[0]
    assert "(was" not in criticals[0]


def test_issue3_weekday_revision_requires_marker_and_preserves_scopes():
    memory = DSPMMemory(budget=250)
    _add(memory, "p0_0", 0, "decision", "Deadline set to Friday")
    _add(memory, "p1_0", 1, "decision", "Deadline set to Monday")
    _add(memory, "p2_0", 2, "decision", "Report deadline moved to Tuesday")
    _add(memory, "p3_0", 3, "decision", "Team lunch moved to Wednesday")

    criticals = [patch.payload for patch in memory.critical_patches]
    assert len(criticals) == 4, criticals
