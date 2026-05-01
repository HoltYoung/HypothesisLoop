"""Phase 9 target-leakage AST guard tests."""

from __future__ import annotations

from hypothesisloop.agent.leakage import check_no_target_leakage


def test_no_leakage_innocent_code_passes():
    code = "df['x'] = df['age'] * 2\n"
    assert check_no_target_leakage(code, "income") is None


def test_subscript_string_key_blocks():
    code = "df['x'] = df['income'].astype(int)\n"
    err = check_no_target_leakage(code, "income")
    assert err is not None
    assert "income" in err
    assert "leakage" in err.lower()


def test_subscript_double_quoted_string_blocks():
    code = 'df["x"] = df["income"]\n'
    err = check_no_target_leakage(code, "income")
    assert err is not None


def test_attribute_access_blocks():
    code = "df['x'] = df.income.astype(int)\n"
    err = check_no_target_leakage(code, "income")
    assert err is not None


def test_case_insensitive_match_blocks():
    code = "df['x'] = df['Income'].astype(int)\n"
    err = check_no_target_leakage(code, "income")
    assert err is not None


def test_whitespace_target_normalized():
    code = "df['x'] = df['income'].astype(int)\n"
    err = check_no_target_leakage(code, "  income  ")
    assert err is not None


def test_syntax_error_returns_none_so_runner_surfaces_it():
    """A SyntaxError should fall through — let the sandbox runner own that."""
    bad = "this is not python !! &&\n"
    assert check_no_target_leakage(bad, "income") is None


def test_target_substring_in_other_column_does_not_match():
    """``df['income_class']`` should NOT trigger when target=='income'."""
    code = "df['x'] = df['income_class'].fillna('low')\n"
    assert check_no_target_leakage(code, "income") is None


def test_blank_target_passes():
    """Empty target column = no rule to enforce."""
    assert check_no_target_leakage("df['income'] = df.income", "") is None
