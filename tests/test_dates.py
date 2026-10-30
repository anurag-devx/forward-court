from dates import extract_date


def test_iso_and_slash_dates():
    assert extract_date("published 2026-04-10 here") == "2026-04-10"
    assert extract_date("dated 2015/04/25.") == "2015-04-25"


def test_month_name_dates():
    assert extract_date("April 10, 2026") == "2026-04-10"
    assert extract_date("10 Apr 2026") == "2026-04-10"
    assert extract_date("1st January 2020") == "2020-01-01"


def test_invalid_and_missing_dates():
    assert extract_date("February 30, 2026") is None
    assert extract_date("no date in this snippet") is None
    assert extract_date("") is None
