import numpy as np
import pandas as pd

from src.utils.transform_rules import (
    ORG_GROUPS, assign_age_group, age_group_series, assign_term_group, categorize_organization,
    hour_group_series, assign_hour_group, term_group_series, term_months, term_months_series,
)

ALL_HOME_CREDIT_ORGS = """Advertising;Agriculture;Bank;Business Entity Type 1;Business Entity Type 2;Business Entity Type 3;
Cleaning;Construction;Culture;Electricity;Emergency;Government;Hotel;Housing;Industry: type 1;Industry: type 10;
Industry: type 11;Industry: type 12;Industry: type 13;Industry: type 2;Industry: type 3;Industry: type 4;Industry: type 5;
Industry: type 6;Industry: type 7;Industry: type 8;Industry: type 9;Insurance;Kindergarten;Legal Services;Medicine;
Military;Mobile;Other;Police;Postal;Realtor;Religion;Restaurant;School;Security;Security Ministries;Self-employed;
Services;Telecom;Trade: type 1;Trade: type 2;Trade: type 3;Trade: type 4;Trade: type 5;Trade: type 6;Trade: type 7;
Transport: type 1;Transport: type 2;Transport: type 3;Transport: type 4;University;XNA""".replace("\n", "").split(";")


def test_every_org_type_maps_to_known_group():
    for org in ALL_HOME_CREDIT_ORGS:
        assert categorize_organization(org) in ORG_GROUPS, org


def test_xna_and_missing_are_unknown_not_other():
    assert categorize_organization("XNA") == "Không xác định"
    assert categorize_organization(None) == "Không xác định"
    assert categorize_organization("Other") == "Khác"


def test_age_group_boundaries_scalar_equals_vectorized():
    ages = pd.Series([18, 24.99, 25, 34.99, 35, 44.99, 45, 54.99, 55, 64.99, 65, 80])
    assert [assign_age_group(a) for a in ages] == age_group_series(ages).tolist()
    assert assign_age_group(25) == "25-34" and assign_age_group(65) == "65+"


def test_term_months_and_groups_consistent():
    assert term_months(120000, 10000) == 12
    assert term_months(1e9, 1) == 120                  # giới hạn trên
    assert term_months(100000, None) == 0 and term_months(100000, 0) == 0 and term_months(np.nan, 5) == 0
    credit = pd.Series([120000, 120000, 100000, 500000])
    ann = pd.Series([10000, 5000, np.nan, 10000])
    months = term_months_series(credit, ann)
    assert months.tolist() == [12, 24, 0, 50]
    assert term_group_series(months).tolist() == [assign_term_group(m) for m in months]
    assert assign_term_group(0) == "Không xác định" and assign_term_group(37) == ">36"


def test_hour_groups_scalar_equals_vectorized():
    hours = pd.Series(range(24))
    assert hour_group_series(hours).tolist() == [assign_hour_group(h) for h in hours]
