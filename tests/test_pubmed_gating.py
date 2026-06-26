"""PubMed is opt-in and API-key-gated; these pin the surface that find_thermo relies on."""

import inspect

from ivette.module import pubmed_client as pm
from ivette.core import params as P


def test_empty_pubmed_result_has_all_report_keys():
    r = pm.empty_pubmed_result()
    assert set(r) == {
        "PubMed_Thermo_Count", "PubMed_Top_PMIDs", "PubMed_PubMed_Links",
        "PubMed_Abstract_Match_Count", "PubMed_Example_Matches", "PubMed_Supplementary_Count",
    }
    assert r["PubMed_Thermo_Count"] == 0
    assert r["PubMed_Abstract_Match_Count"] == 0


def test_pubmed_functions_accept_api_key():
    assert "api_key" in inspect.signature(pm.pubmed_search).parameters
    assert "api_key" in inspect.signature(pm.pubmed_fetch_abstracts).parameters
    assert "api_key" in inspect.signature(pm.analyze_pubmed).parameters


def test_pubmed_is_off_by_default_in_dataset_params():
    assert P.DatasetParams().fetch_pubmed is False
