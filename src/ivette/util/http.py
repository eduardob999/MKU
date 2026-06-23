"""HTTP helpers and PubChem REST endpoints.

Centralizes the timeout / ``raise_for_status`` / retry-and-backoff boilerplate
that was previously copy-pasted across every client and downloader.
"""

import sys
import time

import requests

DEFAULT_TIMEOUT = 30

PUBCHEM_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound"
PUBCHEM_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/compound"

# A desktop browser User-Agent for endpoints that block default clients.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120 Safari/537.36"
    )
}

# Transient network exceptions worth retrying.
RETRYABLE_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
)
RETRYABLE_STATUS = (429, 500, 502, 503, 504)


def get(url, *, params=None, headers=None, timeout=DEFAULT_TIMEOUT):
    """GET returning the raw :class:`requests.Response` (no ``raise_for_status``)."""
    return requests.get(url, params=params, headers=headers, timeout=timeout)


def get_json(url, *, params=None, headers=None, timeout=DEFAULT_TIMEOUT):
    """GET, raise on HTTP error, return the parsed JSON body."""
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_text(url, *, params=None, headers=None, timeout=DEFAULT_TIMEOUT):
    """GET, raise on HTTP error, return the response text."""
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def pubchem_fetch_properties(cids, properties, *, max_retries=5, timeout=60):
    """Fetch PubChem compound ``properties`` for ``cids`` with retry/backoff.

    Returns the raw ``PropertyTable.Properties`` list. On a 400 with more than
    one CID the batch is split in half and retried. Shared by the physchem
    downloader and the InChIKey fetcher.
    """
    if not cids:
        return []
    cid_str = ",".join(str(c) for c in cids)
    prop_str = ",".join(properties)
    url = f"{PUBCHEM_PUG}/compound/cid/{cid_str}/property/{prop_str}/JSON"

    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("PropertyTable", {}).get("Properties", [])
        except RETRYABLE_ERRORS as exc:
            wait = 2 ** attempt
            print(
                f"  Connection error (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
        except requests.HTTPError:
            if resp is not None and resp.status_code == 400 and len(cids) > 1:
                mid = len(cids) // 2
                return (
                    pubchem_fetch_properties(
                        cids[:mid], properties,
                        max_retries=max_retries, timeout=timeout,
                    )
                    + pubchem_fetch_properties(
                        cids[mid:], properties,
                        max_retries=max_retries, timeout=timeout,
                    )
                )
            if resp is not None and resp.status_code in RETRYABLE_STATUS:
                wait = 2 ** attempt
                print(
                    f"  HTTP {resp.status_code} (attempt "
                    f"{attempt + 1}/{max_retries}), retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise

    raise requests.exceptions.ConnectionError(
        f"Failed to fetch properties after {max_retries} attempts "
        f"for {len(cids)} CIDs"
    )
