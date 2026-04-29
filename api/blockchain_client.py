"""
client.py
Centralised HTTP client for Bitcoin blockchain data.

Sources used:
  - Blockstream API:    https://blockstream.info/api
  - Mempool.space API:  https://mempool.space/api
  - Blockchain.info:    https://blockchain.info

Design decisions:
  - requests.Session for connection pooling and retry logic.
  - Simple in-memory TTL cache (dict + timestamp) — no external deps.
  - Raises BlockchainAPIError on permanent failure so callers can fall
    back to cached Streamlit session_state data (C3 criterion).
"""

import time
import logging
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKSTREAM_BASE = "https://blockstream.info/api"
MEMPOOL_BASE = "https://mempool.space/api"

CACHE_TTL_SECONDS = 60          # re-fetch at most once per minute
REQUEST_TIMEOUT = 10            # seconds per individual HTTP call
MAX_RETRIES = 3
BACKOFF_FACTOR = 2              # exponential backoff: 2s, 4s, 8s


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class BlockchainAPIError(RuntimeError):
    """Raised when all retry attempts to the blockchain API have failed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """
    Build a requests.Session with automatic retry on 5xx and network errors.

    The Retry adapter uses exponential back-off so we don't hammer the API
    when it is temporarily overloaded.
    """
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "CryptoChainAnalyzer/1.0 (university project)",
        "Accept": "application/json",
    })
    return session


# Module-level session and cache — shared across all calls in one process.
_session = _make_session()
_cache: dict[str, tuple[float, Any]] = {}  # key → (timestamp, value)


def _cached_get(url: str, raw: bool = False) -> Any:
    """
    GET `url`, returning the cached result if it is still fresh.

    Parameters
    ----------
    url  : Full URL to fetch.
    raw  : If True, return response.content (bytes) instead of .json().
    """
    now = time.monotonic()
    if url in _cache:
        ts, value = _cache[url]
        if now - ts < CACHE_TTL_SECONDS:
            return value

    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise BlockchainAPIError(f"Request failed for {url}: {exc}") from exc

    if raw:
        value = resp.content
    else:
        try:
            value = resp.json()
        except ValueError as exc:
            raise BlockchainAPIError(f"Non-JSON response from {url}: {exc}") from exc
    _cache[url] = (now, value)
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_latest_block() -> dict:
    """
    Return metadata for the most recently confirmed Bitcoin block.

    Fetches /blocks (list of latest blocks) and returns the first item,
    which is the tip of the chain.
    """
    blocks = _cached_get(f"{BLOCKSTREAM_BASE}/blocks")
    if not blocks:
        raise BlockchainAPIError("Empty block list returned by API")
    return blocks[0]


def get_block_by_hash(block_hash: str) -> dict:
    """Return full metadata for a block identified by its hash."""
    return _cached_get(f"{BLOCKSTREAM_BASE}/block/{block_hash}")


# Alias used by the M2 starter stub
get_block = get_block_by_hash


def get_raw_header(block_hash: str) -> bytes:
    """
    Return the raw 80-byte block header for *block_hash*.

    The Blockstream endpoint returns a hex-encoded string; we decode it
    to bytes here so callers always receive actual bytes (not hex).

    Byte order note: the raw bytes are in the on-disk / wire format
    (little-endian for multi-byte fields). Callers that want to display
    hashes must reverse the bytes to obtain big-endian / RPC byte order.
    """
    hex_str = _cached_get(
        f"{BLOCKSTREAM_BASE}/block/{block_hash}/header",
        raw=True,
    )
    # The endpoint returns the hex string as plain text (bytes object here).
    if isinstance(hex_str, bytes):
        hex_str = hex_str.decode().strip()
    raw = bytes.fromhex(hex_str)
    if len(raw) != 80:
        raise BlockchainAPIError(
            f"Expected 80-byte header, got {len(raw)} bytes for {block_hash}"
        )
    return raw


def get_last_n_blocks(n: int = 50) -> list[dict]:
    """
    Return the last *n* confirmed blocks, most-recent first.

    Blockstream's /blocks endpoint returns 10 blocks per page. We
    iterate pages until we have enough.
    """
    results: list[dict] = []
    # First page — no start_height parameter needed.
    page = _cached_get(f"{BLOCKSTREAM_BASE}/blocks")
    results.extend(page)

    while len(results) < n:
        oldest_height = results[-1]["height"] - 1
        url = f"{BLOCKSTREAM_BASE}/blocks/{oldest_height}"
        try:
            page = _cached_get(url)
        except BlockchainAPIError:
            break
        if not page:
            break
        results.extend(page)

    return results[:n]


def get_difficulty_history() -> pd.DataFrame:
    """
    Return historical Bitcoin difficulty as a DataFrame.

    Columns: timestamp (datetime64), difficulty (float64).

    Data source: mempool.space mining hashrate endpoint, which includes
    one difficulty data point per adjustment epoch (every 2016 blocks).
    """
    url = f"{MEMPOOL_BASE}/v1/mining/hashrate/2y"
    data = _cached_get(url)
    difficulty_points = data.get("difficulty", [])
    if not difficulty_points:
        raise BlockchainAPIError("No difficulty data in mempool.space response")
    df = pd.DataFrame(difficulty_points)[["time", "difficulty"]]
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df
