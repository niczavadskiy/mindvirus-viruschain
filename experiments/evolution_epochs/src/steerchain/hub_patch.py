"""Bypass broken httpx2 Hub streaming by forcing requests-based http_get.

Works with huggingface_hub 0.x and 1.x/2.x — keep whatever transformers needs;
only replace the download body transport.
"""

from __future__ import annotations

_PATCHED = False


def patch_hub_http_get() -> None:
    """Replace huggingface_hub.file_download.http_get with a requests implementation."""
    global _PATCHED
    if _PATCHED:
        return

    import requests

    from huggingface_hub import file_download

    try:
        from huggingface_hub.utils import tqdm as hub_tqdm
    except Exception:  # noqa: BLE001
        from tqdm.auto import tqdm as hub_tqdm

    def http_get(  # type: ignore[no-untyped-def]
        url,
        temp_file,
        *,
        proxies=None,
        resume_size=0,
        headers=None,
        expected_size=None,
        displayed_filename=None,
        tqdm_class=None,
        _tqdm_bar=None,
        timeout=None,
        **_kwargs,
    ):
        headers = dict(headers or {})
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        # Skip compressed encodings that trip the broken httpx2 zlib wrapper.
        headers["Accept-Encoding"] = "identity"

        timeout = timeout if timeout is not None else 120
        bar_cls = tqdm_class or hub_tqdm
        total = expected_size + resume_size if expected_size is not None else None

        with requests.get(
            url,
            stream=True,
            proxies=proxies,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            progress = bar_cls(
                unit="B",
                unit_scale=True,
                total=total,
                initial=resume_size,
                desc=displayed_filename or url.split("/")[-1],
                disable=False,
            )
            try:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        temp_file.write(chunk)
                        progress.update(len(chunk))
            finally:
                progress.close()

    file_download.http_get = http_get  # type: ignore[assignment]
    for mod_name in (
        "huggingface_hub.file_download",
        "huggingface_hub.utils._http",
        "transformers.utils.hub",
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            if hasattr(mod, "http_get"):
                setattr(mod, "http_get", http_get)
        except Exception:
            pass
    _PATCHED = True


def prepare_hub_downloads() -> dict:
    """Call once per process before any from_pretrained / hf_hub_download."""
    patch_hub_http_get()
    return {"patched": True, "transport": "requests"}
