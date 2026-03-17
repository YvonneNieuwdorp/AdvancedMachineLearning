import re
from pathlib import Path
from html import unescape
from typing import Generator


__all__ = [
    "iter_sgm_files",
]

SRC_FOLDER = Path('reuters')


def extract_attributes(tag: str) -> dict:
    """Extract attributes from a tag into a lowercase-key dictionary.

    Args:
        tag: Raw attribute string or full tag text containing key="value" pairs.

    Returns:
        A dictionary mapping lowercase attribute names to their string values.
    """
    return {
        k.lower(): v
        for k, v in re.findall(r'([A-Za-z_:][A-Za-z0-9_:\-.]*)="([^"]*)"', tag)
    }


def extract_tag_text(block: str, tag_name: str) -> str | None:
    """Extract and unescape the inner text of the first matching subtag.

    Args:
        block: Text block that may contain the target tag.
        tag_name: Tag name to extract, for example ``TOPICS`` or ``BODY``.

    Returns:
        The stripped inner text for the first match, or ``None`` if missing.
    """
    m = re.search(rf"<{tag_name}[^>]*>(.*?)</{tag_name}>", block, flags=re.DOTALL)
    return unescape(m.group(1)).strip() if m else None


def extract_topics_list(topics_text: str | None) -> list[str]:
    """Parse Reuters topic labels from TOPICS inner XML content.

    Args:
        topics_text: TOPICS content, typically containing repeated ``<D>...</D>`` nodes.

    Returns:
        A list of topic labels. Returns an empty list when input is empty.
    """
    if not topics_text:
        return []
    return re.findall(r"<D>(.*?)</D>", topics_text, flags=re.DOTALL)


def iter_reuters_records(filename: Path) -> Generator[dict, None, None]:
    """Yield filtered Reuters records from one SGML file.

    The function keeps only records where the REUTERS attributes satisfy
    ``TOPICS == "YES"`` and ``LEWISSPLIT in {"TEST", "TRAIN"}``.

    Args:
        filename: Path to a Reuters ``.sgm`` file.

    Yields:
        Dictionaries containing selected metadata and extracted text fields.
    """
    text = Path(filename).read_text(encoding="latin-1", errors="ignore")
    for m in re.finditer(r"<REUTERS\b([^>]*)>(.*?)</REUTERS>", text, flags=re.DOTALL):
        attributes_text, reuters_tag_content = m.group(1), m.group(2)
        attributes = extract_attributes(attributes_text)
        if attributes.get("topics") == "YES" and attributes.get("lewissplit") in ("TEST", "TRAIN"):
            yield {
                "topics_present": attributes.get("topics"),
                "lewissplit": attributes.get("lewissplit"),
                "topics": ','.join(
                    extract_topics_list(extract_tag_text(reuters_tag_content, "TOPICS"))
                ),
                "body_text": extract_tag_text(reuters_tag_content, "BODY"),
            }


def iter_sgm_files(folder: Path) -> Generator[dict, None, None]:
    """Iterate over Reuters SGML files and yield enriched record dictionaries.

    Example of usage:
    df = pd.DataFrame(iter_sgm_files(SRC_FOLDER))

    Args:
        folder: Folder containing Reuters ``.sgm`` files.

    Yields:
        Record dictionaries from ``iter_reuters_records`` with added filename and
        per-file record index.
    """
    for filename in sorted(folder.glob('*.sgm')):
        # print(f"Processing {filename}...")
        for ix, record in enumerate(iter_reuters_records(filename), 1):
            record["filename"] = filename.name
            record["record_ix"] = ix
            yield record
