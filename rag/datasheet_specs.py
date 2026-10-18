"""Structured extraction from turbine datasheet PDFs, and a cross-check against the turbine
specs hand-coded in data_sources/farms.py.

Extraction runs once at index-build time (rag/pdf_index_build.py): Nova reads the datasheet's
pages and returns each field with the page it read it from. Nothing the LLM returns is taken on
trust - every value is re-checked against the text of the page it cites (`verified`), so a
hallucinated or misread number shows up as unverified instead of as a fact. The API only reads
the resulting JSON (slim, no PyMuPDF), see crosscheck().
"""
import json
import re
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "pdf_index"
SPECS_PATH = INDEX_DIR / "datasheet_specs.json"

SPEC_FIELDS = {
    "rated_power_kw": "nominal/rated electrical power, in kW (number)",
    "rotor_diameter_m": "rotor diameter in metres (number)",
    "swept_area_m2": "rotor swept area in square metres (number)",
    "blade_length_m": "blade length in metres (number)",
    "rotor_speed_rpm": "rotor speed range as printed, e.g. '6-18'",
    "cut_in_wind_speed_ms": "cut-in wind speed in m/s as printed (may be a range)",
    "rated_wind_speed_ms": "wind speed at which rated power is reached, m/s as printed",
    "cut_out_wind_speed_ms": "cut-out wind speed in m/s (number)",
    "hub_height_m": "hub height(s) as printed",
    "iec_class": "IEC wind class",
    "gearbox": "gearbox type and ratio as printed",
    "generator": "generator type",
}

_EXTRACTION_PROMPT = """You are reading a wind turbine manufacturer's datasheet. Pages are marked [page N].

{pages}

Extract these fields: {fields}

Return ONLY a JSON object: each key is one field name above, each value is an object
{{"value": <value exactly as stated, number where the field says number>, "page": <page number it is on>}},
or null if the document does not state it. Never infer or compute a value that is not printed."""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    out = set()
    for tok in _NUMBER.findall(text):
        try:
            out.add(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def verify(field: str, value, page_text: str) -> bool:
    """True when the value is actually printed on the cited page: every number in it appears
    there (rated power may be printed in MW instead of kW), or for a text value, the text does."""
    if value is None or not page_text:
        return False
    page_numbers = _numbers(page_text)
    value_numbers = _numbers(str(value))
    if value_numbers:
        def present(v: float) -> bool:
            return v in page_numbers or (field == "rated_power_kw" and v / 1000 in page_numbers)

        return all(present(v) for v in value_numbers)
    def squash(t: str) -> str:  # PDF text wraps mid-phrase ("IEC\nClass IA")
        return " ".join(t.split()).lower()

    return squash(str(value)) in squash(page_text)


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def extract_specs(pages: list, complete) -> dict:
    """pages: rag.pdf_corpus.Page list for one datasheet; complete: agents.llm_router.complete."""
    page_text = {p.page: p.text for p in pages}
    prompt = _EXTRACTION_PROMPT.format(
        pages="\n\n".join(f"[page {p.page}]\n{p.text}" for p in pages),
        fields=json.dumps(SPEC_FIELDS, indent=1),
    )
    response = complete([{"role": "user", "content": prompt}], temperature=0)
    raw = _parse_json(response.choices[0].message.content)

    specs = {}
    for field in SPEC_FIELDS:
        entry = raw.get(field)
        if not isinstance(entry, dict) or entry.get("value") in (None, ""):
            continue
        page = entry.get("page")
        page = int(page) if isinstance(page, (int, float, str)) and str(page).isdigit() else None
        spec = {"value": entry["value"], "page": page, "verified": verify(field, entry["value"], page_text.get(page, ""))}
        if not spec["verified"]:
            # A brochure's printed page numbers rarely match the PDF's: the model often cites the
            # printed one. If the value is on another page, cite that page and record the fix.
            found = next((n for n, text in page_text.items() if verify(field, entry["value"], text)), None)
            if found is not None:
                spec.update(page=found, verified=True, cited_page=page)
        specs[field] = spec
    return specs


def load_specs() -> dict:
    if not SPECS_PATH.exists():
        return {}
    return json.loads(SPECS_PATH.read_text())


def _as_float(value) -> float | None:
    nums = _numbers(str(value))
    return next(iter(nums)) if len(nums) == 1 else None


def crosscheck(farm) -> dict | None:
    """Datasheet vs data_sources/farms.py for one Farm, or None when no datasheet for its
    turbine model is indexed. Only fields the datasheet states AND that were verified on the
    cited page are compared."""
    from rag.pdf_sources import SOURCES

    source = next((s for s in SOURCES if s.doc_type == "datasheet" and s.turbine_model == farm.manufacturer_model), None)
    specs = load_specs().get(source.doc_id) if source else None
    if not specs:
        return None

    checks = []
    for field, configured in (("rated_power_kw", farm.rated_power_kw), ("rotor_diameter_m", farm.rotor_diameter_m)):
        entry = specs.get(field)
        stated = _as_float(entry["value"]) if entry and entry["verified"] else None
        if stated is None:
            continue
        if field == "rated_power_kw" and stated < 100:  # printed in MW
            stated *= 1000
        checks.append({
            "field": field,
            "datasheet": stated,
            "configured": configured,
            "page": entry["page"],
            "status": "match" if round(stated, 1) == round(configured, 1) else "differs",
        })
    return {
        "turbine_model": farm.manufacturer_model,
        "doc_id": source.doc_id,
        "title": source.title,
        "url": source.url,
        "checks": checks,
        "specs": specs,
    }
