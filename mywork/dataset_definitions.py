from datasets import load_dataset

COLUMN_DEF_DS = load_dataset("jessieszec/column_definition_final_v")["train"]

COLUMN_DEFINITION_MAP: dict[str, dict] = {}

for row in COLUMN_DEF_DS:
    # Try multiple possible field names
    raw_name = row.get("column_name") or row.get("column") or ""
    name = raw_name.strip() if isinstance(raw_name, str) else ""

    # Skip rows without a valid column name
    if not name:
        continue

    COLUMN_DEFINITION_MAP[name] = {
        "definition": (row.get("definition") or "").strip(),
        "category": (row.get("category") or "").strip(),
        "example": (row.get("example") or "").strip(),
    }


def get_column_definition(column_name: str):
    """
    Return a structured definition for a dataset column.
    """
    col = column_name.strip().lower()

    # Exact match (case-insensitive)
    for key in COLUMN_DEFINITION_MAP:
        if key.lower() == col:
            return {
                "column_name": key,
                **COLUMN_DEFINITION_MAP[key],
            }

    # Fuzzy contains
    for key in COLUMN_DEFINITION_MAP:
        if col in key.lower():
            return {
                "column_name": key,
                **COLUMN_DEFINITION_MAP[key],
            }

    return {
        "column_name": column_name,
        "definition": "No definition found for this column.",
        "category": None,
        "example": None,
    }
