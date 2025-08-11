import re
import unicodedata

def sanitize_collection_name(name: str) -> str:
    # Normalize Unicode (remove fancy characters like “‐”)
    name = unicodedata.normalize("NFKD", name)
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove disallowed characters
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    # Ensure it starts/ends with alphanumeric
    name = re.sub(r"^[^a-zA-Z0-9]+", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+$", "", name)
    # Truncate if too long
    return name[:100]

def sanitize_text(text: str) -> str:
    """
    Remove problematic characters from text that would break PostgreSQL.
    """
    if not isinstance(text, str):
        return ""
    # Remove NULL bytes and trim
    return text.replace("\x00", "").strip()

