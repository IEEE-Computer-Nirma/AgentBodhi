import json
import re


def clean_query(q: str) -> str:
    q = re.sub(r"[\*\[\]\(\)\"\'`]", '', q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()


def extract_json(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError('No valid JSON found')


