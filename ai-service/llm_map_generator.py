import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

from fastapi import HTTPException

DEFAULT_OPENAI_MAP_MODEL = "gpt-5.6-terra"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def configured_openai_map_model() -> str:
    return os.getenv("OPENAI_MAP_MODEL", DEFAULT_OPENAI_MAP_MODEL)


def venue_map_json_schema(target_map_id: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "name", "image", "width", "height", "nodes", "checkpoints", "edges"],
        "properties": {
            "id": {"type": "string", "const": target_map_id},
            "name": {"type": "string"},
            "image": {"type": "string"},
            "width": {"type": "number", "exclusiveMinimum": 0},
            "height": {"type": "number", "exclusiveMinimum": 0},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "name", "x", "y", "type", "selectable"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "type": {
                            "type": "string",
                            "enum": ["entrance", "exit", "junction", "facility", "booth"],
                        },
                        "selectable": {"type": "boolean"},
                    },
                },
            },
            "checkpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "name", "node_id", "region"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "node_id": {"type": "string"},
                        "region": {"type": "string", "enum": ["lobby", "booth"]},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "id",
                        "from",
                        "to",
                        "distance",
                        "widthM",
                        "zone",
                        "crowdRegion",
                        "bidirectional",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "distance": {"type": "number", "exclusiveMinimum": 0},
                        "widthM": {"type": "number", "exclusiveMinimum": 0},
                        "zone": {
                            "type": "string",
                            "enum": ["lobby", "booth", "gate", "facility"],
                        },
                        "crowdRegion": {
                            "type": "string",
                            "enum": ["west", "central", "north", "booth"],
                        },
                        "bidirectional": {"type": "boolean"},
                    },
                },
            },
        },
    }


def source_image_data_url(job: dict[str, Any]) -> str:
    with open(job["source_image_path"], "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode("ascii")
    return f"data:{job['source_mime_type']};base64,{encoded_image}"


def build_map_generation_prompt(
    job: dict[str, Any],
    extra_instructions: str | None = None,
) -> str:
    notes = job.get("notes") or "No extra notes."
    extra = extra_instructions or "No extra instructions."
    return (
        "You are generating an indoor navigation map JSON from a floorplan image.\n"
        "Return only JSON matching the provided schema.\n"
        f"Map id must be exactly {job['target_map_id']}.\n"
        f"Use image path {job['source_image_url']} in the image field.\n"
        "Create nodes for entrances, exits, facilities, booth aisles, junctions, and destinations.\n"
        "Create checkpoints for QR-readable positions when sensible.\n"
        "Create bidirectional edges for walkable corridors only; do not route through walls or booths.\n"
        "Use pixel coordinates in the source image coordinate system.\n"
        "Prefer a connected graph between all selectable nodes.\n"
        "Use Korean names if labels are visible in Korean.\n"
        f"Operator notes: {notes}\n"
        f"Extra instructions: {extra}"
    )


def extract_response_text(response_data: dict[str, Any]) -> str:
    if isinstance(response_data.get("output_text"), str):
        return response_data["output_text"]

    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]

    raise HTTPException(
        status_code=502,
        detail="OpenAI response did not contain text output.",
    )


def request_openai_map_draft(
    job: dict[str, Any],
    model: str,
    image_detail: str,
    extra_instructions: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENAI_API_KEY is not configured. Use /draft-map manually or set "
                "OPENAI_API_KEY before calling generate-draft."
            ),
        )

    request_body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_map_generation_prompt(job, extra_instructions),
                    },
                    {
                        "type": "input_image",
                        "image_url": source_image_data_url(job),
                        "detail": image_detail,
                    },
                ],
            }
        ],
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "venue_map",
                "schema": venue_map_json_schema(job["target_map_id"]),
                "strict": False,
            }
        },
    }

    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error: {error.code} {error_body}",
        )
    except urllib.error.URLError as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API request failed: {error.reason}",
        )

    try:
        draft_map = json.loads(extract_response_text(response_data))
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI response was not valid JSON: {error}",
        )

    if not isinstance(draft_map, dict):
        raise HTTPException(
            status_code=502,
            detail="OpenAI response JSON must be an object.",
        )

    draft_map["id"] = job["target_map_id"]
    draft_map["image"] = job["source_image_url"]
    return draft_map
