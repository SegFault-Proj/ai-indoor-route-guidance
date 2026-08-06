import base64
import binascii
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from map_store import (
    PROTECTED_MAP_IDS,
    save_map_data,
    validate_map_data,
    validate_map_id,
)

JOB_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
MAX_SOURCE_IMAGE_BYTES = 8 * 1024 * 1024

GENERATION_JOBS_DIR = Path(__file__).with_name("map_generation_jobs")
GENERATED_ASSETS_DIR = Path(__file__).with_name("generated_assets")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_generation_store():
    GENERATION_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def validate_job_id(job_id: str):
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="Invalid generation job id.")


def job_file_path(job_id: str) -> Path:
    validate_job_id(job_id)
    path = (GENERATION_JOBS_DIR / f"{job_id}.json").resolve()
    if path.parent != GENERATION_JOBS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid generation job path.")
    return path


def draft_file_path(job_id: str) -> Path:
    validate_job_id(job_id)
    path = (GENERATION_JOBS_DIR / f"{job_id}.draft.json").resolve()
    if path.parent != GENERATION_JOBS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid draft map path.")
    return path


def decode_source_image(source_image_base64: str, mime_type: str) -> tuple[bytes, str]:
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="mime_type must be image/png or image/jpeg.",
        )

    try:
        image_bytes = base64.b64decode(source_image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="source_image_base64 is invalid.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="source image is empty.")
    if len(image_bytes) > MAX_SOURCE_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="source image must be 8MB or smaller.",
        )

    return image_bytes, ALLOWED_IMAGE_TYPES[mime_type]


def source_image_path(job_id: str, extension: str) -> Path:
    validate_job_id(job_id)
    path = (GENERATED_ASSETS_DIR / f"{job_id}{extension}").resolve()
    if path.parent != GENERATED_ASSETS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid source image path.")
    return path


def persist_job(job: dict[str, Any]):
    init_generation_store()
    with job_file_path(job["job_id"]).open("w", encoding="utf-8") as file:
        json.dump(job, file, ensure_ascii=False, indent=2)
        file.write("\n")


def create_generation_job(
    source_image_base64: str,
    filename: str,
    mime_type: str,
    target_map_id: str,
    notes: str | None = None,
) -> dict[str, Any]:
    validate_map_id(target_map_id)
    if target_map_id in PROTECTED_MAP_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Protected map cannot be used as a generation target: {target_map_id}",
        )
    image_bytes, extension = decode_source_image(source_image_base64, mime_type)
    job_id = uuid4().hex
    init_generation_store()

    image_path = source_image_path(job_id, extension)
    image_path.write_bytes(image_bytes)

    now = utc_now()
    job = {
        "job_id": job_id,
        "status": "needs_llm",
        "target_map_id": target_map_id,
        "source_filename": filename,
        "source_mime_type": mime_type,
        "source_image_path": str(image_path),
        "source_image_url": f"/generated-assets/{image_path.name}",
        "notes": notes,
        "created_at": now,
        "updated_at": now,
        "validation": None,
        "saved_map_id": None,
        "next_step": (
            "Call an LLM/vision model with the source image and /maps/schema, "
            "then submit the generated map JSON to this job's draft endpoint."
        ),
    }
    persist_job(job)
    return job


def get_generation_job(job_id: str) -> dict[str, Any]:
    path = job_file_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Map generation job not found: {job_id}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_generation_jobs(limit: int = 20) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    init_generation_store()
    jobs: list[dict[str, Any]] = []
    for path in GENERATION_JOBS_DIR.glob("*.json"):
        if path.name.endswith(".draft.json"):
            continue
        with path.open("r", encoding="utf-8") as file:
            jobs.append(json.load(file))

    jobs.sort(key=lambda job: job["updated_at"], reverse=True)
    return jobs[:limit]


def get_draft_map(job_id: str) -> dict[str, Any]:
    get_generation_job(job_id)
    path = draft_file_path(job_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Draft map not found for generation job: {job_id}",
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def attach_draft_map(job_id: str, venue_map: dict[str, Any]) -> dict[str, Any]:
    job = get_generation_job(job_id)
    venue_map["id"] = job["target_map_id"]
    validation = {
        "map_id": venue_map.get("id"),
        **validate_map_data(venue_map),
    }

    with draft_file_path(job_id).open("w", encoding="utf-8") as file:
        json.dump(venue_map, file, ensure_ascii=False, indent=2)
        file.write("\n")

    job["status"] = "draft_valid" if validation["valid"] else "draft_invalid"
    job["validation"] = validation
    job["updated_at"] = utc_now()
    job["next_step"] = (
        f"Save the draft map through /map-generation/jobs/{job_id}/save-map."
        if validation["valid"]
        else "Fix the draft map JSON and submit it again."
    )
    persist_job(job)
    return job


def save_generation_job_map(job_id: str, overwrite: bool = False) -> dict[str, Any]:
    job = get_generation_job(job_id)
    draft_path = draft_file_path(job_id)
    if not draft_path.exists():
        raise HTTPException(status_code=400, detail="No draft map is attached to this job.")

    with draft_path.open("r", encoding="utf-8") as file:
        venue_map = json.load(file)

    validation = validate_map_data(venue_map)
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Draft map is invalid.",
                "validation": {"map_id": venue_map.get("id"), **validation},
            },
        )

    save_result = save_map_data(venue_map, overwrite=overwrite)
    job["status"] = "map_saved"
    job["validation"] = save_result["validation"]
    job["saved_map_id"] = save_result["map_id"]
    job["updated_at"] = utc_now()
    job["next_step"] = "Use the saved map id in /route and navigation APIs."
    persist_job(job)

    return {
        "job": job,
        "save_result": save_result,
    }


def reset_generation_store():
    for directory in [GENERATION_JOBS_DIR, GENERATED_ASSETS_DIR]:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
