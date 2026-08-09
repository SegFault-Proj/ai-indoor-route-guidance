import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

Region = Literal["lobby", "booth"]

DB_PATH = Path(__file__).with_name("crowd_signals.sqlite3")
EVENT_WINDOW_MINUTES = 10
DEFAULT_LOBBY_PEOPLE = 25
DEFAULT_BOOTH_PEOPLE = 55
DEFAULT_EVENT_PHASE = 2


def current_utc() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_text(value: datetime) -> str:
    return value.isoformat()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_crowd_store():
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS qr_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    map_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS route_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    map_id TEXT NOT NULL,
                    destination_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_estimates (
                    map_id TEXT PRIMARY KEY,
                    lobby_people INTEGER NOT NULL,
                    booth_people INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS navigation_sessions (
                    session_id TEXT PRIMARY KEY,
                    map_id TEXT NOT NULL,
                    start_node_id TEXT NOT NULL,
                    current_node_id TEXT NOT NULL,
                    destination_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS navigation_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    checkpoint_id TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )


def prune_events(now: datetime | None = None):
    init_crowd_store()
    now = now or current_utc()
    cutoff = timestamp_text(now - timedelta(minutes=EVENT_WINDOW_MINUTES))

    with closing(connect()) as connection:
        with connection:
            connection.execute("DELETE FROM qr_events WHERE timestamp < ?", (cutoff,))
            connection.execute("DELETE FROM route_intents WHERE timestamp < ?", (cutoff,))


def record_qr_scan(map_id: str, checkpoint_id: str, region: Region, count: int):
    init_crowd_store()
    prune_events()
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO qr_events (map_id, checkpoint_id, region, count, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (map_id, checkpoint_id, region, count, timestamp_text(current_utc())),
            )


def record_route_intent(map_id: str, destination_id: str, region: Region):
    init_crowd_store()
    prune_events()
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO route_intents (map_id, destination_id, region, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (map_id, destination_id, region, timestamp_text(current_utc())),
            )


def set_manual_estimate(map_id: str, lobby_people: int, booth_people: int):
    init_crowd_store()
    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO manual_estimates (
                    map_id,
                    lobby_people,
                    booth_people,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(map_id) DO UPDATE SET
                    lobby_people = excluded.lobby_people,
                    booth_people = excluded.booth_people,
                    updated_at = excluded.updated_at
                """,
                (map_id, lobby_people, booth_people, timestamp_text(current_utc())),
            )


def region_for_destination(destination_id: str) -> Region:
    if destination_id.startswith("BOOTH_"):
        return "booth"
    return "lobby"


def region_for_node(node_id: str) -> Region:
    if node_id.startswith("BOOTH_"):
        return "booth"
    return "lobby"


def manual_estimate(map_id: str) -> dict[str, int]:
    init_crowd_store()
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT lobby_people, booth_people
            FROM manual_estimates
            WHERE map_id = ?
            """,
            (map_id,),
        ).fetchone()

    if row is None:
        return {
            "lobby_people": DEFAULT_LOBBY_PEOPLE,
            "booth_people": DEFAULT_BOOTH_PEOPLE,
        }

    return {
        "lobby_people": int(row["lobby_people"]),
        "booth_people": int(row["booth_people"]),
    }


def count_qr_scans(map_id: str, region: Region) -> int:
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(count), 0) AS total
            FROM qr_events
            WHERE map_id = ? AND region = ?
            """,
            (map_id, region),
        ).fetchone()
    return int(row["total"])


def count_route_intents(map_id: str, region: Region) -> int:
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM route_intents
            WHERE map_id = ? AND region = ?
            """,
            (map_id, region),
        ).fetchone()
    return int(row["total"])


def count_active_navigation_sessions(map_id: str, region: Region) -> int:
    cutoff = timestamp_text(current_utc() - timedelta(minutes=EVENT_WINDOW_MINUTES))
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM navigation_sessions
            WHERE map_id = ?
              AND status = 'active'
              AND updated_at >= ?
              AND (
                  CASE
                      WHEN destination_id LIKE 'BOOTH_%' THEN 'booth'
                      ELSE 'lobby'
                  END
              ) = ?
            """,
            (map_id, cutoff, region),
        ).fetchone()
    return int(row["total"])


def count_recent_navigation_updates(map_id: str, region: Region) -> int:
    cutoff = timestamp_text(current_utc() - timedelta(minutes=EVENT_WINDOW_MINUTES))
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM navigation_updates AS updates
            INNER JOIN navigation_sessions AS sessions
                ON sessions.session_id = updates.session_id
            WHERE sessions.map_id = ?
              AND updates.timestamp >= ?
              AND (
                  CASE
                      WHEN updates.node_id LIKE 'BOOTH_%' THEN 'booth'
                      ELSE 'lobby'
                  END
              ) = ?
            """,
            (map_id, cutoff, region),
        ).fetchone()
    return int(row["total"])


def recent_qr_events(map_id: str, limit: int) -> list[dict[str, Any]]:
    init_crowd_store()
    prune_events()
    with closing(connect()) as connection:
        rows = connection.execute(
            """
            SELECT checkpoint_id, region, count, timestamp
            FROM qr_events
            WHERE map_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (map_id, limit),
        ).fetchall()

    return [
        {
            "checkpoint_id": row["checkpoint_id"],
            "region": row["region"],
            "count": int(row["count"]),
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def recent_route_intents(map_id: str, limit: int) -> list[dict[str, Any]]:
    init_crowd_store()
    prune_events()
    with closing(connect()) as connection:
        rows = connection.execute(
            """
            SELECT destination_id, region, timestamp
            FROM route_intents
            WHERE map_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (map_id, limit),
        ).fetchall()

    return [
        {
            "destination_id": row["destination_id"],
            "region": row["region"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def telemetry_snapshot(map_id: str, limit: int = 20) -> dict[str, Any]:
    return {
        "map_id": map_id,
        "crowd_estimate": estimate_crowd_inputs(
            map_id=map_id,
            hour=current_utc().hour,
            event_phase=DEFAULT_EVENT_PHASE,
        ),
        "recent_qr_scans": recent_qr_events(map_id, limit),
        "recent_route_intents": recent_route_intents(map_id, limit),
    }


def create_navigation_session(
    map_id: str,
    start_node_id: str,
    destination_id: str,
) -> dict[str, Any]:
    init_crowd_store()
    session_id = uuid4().hex
    now = timestamp_text(current_utc())
    status = "arrived" if start_node_id == destination_id else "active"

    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO navigation_sessions (
                    session_id,
                    map_id,
                    start_node_id,
                    current_node_id,
                    destination_id,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    map_id,
                    start_node_id,
                    start_node_id,
                    destination_id,
                    status,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO navigation_updates (
                    session_id,
                    node_id,
                    source,
                    checkpoint_id,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, start_node_id, "start", None, now),
            )

    return get_navigation_session(session_id)


def update_navigation_session_position(
    session_id: str,
    node_id: str,
    source: str,
    checkpoint_id: str | None = None,
) -> dict[str, Any] | None:
    init_crowd_store()
    session = get_navigation_session(session_id)
    if session is None:
        return None

    now = timestamp_text(current_utc())
    status = "arrived" if node_id == session["destination_id"] else "active"

    with closing(connect()) as connection:
        with connection:
            connection.execute(
                """
                UPDATE navigation_sessions
                SET current_node_id = ?, status = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (node_id, status, now, session_id),
            )
            connection.execute(
                """
                INSERT INTO navigation_updates (
                    session_id,
                    node_id,
                    source,
                    checkpoint_id,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, node_id, source, checkpoint_id, now),
            )

    return get_navigation_session(session_id)


def get_navigation_session(session_id: str) -> dict[str, Any] | None:
    init_crowd_store()
    with closing(connect()) as connection:
        row = connection.execute(
            """
            SELECT
                session_id,
                map_id,
                start_node_id,
                current_node_id,
                destination_id,
                status,
                created_at,
                updated_at
            FROM navigation_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "session_id": row["session_id"],
        "map_id": row["map_id"],
        "start_node_id": row["start_node_id"],
        "current_node_id": row["current_node_id"],
        "destination_id": row["destination_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def recent_navigation_updates(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    init_crowd_store()
    with closing(connect()) as connection:
        rows = connection.execute(
            """
            SELECT node_id, source, checkpoint_id, timestamp
            FROM navigation_updates
            WHERE session_id = ?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    return [
        {
            "node_id": row["node_id"],
            "source": row["source"],
            "checkpoint_id": row["checkpoint_id"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def navigation_updates_chronological(
    session_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    init_crowd_store()
    with closing(connect()) as connection:
        rows = connection.execute(
            """
            SELECT node_id, source, checkpoint_id, timestamp
            FROM navigation_updates
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    return [
        {
            "node_id": row["node_id"],
            "source": row["source"],
            "checkpoint_id": row["checkpoint_id"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def estimate_crowd_inputs(map_id: str, hour: int, event_phase: int) -> dict[str, Any]:
    init_crowd_store()
    prune_events()

    manual = manual_estimate(map_id)
    recent_qr_lobby = count_qr_scans(map_id, "lobby")
    recent_qr_booth = count_qr_scans(map_id, "booth")
    recent_route_lobby = count_route_intents(map_id, "lobby")
    recent_route_booth = count_route_intents(map_id, "booth")
    active_nav_lobby = count_active_navigation_sessions(map_id, "lobby")
    active_nav_booth = count_active_navigation_sessions(map_id, "booth")
    recent_nav_lobby = count_recent_navigation_updates(map_id, "lobby")
    recent_nav_booth = count_recent_navigation_updates(map_id, "booth")

    lobby_people = min(
        1000,
        manual["lobby_people"]
        + recent_qr_lobby
        + round(recent_route_lobby * 0.6)
        + round(active_nav_lobby * 0.7)
        + round(recent_nav_lobby * 0.25),
    )
    booth_people = min(
        1000,
        manual["booth_people"]
        + recent_qr_booth
        + round(recent_route_booth * 0.8)
        + round(active_nav_booth * 0.9)
        + round(recent_nav_booth * 0.3),
    )
    recent_inflow = min(
        1000,
        recent_qr_lobby
        + recent_qr_booth
        + recent_route_lobby
        + recent_route_booth
        + active_nav_lobby
        + active_nav_booth
        + recent_nav_lobby
        + recent_nav_booth,
    )

    return {
        "map_id": map_id,
        "lobby_people": lobby_people,
        "booth_people": booth_people,
        "recent_inflow": recent_inflow,
        "hour": hour,
        "event_phase": event_phase,
        "signals": {
            "window_minutes": EVENT_WINDOW_MINUTES,
            "qr_scans": {
                "lobby": recent_qr_lobby,
                "booth": recent_qr_booth,
            },
            "route_intents": {
                "lobby": recent_route_lobby,
                "booth": recent_route_booth,
            },
            "navigation_sessions": {
                "lobby": active_nav_lobby,
                "booth": active_nav_booth,
            },
            "navigation_updates": {
                "lobby": recent_nav_lobby,
                "booth": recent_nav_booth,
            },
            "manual_estimate": manual,
        },
    }


def reset_crowd_store():
    init_crowd_store()
    with closing(connect()) as connection:
        with connection:
            connection.execute("DELETE FROM qr_events")
            connection.execute("DELETE FROM route_intents")
            connection.execute("DELETE FROM manual_estimates")
            connection.execute("DELETE FROM navigation_updates")
            connection.execute("DELETE FROM navigation_sessions")


init_crowd_store()
