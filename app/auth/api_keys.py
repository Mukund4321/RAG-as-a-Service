import secrets
import hashlib
from datetime import datetime
from app.database import db_cursor
from app.models import new_id


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_api_key(tenant_name: str) -> dict:
    raw_key = secrets.token_urlsafe(32)
    hashed = _hash_key(raw_key)
    tenant_id = new_id()

    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO tenants (tenant_id, tenant_name, api_key) VALUES (?,?,?)",
            (tenant_id, tenant_name, hashed),
        )

    return {
        "api_key": raw_key,
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "created_at": datetime.utcnow(),
    }


def validate_api_key(raw_key: str) -> dict | None:
    hashed = _hash_key(raw_key)
    with db_cursor() as cur:
        cur.execute(
            "SELECT tenant_id, tenant_name FROM tenants WHERE api_key=?",
            (hashed,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"tenant_id": row["tenant_id"], "tenant_name": row["tenant_name"]}
