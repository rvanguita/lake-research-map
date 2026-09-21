"""Provision least-privilege runtime users for the shared MySQL databases.

Only tables whose names start with ``lit_`` are considered.  MySQL does not
support prefix-scoped table grants, so this command must be rerun after the
bootstrap creates a new project table.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text

from lake_research_map.config import LAYERS, get_settings


def _safe_identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"unsafe MySQL identifier {value!r}")
    return value


def provision_runtime_roles() -> dict[str, int]:
    """Create/update pipeline and dashboard users and grant only ``lit_*`` tables."""
    pipeline_user = _safe_identifier(os.environ["MYSQL_PIPELINE_USER"])
    dashboard_user = _safe_identifier(os.environ["MYSQL_DASHBOARD_USER"])
    pipeline_password = os.environ["MYSQL_PIPELINE_PASSWORD"]
    dashboard_password = os.environ["MYSQL_DASHBOARD_PASSWORD"]
    settings = get_settings()
    engine = create_engine(settings.server_url(), future=True)
    counts = {"pipeline_tables": 0, "dashboard_tables": 0}
    with engine.begin() as conn:
        conn.execute(
            text(f"CREATE USER IF NOT EXISTS `{pipeline_user}`@'%' IDENTIFIED BY :password"),
            {"password": pipeline_password},
        )
        conn.execute(
            text(f"ALTER USER `{pipeline_user}`@'%' IDENTIFIED BY :password"),
            {"password": pipeline_password},
        )
        conn.execute(
            text(f"CREATE USER IF NOT EXISTS `{dashboard_user}`@'%' IDENTIFIED BY :password"),
            {"password": dashboard_password},
        )
        conn.execute(
            text(f"ALTER USER `{dashboard_user}`@'%' IDENTIFIED BY :password"),
            {"password": dashboard_password},
        )
        for layer in LAYERS:
            database = settings.database_name(layer)
            table_names = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :database AND table_name LIKE 'lit\\_%'"
                ),
                {"database": database},
            ).scalars()
            for table_name in table_names:
                table_name = _safe_identifier(table_name)
                conn.execute(
                    text(
                        f"GRANT SELECT, INSERT, UPDATE, DELETE ON `{database}`.`{table_name}` "
                        f"TO `{pipeline_user}`@'%'"
                    )
                )
                counts["pipeline_tables"] += 1
                conn.execute(
                    text(f"GRANT SELECT ON `{database}`.`{table_name}` TO `{dashboard_user}`@'%'")
                )
                counts["dashboard_tables"] += 1
    engine.dispose()
    return counts


def main() -> None:
    print(provision_runtime_roles())


if __name__ == "__main__":
    main()
