import os
import sqlite3


class DocStore:
    """
    Parent chunk'ları embed etmeden id -> text olarak saklayan
    basit, SQLite tabanlı, disk'e persist eden key-value store.
    """

    def __init__(self, persist_path: str | None = None):
        if persist_path is None:
            try:
                from config_loader import load_appcfg, load_retcfg

                ret_cfg = load_retcfg()
                app_cfg = load_appcfg()
                persist_path = (
                    getattr(ret_cfg, "docstore", {}).get("persist_path")
                    or getattr(app_cfg, "paths", {}).get("docstore_path")
                    or "./docstore.db"
                )
            except Exception:
                persist_path = "./docstore.db"

        self.persist_path = persist_path
        dirname = os.path.dirname(self.persist_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with sqlite3.connect(self.persist_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            conn.commit()

    def add(self, ids: list[str], texts: list[str]):
        with sqlite3.connect(self.persist_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)",
                zip(ids, texts)
            )
            conn.commit()

    def get(self, id_: str) -> str | None:
        with sqlite3.connect(self.persist_path) as conn:
            cursor = conn.execute("SELECT text FROM docs WHERE id = ?", (id_,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_many(self, ids: list[str]) -> dict[str, str]:
        results = {}
        with sqlite3.connect(self.persist_path) as conn:
            for i in range(0, len(ids), 999):
                chunk = ids[i:i + 999]
                placeholders = ",".join(["?"] * len(chunk))
                cursor = conn.execute(
                    f"SELECT id, text FROM docs WHERE id IN ({placeholders})", chunk
                )
                for row_id, text in cursor.fetchall():
                    results[row_id] = text
        return results

    def count(self) -> int:
        with sqlite3.connect(self.persist_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM docs")
            return cursor.fetchone()[0]