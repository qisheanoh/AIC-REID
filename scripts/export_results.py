from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path("data/retail.db")

def q(sql: str, args=()):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(sql, args)
    rows = cur.fetchall()
    conn.close()
    return rows

def print_zone_summary(cam: str):
    rows = q("""
    SELECT zone_id,
           COUNT(*) AS visits,
           ROUND(SUM(dwell_s),2) AS total_dwell,
           ROUND(AVG(dwell_s),2) AS avg_dwell
    FROM events
    WHERE camera_id=?
    GROUP BY zone_id
    ORDER BY zone_id;
    """, (cam,))
    print(f"\n=== Zone summary ({cam}) ===")
    for r in rows:
        print(f"{r[0]:15s} visits={r[1]:3d} total_dwell={r[2]:7.2f}s avg={r[3]:5.2f}s")

def print_person_summary(cam: str):
    rows = q("""
    SELECT global_id,
           COUNT(*) AS visits,
           ROUND(SUM(dwell_s),2) AS total_dwell
    FROM events
    WHERE camera_id=?
    GROUP BY global_id
    ORDER BY global_id;
    """, (cam,))
    print(f"\n=== Person summary ({cam}) ===")
    for r in rows:
        print(f"gid={r[0]} visits={r[1]:3d} total_dwell={r[2]:7.2f}s")

def print_ab(camA: str, camB: str):
    A = q("SELECT ROUND(SUM(dwell_s),2), COUNT(*) FROM events WHERE camera_id=?;", (camA,))
    B = q("SELECT ROUND(SUM(dwell_s),2), COUNT(*) FROM events WHERE camera_id=?;", (camB,))
    a_dwell, a_vis = (A[0][0] or 0.0), (A[0][1] or 0)
    b_dwell, b_vis = (B[0][0] or 0.0), (B[0][1] or 0)
    print("\n=== A/B comparison ===")
    print(f"{camA}: total_dwell={a_dwell:.2f}s visits={a_vis}")
    print(f"{camB}: total_dwell={b_dwell:.2f}s visits={b_vis}")

if __name__ == "__main__":
    print_zone_summary("cam1")
    print_person_summary("cam1")
    print_zone_summary("cam1_hot")
    print_person_summary("cam1_hot")
    print_ab("cam1", "cam1_hot")
