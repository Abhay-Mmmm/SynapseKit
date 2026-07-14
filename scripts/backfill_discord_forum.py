#!/usr/bin/env python3
"""Backfill open GitHub issues + PRs into a SINGLE Discord forum channel,
creating one thread per label and posting all matching issues inside.

Usage:
    DISCORD_FORUM_WEBHOOK=https://discord.com/api/webhooks/.../... \
        python scripts/backfill_discord_forum.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = os.environ.get("BACKFILL_REPO", "SynapseKit/SynapseKit")
WEBHOOK = os.environ["DISCORD_FORUM_WEBHOOK"]
STATE_FILE = Path(".discord_forum_state.json")
SLEEP = 1.2

UA = "SynapseKit-Backfill (https://github.com/SynapseKit/SynapseKit, 1.0)"

LABEL_CONFIG: dict[str, dict[str, Any]] = {
    "v2.0":          {"thread": "🚀 v2.0 — The Futuristic Leap",        "emoji": "🚀",  "color": 10038562},
    "v2.1":          {"thread": "🧠 v2.1 — Local-First Knowledge Mesh", "emoji": "🧠",  "color": 5793266},
    "post-v2.0":     {"thread": "📚 post-v2.0 — Content & Benchmarks",  "emoji": "📚",  "color": 15105570},
    "v1.8":          {"thread": "🛠 v1.8 — Current Release",            "emoji": "🛠",  "color": 3447003},
    "research":      {"thread": "🔬 research",                          "emoji": "🔬",  "color": 10181046},
    "benchmark":     {"thread": "📊 benchmark",                         "emoji": "📊",  "color": 3066993},
    "agents":        {"thread": "🤖 agents",                            "emoji": "🤖",  "color": 15844367},
    "retrieval":     {"thread": "🔍 retrieval",                         "emoji": "🔍",  "color": 1752220},
    "observability": {"thread": "📈 observability",                     "emoji": "📈",  "color": 2123412},
    "security":      {"thread": "🔒 security",                          "emoji": "🔒",  "color": 15158332},
    "performance":   {"thread": "⚡ performance",                       "emoji": "⚡",  "color": 16776960},
    "ui":            {"thread": "🎨 ui",                                "emoji": "🎨",  "color": 15418782},
    "book":          {"thread": "📖 book",                              "emoji": "📖",  "color": 8421504},
    "paper":         {"thread": "📝 paper",                             "emoji": "📝",  "color": 9807270},
    "documentation": {"thread": "📘 documentation",                     "emoji": "📘",  "color": 3447003},
    "ambient":       {"thread": "🌌 ambient",                           "emoji": "🌌",  "color": 7506394},
    "protocol":      {"thread": "🔗 protocol",                          "emoji": "🔗",  "color": 1942002},
    "community":     {"thread": "👥 community",                         "emoji": "👥",  "color": 5763719},
}


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"threads": {}, "posted": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fetch_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for cmd_kind, kind_label in (("issue", "Issue"), ("pr", "PR")):
        out = subprocess.check_output([
            "gh", cmd_kind, "list", "--repo", REPO,
            "--state", "open", "--limit", "500",
            "--json", "number,title,labels,url,author",
        ])
        for item in json.loads(out):
            item["kind"] = kind_label
            items.append(item)
    return items


def http_post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body) if body else {}
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, body


def embed_for(item: dict[str, Any], label: str, cfg: dict[str, Any]) -> dict[str, Any]:
    author = (item.get("author") or {}).get("login") or "anonymous"
    labels = ", ".join(label_["name"] for label_ in item.get("labels", [])) or "—"
    return {
        "title": f"{cfg['emoji']} {item['kind']} #{item['number']}",
        "description": f"**{item['title']}**\n\n_labels:_ {labels}\n\n[View on GitHub]({item['url']})",
        "color": cfg["color"],
        "footer": {"text": f"by {author}"},
    }


def create_thread(label: str, cfg: dict[str, Any], first: dict[str, Any]) -> str | None:
    url = WEBHOOK + "?wait=true"
    payload = {
        "thread_name": cfg["thread"][:100],
        "embeds": [embed_for(first, label, cfg)],
    }
    status, body = http_post(url, payload)
    if 200 <= status < 300 and isinstance(body, dict):
        thread_id = body.get("channel_id") or body.get("id")
        return str(thread_id) if thread_id else None
    print(f"  ✗ create thread '{cfg['thread']}' HTTP {status}: {str(body)[:200]}")
    return None


def post_to_thread(thread_id: str, item: dict[str, Any], label: str, cfg: dict[str, Any]) -> bool:
    url = f"{WEBHOOK}?thread_id={urllib.parse.quote(thread_id)}&wait=true"
    payload = {"embeds": [embed_for(item, label, cfg)]}
    status, body = http_post(url, payload)
    if 200 <= status < 300:
        return True
    print(f"  ✗ post #{item['number']} to thread {thread_id} HTTP {status}: {str(body)[:200]}")
    return False


def main() -> int:
    state = load_state()
    threads: dict[str, str] = state["threads"]
    posted: set[str] = set(state["posted"])

    items = fetch_items()
    print(f"Found {len(items)} open items.")

    by_label: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        for label_obj in item.get("labels", []):
            name = label_obj["name"]
            if name in LABEL_CONFIG:
                by_label.setdefault(name, []).append(item)

    for label, group in by_label.items():
        cfg = LABEL_CONFIG[label]
        # Sort by issue number ascending so threads read chronologically
        group.sort(key=lambda x: x["number"])
        print(f"\n=== {cfg['thread']} ({len(group)} items) ===")

        thread_id = threads.get(label)
        i = 0
        if thread_id is None:
            first = group[0]
            print(f"  → creating thread with #{first['number']}")
            thread_id = create_thread(label, cfg, first)
            if thread_id is None:
                continue
            threads[label] = thread_id
            posted.add(f"{label}::{first['number']}")
            i = 1
            state["threads"] = threads
            state["posted"] = sorted(posted)
            save_state(state)
            time.sleep(SLEEP)

        for item in group[i:]:
            key = f"{label}::{item['number']}"
            if key in posted:
                continue
            if post_to_thread(thread_id, item, label, cfg):
                print(f"  ✓ #{item['number']}")
                posted.add(key)
                state["posted"] = sorted(posted)
                save_state(state)
            time.sleep(SLEEP)

    print(f"\nDone. {len(posted)} total posts across {len(threads)} threads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
