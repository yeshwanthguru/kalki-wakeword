"""Save this repo's traffic, release downloads and star counts so the history outlives GitHub's 14-day window.

    python scripts/traffic_snapshot.py                  # snapshot the repo of the current folder
    python scripts/traffic_snapshot.py --repo owner/name --out my_history

Needs the GitHub CLI signed in as someone with push access (`gh auth login`): GitHub only shows traffic to maintainers.
Run it every week or so (it is safe to run more than once a day). Files are written as CSV to traffic_history/,
which is git-ignored, because visitor data is yours to keep, not something to publish.

    views.csv       one row per day: page views and unique visitors   (a day's row is updated until it is complete)
    clones.csv      one row per day: git clones and unique cloners
    referrers.csv   one row per snapshot per site: where visitors came from (rolling 14-day totals)
    paths.csv       one row per snapshot per page: what they looked at (rolling 14-day totals)
    downloads.csv   one row per snapshot per release file: total downloads so far
    stars.csv       one row per snapshot: stars, forks, watchers, open issues
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def gh_api(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"`gh api {path}` failed: {r.stderr.strip() or r.stdout.strip()}\n"
                         "Check `gh auth status`; traffic data needs push access to the repo.")
    return json.loads(r.stdout)


def detect_repo():
    r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit("Could not detect the repo. Run from inside the clone, or pass --repo owner/name.")
    return r.stdout.strip()


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def upsert_daily(existing, fresh, key="date"):
    """Merge GitHub's rolling 14-day daily rows into the saved history; the newest value wins for each day."""
    merged = {r[key]: dict(r) for r in existing}
    for r in fresh:
        merged[r[key]] = dict(r)
    return [merged[k] for k in sorted(merged)]


def daily_rows(items):
    return [{"date": i["timestamp"][:10], "count": i["count"], "uniques": i["uniques"]} for i in items]


def total(rows, col="count"):
    return sum(int(r[col]) for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="owner/name (default: the repo of the current folder)")
    ap.add_argument("--out", default="traffic_history", help="folder for the CSV files")
    a = ap.parse_args()
    repo = a.repo or detect_repo()
    out = Path(a.out)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    views = gh_api(f"repos/{repo}/traffic/views")
    clones = gh_api(f"repos/{repo}/traffic/clones")
    referrers = gh_api(f"repos/{repo}/traffic/popular/referrers")
    paths = gh_api(f"repos/{repo}/traffic/popular/paths")
    info = gh_api(f"repos/{repo}")
    releases = gh_api(f"repos/{repo}/releases?per_page=100")

    daily_fields = ["date", "count", "uniques"]
    v_hist = upsert_daily(read_rows(out / "views.csv"), daily_rows(views.get("views", [])))
    c_hist = upsert_daily(read_rows(out / "clones.csv"), daily_rows(clones.get("clones", [])))
    write_rows(out / "views.csv", daily_fields, v_hist)
    write_rows(out / "clones.csv", daily_fields, c_hist)

    ref_rows = read_rows(out / "referrers.csv") + [
        {"snapshot": stamp, "referrer": r["referrer"], "count": r["count"], "uniques": r["uniques"]} for r in referrers]
    write_rows(out / "referrers.csv", ["snapshot", "referrer", "count", "uniques"], ref_rows)
    path_rows = read_rows(out / "paths.csv") + [
        {"snapshot": stamp, "path": p["path"], "count": p["count"], "uniques": p["uniques"]} for p in paths]
    write_rows(out / "paths.csv", ["snapshot", "path", "count", "uniques"], path_rows)

    prev_dl = read_rows(out / "downloads.csv")
    dl_now = [{"snapshot": stamp, "tag": rel["tag_name"], "asset": asset["name"], "downloads": asset["download_count"]}
              for rel in releases for asset in rel.get("assets", [])]
    write_rows(out / "downloads.csv", ["snapshot", "tag", "asset", "downloads"], prev_dl + dl_now)

    prev_stars = read_rows(out / "stars.csv")
    star_row = {"snapshot": stamp, "stars": info["stargazers_count"], "forks": info["forks_count"],
                "watchers": info["subscribers_count"], "open_issues": info["open_issues_count"]}
    write_rows(out / "stars.csv", list(star_row), prev_stars + [star_row])

    # ---- summary
    last_snapshot = prev_dl[-1]["snapshot"] if prev_dl else None
    prev_total_dl = sum(int(r["downloads"]) for r in prev_dl if r["snapshot"] == last_snapshot) if prev_dl else None
    total_dl = sum(r["downloads"] for r in dl_now)
    print(f"\n{repo}  |  snapshot {stamp}  |  history in {out}/\n")
    print(f"Views    from {v_hist[0]['date'] if v_hist else 'n/a'}: {total(v_hist)} page views   "
          f"(last 14 days: {views.get('count', 0)} views, {views.get('uniques', 0)} unique visitors)")
    print(f"Clones   from {c_hist[0]['date'] if c_hist else 'n/a'}: {total(c_hist)} clones   "
          f"(last 14 days: {clones.get('count', 0)} clones, {clones.get('uniques', 0)} unique cloners)")
    print("  (unique visitors from different snapshots cannot be added up, so only the 14-day figure is shown)")
    change = "" if prev_total_dl is None else f"  ({total_dl - prev_total_dl:+d} since last snapshot)"
    print(f"Downloads of release files: {total_dl}{change}")
    for r in dl_now:
        print(f"    {r['tag']}  {r['asset']}: {r['downloads']}")
    delta = lambda col: "" if not prev_stars else f" ({star_row[col] - int(prev_stars[-1][col]):+d})"
    print(f"Stars    {star_row['stars']}{delta('stars')}   Forks {star_row['forks']}{delta('forks')}   "
          f"Watchers {star_row['watchers']}   Open issues {star_row['open_issues']}")
    print("Where visitors came from (last 14 days):")
    for r in referrers[:8]:
        print(f"    {r['referrer']}: {r['count']} views, {r['uniques']} unique")
    if not referrers:
        print("    (none yet)")
    print("Most viewed pages (last 14 days):")
    for p in paths[:5]:
        print(f"    {p['path']}: {p['count']} views, {p['uniques']} unique")
    if not paths:
        print("    (none yet)")


if __name__ == "__main__":
    sys.exit(main())
