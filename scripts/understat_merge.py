import argparse
import sys
import pandas as pd
import soccerdata as sd

CROSSWALK = {
    "ENG-Premier League": {
        "Bournemouth": 349, "Arsenal": 359, "Aston Villa": 362, "Brentford": 337,
        "Brighton": 331, "Burnley": 379, "Chelsea": 363, "Crystal Palace": 384,
        "Everton": 368, "Fulham": 370, "Ipswich": 373, "Leeds": 357,
        "Leicester": 375, "Liverpool": 364, "Manchester City": 382,
        "Manchester United": 360, "Newcastle United": 361,
        "Nottingham Forest": 393, "Southampton": 376, "Sunderland": 366,
        "Tottenham": 367, "West Ham": 371, "Wolverhampton Wanderers": 380,
    },
    # "ESP-La Liga": { ... },  # fill from your teams.csv when you expand
    # "GER-Bundesliga": { ... },
    # "ITA-Serie A": { ... },
    # "FRA-Ligue 1": { ... },
}

# Understat league code -> your ESPN leagueId (for filtering fixtures.csv)
LEAGUE_ID = {
    "ENG-Premier League": 700,
    "ESP-La Liga": 740,
    "GER-Bundesliga": 720,
    "ITA-Serie A": 730,
    "FRA-Ligue 1": 710,
}


def load_understat(league, seasons):
    """Pull match-level xG from Understat."""
    us = sd.Understat(leagues=league, seasons=seasons)
    sched = us.read_schedule().reset_index()
    # keep only matches that actually have xG
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "home_xg", "away_xg"]
    sched = sched[[c for c in cols if c in sched.columns]].copy()
    sched = sched.dropna(subset=["home_xg", "away_xg"])
    sched["date_only"] = pd.to_datetime(sched["date"]).dt.date
    return sched


def load_fixtures(fixtures_path, league):
    fx = pd.read_csv(fixtures_path)
    fx = fx[fx["leagueId"] == LEAGUE_ID[league]].copy()
    fx["date_only"] = pd.to_datetime(fx["date"]).dt.date
    return fx


def apply_crosswalk(sched, league):
    """Map Understat names -> ESPN teamId. Report anything unmapped."""
    cw = CROSSWALK.get(league)
    if not cw:
        sys.exit(f"No crosswalk for {league}. Fill CROSSWALK first "
                 f"(use --dump-understat-names to see the names).")
    sched["home_id"] = sched["home_team"].map(cw)
    sched["away_id"] = sched["away_team"].map(cw)
    unmapped = set(sched.loc[sched["home_id"].isna(), "home_team"]) | \
               set(sched.loc[sched["away_id"].isna(), "away_team"])
    if unmapped:
        print("\n!! UNMAPPED Understat team names (add to CROSSWALK):")
        for n in sorted(unmapped):
            print(f"     {n!r}")
        print()
    return sched.dropna(subset=["home_id", "away_id"])


def merge_with_validation(fx, sched, date_tolerance_days=1):
    """
    Join xG onto fixtures by (home_id, away_id) and nearest date within
    tolerance. Validate each join against goals before trusting it.
    """
    sched = sched.copy()
    sched["home_id"] = sched["home_id"].astype(int)
    sched["away_id"] = sched["away_id"].astype(int)

    out_rows = []
    for _, f in fx.iterrows():
        cand = sched[(sched["home_id"] == f["homeTeamId"]) &
                     (sched["away_id"] == f["awayTeamId"])]
        if cand.empty:
            out_rows.append({**f, "home_xg": None, "away_xg": None,
                             "xg_join": "no_match"})
            continue
        # pick the candidate closest in date
        cand = cand.assign(
            ddiff=cand["date_only"].apply(lambda d: abs((d - f["date_only"]).days)))
        best = cand.sort_values("ddiff").iloc[0]
        if best["ddiff"] > date_tolerance_days:
            out_rows.append({**f, "home_xg": None, "away_xg": None,
                             "xg_join": "date_too_far"})
            continue
        # VALIDATE: goals must agree between sources, else it's a bad join
        goals_ok = (best["home_goals"] == f["homeTeamScore"] and
                    best["away_goals"] == f["awayTeamScore"])
        out_rows.append({
            **f,
            "home_xg": best["home_xg"], "away_xg": best["away_xg"],
            "xg_join": "ok" if goals_ok else "GOALS_MISMATCH",
        })
    return pd.DataFrame(out_rows)


def report(merged):
    n = len(merged)
    counts = merged["xg_join"].value_counts().to_dict()
    print("=" * 60)
    print(f"Fixtures processed:        {n}")
    print(f"  clean xG joins (ok):     {counts.get('ok', 0)}")
    print(f"  GOALS_MISMATCH (bad!):   {counts.get('GOALS_MISMATCH', 0)}")
    print(f"  no Understat match:      {counts.get('no_match', 0)}")
    print(f"  date out of tolerance:   {counts.get('date_too_far', 0)}")
    print("=" * 60)
    bad = merged[merged["xg_join"] == "GOALS_MISMATCH"]
    if len(bad):
        print("\nInspect these GOALS_MISMATCH rows (do NOT trust their xG):")
        print(bad[["date", "homeTeamId", "awayTeamId",
                   "homeTeamScore", "awayTeamScore", "home_xg", "away_xg"]]
              .head(10).to_string(index=False))
    print("\nRule: only rows with xg_join == 'ok' should feed the model.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    p.add_argument("--teams", required=False)
    p.add_argument("--league", default="ENG-Premier League")
    p.add_argument("--seasons", default="2024,2025",
                   help="Understat season start years, comma-separated "
                        "(2024 = 2024/25 season).")
    p.add_argument("--out", default="fixtures_with_xg.csv")
    p.add_argument("--dump-understat-names", action="store_true",
                   help="Print Understat team names for this league and exit.")
    args = p.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",")]

    sched = load_understat(args.league, seasons)

    if args.dump_understat_names:
        names = sorted(set(sched["home_team"]) | set(sched["away_team"]))
        print(f"\nUnderstat team names for {args.league}:")
        for nm in names:
            print(f"    {nm!r}: ,")
        return

    sched = apply_crosswalk(sched, args.league)
    fx = load_fixtures(args.fixtures, args.league)
    merged = merge_with_validation(fx, sched)
    report(merged)

    merged.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} (filter to xg_join=='ok' before training).")


if __name__ == "__main__":
    main()