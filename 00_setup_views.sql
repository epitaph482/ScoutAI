-- ==========================================================
-- SCOUT AI DATABASE SETUP
-- Creates all views used by the notebooks, ending with
-- view_scout_master (the single source of truth every
-- notebook reads from).
--
-- Run this file top to bottom, once, after loading the source
-- CSVs (players, appearances, clubs, competitions, transfers,
-- player_valuations, national_teams) into PostgreSQL.
-- ==========================================================

DROP VIEW IF EXISTS view_scout_master CASCADE;
DROP VIEW IF EXISTS transfer_stats CASCADE;
DROP VIEW IF EXISTS club_info CASCADE;
DROP VIEW IF EXISTS latest_player_value CASCADE;
DROP VIEW IF EXISTS player_stats CASCADE;

-- ==========================================================
-- 1. PLAYER PERFORMANCE SUMMARY
-- Aggregates raw match-by-match appearances into one row per player.
-- ==========================================================

CREATE VIEW player_stats AS
SELECT
    a.player_id,
    COUNT(*) AS total_appearances,
    COALESCE(SUM(a.minutes_played), 0) AS total_minutes,
    COALESCE(SUM(a.goals), 0) AS total_goals,
    COALESCE(SUM(a.assists), 0) AS total_assists,
    COALESCE(SUM(a.yellow_cards), 0) AS total_yellow_cards,
    COALESCE(SUM(a.red_cards), 0) AS total_red_cards,
    ROUND(COALESCE(AVG(a.minutes_played), 0), 2) AS minutes_per_match,
    ROUND(
        COALESCE(SUM(a.goals) * 90.0 / NULLIF(SUM(a.minutes_played), 0), 0),
        3
    ) AS goals_per_90,
    ROUND(
        COALESCE(SUM(a.assists) * 90.0 / NULLIF(SUM(a.minutes_played), 0), 0),
        3
    ) AS assists_per_90
FROM appearances a
GROUP BY a.player_id;


-- ==========================================================
-- 2. LATEST PLAYER MARKET VALUE
-- Picks the single most recent valuation per player.
-- ==========================================================

CREATE VIEW latest_player_value AS
SELECT
    player_id,
    market_value_in_eur AS current_market_value
FROM (
    SELECT
        player_id,
        market_value_in_eur,
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY date::date DESC
        ) AS rn
    FROM player_valuations
) ranked
WHERE rn = 1;


-- ==========================================================
-- 3. TRANSFER SUMMARY
--
-- IMPORTANT: has_transfer_history only counts transfers with a real
-- paid fee (fee > 0). Without this filter, academy promotions and
-- free/zero-fee records (which the transfers table also logs) get
-- counted as a "transfer," incorrectly flagging players who were
-- never actually sold -- e.g. academy graduates like Lamine Yamal.
--
-- most_recent_transfer_fee (vs. max_career_transfer_fee) captures the
-- LATEST paid fee rather than the career peak, so a player who was
-- once expensive but has since declined isn't over-valued by a
-- transfer fee that no longer reflects their current level.
-- ==========================================================

CREATE VIEW transfer_stats AS
WITH paid_transfers AS (
    SELECT
        player_id,
        transfer_date,
        COALESCE(transfer_fee, 0) AS transfer_fee,
        ROW_NUMBER() OVER (
            PARTITION BY player_id
            ORDER BY transfer_date DESC
        ) AS rn
    FROM transfers
    WHERE COALESCE(transfer_fee, 0) > 0
)
SELECT
    t.player_id,
    COUNT(*) AS transfer_count,
    MAX(COALESCE(t.transfer_fee, 0)) AS max_career_transfer_fee,
    COALESCE(pt.transfer_fee, 0) AS most_recent_transfer_fee,
    CASE
        WHEN COUNT(*) FILTER (WHERE COALESCE(t.transfer_fee, 0) > 0) > 0 THEN 1
        ELSE 0
    END AS has_transfer_history
FROM transfers t
LEFT JOIN paid_transfers pt
    ON pt.player_id = t.player_id AND pt.rn = 1
GROUP BY t.player_id, pt.transfer_fee;


-- ==========================================================
-- 4. CLUB INFO
-- Club-level context joined with competition/league name.
-- ==========================================================

CREATE VIEW club_info AS
SELECT
    c.club_id,
    c.name AS club_name,
    c.total_market_value::numeric AS club_total_market_value,
    c.squad_size AS club_squad_size,
    c.average_age AS club_avg_age,
    c.stadium_seats,
    c.foreigners_percentage,
    comp.name AS competition_name
FROM clubs c
LEFT JOIN competitions comp
    ON c.domestic_competition_id = comp.competition_id;


-- ==========================================================
-- 5. VIEW_SCOUT_MASTER
-- The single source of truth every notebook reads from.
--
-- Filters applied (each one fixes a real data quality issue found
-- during development -- see README for the full story):
--   - current_market_value IS NOT NULL: only players with a real
--     valuation to predict.
--   - date_of_birth IS NOT NULL / <= CURRENT_DATE: a small number of
--     players had a missing birth date, which silently became age=0
--     downstream and made the model treat them as infant wonderkids.
--   - last_season >= 2023: excludes retired / long-inactive players,
--     whose frozen market value only added noise. Removing them
--     improved both models' R^2 despite roughly halving the dataset.
-- ==========================================================

CREATE VIEW view_scout_master AS
SELECT
    p.player_id,
    p.name AS player_name,
    ci.club_name,
    ci.competition_name,
    lp.current_market_value,
    LN(lp.current_market_value + 1) AS log_market_value,

    EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.date_of_birth::date)) AS age,

    p.height_in_cm::integer AS height_in_cm,
    p.foot,
    p.position AS position_group,
    p.sub_position,

    COALESCE(ps.total_appearances, 0) AS total_appearances,
    COALESCE(ps.total_minutes, 0) AS total_minutes,
    COALESCE(ps.total_goals, 0) AS total_goals,
    COALESCE(ps.total_assists, 0) AS total_assists,
    COALESCE(ps.goals_per_90, 0) AS goals_per_90,
    COALESCE(ps.assists_per_90, 0) AS assists_per_90,
    COALESCE(ps.minutes_per_match, 0) AS minutes_per_match,
    COALESCE(ps.total_yellow_cards, 0) AS total_yellow_cards,
    COALESCE(ps.total_red_cards, 0) AS total_red_cards,

    COALESCE(NULLIF(p.international_caps, '')::integer, 0) AS international_caps,
    COALESCE(NULLIF(p.international_goals, '')::integer, 0) AS international_goals,

    ci.club_total_market_value,
    ci.club_squad_size,
    ci.club_avg_age,
    ci.stadium_seats,
    ci.foreigners_percentage,

    COALESCE(nt.fifa_ranking, 999) AS passport_power_rank,

    COALESCE(ts.transfer_count, 0) AS transfer_count,
    COALESCE(ts.has_transfer_history, 0) AS has_transfer_history,
    COALESCE(ts.max_career_transfer_fee, 0) AS max_career_transfer_fee,
    COALESCE(ts.most_recent_transfer_fee, 0) AS most_recent_transfer_fee,

    CASE
        WHEN p.contract_expiration_date IS NULL OR p.contract_expiration_date = ''
        THEN 0
        ELSE (
            EXTRACT(YEAR FROM AGE(p.contract_expiration_date::date, CURRENT_DATE)) * 12
            + EXTRACT(MONTH FROM AGE(p.contract_expiration_date::date, CURRENT_DATE))
        )
    END AS contract_months_remaining

FROM players p
LEFT JOIN player_stats ps ON p.player_id = ps.player_id
LEFT JOIN latest_player_value lp ON p.player_id = lp.player_id
LEFT JOIN club_info ci ON p.current_club_id = ci.club_id
LEFT JOIN transfer_stats ts ON p.player_id = ts.player_id
LEFT JOIN national_teams nt ON nt.country_name = p.country_of_citizenship

WHERE lp.current_market_value IS NOT NULL
  AND p.date_of_birth IS NOT NULL
  AND p.date_of_birth::date <= CURRENT_DATE
  AND p.last_season >= 2023;


-- ==========================================================
-- VERIFICATION (optional -- comment out or delete before committing
-- if you don't want query output when running this file)
-- ==========================================================

SELECT * FROM view_scout_master LIMIT 5;

SELECT has_transfer_history, COUNT(*) AS player_count
FROM view_scout_master
GROUP BY has_transfer_history;