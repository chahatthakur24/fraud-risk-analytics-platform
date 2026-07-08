-- =============================================================================
-- CREDIT CARD FRAUD DETECTION — SQL Analytics (17 Queries)
-- Database: PostgreSQL | Table: transactions
-- Author: Chahat Thakur | github.com/chahatthakur24
-- =============================================================================

-- TABLE SCHEMA (for reference)
-- CREATE TABLE transactions (
--     id           SERIAL PRIMARY KEY,
--     time_seconds FLOAT,
--     amount       FLOAT,
--     v1 FLOAT, v2 FLOAT, ..., v28 FLOAT,
--     risk_score   INT,
--     risk_tier    VARCHAR(10),
--     class        INT,        -- 0 = legit, 1 = fraud
--     ml_prob      FLOAT,      -- XGBoost fraud probability
--     is_fraud_ml  BOOLEAN
-- );

-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 01: Overall fraud summary
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(class)                                        AS total_fraud,
    ROUND(100.0 * SUM(class) / COUNT(*), 4)           AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                             AS avg_transaction_amount,
    ROUND(SUM(CASE WHEN class=1 THEN amount END), 2)  AS total_fraud_amount
FROM transactions;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 02: Class distribution
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE WHEN class = 1 THEN 'Fraud' ELSE 'Legitimate' END AS transaction_type,
    COUNT(*)                                               AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4)    AS pct_of_total,
    ROUND(AVG(amount), 2)                                  AS avg_amount,
    ROUND(MAX(amount), 2)                                  AS max_amount
FROM transactions
GROUP BY class
ORDER BY class DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 03: Amount bucket analysis
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN amount < 10    THEN '< $10'
        WHEN amount < 50    THEN '$10 – $50'
        WHEN amount < 100   THEN '$50 – $100'
        WHEN amount < 500   THEN '$100 – $500'
        WHEN amount < 1000  THEN '$500 – $1000'
        ELSE                     '> $1000'
    END AS amount_bucket,
    COUNT(*)                                        AS total,
    SUM(class)                                      AS frauds,
    ROUND(100.0 * SUM(class) / COUNT(*), 2)         AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                           AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY MIN(amount);


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 04: Hourly fraud rate
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    (FLOOR(time_seconds / 3600) % 24)::INT          AS hour_of_day,
    COUNT(*)                                         AS total_txns,
    SUM(class)                                       AS fraud_count,
    ROUND(100.0 * SUM(class) / COUNT(*), 4)          AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                            AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 05: Risk tier performance
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    risk_tier,
    COUNT(*)                                         AS total,
    SUM(class)                                       AS actual_fraud,
    ROUND(100.0 * SUM(class) / COUNT(*), 2)          AS precision_pct,
    ROUND(AVG(amount), 2)                            AS avg_amount,
    ROUND(AVG(risk_score), 1)                        AS avg_risk_score
FROM transactions
GROUP BY risk_tier
ORDER BY avg_risk_score DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 06: ML model vs rule-based comparison
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN class = 1 AND is_fraud_ml = TRUE  THEN 'True Positive'
        WHEN class = 0 AND is_fraud_ml = FALSE THEN 'True Negative'
        WHEN class = 0 AND is_fraud_ml = TRUE  THEN 'False Positive'
        WHEN class = 1 AND is_fraud_ml = FALSE THEN 'False Negative'
    END AS outcome,
    COUNT(*) AS count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM transactions
GROUP BY 1
ORDER BY count DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 07: Rolling 6-hour fraud average (window function)
-- ─────────────────────────────────────────────────────────────────────────────
WITH hourly_txns AS (
    SELECT
        (FLOOR(time_seconds / 3600))::INT AS hour_bucket,
        COUNT(*)                           AS total_txns,
        SUM(class)                         AS fraud_count
    FROM transactions
    GROUP BY 1
)
SELECT
    hour_bucket,
    fraud_count,
    total_txns,
    ROUND(100.0 * fraud_count / NULLIF(total_txns, 0), 4)  AS fraud_rate_pct,
    AVG(fraud_count) OVER (
        ORDER BY hour_bucket
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    )                                                       AS rolling_6h_fraud_avg,
    SUM(total_txns) OVER (ORDER BY hour_bucket)             AS cumulative_txns
FROM hourly_txns
ORDER BY hour_bucket;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 08: Top 10 highest-probability fraud transactions
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    id,
    amount,
    ROUND(ml_prob, 4)                                        AS fraud_probability,
    risk_score,
    risk_tier,
    class                                                    AS actual_label,
    (FLOOR(time_seconds / 3600) % 24)::INT                  AS hour_of_day
FROM transactions
ORDER BY ml_prob DESC
LIMIT 10;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 09: Percentile analysis of fraud probabilities
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ml_prob) AS p50_prob,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ml_prob) AS p75_prob,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ml_prob) AS p90_prob,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ml_prob) AS p95_prob,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ml_prob) AS p99_prob,
    ROUND(AVG(ml_prob), 6)                                 AS mean_prob
FROM transactions;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 10: CTE — High risk transactions in top 5% probability
-- ─────────────────────────────────────────────────────────────────────────────
WITH prob_cutoff AS (
    SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ml_prob) AS cutoff
    FROM transactions
),
high_risk AS (
    SELECT t.*, p.cutoff
    FROM transactions t, prob_cutoff p
    WHERE t.ml_prob >= p.cutoff
)
SELECT
    COUNT(*)                                         AS high_risk_count,
    SUM(class)                                       AS actual_frauds,
    ROUND(100.0 * SUM(class) / COUNT(*), 2)          AS precision_pct,
    ROUND(AVG(amount), 2)                            AS avg_fraud_amount,
    ROUND(MIN(cutoff), 4)                            AS probability_threshold
FROM high_risk;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 11: Fraud amount exposure by risk tier
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    risk_tier,
    SUM(CASE WHEN class = 1 THEN amount ELSE 0 END)   AS fraud_exposure,
    COUNT(CASE WHEN class = 1 THEN 1 END)             AS fraud_cases,
    ROUND(AVG(CASE WHEN class = 1 THEN amount END), 2) AS avg_fraud_amount,
    ROUND(
        100.0 * SUM(CASE WHEN class=1 THEN amount ELSE 0 END) /
        NULLIF(SUM(SUM(CASE WHEN class=1 THEN amount ELSE 0 END)) OVER (), 0),
    2)                                                AS pct_of_total_exposure
FROM transactions
GROUP BY risk_tier
ORDER BY fraud_exposure DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 12: Day-over-day fraud trend (48-hour dataset)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE WHEN time_seconds < 86400 THEN 'Day 1' ELSE 'Day 2' END AS day,
    COUNT(*)                                                      AS total,
    SUM(class)                                                    AS frauds,
    ROUND(100.0 * SUM(class) / COUNT(*), 4)                       AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                                         AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 13: RANK — fraud probability decile analysis
-- ─────────────────────────────────────────────────────────────────────────────
WITH deciles AS (
    SELECT
        id, class, ml_prob, amount,
        NTILE(10) OVER (ORDER BY ml_prob DESC) AS prob_decile
    FROM transactions
)
SELECT
    prob_decile,
    COUNT(*)                                      AS total,
    SUM(class)                                    AS fraud_count,
    ROUND(100.0 * SUM(class) / COUNT(*), 2)       AS fraud_rate_pct,
    ROUND(AVG(ml_prob), 4)                        AS avg_prob,
    ROUND(AVG(amount), 2)                         AS avg_amount
FROM deciles
GROUP BY prob_decile
ORDER BY prob_decile;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 14: Confusion matrix at different thresholds
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    threshold_val,
    SUM(CASE WHEN class=1 AND ml_prob >= threshold_val THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN class=0 AND ml_prob >= threshold_val THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN class=1 AND ml_prob < threshold_val  THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN class=0 AND ml_prob < threshold_val  THEN 1 ELSE 0 END) AS tn,
    ROUND(
        100.0 * SUM(CASE WHEN class=1 AND ml_prob >= threshold_val THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN ml_prob >= threshold_val THEN 1 ELSE 0 END), 0), 2
    ) AS precision_pct,
    ROUND(
        100.0 * SUM(CASE WHEN class=1 AND ml_prob >= threshold_val THEN 1 ELSE 0 END) /
        NULLIF(SUM(class), 0), 2
    ) AS recall_pct
FROM transactions
CROSS JOIN (VALUES (0.3),(0.4),(0.5),(0.6),(0.7),(0.8)) AS t(threshold_val)
GROUP BY threshold_val
ORDER BY threshold_val;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 15: V14 anomaly scoring (strongest fraud feature)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN v14 < -10 THEN 'Extreme (<-10)'
        WHEN v14 < -5  THEN 'Very low (-10 to -5)'
        WHEN v14 < 0   THEN 'Below zero (-5 to 0)'
        ELSE                'Normal (>= 0)'
    END AS v14_range,
    COUNT(*)                                       AS total,
    SUM(class)                                     AS fraud_count,
    ROUND(100.0 * SUM(class) / COUNT(*), 2)        AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                          AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 16: Night vs day transaction comparison
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN (FLOOR(time_seconds / 3600) % 24) BETWEEN 0 AND 4
          OR (FLOOR(time_seconds / 3600) % 24) >= 22
        THEN 'Night (10 PM – 5 AM)'
        ELSE 'Day (5 AM – 10 PM)'
    END AS time_period,
    COUNT(*)                                        AS total,
    SUM(class)                                      AS frauds,
    ROUND(100.0 * SUM(class) / COUNT(*), 4)         AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                           AS avg_amount,
    ROUND(AVG(ml_prob), 4)                          AS avg_ml_prob
FROM transactions
GROUP BY 1;


-- ─────────────────────────────────────────────────────────────────────────────
-- QUERY 17: Final summary dashboard view
-- ─────────────────────────────────────────────────────────────────────────────
WITH stats AS (
    SELECT
        COUNT(*)                                         AS total_txns,
        SUM(class)                                       AS total_fraud,
        SUM(CASE WHEN is_fraud_ml = TRUE THEN 1 END)     AS ml_flagged,
        SUM(CASE WHEN risk_tier = 'High' THEN 1 END)     AS rule_flagged,
        SUM(CASE WHEN class=1 THEN amount END)           AS fraud_exposure,
        AVG(ml_prob)                                     AS avg_ml_prob
    FROM transactions
)
SELECT
    total_txns,
    total_fraud,
    ROUND(100.0 * total_fraud / total_txns, 4)           AS actual_fraud_rate_pct,
    ml_flagged,
    ROUND(100.0 * total_fraud / NULLIF(ml_flagged, 0), 2) AS ml_precision_pct,
    rule_flagged,
    ROUND(fraud_exposure, 2)                             AS total_fraud_exposure_usd,
    ROUND(avg_ml_prob * 100, 4)                          AS avg_fraud_probability_pct
FROM stats;
