-- 03_monthly_kpis: the numbers an operations lead looks at every month.
--
-- Input:  model_input
-- Output: kpi_monthly           one row per month
--         kpi_monthly_segments  one row per month x segment (channel, device, age group)

CREATE OR REPLACE TABLE kpi_monthly AS
SELECT
    month,
    any_value(split)                                     AS split,
    count(*)                                             AS applications,
    sum(is_fraud)                                        AS frauds,
    avg(is_fraud)                                        AS fraud_rate,
    avg(CASE WHEN age_group = '50+' THEN 1 ELSE 0 END)   AS share_50_plus,
    avg(CASE WHEN source = 'TELEAPP' THEN 1 ELSE 0 END)  AS share_teleapp,
    avg(email_is_free)                                   AS share_free_email,
    avg(prev_address_missing::INTEGER)                   AS prev_address_missing_rate,
    avg(intended_balcon_missing::INTEGER)                AS intended_balcon_missing_rate,
    avg(bank_months_missing::INTEGER)                    AS bank_months_missing_rate,
    median(credit_risk_score)                            AS median_credit_risk_score,
    median(velocity_6h)                                  AS median_velocity_6h
FROM model_input
GROUP BY month
ORDER BY month;

CREATE OR REPLACE TABLE kpi_monthly_segments AS
WITH long AS (
    SELECT month, 'channel'   AS dimension, source    AS segment, is_fraud FROM model_input
    UNION ALL
    SELECT month, 'device'    AS dimension, device_os AS segment, is_fraud FROM model_input
    UNION ALL
    SELECT month, 'age group' AS dimension, age_group AS segment, is_fraud FROM model_input
)
SELECT
    month,
    dimension,
    segment,
    count(*)                                                       AS applications,
    count(*) / sum(count(*)) OVER (PARTITION BY month, dimension)  AS share_of_month,
    sum(is_fraud)                                                  AS frauds,
    avg(is_fraud)                                                  AS fraud_rate
FROM long
GROUP BY month, dimension, segment
ORDER BY dimension, segment, month;
