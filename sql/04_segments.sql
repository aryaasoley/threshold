-- 04_segments: what the default policy does to each segment, month by month.
--
-- Input:  model_input, decisions (application_id, month, p, action), written by the
--         pipeline after scoring the calibration and test months with the default policy.
-- Output: kpi_policy_segments
--
-- Approval and fraud rates by month, channel, device and age group: the table a Power BI
-- page or the monitoring view would sit on.

CREATE OR REPLACE TABLE kpi_policy_segments AS
WITH d AS (
    SELECT m.month, m.source, m.device_os, m.age_group, m.is_fraud, d.action, d.p
    FROM decisions d
    JOIN model_input m USING (application_id)
),
long AS (
    SELECT month, 'all'       AS dimension, 'all'     AS segment, is_fraud, action, p FROM d
    UNION ALL
    SELECT month, 'channel'   AS dimension, source    AS segment, is_fraud, action, p FROM d
    UNION ALL
    SELECT month, 'device'    AS dimension, device_os AS segment, is_fraud, action, p FROM d
    UNION ALL
    SELECT month, 'age group' AS dimension, age_group AS segment, is_fraud, action, p FROM d
)
SELECT
    month,
    dimension,
    segment,
    count(*)                                                       AS applications,
    avg(CASE WHEN action = 'approve' THEN 1 ELSE 0 END)            AS approve_rate,
    avg(CASE WHEN action = 'verify'  THEN 1 ELSE 0 END)            AS verify_rate,
    avg(CASE WHEN action = 'review'  THEN 1 ELSE 0 END)            AS review_rate,
    avg(CASE WHEN action = 'decline' THEN 1 ELSE 0 END)            AS decline_rate,
    avg(is_fraud)                                                  AS fraud_rate,
    avg(is_fraud) FILTER (WHERE action = 'approve')                AS fraud_rate_among_approved,
    avg(p)                                                         AS mean_predicted_fraud,
    avg(CASE WHEN action <> 'approve' THEN 1 ELSE 0 END)
        FILTER (WHERE is_fraud = 0)                                AS genuine_checked_rate
FROM long
GROUP BY month, dimension, segment
ORDER BY dimension, segment, month;
