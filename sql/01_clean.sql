-- 01_clean: one tidy row per application.
--
-- Input:  raw_applications (a view over Base.csv, created by threshold.data)
-- Output: applications
--
-- The raw file marks "missing" with -1 in several columns, and with any negative value in
-- intended_balcon_amount and session_length_in_minutes. Treating -1 as a real number is a
-- silent bug (an address held for "-1 months"), so every sentinel becomes NULL here.
-- tests/test_data.py checks that no sentinel survives.

CREATE OR REPLACE TABLE applications AS
SELECT
    CAST(row_number() OVER () AS BIGINT)                          AS application_id,
    CAST(month AS INTEGER)                                        AS month,
    CAST(fraud_bool AS INTEGER)                                   AS is_fraud,

    -- what the applicant tells us
    income,
    customer_age,
    -- 50 matches fairness.older_group_min_age in config/assumptions.yaml
    CASE WHEN customer_age >= 50 THEN '50+' ELSE 'under 50' END   AS age_group,
    employment_status,
    housing_status,
    payment_type,
    proposed_credit_limit,
    NULLIF(prev_address_months_count, -1)                         AS prev_address_months_count,
    NULLIF(current_address_months_count, -1)                      AS current_address_months_count,
    CASE WHEN intended_balcon_amount < 0 THEN NULL
         ELSE intended_balcon_amount END                          AS intended_balcon_amount,
    days_since_request,
    source,

    -- internal activity checks (counts over recent windows)
    zip_count_4w,
    velocity_6h,
    velocity_24h,
    velocity_4w,
    bank_branch_count_8w,
    date_of_birth_distinct_emails_4w,

    -- device and session
    device_os,
    NULLIF(device_distinct_emails_8w, -1)                         AS device_distinct_emails_8w,
    device_fraud_count,
    keep_alive_session,
    CASE WHEN session_length_in_minutes < 0 THEN NULL
         ELSE session_length_in_minutes END                       AS session_length_in_minutes,
    foreign_request,

    -- credit bureau
    credit_risk_score,
    has_other_cards,
    NULLIF(bank_months_count, -1)                                 AS bank_months_count,

    -- email and phone checks
    name_email_similarity,
    email_is_free,
    phone_home_valid,
    phone_mobile_valid
FROM raw_applications;
