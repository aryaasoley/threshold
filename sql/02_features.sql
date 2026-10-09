-- 02_features: the model-ready table.
--
-- Input:  applications
-- Output: model_input
--
-- Adds the time-based split and a flag for every column that can be missing. The split is
-- fixed here, in one place: learn on months 0-4, calibrate and choose cut-offs on month 5,
-- test on months 6-7 as if they were the future. A random split would let the model see
-- future patterns and look better than it is.

CREATE OR REPLACE TABLE model_input AS
SELECT
    a.*,
    CASE
        WHEN month <= 4 THEN 'train'
        WHEN month = 5  THEN 'calibration'
        ELSE 'test'
    END                                              AS split,
    prev_address_months_count    IS NULL             AS prev_address_missing,
    current_address_months_count IS NULL             AS current_address_missing,
    intended_balcon_amount       IS NULL             AS intended_balcon_missing,
    bank_months_count            IS NULL             AS bank_months_missing,
    session_length_in_minutes    IS NULL             AS session_length_missing,
    device_distinct_emails_8w    IS NULL             AS device_emails_missing
FROM applications a;
