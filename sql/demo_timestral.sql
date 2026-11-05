-- Timestral demo. Run with a Timestral checkpoint directory:
--
--   pgstart
--   psql -U postgres -d postgres \
--        -v model_dir="$(readlink -f ~/.cache/huggingface/hub/models--mistralai--Timestral-28M/snapshots/*/)" \
--        -f sql/demo_timestral.sql
--
-- Timestral weights are Modified MIT with a revenue cap. Download them
-- yourself and accept the model terms; the extension never downloads anything.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS tsfm;
SET tsfm.model = 'timestral';
SET tsfm.model_dir = :'model_dir';
SET tsfm.threads = 8;

SELECT tsfm_version();
SELECT tsfm_load();
SELECT jsonb_pretty(tsfm_model_info());

-- Two variates, 20 steps, one past-only and one known-future covariate.
-- Arrays are flattened row-major: V x T for history, C x T for past,
-- C x (T + horizon) for known. NULL / NaN marks a missing value.
WITH data AS (
    SELECT ARRAY[
        100, 100.98, 101.83, 102.46, 102.80, 102.82, 102.55, 102.07,
        101.48, 100.91, 100.49, 100.30, 100.41, 100.83, 101.54, 102.44,
        103.43, 104.39, 105.19, 105.74,
         51.5, 51.53, 51.44, 51.23, 50.94, 50.61, 50.26, 49.94,
         49.69, 49.54, 49.52, 49.62, 49.85, 50.21, 50.66, 51.18,
         51.73, 52.27, 52.75, 53.15
    ]::float8[] AS history,
    ARRAY[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]::float8[] AS past,
    ARRAY[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]::float8[] AS known
)
SELECT f.variate, f.step, f.quantile, round(f.value::numeric, 3) AS value
FROM data,
     LATERAL tsfm_forecast_ex(data.history, 4, 2, data.past, data.known, NULL, NULL) f
WHERE f.quantile IN (0.1, 0.5, 0.9)
ORDER BY f.variate, f.step, f.quantile;

-- Categorical covariates use the same shapes, with category codes and NaN for
-- missing: pass them as cat_past / cat_known.
SELECT count(*) AS categorical_rows
FROM tsfm_forecast_ex(
    ARRAY[10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]::float8[],
    2,
    2,
    NULL,
    NULL,
    ARRAY[0, 1, 2, 0, 1, 2]::float8[],
    ARRAY[1, 2, 3, 0, 1, 2, 3, 0]::float8[]);
