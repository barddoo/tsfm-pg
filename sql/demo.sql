-- tsfm demo. Run with a model directory:
--
--   pgstart
--   psql -U postgres -d postgres \
--        -v model_dir="$(readlink -f ~/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/*/)" \
--        -f sql/demo.sql
--
-- A new session is needed to change tsfm.model_dir or tsfm.threads after the
-- engine has loaded.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS tsfm;
SET tsfm.model_dir = :'model_dir';
SET tsfm.threads = 8;

SELECT tsfm_version();
SELECT tsfm_load();
SELECT jsonb_pretty(tsfm_model_info());

-- One synthetic series: 365 observations, forecast 12 steps.
WITH history AS (
    SELECT array_agg(100 + i * 0.1 + sin(i * 0.3) * 2 ORDER BY i) AS h
    FROM generate_series(1, 365) AS i
)
SELECT step,
       quantile,
       round(value::numeric, 3) AS value
FROM history,
     LATERAL tsfm_forecast(h, 12)
WHERE quantile IN (0.1, 0.5, 0.9)
ORDER BY step, quantile;

-- The same forecast as one step-major [step][quantile] array.
SELECT tsfm_quantiles() AS levels;
SELECT array_length(tsfm_forecast_matrix(ARRAY[1, 2, 3, 4, 5]::float8[], 4), 1) AS values;
SELECT (tsfm_forecast_matrix(ARRAY[1, 2, 3, 4, 5]::float8[], 4))[1:6] AS first_quantiles;

-- A table of series: one forecast per group via LATERAL.
DROP TABLE IF EXISTS demo_metrics;
CREATE TABLE demo_metrics (name text, ts integer, value double precision);
INSERT INTO demo_metrics
SELECT t.name,
       i,
       t.base + i * 0.2 + sin(i * 0.3 + t.phase) * 3
FROM (VALUES ('sku-1', 100.0, 0.0),
             ('sku-2',  50.0, 1.5)) AS t(name, base, phase),
     generate_series(1, 180) AS i;

SELECT m.name,
       f.step,
       round(f.value::numeric, 3) AS median
FROM (SELECT name, array_agg(value ORDER BY ts) AS h
      FROM demo_metrics
      GROUP BY name) m,
     LATERAL tsfm_forecast(m.h, 30) f
WHERE f.quantile = 0.5
ORDER BY m.name, f.step;

-- Stub mode: no weights, naive last-value. Useful for a quick SQL smoke test.
SET tsfm.stub = on;
SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], 3)
WHERE quantile IN (0.1, 0.5, 0.9);
RESET tsfm.stub;

-- Privileges: the extension revokes EXECUTE from PUBLIC.
-- GRANT EXECUTE ON FUNCTION tsfm_forecast(double precision[], integer) TO analyst;

-- For Timestral (multivariate + covariates), see sql/demo_timestral.sql.
