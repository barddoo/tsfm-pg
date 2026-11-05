-- tsfm regression tests. These run without model weights: tsfm.stub uses a
-- naive last-value engine and the same quantile levels as Chronos-2.
CREATE EXTENSION tsfm;

SET tsfm.stub = on;

-- Metadata -------------------------------------------------------------------

SELECT tsfm_version() LIKE 'tsfm-pg %' AS version_ok;
SELECT tsfm_load();

SELECT j->>'model' AS model,
       j->>'stub' AS stub,
       j->>'loaded' AS loaded,
       j->>'threads' AS threads,
       coalesce(j->>'context_length', 'null') AS context_length,
       coalesce(j->>'prediction_length', 'null') AS prediction_length
FROM (SELECT tsfm_model_info() AS j) s;

SELECT (tsfm_model_info()->>'simd_f32_lanes')::int >= 1 AS simd_lanes_reported,
       tsfm_model_info()->>'cpu_arch' <> '' AS cpu_arch_reported;

SELECT tsfm_quantiles();

-- Forecast shape -------------------------------------------------------------

SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], 2)
WHERE quantile IN (0.1, 0.5, 0.9);

-- NULL and NaN elements are missing values; the stub uses the last finite one.
SELECT * FROM tsfm_forecast(ARRAY[1, NULL, 3, 'NaN'::float8]::float8[], 1)
WHERE quantile IN (0.1, 0.5, 0.9);

-- tsfm.max_context keeps only the most recent N values.
SET tsfm.max_context = 1;
SELECT * FROM tsfm_forecast(ARRAY[10, 20, 30]::float8[], 1)
WHERE quantile = 0.5;
RESET tsfm.max_context;

-- The matrix form is one step-major array: [step][quantile].
SELECT (tsfm_forecast_matrix(ARRAY[1, 2, 3]::float8[], 2))[1:3] AS head;
SELECT array_length(tsfm_forecast_matrix(ARRAY[1, 2, 3]::float8[], 2), 1) AS len;

-- Extended matrix: [variate][step][quantile].
SELECT array_length(tsfm_forecast_matrix_ex(ARRAY[1, 2, 3, 10, 20, 30]::float8[], 2, 2), 1) AS len;
SELECT (tsfm_forecast_matrix_ex(ARRAY[1, 2, 3, 10, 20, 30]::float8[], 1, 2))[1:3] AS head;

-- Batch of independent series, with a series column.
SELECT series, variate, step, value
FROM tsfm_forecast_batch(ARRAY[1, 2, 3, 10, 20, 30]::float8[], 2, 2, 1)
WHERE quantile = 0.5
ORDER BY series, variate, step;

-- Timestamp-aware wrapper.
SET timezone = 'UTC';
SELECT ts, step, value
FROM tsfm_forecast_ts(
    ARRAY['2024-01-01 00:00:00+00', '2024-01-02 00:00:00+00']::timestamptz[],
    ARRAY[1, 2]::float8[], 2, interval '1 day')
WHERE quantile = 0.5
ORDER BY step;

-- Errors ---------------------------------------------------------------------

SELECT * FROM tsfm_forecast(NULL::float8[], 2);
SELECT * FROM tsfm_forecast(ARRAY[]::float8[], 2);
SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], 0);
SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], 1025);
SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], NULL);

-- GUC clamping and the kill switch ------------------------------------------

SET tsfm.threads = 1000;
SHOW tsfm.threads;
SET tsfm.threads = 0;
SHOW tsfm.threads;

SET tsfm.enabled = off;
SELECT * FROM tsfm_forecast(ARRAY[1, 2, 3]::float8[], 2);
SET tsfm.enabled = on;

-- Timestral is selectable; the stub cannot serve covariates/multivariate.
SET tsfm.model = 'timestral';
SHOW tsfm.model;
RESET tsfm.model;

SET tsfm.stub = on;
SELECT * FROM tsfm_forecast_ex(ARRAY[1, 2, 3]::float8[], 2, 1, ARRAY[1, 2, 3]::float8[]);
SELECT * FROM tsfm_forecast_batch(ARRAY[1, 2, 3, 10, 20, 30]::float8[], 2, 2, 1, ARRAY[1, 2, 3]::float8[]);

SHOW tsfm.block_patches;
SHOW tsfm.symmetric;
SHOW tsfm.clip_nonnegative;
SHOW tsfm.quantile_real_cap_k;
SHOW tsfm.extreme_value_threshold;

-- Privileges: the model and a CPU core are not reachable by PUBLIC.
SELECT has_function_privilege('public', 'tsfm_forecast(double precision[], integer)', 'execute') AS public_can_execute;
SELECT has_function_privilege('public', 'tsfm_forecast_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[])', 'execute') AS public_can_forecast_ex;
SELECT has_function_privilege('public', 'tsfm_forecast_batch(double precision[], integer, integer, integer, double precision[], double precision[], double precision[], double precision[])', 'execute') AS public_can_batch;
SELECT has_function_privilege('public', 'tsfm_forecast_matrix(double precision[], integer)', 'execute') AS public_can_matrix;
SELECT has_function_privilege('public', 'tsfm_forecast_matrix_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[])', 'execute') AS public_can_matrix_ex;
SELECT has_function_privilege('public', 'tsfm_forecast_ts(timestamptz[], double precision[], integer, interval)', 'execute') AS public_can_ts;
SELECT has_function_privilege('public', 'tsfm_model_info()', 'execute') AS public_can_model_info;

-- Model directory diagnostics: read-only, no model needed.
SELECT s->>'ready' AS ready, s->>'note' AS note
FROM (SELECT tsfm_model_status() AS s) x;           -- stub is on

SET tsfm.stub = off;
SET tsfm.model_dir = '/nonexistent/tsfm-models';
SELECT s->>'ready' AS ready, s->>'config_json' AS config_json, s->>'weights' AS weights,
       s->>'weights_bytes' AS weights_bytes, s->>'detected_model' AS detected,
       s->>'problem' AS problem,
       (s->>'fix') LIKE 'scripts/fetch-model.sh /nonexistent/tsfm-models %' AS fix_has_command
FROM (SELECT tsfm_model_status() AS s) x;

-- A directory that exists but holds no checkpoint reports the same problem.
SET tsfm.model_dir = '/';
SELECT s->>'ready' AS ready, s->>'problem' AS problem
FROM (SELECT tsfm_model_status() AS s) x;

-- The load error now points at the diagnostic.
SELECT tsfm_load();
RESET tsfm.model_dir;
SET tsfm.stub = on;

-- Kill switch covers every entry point, including the matrix forms and warmup.
SET tsfm.enabled = off;
SELECT tsfm_forecast_matrix(ARRAY[1, 2, 3]::float8[], 2);
SELECT tsfm_forecast_matrix_ex(ARRAY[1, 2, 3]::float8[], 2, 1);
SELECT tsfm_load();
SELECT tsfm_quantiles();
SET tsfm.enabled = on;

-- Argument validation for the shape arguments.
SELECT * FROM tsfm_forecast_batch(ARRAY[1, 2, 3]::float8[], 2, NULL, 1);
SELECT * FROM tsfm_forecast_batch(ARRAY[1, 2, 3]::float8[], 2, 2, 1);
SELECT tsfm_forecast_matrix_ex(ARRAY[1, 2, 3]::float8[], 2, NULL);

-- Settings that choose what runs or what is read are superuser-only; the
-- decoding knobs stay available to ordinary roles.
CREATE ROLE tsfm_test_user;
GRANT EXECUTE ON FUNCTION tsfm_forecast(double precision[], integer) TO tsfm_test_user;
SET ROLE tsfm_test_user;
SET tsfm.model_dir = '/etc';
SET tsfm.enabled = on;
SET tsfm.stub = off;
SET tsfm.threads = 2;
SET tsfm.model = 'timestral';
SET tsfm.max_context = 5;
SET tsfm.clip_nonnegative = off;
RESET ROLE;
DROP OWNED BY tsfm_test_user;
DROP ROLE tsfm_test_user;

DROP EXTENSION tsfm;
