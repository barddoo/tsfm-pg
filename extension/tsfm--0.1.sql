\echo Use "CREATE EXTENSION tsfm" to load this file. \quit

-- Metadata -------------------------------------------------------------------

CREATE FUNCTION tsfm_version() RETURNS text
AS '$libdir/tsfm'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

-- Implementation detail of tsfm_model_info(); the extension has no jsonb
-- converter, so the C function returns text and the SQL wrapper casts it.
CREATE FUNCTION tsfm_model_info_text() RETURNS text
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

CREATE FUNCTION tsfm_model_info() RETURNS jsonb
LANGUAGE sql STABLE PARALLEL RESTRICTED
AS $$ SELECT @extschema@.tsfm_model_info_text()::jsonb $$;

-- Read-only check of the configured model directory: is a usable checkpoint
-- there, and if not, what to do. Loads nothing, writes nothing, and never uses
-- the network. Look at `ready`, `problem` and `fix`.
CREATE FUNCTION tsfm_model_status_text() RETURNS text
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

CREATE FUNCTION tsfm_model_status() RETURNS jsonb
LANGUAGE sql STABLE PARALLEL RESTRICTED
AS $$ SELECT @extschema@.tsfm_model_status_text()::jsonb $$;

CREATE FUNCTION tsfm_quantiles() RETURNS double precision[]
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- Explicit warmup. The engine is process-global and loaded on first use.
CREATE FUNCTION tsfm_load() RETURNS boolean
AS '$libdir/tsfm'
LANGUAGE C VOLATILE PARALLEL RESTRICTED;

-- Forecasting ----------------------------------------------------------------

-- history is ordered oldest -> newest. NULL or 'NaN' elements are missing
-- values. `step` is 1-based; rows are ordered step, then quantile.
CREATE FUNCTION tsfm_forecast(history double precision[], horizon integer)
RETURNS TABLE (step integer, quantile double precision, value double precision)
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- The same forecast as a single [step][quantile] array (step-major).
CREATE FUNCTION tsfm_forecast_matrix(history double precision[], horizon integer)
RETURNS double precision[]
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- Extended forecast: multiple target variates and optional covariates.
-- Arrays are flattened row-major. `history` is V x T; `past` is C x T;
-- `known` is C x (T + horizon); categorical arrays use the same shapes with
-- category codes. NULL or 'NaN' marks a missing value. Timestral supports all
-- of it; Chronos-2 accepts only n_variates = 1 with no covariates.
CREATE FUNCTION tsfm_forecast_ex(
    history double precision[],
    horizon integer,
    n_variates integer DEFAULT 1,
    past double precision[] DEFAULT NULL,
    known double precision[] DEFAULT NULL,
    cat_past double precision[] DEFAULT NULL,
    cat_known double precision[] DEFAULT NULL)
RETURNS TABLE (variate integer, step integer, quantile double precision, value double precision)
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- Batch independent series in one engine call. history is B x V x T and the
-- covariates are B x C x T / B x C x (T + horizon). The result has a series
-- column; the engine checks for query cancellation between chunks.
CREATE FUNCTION tsfm_forecast_batch(
    history double precision[],
    horizon integer,
    n_series integer,
    n_variates integer DEFAULT 1,
    past double precision[] DEFAULT NULL,
    known double precision[] DEFAULT NULL,
    cat_past double precision[] DEFAULT NULL,
    cat_known double precision[] DEFAULT NULL)
RETURNS TABLE (series integer, variate integer, step integer, quantile double precision, value double precision)
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- The extended forecast as one [variate][step][quantile] array.
CREATE FUNCTION tsfm_forecast_matrix_ex(
    history double precision[],
    horizon integer,
    n_variates integer DEFAULT 1,
    past double precision[] DEFAULT NULL,
    known double precision[] DEFAULT NULL,
    cat_past double precision[] DEFAULT NULL,
    cat_known double precision[] DEFAULT NULL)
RETURNS double precision[]
AS '$libdir/tsfm'
LANGUAGE C STABLE PARALLEL RESTRICTED;

-- Timestamp-aware forecast: attach future timestamps to tsfm_forecast rows.
CREATE FUNCTION tsfm_forecast_ts(
    timestamps timestamptz[],
    history double precision[],
    horizon integer,
    step_size interval)
RETURNS TABLE (ts timestamptz, step integer, quantile double precision, value double precision)
LANGUAGE plpgsql STABLE PARALLEL RESTRICTED
AS $$
DECLARE
    n integer;
    last_ts timestamptz;
BEGIN
    IF timestamps IS NULL OR history IS NULL OR horizon IS NULL OR step_size IS NULL THEN
        RAISE EXCEPTION 'timestamps, history, horizon and step_size must not be NULL'
            USING ERRCODE = '22023';
    END IF;
    IF array_ndims(timestamps) <> 1 OR array_ndims(history) <> 1 THEN
        RAISE EXCEPTION 'timestamps and history must be one-dimensional arrays'
            USING ERRCODE = '22023';
    END IF;
    n := array_length(timestamps, 1);
    IF n IS NULL OR n = 0 OR n <> array_length(history, 1) THEN
        RAISE EXCEPTION 'timestamps and history must have the same non-zero length'
            USING ERRCODE = '22023';
    END IF;
    IF step_size <= interval '0' THEN
        RAISE EXCEPTION 'step_size must be positive' USING ERRCODE = '22023';
    END IF;
    last_ts := timestamps[array_upper(timestamps, 1)];
    IF last_ts IS NULL THEN
        RAISE EXCEPTION 'the last timestamp must not be NULL' USING ERRCODE = '22023';
    END IF;
    RETURN QUERY
        SELECT last_ts + (f.step * step_size), f.step, f.quantile, f.value
        FROM @extschema@.tsfm_forecast(history, horizon) AS f;
END;
$$;

-- A model and a CPU core per call should not be reachable by every role by
-- default. The admin grants EXECUTE explicitly.
REVOKE ALL ON FUNCTION tsfm_version() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_model_info_text() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_model_info() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_model_status_text() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_model_status() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_quantiles() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_load() FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast(double precision[], integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast_batch(double precision[], integer, integer, integer, double precision[], double precision[], double precision[], double precision[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast_matrix(double precision[], integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast_matrix_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION tsfm_forecast_ts(timestamptz[], double precision[], integer, interval) FROM PUBLIC;

COMMENT ON FUNCTION tsfm_forecast(double precision[], integer) IS
'Forecast horizon steps from an ordered history; one row per step and quantile.';
COMMENT ON FUNCTION tsfm_forecast_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) IS
'Extended forecast: flattened V x T history, optional C x T / C x (T+horizon) covariates; one row per variate, step and quantile.';
COMMENT ON FUNCTION tsfm_forecast_batch(double precision[], integer, integer, integer, double precision[], double precision[], double precision[], double precision[]) IS
'Batch forecast: B independent series, flattened B x V x T history; one row per series, variate, step and quantile.';
COMMENT ON FUNCTION tsfm_forecast_matrix(double precision[], integer) IS
'Forecast as a step-major [step][quantile] double precision array.';
COMMENT ON FUNCTION tsfm_forecast_matrix_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) IS
'Extended forecast as a [variate][step][quantile] double precision array.';
COMMENT ON FUNCTION tsfm_forecast_ts(timestamptz[], double precision[], integer, interval) IS
'Forecast and attach future timestamps using the last timestamp and step_size.';
