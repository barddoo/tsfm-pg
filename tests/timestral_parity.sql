\set ON_ERROR_STOP on
-- Timestral parity: 2 variates, one past-only and one known-future covariate.
-- Expected values come from the tsfm-zig CLI on identical inputs.
CREATE EXTENSION IF NOT EXISTS tsfm;
SET tsfm.stub = off;
SET tsfm.model = 'timestral';
SET tsfm.model_dir = :'model_dir';
SET tsfm.threads = :threads;
SET tsfm.sort_quantiles = on;

CREATE TEMP TABLE expected (variate integer, step integer, quantile double precision, value double precision);
INSERT INTO expected VALUES (1, 1, 0.1, 105.55461);
INSERT INTO expected VALUES (1, 1, 0.5, 105.891);
INSERT INTO expected VALUES (1, 1, 0.9, 106.18863);
INSERT INTO expected VALUES (1, 2, 0.1, 105.30765);
INSERT INTO expected VALUES (1, 2, 0.5, 106.08233);
INSERT INTO expected VALUES (1, 2, 0.9, 106.71208);
INSERT INTO expected VALUES (1, 3, 0.1, 104.67754);
INSERT INTO expected VALUES (1, 3, 0.5, 106.089096);
INSERT INTO expected VALUES (1, 3, 0.9, 107.179794);
INSERT INTO expected VALUES (1, 4, 0.1, 103.7518);
INSERT INTO expected VALUES (1, 4, 0.5, 106.1558);
INSERT INTO expected VALUES (1, 4, 0.9, 107.633644);
INSERT INTO expected VALUES (2, 1, 0.1, 53.19394);
INSERT INTO expected VALUES (2, 1, 0.5, 53.33197);
INSERT INTO expected VALUES (2, 1, 0.9, 53.48104);
INSERT INTO expected VALUES (2, 2, 0.1, 53.183037);
INSERT INTO expected VALUES (2, 2, 0.5, 53.51095);
INSERT INTO expected VALUES (2, 2, 0.9, 53.78115);
INSERT INTO expected VALUES (2, 3, 0.1, 52.979805);
INSERT INTO expected VALUES (2, 3, 0.5, 53.575584);
INSERT INTO expected VALUES (2, 3, 0.9, 54.06143);
INSERT INTO expected VALUES (2, 4, 0.1, 52.477165);
INSERT INTO expected VALUES (2, 4, 0.5, 53.614082);
INSERT INTO expected VALUES (2, 4, 0.9, 54.299355);

DO $$
DECLARE
    got integer := 0;
    bad integer := 0;
    r record;
BEGIN
    FOR r IN
        SELECT f.variate, f.step, f.quantile, f.value, e.value AS want
        FROM tsfm_forecast_ex(ARRAY[100,100.9788367,101.8347122,102.4640782,102.7991472,102.8185949,102.5509264,102.0699763,101.4832517,100.9149591,100.486395,100.2967959,100.4076708,100.8330907,101.5374667,102.441169,103.4330984,104.3882267,105.1873357,105.7358393,51.5,51.53300473,51.43800342,51.23241495,50.94353663,50.6061058,50.25919686,49.94273084,49.69390943,49.54389179,49.51501126,49.61878035,49.85486238,50.21110154,50.66460877,51.1838063,51.73124848,52.26696661,52.75203931,53.15206918]::float8[], 4, 2, ARRAY[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0]::float8[], ARRAY[1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0]::float8[], NULL, NULL) f
        JOIN expected e USING (variate, step, quantile)
    LOOP
        got := got + 1;
        IF abs(r.value - r.want) > 1e-3 THEN
            RAISE WARNING 'mismatch variate=% step=% quantile=% got=% want=%', r.variate, r.step, r.quantile, r.value, r.want;
            bad := bad + 1;
        END IF;
    END LOOP;
    IF got <> (SELECT count(*) FROM expected) THEN
        RAISE EXCEPTION 'expected % values, compared %', (SELECT count(*) FROM expected), got;
    END IF;
    IF bad > 0 THEN
        RAISE EXCEPTION '% of % Timestral parity values differ by more than 1e-3', bad, got;
    END IF;
    RAISE NOTICE 'timestral parity ok: % values within 1e-3 (threads=%)', got, current_setting('tsfm.threads');
END $$;
