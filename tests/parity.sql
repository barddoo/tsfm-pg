\set ON_ERROR_STOP on
-- Numerical parity with tsfm-zig / Chronos-2. Run through tests/run_parity.sh,
-- which supplies :model_dir and :threads. The fixture is 64 deterministic steps,
-- forecast 8 steps; expected values come from the CLI (same engine, no Postgres).
CREATE EXTENSION IF NOT EXISTS tsfm;
SET tsfm.stub = off;
SET tsfm.model_dir = :'model_dir';
SET tsfm.threads = :threads;
SET tsfm.sort_quantiles = on;

CREATE TEMP TABLE history (i integer, v double precision);
INSERT INTO history VALUES (1, 100.69104);
INSERT INTO history VALUES (2, 101.329285);
INSERT INTO history VALUES (3, 101.866654);
INSERT INTO history VALUES (4, 102.264078);
INSERT INTO history VALUES (5, 102.49499);
INSERT INTO history VALUES (6, 102.547695);
INSERT INTO history VALUES (7, 102.426419);
INSERT INTO history VALUES (8, 102.150926);
INSERT INTO history VALUES (9, 101.75476);
INSERT INTO history VALUES (10, 101.28224);
INSERT INTO history VALUES (11, 100.784509);
INSERT INTO history VALUES (12, 100.314959);
INSERT INTO history VALUES (13, 99.924468);
INSERT INTO history VALUES (14, 99.656848);
INSERT INTO history VALUES (15, 99.54494);
INSERT INTO history VALUES (16, 99.607671);
INSERT INTO history VALUES (17, 99.848371);
INSERT INTO history VALUES (18, 100.254471);
INSERT INTO history VALUES (19, 100.798629);
INSERT INTO history VALUES (20, 101.441169);
INSERT INTO history VALUES (21, 102.133628);
INSERT INTO history VALUES (22, 102.823083);
INSERT INTO history VALUES (23, 103.45688);
INSERT INTO history VALUES (24, 103.987336);
INSERT INTO history VALUES (25, 104.376);
INSERT INTO history VALUES (26, 104.597087);
INSERT INTO history VALUES (27, 104.63978);
INSERT INTO history VALUES (28, 104.509198);
INSERT INTO history VALUES (29, 104.225938);
INSERT INTO history VALUES (30, 103.824237);
INSERT INTO history VALUES (31, 103.348909);
INSERT INTO history VALUES (32, 102.851346);
INSERT INTO history VALUES (33, 102.384928);
INSERT INTO history VALUES (34, 102.000251);
INSERT INTO history VALUES (35, 101.740608);
INSERT INTO history VALUES (36, 101.638128);
INSERT INTO history VALUES (37, 101.710895);
INSERT INTO history VALUES (38, 101.961343);
INSERT INTO history VALUES (39, 102.376033);
INSERT INTO history VALUES (40, 102.926854);
INSERT INTO history VALUES (41, 103.573536);
INSERT INTO history VALUES (42, 104.267246);
INSERT INTO history VALUES (43, 104.954949);
INSERT INTO history VALUES (44, 105.584147);
INSERT INTO history VALUES (45, 106.107569);
INSERT INTO history VALUES (46, 106.487391);
INSERT INTO history VALUES (47, 106.698619);
INSERT INTO history VALUES (48, 106.731316);
INSERT INTO history VALUES (49, 106.591494);
INSERT INTO history VALUES (50, 106.300576);
INSERT INTO history VALUES (51, 105.893481);
INSERT INTO history VALUES (52, 105.415507);
INSERT INTO history VALUES (53, 104.918283);
INSERT INTO history VALUES (54, 104.455156);
INSERT INTO history VALUES (55, 104.076429);
INSERT INTO history VALUES (56, 103.824866);
INSERT INTO history VALUES (57, 103.73187);
INSERT INTO history VALUES (58, 103.814681);
INSERT INTO history VALUES (59, 104.074835);
INSERT INTO history VALUES (60, 104.498026);
INSERT INTO history VALUES (61, 105.055383);
INSERT INTO history VALUES (62, 105.706053);
INSERT INTO history VALUES (63, 106.400845);
INSERT INTO history VALUES (64, 107.08663);

CREATE TEMP TABLE expected (step integer, quantile double precision, value double precision);
INSERT INTO expected VALUES (1, 0.1, 107.189896);
INSERT INTO expected VALUES (1, 0.5, 107.47388);
INSERT INTO expected VALUES (1, 0.9, 107.71905);
INSERT INTO expected VALUES (2, 0.1, 107.71369);
INSERT INTO expected VALUES (2, 0.5, 107.97313);
INSERT INTO expected VALUES (2, 0.9, 108.227295);
INSERT INTO expected VALUES (3, 0.1, 108.07124);
INSERT INTO expected VALUES (3, 0.5, 108.31667);
INSERT INTO expected VALUES (3, 0.9, 108.55225);
INSERT INTO expected VALUES (4, 0.1, 108.282455);
INSERT INTO expected VALUES (4, 0.5, 108.47672);
INSERT INTO expected VALUES (4, 0.9, 108.80994);
INSERT INTO expected VALUES (5, 0.1, 108.16565);
INSERT INTO expected VALUES (5, 0.5, 108.451164);
INSERT INTO expected VALUES (5, 0.9, 108.75329);
INSERT INTO expected VALUES (6, 0.1, 107.9501);
INSERT INTO expected VALUES (6, 0.5, 108.19583);
INSERT INTO expected VALUES (6, 0.9, 108.538284);
INSERT INTO expected VALUES (7, 0.1, 107.6399);
INSERT INTO expected VALUES (7, 0.5, 107.93172);
INSERT INTO expected VALUES (7, 0.9, 108.21702);
INSERT INTO expected VALUES (8, 0.1, 107.27057);
INSERT INTO expected VALUES (8, 0.5, 107.5793);
INSERT INTO expected VALUES (8, 0.9, 107.86558);

DO $$
DECLARE
    got integer := 0;
    bad integer := 0;
    r record;
BEGIN
    IF (SELECT count(*) FROM tsfm_forecast((SELECT array_agg(v ORDER BY i) FROM history), 8)) <> 8 * 21 THEN
        RAISE EXCEPTION 'unexpected forecast row count';
    END IF;

    FOR r IN
        SELECT f.step, f.quantile, f.value, e.value AS want
        FROM tsfm_forecast((SELECT array_agg(v ORDER BY i) FROM history), 8) f
        JOIN expected e USING (step, quantile)
    LOOP
        got := got + 1;
        IF abs(r.value - r.want) > 1e-3 THEN
            RAISE WARNING 'mismatch step=% quantile=% got=% want=%', r.step, r.quantile, r.value, r.want;
            bad := bad + 1;
        END IF;
    END LOOP;

    IF got <> (SELECT count(*) FROM expected) THEN
        RAISE EXCEPTION 'expected % values, compared %', (SELECT count(*) FROM expected), got;
    END IF;
    IF bad > 0 THEN
        RAISE EXCEPTION '% of % parity values differ by more than 1e-3', bad, got;
    END IF;
    RAISE NOTICE 'parity ok: % values within 1e-3 (threads=%)', got, current_setting('tsfm.threads');
END $$;
