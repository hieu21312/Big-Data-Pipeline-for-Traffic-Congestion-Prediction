CREATE KEYSPACE IF NOT EXISTS traffic WITH replication =
  {'class':'SimpleStrategy','replication_factor':1};
USE traffic;

CREATE TABLE IF NOT EXISTS readings_by_sensor_day (
  sensor_id text, day date, ts timestamp, speed double,
  PRIMARY KEY ((sensor_id, day), ts)
) WITH CLUSTERING ORDER BY (ts DESC);

CREATE TABLE IF NOT EXISTS sensors (
  sensor_id text PRIMARY KEY,
  latitude double, longitude double,
  fwy text, dir text, abs_pm double, lanes int,
  free_flow double
);

CREATE TABLE IF NOT EXISTS sensor_neighbors (
  sensor_id text, neighbor_id text, weight double,
  PRIMARY KEY ((sensor_id), weight, neighbor_id)
) WITH CLUSTERING ORDER BY (weight DESC, neighbor_id ASC);

CREATE TABLE IF NOT EXISTS network_snapshot (
  bucket text, ts timestamp, sensor_id text,
  latitude double, longitude double, speed double, congested boolean,
  PRIMARY KEY ((bucket), ts, sensor_id)
) WITH CLUSTERING ORDER BY (ts DESC, sensor_id ASC);

CREATE TABLE IF NOT EXISTS predictions (
  sensor_id text, ts timestamp, horizon_min int,
  pred_speed double, pred_congested boolean, prob double,
  PRIMARY KEY ((sensor_id), ts, horizon_min)
) WITH CLUSTERING ORDER BY (ts DESC, horizon_min ASC);

CREATE TABLE IF NOT EXISTS signal_plans (
  intersection_id text, ts timestamp,
  cycle_sec int, green_ns int, green_ew int, est_delay double,
  PRIMARY KEY ((intersection_id), ts)
) WITH CLUSTERING ORDER BY (ts DESC);