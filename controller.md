# Simple Controller

This controller is shared by `run_scenario.py --controller simple` and `run_simple_circle.py`. Both call the same control law in [`jsbsim_gym/simple_controller.py`](/Users/dmrc/git/jsbsim-gym/jsbsim_gym/simple_controller.py) and read the same shared gains from [`output/simple_controller/simple_controller_optuna_best.json`](/Users/dmrc/git/jsbsim-gym/output/simple_controller/simple_controller_optuna_best.json). The only scenario difference is the reference path: canyon centerline vs. circle.

## Block Diagram

```text
reference path + aircraft state
            |
            v
 nearest-point / lookahead projection
            |
            v
  lateral error e_y, heading error e_psi
            |
            v
 track-normal acceleration command

 a_track =
   k_y    * e_y
 + k_ydot * e_y_dot
 + k_psi  * V * sin(e_psi)

            |
      +-----+-----+
      |           |
      v           v
   roll loop   load-factor loop

 phi_des = atan2(a_track, g)
 aileron = k_phi * (phi_des - phi) + k_p * p

 Nz_des = sqrt(1 + (a_track / g)^2) + altitude_bias
 altitude_bias = clip(-k_h * altitude_error, +/- h_bias_max)
 elevator = k_nz * (Nz_des - Nz) + k_q * q

            |
            v
 side-acceleration regulation

 rudder = k_r * r + k_ny * Ny

 speed hold

 throttle = throttle_base + k_v * (V_target - V)
```

## Exact Control Structure

1. Project the aircraft onto the reference trajectory and pick a lookahead sample `lookahead_rows` ahead.
2. Compute perpendicular-to-track acceleration:

   `a_track = k_y * lateral_error + k_ydot * lateral_error_rate + k_psi * speed * sin(heading_error)`

3. Convert `a_track` into desired bank:

   `phi_des = atan2(a_track, g)`

4. Aileron tracks `phi_des` with roll-angle error plus roll-rate damping.
5. Elevator tracks desired normal load:

   `Nz_des = sqrt(1 + (a_track / g)^2) + altitude_bias`

   `altitude_bias = clip(-k_h * altitude_error, +/- nz_altitude_max_bias)`

6. Rudder drives body-frame side acceleration `Ny` toward zero.
7. Throttle regulates speed around `target_speed_fps`.

## Shared Active Gains

The current implementation uses these shared parameters and no others:

- `lookahead_rows`
- `track_accel_lateral_gain`
- `track_accel_lateral_rate_gain`
- `track_accel_heading_gain`
- `track_accel_max_fps2`
- `roll_p_gain`
- `roll_rate_damping`
- `roll_max_rad`
- `pitch_nz_gain`
- `pitch_q_damping`
- `nz_altitude_gain`
- `nz_altitude_max_bias`
- `nz_min_cmd`
- `nz_max_cmd`
- `yaw_ny_gain`
- `yaw_rate_damping`
- `yaw_max_cmd`
- `throttle_base`
- `throttle_speed_gain`
- `throttle_max`

Scenario-specific settings such as `target_speed_fps` and `use_dem_centerline` are not part of the shared gain file.
