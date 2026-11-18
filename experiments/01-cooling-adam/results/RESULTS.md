# Cooling PINN results

These runs solve one forward cooling problem on 0–10 minutes.

Each seed used 6000 Adam steps. Checkpoints were selected using validation residual loss plus the soft initial-condition loss. Exact temperatures were used only after selection.

| Seed | Selected step | MAE (°C) | RMSE (°C) | Max error (°C) | IC error (°C) | Residual RMS (°C/min) |
|---|---|---|---|---|---|---|
| 7 | 6000 | 0.00252663 | 0.00351887 | 0.00727131 | 0.000162641 | 0.00294568 |
| 21 | 5900 | 0.00226556 | 0.00269999 | 0.00527257 | 0.000168197 | 0.0043825 |
| 42 | 6000 | 0.00376351 | 0.00459602 | 0.00863944 | 0.000115526 | 0.00412952 |

Across seeds (sample standard deviation uses n−1):

- mae_c: mean 0.0028519; sample SD 0.000800194; range [0.00226556, 0.00376351].
- rmse_c: mean 0.00360496; sample SD 0.000950944; range [0.00269999, 0.00459602].
- max_abs_error_c: mean 0.00706111; sample SD 0.00169325; range [0.00527257, 0.00863944].
- initial_abs_error_c: mean 0.000148788; sample SD 2.89393e-05; range [0.000115526, 0.000168197].
- test_residual_rms_c_per_min: mean 0.00381923; sample SD 0.00076702; range [0.00294568, 0.0043825].

All saved checkpoints reproduced their in-memory predictions exactly on 101 probe times after reload.

The 1,001-point final grid checks solution error and residuals within the training interval. Its interior points are held out; the initial-condition endpoint is a known constraint. It does not establish an error bound between points. These 3 seeds describe this small experiment; they do not establish broad reliability. No changed coefficients, inverse estimation, noisy observations, or extrapolation were tested.
