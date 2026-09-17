# Self-contact solver fix, on top of `physics-dev`

Small follow-up on `d3e8015` ("new physics parameters, astropy based,
jittering seems fixed"). That fixed the original jitter, but a related
explosion still happened on a subset of bodies — mainly repeated-segment
ones (`centipede_3/4/5`, `gecko`).

## The bug

When two non-adjacent modules swing into an exactly-parallel, flush pose
during a gait, MuJoCo's mesh↔mesh convex collider can misreport a stable
near-zero gap as tens of mm of penetration in a single step. The stiff 5 ms
self-contact `solref` then ejects the robot violently (`qvel` > 100 rad/s in
one 2 ms step). `margin` tweaks and non-shared mesh assets didn't help; a
minimal 2-body repro couldn't reproduce it either — it needs the full
multi-body contact context of a real robot.

**Fix:** soften the *self*-contact `solref` time constant only (robot↔robot
geom pairs, separate from floor contacts) -
`MujocoConfig.self_contact_solref_timeconst` in `mujoco_params.py`, applied
in `_base_world.py`.

## Bisection (on `centipede_3`, ~90-100% explosion rate at 5 ms)

| solref | Result |
|---|---|
| 10 ms | still exploding (10/10) |
| 12 ms | still exploding (9/10) |
| 14 ms | **0/10** |
| 16-20 ms | clean |

**14 ms** used — lowest value that fully fixes it. Re-confirmed 0/40 on
`centipede_3` (two seed batches) and 0/10 each on `gecko`, `centipede_4`,
`centipede_5`. Regression check on the 9 bodies that already worked: fitness
swung ±19-27% with no directional harm: normal CMA-ES run noise.

**Known tradeoff:** `solref` is geom-level, so MuJoCo's `solmix` averaging
also softens floor↔robot contacts (5 ms → 9.5 ms). Still well above the
4 ms stability floor and no measured harm, but a `<pair>`-based override
would be more surgical if that ever matters — not implemented here.

## Results after the fix

CMA-ES controller evolution across all 13 `john_set` bodies, 4 seeds each,
300 generations, in `OlympicArena` (`__data__/claude_neuroevo_sweep.py`).

**15,600 evals total, 0 explosions**, including on all 4 previously-exploding
bodies. Fitness = forward displacement (m) over an 8 s rollout; "mean best"
is the mean of each seed's best individual, across 4 seeds:

| body | actuators | mean best (m) | range | exploded |
|---|---|---|---|---|
| baby_a | 8 | 0.204 | [0.083, 0.338] | 0 |
| baby_b | 10 | 0.147 | [0.135, 0.157] | 0 |
| **gecko** | 6 | 0.132 | [0.083, 0.244] | 0 |
| linkin_modified | 12 | 0.175 | [0.159, 0.192] | 0 |
| snake | 8 | 0.224 | [0.188, 0.256] | 0 |
| turtle | 13 | 0.306 | [0.251, 0.412] | 0 |
| iguana | 8 | 0.204 | [0.069, 0.585] | 0 |
| spider_8 | 8 | 0.327 | [0.240, 0.452] | 0 |
| spider_12 | 12 | 0.206 | [0.183, 0.240] | 0 |
| spider_16 | 16 | 0.199 | [0.180, 0.216] | 0 |
| **centipede_3** | 11 | 0.154 | [0.132, 0.181] | 0 |
| **centipede_4** | 14 | 0.195 | [0.113, 0.271] | 0 |
| **centipede_5** | 17 | 0.168 | [0.146, 0.195] | 0 |

*(bold = previously exploding pre-fix)*


## Files changed

- `mujoco_params.py` / `_base_world.py` — the fix.
- `.claude/dev/testing_and_tooling.md`, `docs/source/ec_course_docs.md`,
  `uv.lock` — unrelated small doc/lockfile touch-ups bundled from the same
  working tree.
