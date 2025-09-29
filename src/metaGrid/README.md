
# Attack Schedule Env


## Key ideas & flow

* **Tasks = “how/when to attack”** (and which lines), nothing else.
* Two **families**:

  * `"random"` → fixed **cooldown** (min time between attacks) and fixed **duration** (outage length).
  * `"geometric"` → **hazard-based timing**: average time between attacks and min duration (attacks last a geometrically distributed number of hours).
* You can **sample** tasks with realistic spreads or **reset** to a specific task to train/evaluate.

---

## Constructor (`__init__`)

* Saves the `env_name`, a RNG (`seed`), and an optional starting `task`.
* Figures out:

  * **steps per hour** from the base env (`delta_time_seconds` → `3600 / dt`). Used to convert *hours → steps* for cooldown/duration.
  * **valid line names** (`name_line`) so tasks can pick real lines.
* Calls `self._make_env_for_current_task()` which builds a Grid2Op env using the current task’s schedule.
* Sets `observation_space` dynamically from a first observation vector, and proxies the **Grid2Op action space** as-is.

---

## Task API

### `sample_tasks(num_tasks)`

* Returns a list of dicts like:

  * Random:

    ```bash
    {
      "family": "random",
      "cooldown_h": 24,     # hours between attacks
      "duration_h": 4,      # outage length (hours)
      "lines_attacked": ["L_0", "L_3"],
      "budget_per_ts": 0.5,
      "init_budget": 0.0
    }
    ```
  * Geometric:

    ```bash
    {
      "family": "geometric",
      "attack_every_h": 24.0,         # avg hours between attacks (hazard)
      "avg_attack_duration_h": 4.0,   # mean outage length
      "min_attack_duration_h": 2,     # minimum outage length
      "lines_attacked": ["L_1"],
      "budget_per_ts": 0.5,
      "init_budget": 0.0
    }
    ```
* It also chooses 1–5 **lines** uniformly from valid line names, and **budgets** from a small set.

### `reset_task(task)`

* Saves the dict and calls `_make_env_for_current_task()` to **recreate** the Grid2Op env with that config.

---

## Turning a task into an opponent env

### `_make_env_for_current_task()`

* Reads common knobs: `lines_attacked`, `budget_per_ts`, `init_budget`.
* If `family == "random"`:

  * Converts `cooldown_h`, `duration_h` → **steps** using `steps_per_hour`.
  * Calls `grid2op.make(..., opponent_class=RandomLineOpponent, opponent_attack_cooldown=..., opponent_attack_duration=..., kwargs_opponent={"lines_attacked": ...}, opponent_budget_per_ts=..., opponent_init_budget=..., opponent_action_class=PowerlineSetAction, opponent_budget_class=BaseActionBudget)`.
* If `family == "geometric"`:

  * Uses hours **directly** in `kwargs_opponent`:

    * `"attack_every_xxx_hour"`, `"average_attack_duration_hour"`, `"minimum_attack_duration_hour"`.
  * Calls `grid2op.make(..., opponent_class=GeometricOpponent, kwargs_opponent={...}, budgets...)`.
* Else → **no opponent** (useful for debugging).
* Resets the env and updates `observation_space` and `action_space` accordingly.

### Budgets (how many attacks happen)

* `opponent_init_budget`: how much **attack budget** you start with (enables immediate attacks).
* `opponent_budget_per_ts`: **regeneration rate** per timestep. Higher → more frequent/longer attacks (subject to cooldown/hazard).

### `lines_attacked`

* Restricts which lines the opponent is allowed to trip. You populate it with valid names from the base env.



---

## Why hours vs steps?

* **Random** schedule uses **env-level** knobs `opponent_attack_cooldown` and `opponent_attack_duration`, which are defined in **steps**. That’s why you compute `steps_per_hour` and multiply.
* **Geometric** opponent expects schedule **in hours** inside its own `kwargs_opponent`, so you pass those floats/ints directly.

---

## Quick mental model

* **Random**: “Every `cooldown_h` hours, if budget allows, start an attack that lasts exactly `duration_h` hours.”
* **Geometric**: “Attacks occur with an average spacing of `attack_every_h` hours. Each attack lasts a random time with mean `avg_attack_duration_h`, but never less than `min_attack_duration_h`.”

---

## Typical usage

```bash
env = AttackScheduleEnv("l2rpn_case14_sandbox", seed=0)

# Train-time: repeatedly sample tasks and adapt
tasks = env.sample_tasks(10)
env.reset_task(tasks[0])
obs = env.reset()
for _ in range(10):
    action = env.action_space()   # or your policy(action|obs)
    obs, r, d, info = env.step(action)
    if d: break
env.close()
```


