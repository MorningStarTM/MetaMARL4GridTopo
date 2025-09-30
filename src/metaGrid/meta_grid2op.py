import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from typing import Dict, Any, List, Optional
from grid2op.Action import PowerlineSetAction
from grid2op import Environment
from grid2op.Opponent import (
    RandomLineOpponent,
    GeometricOpponentMultiArea,
    GeometricOpponent,
    BaseActionBudget,
    get_kwargs_no_opponent,
)

class AttackScheduleEnv(object):
    """
    Grid2Op-based meta-env focused ONLY on opponent attack schedules.

    • No Gym inheritance.
    • reset()/step() return Grid2Op-native Observation, reward, done, info.
    • sample_tasks(...) returns schedule dicts.
    • reset_task(...) applies a schedule (with built-in sanitization).

    Grid rules enforced:
      - Geometric: avg_attack_duration_hour > min_attack_duration_hour (strict).
      - Random: cooldown_h >= duration_h (to avoid degenerate schedules).
      - lines_attacked is never empty and only uses valid line names.
    """

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        task: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        backend=None,  # e.g., backend=LightSimBackend()
    ):
        self.env_name = env_name
        self._rng = np.random.RandomState(seed)
        self._task: Dict[str, Any] = task or {}
        self.env = None

        # Cache basics from a no-opponent env
        self._steps_per_hour = self._infer_steps_per_hour()
        self._line_names = self._infer_line_names()

        # Build the working env with current task (or no-opponent if empty)
        self._make_env_for_current_task()

    # ---------- expose spaces ----------
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    # ---------- public meta-API ----------
    def sample_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        tasks = []
        for _ in range(num_tasks):
            family = self._rng.choice(["random", "geometric"])

            k = int(self._rng.randint(1, max(2, min(5, len(self._line_names)) + 1)))
            lines = self._rng.choice(self._line_names, size=k, replace=False).tolist()
            budget_per_ts = float(self._rng.choice([0.25, 0.5, 1.0]))
            init_budget = float(self._rng.choice([0.0, 0.5, 1.0]))

            if family == "random":
                duration_h = int(self._rng.randint(1, 9))            # 1–8 h
                cooldown_h = int(self._rng.randint(duration_h, 49))  # ensure cooldown ≥ duration
                t = dict(
                    family="random",
                    cooldown_h=cooldown_h,
                    duration_h=duration_h,
                    lines_attacked=lines,
                    budget_per_ts=budget_per_ts,
                    init_budget=init_budget,
                )
            else:
                # Pick min first (1–4), then sample avg strictly greater, cap to 8
                min_h = int(self._rng.randint(1, 5))
                # ensure avg > min by at least eps
                eps = 1e-3
                max_avg = max(min_h + 0.5, 8.0)  # don't shrink the range too much
                avg_h = float(self._rng.uniform(min_h + 0.5, 8.0))
                if avg_h <= min_h:
                    avg_h = min_h + 0.5
                # Average spacing between attacks (no strict rule vs durations; keep broad)
                attack_every_h = float(self._rng.uniform(6.0, 72.0))
                t = dict(
                    family="geometric",
                    attack_every_h=attack_every_h,
                    avg_attack_duration_h=avg_h,
                    min_attack_duration_h=min_h,
                    lines_attacked=lines,
                    budget_per_ts=budget_per_ts,
                    init_budget=init_budget,
                )
            tasks.append(self._sanitize_task(t))
        return tasks

    def reset_task(self, task: Dict[str, Any]):
        """Rebuild the underlying Grid2Op env with the new opponent schedule."""
        self._task = self._sanitize_task(task)
        self._make_env_for_current_task()

    # ---------- Grid2Op-style API ----------
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info)
        info["task"] = self._task
        return obs, float(reward), bool(done), info

    def render(self, *args, **kwargs):
        return getattr(self.env, "render", lambda *a, **k: None)(*args, **kwargs)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ---------- helpers ----------
    def _sanitize_task(self, t: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp / adjust task params to satisfy Grid2Op opponent constraints."""
        t = dict(t)  # copy
        family = t.get("family", "random")
        # Ensure valid, non-empty line set
        lines = t.get("lines_attacked", self._line_names)
        if not lines:
            lines = self._line_names
        else:
            # keep only valid names
            lines = [ln for ln in lines if ln in self._line_names] or self._line_names
        t["lines_attacked"] = lines

        # Budgets sane non-negatives
        t["budget_per_ts"] = float(max(0.0, t.get("budget_per_ts", 0.5)))
        t["init_budget"] = float(max(0.0, t.get("init_budget", 0.0)))

        if family == "random":
            dur = int(max(1, int(t.get("duration_h", 4))))
            cool = int(max(dur, int(t.get("cooldown_h", 24))))  # cooldown ≥ duration
            t["duration_h"] = dur
            t["cooldown_h"] = cool
            t["family"] = "random"
        else:
            # geometric: avg > min (strict)
            min_h = int(max(1, int(t.get("min_attack_duration_h", 2))))
            avg_h = float(t.get("avg_attack_duration_h", max(min_h + 0.5, 4.0)))
            eps = 1e-6
            if avg_h <= min_h:
                avg_h = float(min_h) + 0.5  # push strictly above
            t["min_attack_duration_h"] = min_h
            t["avg_attack_duration_h"] = avg_h
            # attack_every_h: keep reasonable positive hours
            t["attack_every_h"] = float(max(0.1, t.get("attack_every_h", 24.0)))
            t["family"] = "geometric"
        return t

    def _make_env_for_current_task(self):
        if self.env is not None:
            self.env.close()

        t = self._task or {}
        lines_attacked = t.get("lines_attacked", self._line_names)
        budget_per_ts = float(t.get("budget_per_ts", 0.5))
        init_budget = float(t.get("init_budget", 0.0))
        family = t.get("family", "random")

        if family == "random":
            cooldown_steps = self._steps_per_hour * int(t.get("cooldown_h", 24))
            duration_steps = self._steps_per_hour * int(t.get("duration_h", 4))
            self.env = grid2op.make(
                self.env_name,
                backend=LightSimBackend(),
                opponent_attack_cooldown=cooldown_steps,
                opponent_attack_duration=duration_steps,
                opponent_budget_per_ts=budget_per_ts,
                opponent_init_budget=init_budget,
                opponent_action_class=PowerlineSetAction,
                opponent_class=RandomLineOpponent,
                opponent_budget_class=BaseActionBudget,
                kwargs_opponent={"lines_attacked": lines_attacked},
            )
        elif family == "geometric":
            self.env = grid2op.make(
                self.env_name,
                backend=LightSimBackend(),
                opponent_budget_per_ts=budget_per_ts,
                opponent_init_budget=init_budget,
                opponent_action_class=PowerlineSetAction,
                opponent_class=GeometricOpponent,
                opponent_budget_class=BaseActionBudget,
                kwargs_opponent={
                    "lines_attacked": lines_attacked,
                    "attack_every_xxx_hour": float(t.get("attack_every_h", 24.0)),
                    "average_attack_duration_hour": float(t.get("avg_attack_duration_h", 4.0)),
                    "minimum_attack_duration_hour": int(t.get("min_attack_duration_h", 2)),
                },
            )
        else:
            # Fallback: no opponent
            self.env = grid2op.make(self.env_name, backend=self.backend, **get_kwargs_no_opponent())

    def _infer_steps_per_hour(self) -> int:
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return int(round(3600.0 / tmp.delta_time_seconds))
        finally:
            tmp.close()

    def _infer_line_names(self) -> List[str]:
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return list(tmp.name_line)
        finally:
            tmp.close()

    def _thermal_limit(self):
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return tmp._thermal_limit_a
        finally:
            tmp.close()






class GeometricMetaEnv:
    """
    Meta-environment where EACH TASK configures a GeometricOpponent.

    Task schema
    -----------
    {
      "family": "geometric",                 # fixed to 'geometric'
      "lines_attacked": ["L_0","L_3"],       # non-empty subset of valid line names
      "attack_every_h": 24.0,                # average hours between attacks (>0)
      "avg_attack_duration_h": 4.0,          # strictly > min_attack_duration_h
      "min_attack_duration_h": 2,            # >=1
      "pmax_pmin_ratio": 4.0,                # >=1.0 (1.0 ~ uniform over lines_attacked)
      "budget_per_ts": 0.5,                  # >=0
      "init_budget": 0.0                     # >=0
    }
    """

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        task: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        backend=None,  # e.g., LightSimBackend()
    ):
        self.env_name = env_name
        self.backend = backend
        self._rng = np.random.RandomState(seed)
        self._task: Dict[str, Any] = task or {}
        self.env = None

        # Cache valid line names
        self._line_names = self._infer_line_names()

        # Build env with current task (or no-opponent if empty)
        self._make_env_for_current_task()

    # -------- spaces passthrough --------
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    # -------- meta-API --------
    def sample_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for _ in range(num_tasks):
            # choose lines subset (1–5)
            k = int(self._rng.randint(1, max(2, min(5, len(self._line_names)) + 1)))
            lines = self._rng.choice(self._line_names, size=k, replace=False).tolist()

            # durations: pick min first, then avg strictly greater
            min_h = int(self._rng.randint(1, 5))                    # 1–4 h
            avg_h = float(self._rng.uniform(min_h + 0.5, 8.0))      # > min_h
            attack_every_h = float(self._rng.uniform(6.0, 72.0))    # spacing between attacks

            # selectivity: how much more likely stressed lines are attacked
            pmax_pmin_ratio = float(self._rng.choice([1.0, 2.0, 4.0, 8.0]))

            # budgets
            budget_per_ts = float(self._rng.choice([0.25, 0.5, 1.0]))
            init_budget = float(self._rng.choice([0.0, 0.5, 1.0]))

            t = dict(
                family="geometric",
                lines_attacked=lines,
                attack_every_h=attack_every_h,
                avg_attack_duration_h=avg_h,
                min_attack_duration_h=min_h,
                pmax_pmin_ratio=pmax_pmin_ratio,
                budget_per_ts=budget_per_ts,
                init_budget=init_budget,
            )
            tasks.append(self._sanitize_task(t))
        return tasks

    def reset_task(self, task: Dict[str, Any]):
        self._task = self._sanitize_task(task)
        self._make_env_for_current_task()

    # -------- Grid2Op-style API --------
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info)
        info["task"] = self._task
        return obs, float(reward), bool(done), info

    def render(self, *args, **kwargs):
        return getattr(self.env, "render", lambda *a, **k: None)(*args, **kwargs)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # -------- internals --------
    def _sanitize_task(self, t: Dict[str, Any]) -> Dict[str, Any]:
        t = dict(t)
        # enforce geometric family
        t["family"] = "geometric"

        # lines_attacked: non-empty, valid names
        lines = t.get("lines_attacked", self._line_names)
        if not lines:
            lines = self._line_names
        else:
            lines = [ln for ln in lines if ln in self._line_names] or self._line_names
        t["lines_attacked"] = lines

        # durations: avg > min (strict)
        min_h = int(max(1, int(t.get("min_attack_duration_h", 2))))
        avg_h = float(t.get("avg_attack_duration_h", max(min_h + 0.5, 4.0)))
        if avg_h <= min_h:
            avg_h = float(min_h) + 0.5
        t["min_attack_duration_h"] = min_h
        t["avg_attack_duration_h"] = avg_h

        # spacing between attacks
        t["attack_every_h"] = float(max(0.1, t.get("attack_every_h", 24.0)))

        # selectivity
        t["pmax_pmin_ratio"] = float(max(1.0, t.get("pmax_pmin_ratio", 1.0)))

        # budgets
        t["budget_per_ts"] = float(max(0.0, t.get("budget_per_ts", 0.5)))
        t["init_budget"] = float(max(0.0, t.get("init_budget", 0.0)))
        return t

    def _make_env_for_current_task(self):
        if self.env is not None:
            self.env.close()

        if not self._task:
            # Fallback: no opponent
            self.env = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
            return

        t = self._task
        self.env = grid2op.make(
            self.env_name,
            backend=LightSimBackend(),
            opponent_budget_per_ts=float(t["budget_per_ts"]),
            opponent_init_budget=float(t["init_budget"]),
            opponent_action_class=PowerlineSetAction,
            opponent_class=GeometricOpponent,
            opponent_budget_class=BaseActionBudget,
            kwargs_opponent={
                "lines_attacked": t["lines_attacked"],
                "attack_every_xxx_hour": float(t["attack_every_h"]),
                "average_attack_duration_hour": float(t["avg_attack_duration_h"]),
                "minimum_attack_duration_hour": int(t["min_attack_duration_h"]),
                "pmax_pmin_ratio": float(t["pmax_pmin_ratio"]),
            },
        )

    def _infer_line_names(self) -> List[str]:
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return list(tmp.name_line)
        finally:
            tmp.close()

    
    def _thermal_limit(self):
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return tmp._thermal_limit_a
        finally:
            tmp.close()



class GeometricMultiAreaMetaEnv:
    """
    Meta-env for GeometricOpponentMultiArea.

    Valid task schema (recommended)
    -------------------------------
    {
      "family": "geometric_multi_area",
      "lines_attacked_areas": [               # list of lists (one inner list per area)
        ["L_0","L_3"],
        ["L_5","L_7","L_9"]
      ],
      "attack_every_h": 24.0,                 # > 0   (scalar, shared by all areas)
      "avg_attack_duration_h": 4.0,           # > min (scalar, shared by all areas)
      "min_attack_duration_h": 2,             # >= 1  (scalar, shared by all areas)
      "pmax_pmin_ratio": 2.0,                 # >= 1  (scalar, shared by all areas)
      "budget_per_ts": 0.5,                   # >= 0
      "init_budget": 0.0                      # >= 0
    }

    Backward-compat (optional):
    If you pass {"areas": [{"lines_attacked": [...], ...}, ...]}, we’ll extract the
    line lists and use a single shared schedule (first area's values) to match the
    class’ API.
    """

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        task: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        backend=None,  # e.g., backend=LightSimBackend()
    ):
        self.env_name = env_name
        self.backend = backend
        self._rng = np.random.RandomState(seed)
        self._task: Dict[str, Any] = task or {}
        self.env = None

        # Cache valid line names
        self._line_names = self._infer_line_names()

        # Build env with current task (or no-opponent if empty)
        self._make_env_for_current_task()

    # -------- spaces passthrough --------
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    # -------- meta-API --------
    def sample_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for _ in range(num_tasks):
            # number of areas: 2–4 (capped by number of lines)
            max_areas = min(4, max(2, len(self._line_names)))
            num_areas = int(self._rng.randint(2, max_areas + 1))

            # draw disjoint-ish line subsets; allow overlaps if we run short
            remaining = self._line_names.copy()
            areas_lines: List[List[str]] = []
            for _a in range(num_areas):
                k = int(self._rng.randint(1, max(2, min(6, len(self._line_names)))))  # 1–5 per area
                if len(remaining) >= k:
                    lines = self._rng.choice(remaining, size=k, replace=False).tolist()
                    remaining = [ln for ln in remaining if ln not in lines]
                else:
                    lines = self._rng.choice(self._line_names, size=k, replace=False).tolist()
                areas_lines.append(lines)

            # one shared schedule for all areas (that’s what the class supports)
            min_h = int(self._rng.randint(1, 5))                    # 1–4
            avg_h = float(self._rng.uniform(min_h + 0.5, 8.0))      # > min
            attack_every_h = float(self._rng.uniform(6.0, 72.0))    # spacing
            p_ratio = float(self._rng.choice([1.0, 2.0, 4.0, 8.0]))

            budget_per_ts = float(self._rng.choice([0.25, 0.5, 1.0]))
            init_budget = float(self._rng.choice([0.0, 0.5, 1.0]))

            t = dict(
                family="geometric_multi_area",
                lines_attacked_areas=areas_lines,         # <- list of lists
                attack_every_h=attack_every_h,            # <- scalars
                avg_attack_duration_h=avg_h,
                min_attack_duration_h=min_h,
                pmax_pmin_ratio=p_ratio,
                budget_per_ts=budget_per_ts,
                init_budget=init_budget,
            )
            tasks.append(self._sanitize_task(t))
        return tasks

    def reset_task(self, task: Dict[str, Any]):
        self._task = self._sanitize_task(task)
        self._make_env_for_current_task()

    # -------- Grid2Op-style API --------
    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info)
        info["task"] = self._task
        return obs, float(reward), bool(done), info

    def render(self, *args, **kwargs):
        return getattr(self.env, "render", lambda *a, **k: None)(*args, **kwargs)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # -------- internals --------
    def _sanitize_task(self, t: Dict[str, Any]) -> Dict[str, Any]:
        t = dict(t)
        t["family"] = "geometric_multi_area"

        # --- Backward-compat: accept {"areas": [{...}, ...]} and convert ---
        if "areas" in t and "lines_attacked_areas" not in t:
            areas = t["areas"] or []
            # gather the line lists
            lines_attacked_areas = []
            for ar in areas:
                lines = ar.get("lines_attacked", self._line_names)
                lines = [ln for ln in lines if ln in self._line_names] or self._line_names
                lines_attacked_areas.append(lines)
            t["lines_attacked_areas"] = lines_attacked_areas
            # pick a single shared schedule from first area (or defaults)
            src = areas[0] if areas else {}
            t["attack_every_h"] = float(src.get("attack_every_h", t.get("attack_every_h", 24.0)))
            t["avg_attack_duration_h"] = float(src.get("avg_attack_duration_h", t.get("avg_attack_duration_h", 4.5)))
            t["min_attack_duration_h"] = int(src.get("min_attack_duration_h", t.get("min_attack_duration_h", 2)))
            t["pmax_pmin_ratio"] = float(src.get("pmax_pmin_ratio", t.get("pmax_pmin_ratio", 1.0)))
            # drop old key
            t.pop("areas", None)

        # lines_attacked_areas: ensure non-empty, valid names
        laa = t.get("lines_attacked_areas")
        if not laa:
            laa = [self._line_names]  # single area with all lines
        laa_clean: List[List[str]] = []
        for sub in laa:
            sub = [ln for ln in (sub or []) if ln in self._line_names]
            laa_clean.append(sub or self._line_names)
        t["lines_attacked_areas"] = laa_clean

        # schedules (shared across areas)
        min_h = int(max(1, int(t.get("min_attack_duration_h", 2))))
        avg_h = float(t.get("avg_attack_duration_h", max(min_h + 0.5, 4.0)))
        if avg_h <= min_h:
            avg_h = float(min_h) + 0.5
        t["min_attack_duration_h"] = min_h
        t["avg_attack_duration_h"] = avg_h
        t["attack_every_h"] = float(max(0.1, t.get("attack_every_h", 24.0)))
        t["pmax_pmin_ratio"] = float(max(1.0, t.get("pmax_pmin_ratio", 1.0)))

        # budgets
        t["budget_per_ts"] = float(max(0.0, t.get("budget_per_ts", 0.5)))
        t["init_budget"] = float(max(0.0, t.get("init_budget", 0.0)))
        return t

    def _make_env_for_current_task(self):
        if self.env is not None:
            self.env.close()

        if not self._task:
            # Fallback: no opponent
            self.env = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
            return

        t = self._task
        self.env = grid2op.make(
            self.env_name,
            backend=LightSimBackend(),
            opponent_budget_per_ts=float(t["budget_per_ts"]),
            opponent_init_budget=float(t["init_budget"]),
            opponent_action_class=PowerlineSetAction,
            opponent_class=GeometricOpponentMultiArea,
            opponent_budget_class=BaseActionBudget,
            # IMPORTANT: pass "lines_attacked" (list of lists) + shared scalars
            kwargs_opponent={
                "lines_attacked": t["lines_attacked_areas"],                 # <- list of lists
                "attack_every_xxx_hour": float(t["attack_every_h"]),         # <- scalar
                "average_attack_duration_hour": float(t["avg_attack_duration_h"]),
                "minimum_attack_duration_hour": int(t["min_attack_duration_h"]),
                "pmax_pmin_ratio": float(t["pmax_pmin_ratio"]),
            },
        )

    def _infer_line_names(self) -> List[str]:
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return list(tmp.name_line)
        finally:
            tmp.close()

    def _thermal_limit(self):
        tmp = grid2op.make(self.env_name, backend=LightSimBackend(), **get_kwargs_no_opponent())
        try:
            return tmp._thermal_limit_a
        finally:
            tmp.close()
