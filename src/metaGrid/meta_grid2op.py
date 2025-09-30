import numpy as np
import grid2op
from lightsim2grid import LightSimBackend
from typing import Dict, Any, List, Optional
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import (
    RandomLineOpponent,
    GeometricOpponent,
    BaseActionBudget,
    get_kwargs_no_opponent,
)



class AttackScheduleEnv:
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
