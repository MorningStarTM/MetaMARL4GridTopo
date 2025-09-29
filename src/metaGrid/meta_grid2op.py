import numpy as np
import grid2op
import gym
from typing import Dict, Any, List, Optional
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import (
    RandomLineOpponent,
    GeometricOpponent,
    BaseActionBudget,
    get_kwargs_no_opponent,
)


class AttackScheduleEnv(gym.Env):
    """
    One-class meta-env focused ONLY on opponent attack schedules for Grid2Op.
    Tasks encode the schedule knobs. Use sample_tasks(...) and reset_task(...).

    Task schema (examples)
    ----------------------
    # Random schedule (cooldown/duration in HOURS)
    {
      "family": "random",
      "cooldown_h": 24,
      "duration_h": 4,
      "lines_attacked": ["L_0", "L_3"],
      "budget_per_ts": 0.5,
      "init_budget": 0.0
    }

    # Geometric schedule (hazard/min duration in HOURS)
    {
      "family": "geometric",
      "attack_every_h": 24.0,
      "avg_attack_duration_h": 4.0,
      "min_attack_duration_h": 2,
      "lines_attacked": ["L_1"],
      "budget_per_ts": 0.5,
      "init_budget": 0.0
    }
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_name: str = "l2rpn_case14_sandbox", task: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        super().__init__()
        self.env_name = env_name
        self._rng = np.random.RandomState(seed)
        self._task: Dict[str, Any] = task or {}
        self.env = None
        self._steps_per_hour = self._infer_steps_per_hour()
        self._line_names = self._infer_line_names()
        self._make_env_for_current_task()
        vec = self._to_vec(self.env.reset())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=vec.shape, dtype=np.float32)
        self.action_space = self.env.action_space

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
                t = dict(
                    family="random",
                    cooldown_h=int(self._rng.randint(6, 49)),   # 6–48 h
                    duration_h=int(self._rng.randint(1, 9)),    # 1–8 h
                    lines_attacked=lines,
                    budget_per_ts=budget_per_ts,
                    init_budget=init_budget,
                )
            else:
                t = dict(
                    family="geometric",
                    attack_every_h=float(self._rng.uniform(6.0, 72.0)),
                    avg_attack_duration_h=float(self._rng.uniform(1.0, 8.0)),
                    min_attack_duration_h=int(self._rng.randint(1, 5)),
                    lines_attacked=lines,
                    budget_per_ts=budget_per_ts,
                    init_budget=init_budget,
                )
            tasks.append(t)
        return tasks

    def reset_task(self, task: Dict[str, Any]):
        self._task = dict(task)
        self._make_env_for_current_task()

    # ---------- gym API ----------
    def reset(self):
        return self._to_vec(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info)
        info["task"] = self._task
        return self._to_vec(obs), float(reward), bool(done), info

    def render(self, mode="human"):
        return getattr(self.env, "render", lambda *a, **k: None)()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # ---------- helpers ----------
    def _make_env_for_current_task(self):
        if self.env is not None:
            self.env.close()
        t = self._task or {}

        # Defaults
        lines_attacked = t.get("lines_attacked", self._line_names)
        budget_per_ts = float(t.get("budget_per_ts", 0.5))
        init_budget = float(t.get("init_budget", 0.0))
        family = t.get("family", "random")

        if family == "random":
            cooldown_steps = self._steps_per_hour * int(t.get("cooldown_h", 24))
            duration_steps = self._steps_per_hour * int(t.get("duration_h", 4))
            self.env = grid2op.make(
                self.env_name,
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
            # no-opponent fallback
            self.env = grid2op.make(self.env_name, **get_kwargs_no_opponent())

    def _infer_steps_per_hour(self) -> int:
        tmp = grid2op.make(self.env_name, **get_kwargs_no_opponent())
        try:
            return int(round(3600.0 / tmp.delta_time_seconds))
        finally:
            tmp.close()

    def _infer_line_names(self) -> List[str]:
        tmp = grid2op.make(self.env_name, **get_kwargs_no_opponent())
        try:
            return list(tmp.name_line)
        finally:
            tmp.close()

    def _to_vec(self, obs) -> np.ndarray:
        # Minimal: just use Grid2Op's numeric vector
        return np.asarray(obs.to_vect(), dtype=np.float32).flatten()


