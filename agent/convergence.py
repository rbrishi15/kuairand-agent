"""[Rishi] Convergence + compute-budget checks for the outer agent loop
(CLAUDE.md §2, §6 step 7).

Stop the loop when ANY of these trigger, whichever comes first:
  - iteration count hits max_iterations
  - wall-clock time since start() hits max_wallclock_hours
  - validation primary has NOT improved by more than epsilon, in ANY of
    the last N iterations, relative to the best score known before each
    of those iterations
"""
import time


class ConvergenceTracker:
    def __init__(self, epsilon=0.002, N=3, max_iterations=50, max_wallclock_hours=6):
        self.epsilon = epsilon
        self.N = N
        self.max_iterations = max_iterations
        self.max_wallclock_hours = max_wallclock_hours
        self.history = []   # [(iteration, primary_score), ...]
        self.best = -1.0
        self._start = None

    def start(self):
        self._start = time.time()

    def record(self, iteration, primary_score):
        self.history.append((iteration, primary_score))
        self.best = max(self.best, primary_score)

    def should_stop(self):
        if len(self.history) >= self.max_iterations:
            return True, 'iteration_cap'
        if self._start is not None and (time.time() - self._start) >= self.max_wallclock_hours * 3600:
            return True, 'wallclock_cap'
        if len(self.history) >= self.N:
            scores = [s for _, s in self.history]
            recent = scores[-self.N:]
            running_best = max(scores[:-self.N]) if len(scores) > self.N else float('-inf')
            improved_any = False
            for s in recent:
                if s > running_best + self.epsilon:
                    improved_any = True
                running_best = max(running_best, s)
            if not improved_any:
                return True, 'converged'
        return False, None
