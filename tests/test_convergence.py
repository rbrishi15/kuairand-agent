import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.convergence import ConvergenceTracker


def test_iteration_cap_fires_even_while_still_improving():
    t = ConvergenceTracker(epsilon=0.002, N=3, max_iterations=5, max_wallclock_hours=6)
    t.start()
    for i in range(1, 6):
        t.record(i, 0.5 + i * 0.01)
    stop, reason = t.should_stop()
    assert stop and reason == 'iteration_cap'


def test_plateau_within_epsilon_converges():
    t = ConvergenceTracker(epsilon=0.002, N=3, max_iterations=50, max_wallclock_hours=6)
    t.start()
    for i, s in enumerate([0.50, 0.55, 0.60, 0.601, 0.6005, 0.6009], start=1):
        t.record(i, s)
    stop, reason = t.should_stop()
    assert stop and reason == 'converged'


def test_still_improving_does_not_converge():
    t = ConvergenceTracker(epsilon=0.002, N=3, max_iterations=50, max_wallclock_hours=6)
    t.start()
    for i, s in enumerate([0.50, 0.55, 0.60, 0.65], start=1):
        t.record(i, s)
    stop, _ = t.should_stop()
    assert not stop


def test_fewer_than_N_iterations_never_converges():
    t = ConvergenceTracker(epsilon=0.002, N=3, max_iterations=50, max_wallclock_hours=6)
    t.start()
    t.record(1, 0.60)
    t.record(2, 0.60)
    stop, _ = t.should_stop()
    assert not stop
