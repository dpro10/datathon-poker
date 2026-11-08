"""sdbot - the strategy engine behind agent.py (Heads-Up Short Deck 5-card PLO)."""
import os

# The tournament gives a bot one CPU. NumPy's BLAS would otherwise start extra threads for
# matrix products (e.g. candidate bucketing), which spin without helping: 3x the CPU for the
# same wall time, and under a 1-CPU quota they slow the main thread. Must run before numpy
# is first imported (agent.py imports sdbot before anything that uses numpy).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
