# sandbox-only shim: this sandbox forbids named pipes, so joblib thread pools cannot start.
# Force joblib's sequential backend. Results are identical because every model has random_state=SEED.
import joblib
joblib.parallel_config(backend="sequential").__enter__()
