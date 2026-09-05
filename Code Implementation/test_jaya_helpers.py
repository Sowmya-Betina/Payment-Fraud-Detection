
import numpy as np
import pytest
 
 
BOUNDS = {
    'path_width':    (16, 64),
    'gru_units':      (64, 256),
    'dropout_rate':   (0.1, 0.5),
    'learning_rate':  (0.0001, 0.01),
    'weight_decay':   (0.00001, 0.001),
    'batch_size':     (16, 64),
}
DISCRETE_CHOICES = {
    'path_width': [16, 32, 64],
    'gru_units': [64, 128, 256],
    'batch_size': [16, 32, 64],
}
 
 
def snap_discrete(key, value):
    choices = DISCRETE_CHOICES[key]
    return int(min(choices, key=lambda c: abs(c - value)))
 
 
def random_candidate(rng):
    hp = {}
    for key, (lo, hi) in BOUNDS.items():
        val = rng.uniform(lo, hi)
        hp[key] = snap_discrete(key, val) if key in DISCRETE_CHOICES else round(val, 6)
    return hp
 
 
def clip_candidate(hp):
    clipped = {}
    for key, (lo, hi) in BOUNDS.items():
        val = min(max(hp[key], lo), hi)
        clipped[key] = snap_discrete(key, val) if key in DISCRETE_CHOICES else round(val, 6)
    return clipped
 
 
# --- tests ---------------------------------------------------------------
 
class TestSnapDiscrete:
 
    def test_exact_match(self):
        assert snap_discrete('gru_units', 128) == 128
 
    def test_rounds_to_nearest(self):
        assert snap_discrete('gru_units', 200) == 256   # closer to 256 than 128
        assert snap_discrete('gru_units', 100) == 128    # closer to 128 than 64
 
    def test_midpoint_ties_to_lower_choice(self):
        # min() with equal distances keeps the first match in the list
        assert snap_discrete('path_width', 24) == 16     # |24-16|=8, |24-32|=8 -> first wins
 
    def test_result_always_in_choices(self):
        for key, choices in DISCRETE_CHOICES.items():
            for v in np.linspace(BOUNDS[key][0], BOUNDS[key][1], 25):
                assert snap_discrete(key, v) in choices
 
    def test_return_type_is_int(self):
        assert isinstance(snap_discrete('batch_size', 50.7), int)
 
 
class TestClipCandidate:
 
    def test_values_within_bounds_are_unchanged(self):
        hp = {'path_width': 32, 'gru_units': 128, 'dropout_rate': 0.3,
              'learning_rate': 0.001, 'weight_decay': 0.0005, 'batch_size': 32}
        assert clip_candidate(hp) == hp
 
    def test_values_below_lower_bound_are_clipped_up(self):
        hp = {'path_width': 5, 'gru_units': 10, 'dropout_rate': -0.2,
              'learning_rate': -1, 'weight_decay': -1, 'batch_size': 5}
        out = clip_candidate(hp)
        assert out['dropout_rate'] == BOUNDS['dropout_rate'][0]
        assert out['learning_rate'] == BOUNDS['learning_rate'][0]
        assert out['path_width'] in DISCRETE_CHOICES['path_width']
 
    def test_values_above_upper_bound_are_clipped_down(self):
        hp = {'path_width': 999, 'gru_units': 999, 'dropout_rate': 5.0,
              'learning_rate': 10.0, 'weight_decay': 10.0, 'batch_size': 999}
        out = clip_candidate(hp)
        assert out['dropout_rate'] == BOUNDS['dropout_rate'][1]
        assert out['learning_rate'] == BOUNDS['learning_rate'][1]
        assert out['gru_units'] == 256   # highest discrete choice
 
    def test_discrete_fields_always_snapped(self):
        hp = {'path_width': 999, 'gru_units': 999, 'dropout_rate': 0.2,
              'learning_rate': 0.001, 'weight_decay': 0.0005, 'batch_size': 999}
        out = clip_candidate(hp)
        for key in DISCRETE_CHOICES:
            assert out[key] in DISCRETE_CHOICES[key]
 
 
class TestRandomCandidate:
 
    def test_reproducible_with_seed(self):
        hp1 = random_candidate(np.random.default_rng(42))
        hp2 = random_candidate(np.random.default_rng(42))
        assert hp1 == hp2   # same seed -> identical candidate
 
    def test_all_keys_present(self):
        hp = random_candidate(np.random.default_rng(0))
        assert set(hp.keys()) == set(BOUNDS.keys())
 
    def test_all_values_within_bounds(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            hp = random_candidate(rng)
            for key, (lo, hi) in BOUNDS.items():
                assert lo <= hp[key] <= hi
 
    def test_discrete_fields_are_valid_choices(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            hp = random_candidate(rng)
            for key in DISCRETE_CHOICES:
                assert hp[key] in DISCRETE_CHOICES[key]
 
 
if __name__ == "__main__":
    pytest.main([__file__, "-v"])