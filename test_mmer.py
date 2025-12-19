import numpy as np
import pickle
from sklearn.linear_model import Ridge
from mmer import MixedEffectRegressor

def test_mmer_redesign():
    print("Generating synthetic data...")
    n_train = 100
    n_test = 50
    m = 2
    
    # Groups
    groups_train = np.random.randint(0, 5, size=(n_train, 1))
    groups_test = np.random.randint(0, 5, size=(n_test, 1)) # Some same groups
    
    # Features
    X_train = np.random.randn(n_train, 3)
    X_test = np.random.randn(n_test, 3)
    
    # Synthetic Outcome (Fixed + Random)
    # y = X @ beta + Z @ u + e
    beta = np.random.randn(3, m)
    
    y_train = X_train @ beta + np.random.randn(n_train, m)
    
    print("Initializing Model...")
    # FE model
    fe = Ridge()
    
    # ME Model
    mmer = MixedEffectRegressor(fixed_effects_model=fe, max_iter=5, patience=2)
    
    print("Fitting Model...")
    mmer.fit(X_train, y_train, groups_train)
    
    print("Model Fitted.")
    # Check attributes
    print(f"Residual Covariance shape: {mmer.residual_term.cov.shape}")
    print(f"Random Effect 0 Covariance shape: {mmer.random_effect_terms[0].cov.shape}")
    
    print("Testing Pickle...")
    saved_model = pickle.dumps(mmer)
    loaded_model = pickle.loads(saved_model)
    print("Pickle successful.")
    
    print("Testing Predict...")
    preds = loaded_model.predict(X_test)
    print(f"Prediction shape: {preds.shape} (Expected {n_test}, {m})")
    assert preds.shape == (n_test, m)
    
    print("Testing Compute Random Effects on New Data...")
    mu, resid = loaded_model.compute_random_effects(X_test, np.zeros((n_test, m)), groups_test)
    print("Compute Random Effects successful.")
    print(f"Mu length: {len(mu)}")
    print(f"Residual shape: {resid.shape} (Expected {m * n_test},)") # resid is raveled

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_mmer_redesign()
