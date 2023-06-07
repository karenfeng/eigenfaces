import numpy as np
import warnings
import nimfa
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import FactorAnalysis


def fit_DR(X_train, X_test, DR, n_component, seed="315"):
    """Generalized function to fit a selected dimension reduction method.

    Args:
        X_train (torch): tensor of training data
        X_test (torch): tensor of test data
        DR (str): "PCA", "NMF", "PPCA" (probabilistic PCA), "PNMF" (probabilistic NMF)
        n_component (int): number of components to fit
        seed (str, optional): Random seed. Defaults to "315".
    """
    X_train = X_train.numpy()
    X_test = X_test.numpy()

    # Fit PCA
    if DR == "PCA":
        dr = PCA(n_components=n_component)
        dr.fit(X_train)
        # X_train_dr = dr.transform(X_train)
        # X_test_dr = dr.transform(X_test)

    # Fit NMF
    elif DR == "NMF":
        X_train_min = np.min(X_train)
        if X_train_min < 0:
            X_train -= X_train_min

        X_test_min = np.min(X_test)
        if X_test_min < 0:
            X_test -= X_test_min

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        dr = NMF(n_components=n_component, random_state=seed)
        dr = dr.fit(X_train)
        # X_train_dr = dr.transform(X_train)
        # X_test_dr = dr.transform(X_test)
        warnings.resetwarnings()

    # Fit probabilistic PCA
    elif DR == "PPCA":
        dr = FactorAnalysis(n_components=n_component, random_state=0)
        dr.fit(X_train)
        # X_train_dr = dr.transform(X_train)
        # X_test_dr = dr.transform(X_test)

    # Fit probabilistic NMF
    elif DR == "PNMF":
        pnmf = nimfa.Pmf(X_train, rank=n_component)
        dr = pnmf()
    #     X_train_dr = pnmf_fit.basis()
    #     X_test_dr = pnmf_fit.coef()

    # return X_train_dr, X_test_dr
    return dr


def transform_DR(X_train, X_test, DR, model):
    X_train = X_train.numpy()
    X_test = X_test.numpy()

    if DR == "PCA" or DR == "NMF" or DR == "PPCA":
        if DR == "NMF":
            X_train_min = np.min(X_train)
            if X_train_min < 0:
                X_train -= X_train_min

            X_test_min = np.min(X_test)
            if X_test_min < 0:
                X_test -= X_test_min

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        X_train_dr = model.transform(X_train)
        X_test_dr = model.transform(X_test)
        warnings.resetwarnings()
    if DR == "PNMF":
        X_train_dr = np.array(model.basis())
        X_test_dr = np.array(model.coef())

    return X_train_dr, X_test_dr
