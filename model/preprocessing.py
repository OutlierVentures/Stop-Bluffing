import numpy as np
import progressbar
import sklearn.preprocessing
from sklearn.mixture import GaussianMixture

from model import fisher_vector


def to_fisher(x, k=3):
    """
    Fisher vector encoding of input
    :param x: Input vector of shape(nb_samples, nb_frames, nb_landmarks, 3)
    :type x: ndarray
    :param k: Number of GMM components
    :type k: int
    :return:
    """
    print("Encoding as Fisher Vector...")
    nb_samples, nb_frames, nb_landmarks, d = x.shape
    fv = np.zeros((nb_samples, nb_landmarks, (k + 2 * d * k)))

    bar = progressbar.ProgressBar()
    for i in bar(range(nb_samples)):
        for j in range(nb_landmarks):
            xx = x[i, :, j, :]

            gmm = GaussianMixture(n_components=k, covariance_type='diag')
            gmm.fit(xx)

            fv[i, j, :] = fisher_vector.fisher_vector(xx, gmm)

    scaler = sklearn.preprocessing.MinMaxScaler()
    rescaled = fv.reshape((nb_samples, -1))
    scaler.fit(rescaled)

    rescaled = scaler.transform(rescaled)
    rescaled = rescaled.reshape((nb_samples, nb_landmarks, (k + 2 * d * k)))

    return rescaled