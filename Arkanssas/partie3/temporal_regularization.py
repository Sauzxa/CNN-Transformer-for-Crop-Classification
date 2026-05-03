"""
Régularisation temporelle pour séries Sentinel-2 (et génériques).

- **Rééchantillonnage linéaire** : passe d'un nombre variable de pas (souvent irréguliers
  dans le temps réel) à une grille fixe [t0..t1] avec le même nombre de points pour tous
  les échantillons — prérequis classique des architectures type TempCNN / MCTNet.

- **Masque d'observation** : canal binaire (ou continu) indiquant si un pas de grille
  correspond à une vraie acquisition ou à une valeur interpolée / synthétique.

Quand vous n'avez que des tenseurs `(N, T, C)` sans dates : on suppose des temps
uniformes sur la saison (0→1). Pour des **dates réelles** (DOY ou jours depuis semis),
passez `t_obs` à `resample_temporal_linear` ou utilisez `interpolate_irregular_to_grid`.
"""

from __future__ import annotations

import numpy as np


def build_valid_timestep_mask(
    X: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Masque booléen (N, T) : True si au moins un canal n'est pas ~nul (observation utile).

    À adapter si votre convention de nodata n'est pas le zéro.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X doit être (N, T, C)")
    return np.any(np.abs(X) > eps, axis=-1)


def resample_temporal_linear(
    X: np.ndarray,
    n_steps: int,
    *,
    t_obs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Interpolation linéaire le long du temps : (N, T_in, C) -> (N, n_steps, C).

    Si `t_obs` est None, les temps d'entrée sont supposés équidistants sur [0, 1].
    Sinon, `t_obs` est un 1D de longueur T_in (ex. DOY normalisés ou jours depuis t0).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X doit être (N, T_in, C)")
    N, T_in, C = X.shape
    if n_steps < 1:
        raise ValueError("n_steps >= 1")
    if T_in == 0:
        raise ValueError("T_in == 0")
    if T_in == 1:
        return np.repeat(X, n_steps, axis=1)

    if t_obs is None:
        t_in = np.linspace(0.0, 1.0, T_in, dtype=np.float64)
    else:
        t_in = np.asarray(t_obs, dtype=np.float64).ravel()
        if t_in.shape[0] != T_in:
            raise ValueError("len(t_obs) doit égaler T_in")

    t_min, t_max = float(t_in[0]), float(t_in[-1])
    if t_max - t_min < 1e-12:
        return np.repeat(X[:, -1:, :], n_steps, axis=1)

    t_out = np.linspace(t_min, t_max, n_steps, dtype=np.float64)
    out = np.empty((N, n_steps, C), dtype=np.float32)

    for k in range(n_steps):
        tk = t_out[k]
        if tk <= t_in[0]:
            out[:, k, :] = X[:, 0, :]
        elif tk >= t_in[-1]:
            out[:, k, :] = X[:, -1, :]
        else:
            i = int(np.searchsorted(t_in, tk, side="right") - 1)
            i = min(max(i, 0), T_in - 2)
            denom = float(t_in[i + 1] - t_in[i])
            alpha = 0.0 if denom < 1e-12 else (tk - float(t_in[i])) / denom
            out[:, k, :] = ((1.0 - alpha) * X[:, i, :] + alpha * X[:, i + 1, :]).astype(np.float32)

    return out


def resample_timestep_mask_nearest(
    valid: np.ndarray,
    n_steps: int,
    *,
    t_obs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remappe un masque (N, T_in) booléen sur la grille `n_steps` par plus proche voisin
    dans le temps (après rééchantillonnage linéaire des features, les pas « proches »
    d'une acquisition réelle restent marqués observés).
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid doit être (N, T_in)")
    N, T_in = valid.shape
    if T_in == 1:
        v = valid[:, 0].astype(np.float32)
        return np.broadcast_to(v[:, None], (N, n_steps)).copy()

    if t_obs is None:
        t_in = np.linspace(0.0, 1.0, T_in, dtype=np.float64)
    else:
        t_in = np.asarray(t_obs, dtype=np.float64).ravel()
        if len(t_in) != T_in:
            raise ValueError("len(t_obs) doit égaler T_in")

    t_min, t_max = float(t_in[0]), float(t_in[-1])
    t_out = np.linspace(t_min, t_max, n_steps, dtype=np.float64)
    out = np.zeros((N, n_steps), dtype=np.float32)

    for k in range(n_steps):
        tk = t_out[k]
        j = int(np.argmin(np.abs(t_in - tk)))
        out[:, k] = valid[:, j].astype(np.float32)

    return out


def append_observation_mask_channel(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatène un canal masque (N, T, 1) à X (N, T, C) -> (N, T, C+1)."""
    X = np.asarray(X, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    if X.shape[:2] != m.shape[:2]:
        raise ValueError("X et mask incompatibles sur (N, T)")
    return np.concatenate([X, m], axis=-1)


def interpolate_irregular_to_grid(
    t_obs: np.ndarray,
    values: np.ndarray,
    t_grid: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Interpolation linéaire 1D par échantillon : pour une série irrégulière.

    Parameters
    ----------
    t_obs : (T_obs,) ou (N, T_obs)
        Instants (strictement croissants par ligne si N > 1).
    values : (N, T_obs, C)
        Valeurs aux instants donnés.
    t_grid : (T_grid,)
        Grille cible commune (ex. pas fixe de 5 jours).

    Returns
    -------
    (N, T_grid, C)
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("values doit être (N, T_obs, C)")
    N, T_obs, C = values.shape
    t_grid = np.asarray(t_grid, dtype=np.float64).ravel()
    t_arr = np.asarray(t_obs, dtype=np.float64)
    if t_arr.ndim == 1:
        t_arr = np.broadcast_to(t_arr, (N, T_obs))

    out = np.full((N, len(t_grid), C), fill_value, dtype=np.float32)
    for n in range(N):
        tn = t_arr[n]
        order = np.argsort(tn)
        tn = tn[order]
        vn = values[n][order]
        for c in range(C):
            out[n, :, c] = np.interp(t_grid, tn, vn[:, c], left=fill_value, right=fill_value).astype(
                np.float32
            )
    return out


__all__ = [
    "append_observation_mask_channel",
    "build_valid_timestep_mask",
    "interpolate_irregular_to_grid",
    "resample_temporal_linear",
    "resample_timestep_mask_nearest",
]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 3, 11)).astype(np.float32)
    Y = resample_temporal_linear(X, 12)
    assert Y.shape == (4, 12, 11)
    vm = build_valid_timestep_mask(X)
    mm = resample_timestep_mask_nearest(vm, 12)
    Z = append_observation_mask_channel(Y, mm)
    assert Z.shape == (4, 12, 12)
    print("temporal_regularization OK:", Y.shape, Z.shape)
