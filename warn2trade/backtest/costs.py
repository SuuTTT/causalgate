"""Transaction-cost model used both inside the differentiable surrogate and in the evaluation backtest.

cost(|dw|) = c0 |dw|                                  commission / fees (bps)
           + s_half |dw|                              half-spread crossing (relative)
           + eta * sigma * sqrt(|dw| * notional / (adv_frac * kappa)) * |dw|   square-root market impact (Almgren et al. 2005)
           + zeta * sigma * sqrt(kappa) * |dw|        expected adverse drift while working the order over kappa bars
All quantities are in log-return units of the traded notional; `notional` is the strategy's notional as a fraction of
the asset's ADV so that participation = |dw| * notional / (adv_frac * kappa).
"""
from __future__ import annotations
import torch


class CostModel:
    def __init__(self, commission_bps: float = 5.0, use_half_spread: bool = True, impact_eta: float = 0.1,
                 delay_zeta: float = 0.05, notional: float = 0.05):
        self.commission_bps, self.use_half_spread = commission_bps, use_half_spread
        self.impact_eta, self.delay_zeta, self.notional = impact_eta, delay_zeta, notional

    def __call__(self, dw_abs: torch.Tensor, half_spread: torch.Tensor, sigma: torch.Tensor, adv_frac: torch.Tensor,
                 kappa: torch.Tensor) -> torch.Tensor:
        c = self.commission_bps * 1e-4 * dw_abs
        if self.use_half_spread:
            c = c + half_spread * dw_abs
        part = dw_abs * self.notional / (adv_frac.clamp_min(1e-6) * kappa.clamp_min(1.0))
        c = c + self.impact_eta * sigma * torch.sqrt(part.clamp_min(0.0)) * dw_abs
        c = c + self.delay_zeta * sigma * torch.sqrt(kappa.clamp_min(1.0)) * dw_abs
        return c

    def round_trip_estimate(self, half_spread: torch.Tensor, sigma: torch.Tensor, adv_frac: torch.Tensor, dw_abs: float = 0.5) -> torch.Tensor:
        """Scalar-ish prior used for the executable-alpha weight (both legs, immediate execution)."""
        d = torch.full_like(half_spread, dw_abs)
        return 2.0 * self(d, half_spread, sigma, adv_frac, torch.ones_like(half_spread)) / dw_abs

    def with_bps(self, bps: float) -> "CostModel":
        return CostModel(bps, self.use_half_spread, self.impact_eta, self.delay_zeta, self.notional)
