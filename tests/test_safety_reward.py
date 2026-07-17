"""Unit tests for the safety-aware reward math (no SUMO needed)."""
from types import SimpleNamespace

import pytest

import env_common as ec


def test_vehicle_vuln_exact_types():
    assert ec._vehicle_vuln("moto") == 1.0
    assert ec._vehicle_vuln("auto") == 0.6
    assert ec._vehicle_vuln("car") == 0.3


def test_vehicle_vuln_distribution_suffix():
    # vType ids may carry a distribution suffix like "moto@0"
    assert ec._vehicle_vuln("moto@0") == 1.0
    assert ec._vehicle_vuln("auto@3") == 0.6


def test_vehicle_vuln_unknown_defaults_to_car():
    assert ec._vehicle_vuln("bus") == ec.DEFAULT_VULN == 0.3
