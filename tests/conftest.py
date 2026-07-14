import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="lance aussi les tests lents (entraînement complet)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test lent (entraînement)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="ajoutez --runslow pour lancer")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
