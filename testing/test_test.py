def test_something():
    assert False


def test_something_else():
    assert False


# region fixtures
import pytest


@pytest.fixture
def sample_data():
    return [1, 2, 3]


def test_sum(sample_data):
    assert sum(sample_data) == 6


@pytest.fixture
def temp_file():
    print("I run before the tests. Like setup")
    yield "foo"
    print("I run after the tests. Like teardown")


@pytest.fixture(autouse=True)
def setup():
    print("I run even if you do not specify")
    yield
    print("I also run after each test")


# endregion

# Marks are handy for CI

# region marks
# @pytest.mark.parametrize("foo", "bar", [[1,2],[3,4]])
# def test_plus_one(foo, bar):
#     print("This mark allows passing like input and expected value pairs")


@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_product(x, y):
    print("This mark allows a Cartesian product between multiple sets. Like all pairs of values, nested loops")


@pytest.mark.skip(reason="I forget lol")
def test_skipped():
    print("This will be skipped because who knows")


@pytest.mark.skipif(True, reason="Becuz")
def test_skipif():
    print("Conditional skip")


@pytest.mark.xfail(reason="Why not")
def test_broken():
    print("This is like a skip also")


# endregion

# region custom marks
@pytest.mark.i_made_this_up()
def test_made_up_tag():
    print("This test has some custom tag for some reason")


# endregion

# region mocking -- fun to deep dive into

from unittest.mock import Mock


def function_to_test(something):
    return something.foo("bar")


@pytest.fixture
def mock_obj():
    mock = Mock()
    mock.foo = lambda x: "something"
    return mock


def test_with_fixture(mock_obj):
    result = function_to_test(mock_obj)
    assert result == "something"

# endregion

# region parallel tests
# pytest-xdist
# run with pytest -n <core_count>
# pytest --dist=loadscope --tx... to distribute tests across machines