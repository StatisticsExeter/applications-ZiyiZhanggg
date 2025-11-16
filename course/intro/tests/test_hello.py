import pytest
from course.intro.hello import greet, add_two_numbers


def test_greet_output(capsys):
    greet()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello, World!"


def test_add_two_numbers_too_simple():
    simple_sum = add_two_numbers(2, 2)
    assert simple_sum == 4


@pytest.mark.parametrize("input_one, input_two , expected", [
    (1, 2, 3),
    (4, 8, 12),
    (-1, -3, -4),
])
def test_add_two_numbers_sensible(input_one, input_two, expected):
    sensible_sum = add_two_numbers(input_one, input_two)
    assert sensible_sum == expected
    
# test push

# test push 2
