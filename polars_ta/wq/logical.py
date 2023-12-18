from functools import reduce

import numpy as np
import polars as pl


def and_(*args):
    """Logical AND operator, returns true if both operands are true and returns false otherwise"""
    return reduce(lambda x, y: np.logical_and(x, y), list(args))


def or_(*args):
    """Logical OR operator returns true if either or both inputs are true and returns false otherwise"""
    return reduce(lambda x, y: np.logical_or(x, y), args)


def equal(input1: pl.Expr, input2: pl.Expr) -> pl.Expr:
    """Returns true if both inputs are same and returns false otherwise"""
    return input1.eq(input2)


def negate(input1: pl.Expr) -> pl.Expr:
    """The result is true if the converted operand is false; the result is false if the converted operand is true"""
    return ~input1


def less(input1: pl.Expr, input2: pl.Expr) -> pl.Expr:
    """If input1 < input2 return true, else return false"""
    # ValueError: Can only compare identically-labeled Series objects
    return input1 - input2 < 0


def if_else(input1: pl.Expr, input2: pl.Expr, input3: pl.Expr) -> pl.Expr:
    """If input1 is true then return input2 else return input3."""
    return pl.when(input1).then(input2).otherwise(input3)


def is_not_nan(input1: pl.Expr) -> pl.Expr:
    """If (input != NaN) return 1 else return 0"""
    return input1.is_not_nan()


def is_nan(input1: pl.Expr) -> pl.Expr:
    """If (input == NaN) return 1 else return 0"""
    return input1.is_nan()


def is_finite(input1: pl.Expr) -> pl.Expr:
    """If (input NaN or input == INF) return 0, else return 1"""
    return input1.is_finite()


def is_not_finite(input1: pl.Expr) -> pl.Expr:
    """If (input NAN or input == INF) return 1 else return 0"""
    return input1.is_infinite()
