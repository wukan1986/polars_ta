from polars import Expr, when


# def cs_bucket(x: Expr) -> Expr:
#     """Convert float values into indexes for user-specified buckets. Bucket is useful for creating group values, which can be passed to group operators as input."""
#     pass

def cut(x: Expr, b: float, *more_bins) -> Expr:
    """分箱"""
    return x.cut([b, *more_bins]).to_physical()


def cs_qcut(x: Expr, q: int = 10) -> Expr:
    """Convert float values into indexes for user-specified buckets. Bucket is useful for creating group values, which can be passed to group operators as input."""
    return x.qcut(q, allow_duplicates=True).to_physical()


def clamp(x: Expr, lower: float = 0, upper: float = 0, inverse: bool = False, mask: float = None) -> Expr:
    """Limits input value between lower and upper bound in inverse = false mode (which is default). Alternatively, when inverse = true, values between bounds are replaced with mask, while values outside bounds are left as is."""
    if inverse:
        # mask is one of: 'nearest_bound', 'mean', 'NAN' or any floating point number
        return when((x < lower) | (x > upper)).then(x).otherwise(mask)
    else:
        return x.clip(lower, upper)


def filter_(x: Expr, h: str = "1, 2, 3, 4", t: str = "0.5") -> Expr:
    """Used to filter the value and allows to create filters like linear or exponential decay."""
    raise


def keep(x: Expr, f: float, period: int = 5) -> Expr:
    """This operator outputs value x when f changes and continues to do that for “period” days after f stopped changing. After “period” days since last change of f, NaN is output."""
    raise


def left_tail(x: Expr, maximum: float = 0) -> Expr:
    """NaN everything greater than maximum, maximum should be constant."""
    return when(x <= maximum).then(x).otherwise(None)


def pasteurize(x: Expr) -> Expr:
    """Set to NaN if x is INF or if the underlying instrument is not in the Alpha universe"""
    # TODO: 不在票池中的的功能无法表示
    # TODO: 与purify好像没啥区别
    return when(x.is_finite()).then(x).otherwise(None)


def purify(x: Expr) -> Expr:
    """Clear infinities (+inf, -inf) by replacing with NaN."""
    return when(x.is_finite()).then(x).otherwise(None)


def fill_nan(x: Expr) -> Expr:
    """fill nan by null
    填充nan为null"""
    return x.fill_nan(None)


def fill_zero(x: Expr) -> Expr:
    """填充null为0"""
    return x.fill_null(0)


def right_tail(x: Expr, minimum: float = 0) -> Expr:
    """NaN everything less than minimum, minimum should be constant."""
    return when(x >= minimum).then(x).otherwise(None)


def sigmoid(x: Expr) -> Expr:
    """Returns 1 / (1 + exp(-x))"""
    return 1 / (1 + (-x).exp())


def tail(x: Expr, lower: float = 0, upper: float = 0, newval: float = 0) -> Expr:
    """If (x > lower AND x < upper) return newval, else return x. Lower, upper, newval should be constants. """
    # TODO 与clamp一样?
    return when((x <= lower) | (x >= upper)).then(x).otherwise(newval)
