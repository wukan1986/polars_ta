from polars import Expr, when


def bucket(x: Expr,
           range: str = "0, 1, 0.1",
           buckets: str = "2,5,6,7,10",
           skipBegin: bool = False, skipEnd: bool = False, skipBoth: bool = False,
           NANGroup: bool = True) -> Expr:
    """Convert float values into indexes for user-specified buckets. Bucket is useful for creating group values, which can be passed to group operators as input."""
    # TODO 未完成
    raise


def clamp(x: Expr, lower: float = 0, upper: float = 0, inverse: bool = False, mask: float = None) -> Expr:
    """Limits input value between lower and upper bound in inverse = false mode (which is default). Alternatively, when inverse = true, values between bounds are replaced with mask, while values outside bounds are left as is."""
    if inverse:
        # mask is one of: 'nearest_bound', 'mean', 'NAN' or any floating point number
        return when((lower < x) & (x < upper)).then(mask).otherwise(x)
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
    return when(x > maximum).then(None).otherwise(x)


def pasteurize(x: Expr) -> Expr:
    """Set to NaN if x is INF or if the underlying instrument is not in the Alpha universe"""
    # TODO: 不在票池中的的功能无法表示
    # TODO: 与purify好像没啥区别
    return when(x.is_infinite()).then(None).otherwise(x)


def right_tail(x: Expr, minimum: float = 0) -> Expr:
    """NaN everything less than minimum, minimum should be constant."""
    return when(x < minimum).then(None).otherwise(x)


def sigmoid(x: Expr) -> Expr:
    """Returns 1 / (1 + exp(-x))"""
    return 1 / (1 + (-x).exp())


def tail(x: Expr, lower: float = 0, upper: float = 0, newval: float = 0) -> Expr:
    """If (x > lower AND x < upper) return newval, else return x. Lower, upper, newval should be constants. """
    # TODO 与clamp一样?
    return when((lower < x) & (x < upper)).then(newval).otherwise(x)
