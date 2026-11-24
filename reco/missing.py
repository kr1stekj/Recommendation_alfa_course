"""Единые правила пропусков без зависимости от pandas."""
import math
from decimal import Decimal, InvalidOperation
from numbers import Number


def is_missing(value):
    """NaN и числовые бесконечности — пропуск; ноль и False сохраняются."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"nan", "+nan", "-nan"}
    if isinstance(value, (Number, Decimal)):
        try:
            return not math.isfinite(value)
        except (TypeError, OverflowError, ValueError):
            return False
    return False


def integer_or_none(value):
    """Не округляет дроби и не превращает некорректные числа в нулевую цену."""
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            return None
        return int(number)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None


def clean_tables(tables):
    """Создаёт копии строк; Cost разбирается отдельно, чтобы отличать NULL от NaN."""
    return {table: [{key: (None if key != "Cost" and is_missing(value) else value)
                     for key, value in row.items()} for row in rows]
            for table, rows in tables.items()}
