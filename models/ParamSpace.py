from dataclasses import dataclass
from typing import Union, List, Any
from enum import Enum


class ParamType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    FLOAT_LOG = "float_log"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParamSpace:
    param_type: ParamType
    min_value: Union[int, float, None] = None
    max_value: Union[int, float, None] = None
    choices: Union[List[Any], None] = None
    default: Any = None

    def __post_init__(self):
        """Validate parameter space configuration"""
        if self.param_type in [ParamType.INTEGER, ParamType.FLOAT, ParamType.FLOAT_LOG]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(
                    f"min_value and max_value required for {self.param_type.value}"
                )
            if self.param_type == ParamType.FLOAT_LOG:
                if float(self.min_value) <= 0 or float(self.max_value) <= 0:
                    raise ValueError(
                        "min_value and max_value must be positive for log-uniform distribution"
                    )
        elif self.param_type == ParamType.CATEGORICAL:
            if not self.choices:
                raise ValueError("choices required for categorical parameters")

    @classmethod
    def integer(cls, min_val: int, max_val: int, default: int):
        """Create an integer parameter space"""
        return cls(ParamType.INTEGER, min_val, max_val, default=default)

    @classmethod
    def float_range(cls, min_val: float, max_val: float, default: float):
        """Create a float parameter space (uniform distribution)"""
        return cls(ParamType.FLOAT, min_val, max_val, default=default)

    @classmethod
    def float_log_range(cls, min_val: float, max_val: float, default: float):
        """Create a float parameter space with log-uniform distribution."""
        if min_val <= 0 or max_val <= 0:
            raise ValueError("min_val and max_val must be positive for log-uniform distribution")
        return cls(ParamType.FLOAT_LOG, min_val, max_val, default=default)

    @classmethod
    def categorical(cls, choices: List[Any], default: Any):
        """Create a categorical parameter space"""
        return cls(ParamType.CATEGORICAL, choices=choices, default=default)

    @classmethod
    def boolean(cls, default: bool):
        """Create a boolean parameter space"""
        return cls(ParamType.BOOLEAN, choices=[True, False], default=default)
