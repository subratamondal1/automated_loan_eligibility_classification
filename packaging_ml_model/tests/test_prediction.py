import pytest
import pandas as pd

from ..prediction_model.config import config
from ..prediction_model.processing.data_handling import load_dataset
from ..prediction_model.predict import generate_prediction

"""
What will be tested?
1. The output is not None.
2. The output returns string datatype.
3. The output is 'Y' or not for sample test data.
"""

@pytest.fixture
def single_prediction() -> dict:
    test_data:pd.DataFrame = load_dataset(filename=config.TEST_FILE)
    single_row:pd.DataFrame = test_data[:1]
    result = generate_prediction(data_input=single_row)
    return result

def test_single_prediction_not_none(single_prediction) -> None:
    """Test that the Output is not None"""
    assert single_prediction is not None

def test_single_prediction_is_str_dtype(single_prediction) -> None:
    """Test that the Output is of 'str' datatype"""
    assert isinstance(single_prediction.get("prediction")[0], str)

def test_single_prediction_validate(single_prediction) -> None:
    """Test whether the Output is 'Y' or not"""
    assert single_prediction.get("prediction")[0] == "Y"