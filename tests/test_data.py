from pathlib import Path

from spectroscopy.data import (
    TRM_PATTERN,
    extract_data,
    get_relevant_filepaths
)

def test_extract_data():
    # TODO: test trm & csv files directory
    test_data_dir = Path('tests/test_samples')
    assert(test_data_dir.exists())
    # TODO: extract data
    extracted_data, extracted_filepaths = extract_data(test_data_dir, concurrent=True)
    # assert that all TRM scan files were extracted correctly
    trm_filepaths = get_relevant_filepaths(test_data_dir, [TRM_PATTERN])
    assert len(extracted_data.index) == len(trm_filepaths)
    all_filepaths = get_relevant_filepaths(test_data_dir)
    assert(extracted_filepaths == all_filepaths)
