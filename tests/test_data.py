from pathlib import Path
import pytest

from spectroscopy.data import (
    extract_data,
    get_relevant_filepaths,
    get_sample_matches,
    LAB_REPORT_PATTERN,
    SpectroscopyDataMonitor,
    TRM_PATTERN,
)
@pytest.fixture
def test_data_dir():
    test_data_dir = Path('tests/test_samples')
    assert(test_data_dir.exists())
    return test_data_dir


@pytest.fixture
def monitor(test_data_dir):
    return SpectroscopyDataMonitor(
        watch_directory=test_data_dir,
        cache=False
    )

def test_extract_data(test_data_dir):
    extracted_data, extracted_filepaths = extract_data(test_data_dir, concurrent=True)
    # assert that all TRM scan files were extracted correctly
    trm_filepaths = get_relevant_filepaths(test_data_dir, TRM_PATTERN)
    assert len(extracted_data.index) == len(trm_filepaths)
    all_filepaths = get_relevant_filepaths(test_data_dir)
    assert(extracted_filepaths == all_filepaths)


def test_get_sample_matches(test_data_dir):
    # get current filepaths in test directory
    trm_filepaths = get_relevant_filepaths(test_data_dir, TRM_PATTERN) 
    lr_filepaths = get_relevant_filepaths(test_data_dir, LAB_REPORT_PATTERN)
    filepaths = trm_filepaths | lr_filepaths
    matches = get_sample_matches(filepaths)
    assert matches == trm_filepaths | lr_filepaths
    


def test_sync_delete_add_data(monitor):
    directory = monitor.watch_directory
    # create temporary directory
    temp_dir = directory / 'temp_dir'
    temp_dir.mkdir(exist_ok=True)

    # get current trm and lab report files in the directory
    trm_filepaths = get_relevant_filepaths(monitor.watch_directory, TRM_PATTERN) 
    lr_filepaths = get_relevant_filepaths(monitor.watch_directory, LAB_REPORT_PATTERN)

    # delete(move) some TRM files
    deleted_trms = set(list(trm_filepaths)[0:5])
    previous_num_extracted = len(monitor.extracted_data.index)
    new_deleted_locations = set()
    for trm in deleted_trms:
        new_location = temp_dir/trm.name
        trm.rename(new_location)
        new_deleted_locations.add(new_location)
    # sync monitor
    extracted_data, has_changed = monitor.sync_data()
    # check that syncing worked correctly
    assert has_changed
    updated_num_extracted = len(extracted_data.index)
    assert previous_num_extracted - len(deleted_trms) == updated_num_extracted
    assert monitor.extracted_filepaths == (lr_filepaths | trm_filepaths) - deleted_trms
    # move back the files
    for trm in new_deleted_locations:
        trm.rename(directory/trm.name)
    # sync monitor
    extracted_data, has_changed = monitor.sync_data()
    # check that syncing worked correctly
    assert has_changed
    previous_num_extracted = updated_num_extracted
    updated_num_extracted = len(monitor.extracted_data.index)
    assert updated_num_extracted == previous_num_extracted + len(deleted_trms)
    # check to make sure that the added trms have their lab reports matched
    assert extracted_data['filename_lr'].isna().sum() == 0
    assert 'filename' not in extracted_data

    