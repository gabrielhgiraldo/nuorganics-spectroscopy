from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re

import pandas as pd
import pickle
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# TODO: write unittests for these functions

AMMONIA_N = 'Ammonia-N'
TOTAL_N = 'N'
PERCENT_MOISTURE = 'Moisture'
PHOSPHORUS = 'P'
POTASSIUM = 'K'
SULFUR = 'S'
AVAILABLE_TARGETS = [
    AMMONIA_N,
    TOTAL_N,
    PERCENT_MOISTURE,
    PHOSPHORUS,
    POTASSIUM,
    SULFUR
]
TRM_PATTERN = "*.TRM"
LAB_REPORT_PATTERN = "Lab Report*.csv"
LAB_REPORT_COLUMNS = ['filename', *AVAILABLE_TARGETS]
FILE_PATTERNS = {TRM_PATTERN, LAB_REPORT_PATTERN}
SCAN_FILE_DATETIME_FORMAT = '%m-%d-%y'

DATA_DIR = Path(__file__).parents[1] / 'data'
INFERENCE_RESULTS_FILENAME = 'inference_results.pkl'
EXTRACTED_DATA_FILENAME='extracted_files.pkl'
EXTRACTED_REFERENCE_FILENAME = '.extracted_filepaths.pkl' # file for caching extracted filepaths

SAMPLE_IDENTIFIER_COLUMNS = ['sample_name', 'sample_date']

MAX_WORKERS = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_sample_ids(df_samples, identifier_columns=SAMPLE_IDENTIFIER_COLUMNS,
                   include_run_number=True):
    if include_run_number:
        identifier_columns = [*identifier_columns, 'run_number']
    return set(zip(*[df_samples[column] for column in identifier_columns]))


# TODO: validate that this works correctly
def get_unmatched_sample_ids(df_lr, df_samples):
    lr_ids = get_sample_ids(df_lr, include_run_number=False)
    samples_ids = get_sample_ids(df_samples, include_run_number=False)
    return samples_ids - lr_ids


def _extract_spect_filename_info(filename):
    sample_name_method, _, remaining = filename.partition('-')
    if '(' in sample_name_method:
        sample_name, _, process_method = sample_name_method.partition('(')
        sample_name = sample_name.strip().lower()
        process_method = process_method.strip()[:-1] # drop )
    else:
        sample_name = sample_name_method.strip().lower()
        process_method = ''
    process_method = process_method.lower()
    sample_date_string = re.search(r'\d+-\d+-\d+',remaining)[0].strip()
    sample_date = pd.to_datetime(sample_date_string, format=SCAN_FILE_DATETIME_FORMAT)
    run_number = filename.partition('#')[2].partition('.')[0]
    return sample_name, process_method, sample_date, run_number


def _extract_integration_time(extra_info):
    return extra_info.partition('->')[2]\
                     .partition('Time:')[2]\
                     .partition('Avg:')[0]\
                     .strip()\
                     .partition('ms')[0]


def _extract_lab_report_filename_info(filename):
    sample_name_date = filename.partition('-')[2]
    sample_name = re.split(r'\d+-\d+-\d+', sample_name_date)[0]\
                     .partition('-')[0]\
                     .strip()
    sample_name = sample_name.strip().lower()
    sample_date_string = re.search(r'\d+-\d+-\d+',sample_name_date)[0].strip()
    sample_date = pd.to_datetime(sample_date_string, format=SCAN_FILE_DATETIME_FORMAT)
    return sample_name, sample_date


def parse_lab_report(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['filename'] = filepath.name
    sample_name, sample_date = _extract_lab_report_filename_info(filepath.name)
    df['sample_name'] = sample_name
    df['sample_date'] = sample_date
    df = df.reset_index(drop=True)
    return df



def parse_spect_file(path):
    path = Path(path)
    df = pd.read_csv(path)
    extra_info = df.columns[0]
    data = df.iloc[:,0]\
             .str.strip()\
             .str.partition(' ')
    wavelengths = data[0].astype(float)
    values = data[2].astype(float)
    sample_df = pd.DataFrame([values.values], columns=wavelengths)
    sample_df['extra_info'] = extra_info
    sample_df['integration_time'] = _extract_integration_time(extra_info)
    sample_df['filename'] = path.name
    sample_name, process_method, sample_date, run_number = _extract_spect_filename_info(path.name)
    sample_df['sample_name'] = sample_name
    sample_df['sample_date'] = sample_date
    sample_df['run_number'] = run_number
    sample_df['process_method'] = process_method
    sample_df = sample_df.reset_index(drop=True)
    return sample_df


def parse_abs_files(directory_path=None) -> pd.DataFrame:
    if directory_path is None:
        directory_path = DATA_DIR
    directory_path = Path(directory_path)
    try:
        return pd.concat([parse_spect_file(filepath) for filepath in directory_path.glob("*.ABS")])
    except ValueError:
        raise FileNotFoundError(f'no .ABS files found at {directory_path}')


def parse_trm_files(data_dir=None, zero_negatives=True, skip_paths=None, concurrent=True) -> pd.DataFrame:
    # TODO: add test for checking whether parsing occurred correctly 
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    try:
        trm_filepaths = set(data_dir.glob(TRM_PATTERN))
        logger.info('parsing trm files')
        if skip_paths is not None:
            logger.warn(f'skipping paths {skip_paths}')
            trm_filepaths = [filepath for filepath in trm_filepaths if filepath not in skip_paths]
            if len(trm_filepaths) <= 0:
                logger.warning('all .TRM files were in skip_paths. no new .TRM files parsed.')
                return pd.DataFrame(), []
        if concurrent:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                dfs = pool.map(parse_spect_file, trm_filepaths)
        else:
            dfs = [parse_spect_file(filepath) for filepath in trm_filepaths]
        df_trms = pd.concat(dfs)
        if zero_negatives:
            # set trms that are < 0 to 0
            num = df_trms._get_numeric_data()
            num[num < 0] = 0
        # make sure that all files were extracted
        assert len(df_trms.index) == len(trm_filepaths)
        return df_trms, trm_filepaths
    except ValueError as e:
        raise FileNotFoundError(f'no .TRM files found at {data_dir}. {e}')


# def validate_extraction(df_trms, data_dir, extracted_filepaths, df_lr=None):
#     trm_filepaths = get_relevant_filepaths(data_dir, [TRM_PATTERN])
    # failed_to_extract_scans = 
#     # TODO: make sure that all files are accounted for
#     # TODO: make sure that all files were matched
#     # TODO: add warnings if something went wrong


def extract_data(data_path=DATA_DIR, extracted_filename=EXTRACTED_DATA_FILENAME,
                cache=True, skip_paths=None, concurrent=True,
                lab_report_columns=LAB_REPORT_COLUMNS):
    # TODO: add warning/check if all files are extracted
    # handle transmittance
    df_trms, trm_filepaths = parse_trm_files(
        data_dir=data_path,
        skip_paths=skip_paths,
        concurrent=concurrent
    )
    # try and see if there's groundtruth for these files
    try:
        df_lr, lr_filepaths = parse_lab_reports(
            data_dir=data_path,
            skip_paths=skip_paths,
            concurrent=concurrent
        )
    except FileNotFoundError as e:
        logger.warning(e)
        df = df_trms
        extracted_files = trm_filepaths
    else:
        # if there's groundtruth, join the groundtruth to the dataset
        unmatched_samples= get_unmatched_sample_ids(df_lr, df_trms)
        if len(unmatched_samples) > 0:
            message = f'unable to match samples {unmatched_samples}'
            logger.warning(message)
            # TODO: add custom exception for this
            raise Exception(message)
        lr_to_join = df_lr.set_index(SAMPLE_IDENTIFIER_COLUMNS)[lab_report_columns]
        df = df_trms.join(lr_to_join, on=SAMPLE_IDENTIFIER_COLUMNS, lsuffix='_trm', rsuffix='_lr')\
                                            .reset_index(drop=True)
        if len(df.index) > len(trm_filepaths):
            filename_counts = df['filename_trm'].value_counts()
            duplicate_files = filename_counts[filename_counts > 1]
            message = f'potential duplicate lab reports detected for samples {duplicate_files}'
            logger.warning(message)
            df = df.drop_duplicates(subset=['filename_trm'])
        assert len(df.index) == len(trm_filepaths)
        extracted_files = set([*trm_filepaths, *lr_filepaths])
    if cache:
        # df.to_csv(data_path/extracted_filename, index=False)
        with open(data_path/extracted_filename, 'wb') as f:
            pickle.dump(df, f)
        with open(data_path/EXTRACTED_REFERENCE_FILENAME, 'wb') as f:
            pickle.dump(extracted_files, f)

    return df, extracted_files


def parse_lab_reports(data_dir=None, skip_paths=None, concurrent=True) -> pd.DataFrame:
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    try:
        lr_filepaths = set(data_dir.glob(LAB_REPORT_PATTERN))
        logger.info('parsing lab report files')
        if skip_paths is not None:
            logger.warn(f'skipping paths {skip_paths}')
            lr_filepaths = [filepath for filepath in lr_filepaths if filepath not in skip_paths]
        if concurrent:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                reports = pool.map(parse_lab_report, lr_filepaths)
        else:
            reports = [parse_lab_report(filepath) for filepath in lr_filepaths]

        df_lr = pd.concat(reports)
        assert len(df_lr.index) == len(lr_filepaths)
        return df_lr, lr_filepaths
    except ValueError:
        raise FileNotFoundError(f'no lab report files found at {data_dir}')


def load_cached_extracted_data(data_dir, extracted_data_filename):

    extracted_data_filepath = data_dir / extracted_data_filename
    with open(extracted_data_filepath, 'rb') as f:
        extracted_data = pickle.load(f)

    extracted_ref_path = data_dir / EXTRACTED_REFERENCE_FILENAME
    with open(extracted_ref_path, 'rb') as f:
        extracted_filepaths = pickle.load(f)

    return extracted_data, extracted_filepaths


# class SpectroscopyDataEventHandler(FileSystemEventHandler):
#     def __init__(self, sync_data, extraction_func):
#         super().__init__()
#         self.sync_data = sync_data
#         self.extraction_func = extraction_func


#     def on_created(self, event):
#         # get files that were created/moved here
#         # new_data = self.extraction_func(event.src_path)
#         # data = pd.concat([new_data, self.sync_data])
#         print(event)
        
 
#     def on_deleted(self, event):
#         # on_created_func()
#         # get files that were delete from here
#         # remove information from the reference
#         # remove related information from cached files
#         # update cache
#         print(event)
#     # TODO: implement other event type handlers?


def get_relevant_filepaths(data_dir=DATA_DIR, file_patterns=FILE_PATTERNS):
    file_paths = set()
    for pattern in file_patterns:
        file_paths |= set(data_dir.glob(pattern))
    return file_paths

class SpectroscopyDataMonitor:
    def __init__(self, watch_directory, extracted_data_filename):
        # extract initial files
        self.syncing = False
        self.watch_directory = watch_directory
        self.extracted_data_filename = extracted_data_filename
        self.extracted_data = pd.DataFrame()
        self.extracted_filepaths = set()
        self.load_data()
        # self.data_updated = True
        # set up file syncing
        # self._event_handler = SpectroscopyDataEventHandler(self.extracted_data, extract_data, on_data_change)
        # self.observer = Observer()
        # self.observer.schedule(
        #     event_handler=self._event_handler,
        #     path=watch_directory,
        #     recursive=False
        # )

        
    def cache_data(self):
        # self.extracted_data.to_csv(self.watch_directory/self.extracted_data_filename, index=False)
        self.extracted_data.to_pickle(self.watch_directory/self.extracted_data_filename)
        with open(self.watch_directory/EXTRACTED_REFERENCE_FILENAME, 'wb') as f:
            pickle.dump(self.extracted_filepaths, f)


    def sync_data(self):
        self.syncing = True
        # get set of files that are in current directory
        # TODO: include abs files?
        has_changed = False
        current_filepaths = get_relevant_filepaths(self.watch_directory)
        # check if the folder is empty of relevant files
        if len(current_filepaths) == 0:
            if len(self.extracted_filepaths) > 0:
                has_changed = True
            self.extracted_filepaths = set()
            self.extracted_data = pd.DataFrame()
            self.syncing = False
            return self.extracted_data, has_changed
        # remove any files that were deleted
        deleted_filepaths = self.extracted_filepaths - current_filepaths
        if len(deleted_filepaths) > 0:
            logger.warn(f'files deleted: {deleted_filepaths}')
            deleted_sample_ids = []
            for deleted_filepath in deleted_filepaths:
                sample_name, _, sample_date, run_number = _extract_spect_filename_info(deleted_filepath.name)
                # get id associated with sample
                sample_id = (sample_name, sample_date, run_number)
                deleted_sample_ids.append(sample_id)
                # get row in sample associated with that ids
            current_sample_ids = pd.Series(zip(
                self.extracted_data['sample_name'],
                self.extracted_data['sample_date']
            ), index=self.extracted_data.index)
            mask = ~current_sample_ids.isin(deleted_sample_ids)
            self.extracted_data = self.extracted_data[mask]
            self.extracted_filepaths -= deleted_filepaths
            has_changed = True

        # add in new files that haven't been extracted
        new_filepaths = current_filepaths - self.extracted_filepaths
        if len(new_filepaths) > 0:
            # TODO: pretty print this or/and display in UI
            logger.warn(f'new files added: {new_filepaths}')
            new_data, _ = extract_data(
                data_path=self.watch_directory,
                cache=False,
                skip_paths=self.extracted_filepaths
            )
            self.extracted_filepaths |= new_filepaths
            self.extracted_data = pd.concat([self.extracted_data, new_data])
            has_changed = True
        if has_changed:
            self.cache_data()
        self.syncing = False
        return self.extracted_data, has_changed
            

    def load_data(self, cache=True, skip_paths=None):
        data_path = self.watch_directory
        try:
            # extract cached data
            logger.info('loading extracted data')
            self.extracted_data, self.extracted_filepaths = load_cached_extracted_data(
                data_dir=self.watch_directory,
                extracted_data_filename=self.extracted_data_filename
            )
            if skip_paths is None:
                skip_paths = self.extracted_filepaths
            else:
                skip_paths = [*skip_paths, *self.extracted_filepaths]
        except FileNotFoundError:
            message = (
                f'no cached extracted data found at {data_path}'
            )
            logger.warning(message)
        else:
            logger.info(f'cached data loaded')
        finally:
            self.sync_data()
            if len(self.extracted_data.index) > 0:
                if cache:
                    self.cache_data()
                self.extracted_samples_ids = get_sample_ids(self.extracted_data)
            else:
                self.extracted_samples_ids = set()
            return self.extracted_data, self.extracted_filepaths
        

