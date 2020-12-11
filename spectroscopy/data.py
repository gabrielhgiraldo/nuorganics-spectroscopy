from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pickle
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# TODO: write unittests for these functions
# TODO: use index of dataframe for sample_ids

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
LAB_REPORT_PATTERN = "Lab Report*"
LAB_REPORT_COLUMNS = ['filename', *AVAILABLE_TARGETS]
FILE_PATTERNS = {TRM_PATTERN, LAB_REPORT_PATTERN}
SCAN_FILE_DATETIME_FORMAT = '%m-%d-%y'

DATA_DIR = Path(__file__).parents[1] / 'data'
INFERENCE_RESULTS_FILENAME = 'inference_results.pkl'
EXTRACTED_DATA_FILENAME='.extracted_files.pkl'
EXTRACTED_REFERENCE_FILENAME = '.extracted_filepaths.pkl' # file for caching extracted filepaths

SAMPLE_IDENTIFIER_COLUMNS = ['sample_name', 'sample_date']
SCAN_IDENTIFIER_COLUMNS = SAMPLE_IDENTIFIER_COLUMNS + ['process_method', 'run_number']

DEFAULT_MAX_WORKERS = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UnmatchedFilesException(Exception):
    def __init__(self, message, unmatched_filepaths):
        self.unmatched_filepaths = unmatched_filepaths
        super().__init__(message)


    def __repr__(self) -> str:
        return f"{super().__repr__()}: unmatched files {self.unmatched_filepaths}" 


def is_spect_file(filepath):
    return filepath.suffix in ['.TRM', '.ABS']


def is_lab_report(filepath):
    return filepath.name.startswith('Lab Report')


def get_relevant_filepaths(data_dir=DATA_DIR, file_patterns=FILE_PATTERNS):
    if not isinstance(file_patterns, Iterable) or isinstance(file_patterns, str):
        file_patterns = [file_patterns]
    file_paths = set()
    for pattern in file_patterns:
        file_paths |= set(data_dir.glob(pattern))
    return file_paths

# TODO: use index?
def get_sample_ids(samples, identifier_columns=SAMPLE_IDENTIFIER_COLUMNS,
                   include_run_number=True, include_process_method=False, unique=True):
    """Get sample ids for given samples.

    Args:
        samples ([pathlib.Path] OR pd.DataFrame): [list of samples to get ids for (could be lab reports or spect files)]
        identifier_columns ([string], optional): [list of columns to use as common identifiers]. Defaults to SAMPLE_IDENTIFIER_COLUMNS.
        include_run_number (bool, optional): [include run_number in id (applicable to spect files)]. Defaults to True.
        include_process_method (bool, optional): [include process_method in id]. Defaults to False.
        unique (bool, optional): [whether to include only the unique ids, or all ids]. Defaults to True.

    Returns:
        [tuple]: [ids for provided samples]
    """
    if isinstance(samples, pd.DataFrame):
        if include_process_method:
            identifier_columns = [*identifier_columns, 'process_method']
        if include_run_number:
            identifier_columns = [*identifier_columns, 'run_number']
        sample_ids = zip(*[samples[column] for column in identifier_columns])
        if unique:
            return set(sample_ids)
        else:
            return pd.Series(sample_ids, index=samples.index)
    else:
        sample_ids = []
        for filepath in samples:
            if is_spect_file(filepath):
                sample_name, process_method, sample_date, run_number = _extract_spect_filename_info(filepath.name)
            else:
                sample_name, sample_date = _extract_lab_report_filename_info(filepath.name)
                run_number = None
                process_method = None
            sample_id = [sample_name, sample_date]
            if include_process_method:
                sample_id.append(process_method)
            if include_run_number:
                sample_id.append(run_number)
            sample_id = tuple(sample_id)
            sample_ids.append(sample_id)
        if unique:
            return set(sample_ids)
        else:
            return sample_ids


# TODO: add test for this
# TODO: use index?
def get_unmatched_sample_ids(df_lr, df_samples):
    lr_ids = get_sample_ids(df_lr, include_run_number=False)
    samples_ids = get_sample_ids(df_samples, include_run_number=False)
    return samples_ids - lr_ids


def _extract_spect_filename_info(filename):
    sample_name_method, _, remaining = filename.partition('-')
    if '(' in sample_name_method:
        sample_name, _, process_method = sample_name_method.partition('(')
        sample_name = sample_name.strip().lower()
        process_method = process_method.strip()[:-1]
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


def parse_abs_files(directory_path=None):
    if directory_path is None:
        directory_path = DATA_DIR
    directory_path = Path(directory_path)
    try:
        return pd.concat([parse_spect_file(filepath) for filepath in directory_path.glob("*.ABS")], ignore_index=True)
    except ValueError:
        raise FileNotFoundError(f'no .ABS files found at {directory_path}')


def parse_trm_files(data_dir=None, zero_negatives=True, skip_paths=None, concurrent=True,
                    max_workers=DEFAULT_MAX_WORKERS):
    # TODO: add test for checking whether parsing occurred correctly (file info was extracted correctly) 
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    try:
        trm_filepaths = set(data_dir.glob(TRM_PATTERN))
        logger.info('parsing trm files')
        if skip_paths is not None:
            logger.warn(f'skipping paths {skip_paths}')
            trm_filepaths = {filepath for filepath in trm_filepaths if filepath not in skip_paths}
            if len(trm_filepaths) <= 0:
                logger.warning('all .TRM files were in skip_paths. no new .TRM files parsed.')
                return pd.DataFrame(), set()
        if concurrent:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                dfs = pool.map(parse_spect_file, trm_filepaths)
        else:
            dfs = [parse_spect_file(filepath) for filepath in trm_filepaths]
        df_trms = pd.concat(dfs, ignore_index=True)
        if zero_negatives:
            # set trms that are < 0 to 0
            num = df_trms._get_numeric_data()
            num[num < 0] = 0
        # make sure that all files were extracted
        assert len(df_trms.index) == len(trm_filepaths)
        return df_trms, set(trm_filepaths)
    except ValueError as e:
        raise FileNotFoundError(f'no .TRM files found at {data_dir}. {e}')


def get_sample_matches(filepaths):
    """Get matching lab report filepaths for trm samples and matching trm samples for 
    lab report filepaths.

    Args:
        filepaths (set[pathlib.Path]): set of filepaths to find matches for

    Returns:
        set[pathlib.Path]: set of matched filepaths for provided filepaths.
    """
    directory = list(filepaths)[0].parent
    # get matching lab reports for each trm path
    trm_filepaths = {filepath for filepath in filepaths if is_spect_file(filepath)}
    # get all available lab reports to try to match the trms
    available_lrs = get_relevant_filepaths(directory, LAB_REPORT_PATTERN)
    # for each provided trm filepath calculate the ids for matching to lab report
    # TODO: use index?
    trm_ids = get_sample_ids(trm_filepaths, include_run_number=False, unique=False)
    available_lr_ids = get_sample_ids(available_lrs, include_run_number=False, unique=False)
    lab_report_lookup = dict(zip(available_lr_ids, available_lrs))
    lr_matches = set()
    for trm_id in trm_ids:
        try:
            lr_matches.add(lab_report_lookup[trm_id])
        except KeyError:
            logger.warning(f'unable to find lab report for trm file {trm_id}')
    # get matching trm for each lab report path
    lr_filepaths = {filepath for filepath in filepaths if is_lab_report(filepath)}
    # for each provided lab report filepath calculate the ids for matching to trm
    # TODO: use index?
    lr_ids = get_sample_ids(lr_filepaths, include_run_number=False, unique=False)
    available_trms = get_relevant_filepaths(directory, TRM_PATTERN)
    available_trm_ids = get_sample_ids(available_trms, include_run_number=False, unique=False)
    trm_lookup = defaultdict(list)
    for trm_id, trm_filepath in zip(available_trm_ids, available_trms):
        trm_lookup[trm_id].append(trm_filepath)

    trm_matches = set()
    for lr_id in lr_ids:
        trm_matches.update(trm_lookup[lr_id])

    return trm_matches | lr_matches


def extract_data(data_path=DATA_DIR,
                skip_paths=None, concurrent=True,
                lab_report_columns=LAB_REPORT_COLUMNS):
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
        extracted_files = set(trm_filepaths)
    else:
        # if there's groundtruth, join the groundtruth to the dataset
        unmatched_samples = get_unmatched_sample_ids(df_lr, df_trms)
        if len(unmatched_samples) > 0:
            message = f'unable to find lab reports for samples {unmatched_samples}'
            logger.warning(message)
            # TODO: only include the unmatched files here
            raise UnmatchedFilesException(message, trm_filepaths|lr_filepaths)
        lr_to_join = df_lr.set_index(SAMPLE_IDENTIFIER_COLUMNS)[lab_report_columns]
        df = df_trms.join(lr_to_join, on=SAMPLE_IDENTIFIER_COLUMNS, lsuffix='_trm', rsuffix='_lr')\
                                            .reset_index(drop=True)
        if len(df.index) > len(trm_filepaths):
            filename_counts = df['filename_trm'].value_counts()
            duplicate_files = filename_counts[filename_counts > 1]
            message = f'potential duplicate lab reports detected for samples:\n {duplicate_files}'
            logger.warning(message)
            df = df.drop_duplicates(subset=['filename_trm'])
        # ensure that matching occurred correctly
        assert len(df.index) == len(trm_filepaths)
        extracted_files = set([*trm_filepaths, *lr_filepaths])

    return df, extracted_files


def parse_lab_reports(data_dir=None, skip_paths=None, concurrent=True,
                      max_workers=DEFAULT_MAX_WORKERS) -> pd.DataFrame:
    if data_dir is None:
        data_dir = DATA_DIR
    data_dir = Path(data_dir)
    try:
        lr_filepaths = set(data_dir.glob(LAB_REPORT_PATTERN))
        logger.info('parsing lab report files')
        if skip_paths is not None:
            logger.warn(f'skipping paths {skip_paths}')
            lr_filepaths = set([filepath for filepath in lr_filepaths if filepath not in skip_paths])
        if concurrent:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                reports = pool.map(parse_lab_report, lr_filepaths)
        else:
            reports = [parse_lab_report(filepath) for filepath in lr_filepaths]

        df_lr = pd.concat(reports, ignore_index=True)
        assert len(df_lr.index) == len(lr_filepaths)
        return df_lr, lr_filepaths
    except ValueError:
        raise FileNotFoundError(f'no lab report files found at {data_dir}')


def load_cached_extracted_data(data_dir, extracted_data_filename):

    extracted_data_filepath = data_dir / extracted_data_filename
    extracted_data = pd.read_pickle(extracted_data_filepath)

    extracted_ref_path = data_dir / EXTRACTED_REFERENCE_FILENAME
    with open(extracted_ref_path, 'rb') as f:
        extracted_filepaths = pickle.load(f)

    return extracted_data, set(extracted_filepaths)


def _is_target_column(column_name):
    return any([str(column_name).endswith(target) for target in AVAILABLE_TARGETS])


class SpectroscopyDataMonitor:
    def __init__(self, watch_directory, extracted_data_filename=EXTRACTED_DATA_FILENAME,
                 column_order=None, cache=True):
        self.syncing = False
        self.watch_directory = watch_directory
        self.extracted_data_filename = extracted_data_filename
        self.extracted_data = pd.DataFrame()
        self.extracted_filepaths = set()
        self.cache = cache
        if column_order is None:
            self.column_order = ['index', *SAMPLE_IDENTIFIER_COLUMNS, 'process_method', 'run_number']
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

    def set_extracted_data(self, extracted_data):
        """update the extracted data attribute for the monitor"""
        if len(extracted_data.index) <= 0:
            self.extracted_data = pd.DataFrame()
        else:
            # TODO: make this the actual index?
            # create explicit index column for sorting
            extracted_data['index'] = get_sample_ids(
                extracted_data,
                unique=False,
                include_process_method=True
            )
            
            if self.column_order is None:
                self.extracted_data = extracted_data
            else:
                # first, remove the explicitly ordered columns
                columns = list(extracted_data.columns)
                for column in self.column_order:
                    columns.remove(column)
                # next remove the target columns
                value_columns = [column for column in columns if _is_target_column(column)]
                for column in value_columns:
                    columns.remove(column)
                # place columns in order
                columns = [
                    *self.column_order, # explicitly ordered columns
                    *value_columns, # target/value columns (Ammonia, N, etc.)
                    *columns # remaining columns
                ]
                # apply column order to dataframe
                self.extracted_data = extracted_data.reindex(columns, axis=1).sort_values('index', axis=0)
            # TODO: if caching becomes slow, put this on a separate thread
            # TODO: if caching becomes slow, add caching only when changes occur
            if self.cache:
                # update cache
                self.cache_data()


    def cache_data(self):
        """store the extracted data as a file in the watch directory.
        """
        # self.extracted_data.to_csv(self.watch_directory/self.extracted_data_filename, index=False)
        self.extracted_data.to_pickle(self.watch_directory/self.extracted_data_filename)
        with open(self.watch_directory/EXTRACTED_REFERENCE_FILENAME, 'wb') as f:
            pickle.dump(self.extracted_filepaths, f)


    def sync_data(self):
        """check for changes in the watch directory and update extracted data accordingly.

        Returns:
            tuple[pd.DataFrame, bool]: extracted data and whether the data changed since last sync.
        """
        self.syncing = True
        # get set of files that are in current directory
        # TODO: include abs files?
        has_changed = False
        current_filepaths = get_relevant_filepaths(self.watch_directory)
        # check if the folder is empty of relevant files
        if len(current_filepaths) <= 0:
            if len(self.extracted_filepaths) > 0:
                self.set_extracted_data(pd.DataFrame())
                self.extracted_filepaths = set()
                has_changed = True
            self.syncing = False
            return self.extracted_data, has_changed
        # remove any files that were deleted
        deleted_filepaths = self.extracted_filepaths - current_filepaths
        if len(deleted_filepaths) > 0:
            deleted_filenames = [fp.name for fp in deleted_filepaths]
            logger.warn(f'files deleted: {deleted_filenames}')
            filename_keys = {'filename_lr', 'filename_trm', 'filename'}
            mask = pd.Series([False]*len(self.extracted_data.index))
            for key in filename_keys:
                if key in self.extracted_data:
                    mask |= self.extracted_data[key].isin(deleted_filenames)        
            self.set_extracted_data(self.extracted_data[~mask])
            self.extracted_filepaths -= deleted_filepaths
            has_changed = True

        # add in new files that haven't been extracted
        new_filepaths = current_filepaths - self.extracted_filepaths
        if len(new_filepaths) > 0:
            # TODO: pretty print this or/and display in UI
            logger.warn(f'new files added: {[fp.name for fp in new_filepaths]}')
            matched_filepaths = get_sample_matches(new_filepaths)
            # extract the data for matched files and added files
            skip_paths = self.extracted_filepaths - (matched_filepaths | new_filepaths)
            # skip_paths = self.extracted_filepaths
            new_data, new_extracted_files = extract_data(
                data_path=self.watch_directory,
                skip_paths=skip_paths,
            )
            self.extracted_filepaths |= new_extracted_files
            # make sure that datetime types align
            for column in new_data.columns:
                if is_datetime(new_data[column]) or str(column).endswith('date'):
                    new_data[column] = pd.to_datetime(
                        arg=new_data[column],
                        format=SCAN_FILE_DATETIME_FORMAT,
                        errors='coerce'
                    ).dt.strftime(SCAN_FILE_DATETIME_FORMAT)

            self.set_extracted_data(
                extracted_data=pd.concat([self.extracted_data, new_data], ignore_index=True),
            )
            has_changed = True
        self.syncing = False
        return self.extracted_data, has_changed
            

    def load_data(self, skip_paths=None):
        data_path = self.watch_directory
        try:
            # extract cached data
            logger.info('loading extracted data')
            extracted_data, self.extracted_filepaths = load_cached_extracted_data(
                data_dir=self.watch_directory,
                extracted_data_filename=self.extracted_data_filename
            )
            self.set_extracted_data(extracted_data)
            if skip_paths is None:
                skip_paths = self.extracted_filepaths
            else:
                skip_paths = {*skip_paths, *self.extracted_filepaths}
        except FileNotFoundError:
            message = (
                f'no cached extracted data found at {data_path}'
            )
            logger.warning(message)
        else:
            logger.info(f'cached data loaded')
        finally:
            self.sync_data()
            return self.extracted_data, self.extracted_filepaths
        

