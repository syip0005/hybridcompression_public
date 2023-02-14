import os
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm
import datetime
import numpy as np

# TODO: write logging diagnostic to calculate how many records or meters removed (choose 1)
# TODO: update all docstrings

class HCPreprocessor():
    
    def __init__(self, raw_dataset_link = './dataset/raw/', target_folder = './dataset/final/'):
    
        """
        Custom-built preprocessor for the CERN, DRED and UMASS dataset.
        Citations to code will be made evident if they are used.
        
        Args:
            raw_dataset_link (string): Directories with all the raw files (downloaded using download_dataset.sh).
            target_folder (string): Directory to which the final .csv files will be written.
        """
        
        self.cern_folder = raw_dataset_link + 'CERN/'
        self.dred_folder = raw_dataset_link + 'DRED/'
        self.umass_folder = raw_dataset_link + 'UMASS_apartment/'

        self.target_folder = target_folder
        
    def cern_preprocess(self, train_test_split = 0.95, meter_pct_zeroes_remove = 0.5, debug=False):
    
        """
        Preprocess CERN dataset for training use.
        The resulting dataset will be written to a '.csv' file for training access.
        Please refer to the notebook documentation for reasoning.
        
        Parameters:
            None
            
        Returns:
            None
        """
        
        print("Processing CERN dataset")
        
        # Initialise various progress bars
        ProgressBar().register()       
        tqdm.pandas()
        
        ######################################################################################################
        # READ
        
        print("(1 / 5)\t\t Reading CERN CSV files.")
        print("-----------------------------------------------------------------------------------------------")
  
        # Obtain list of all text file locations.
        all_files = [self.cern_folder + file for file in os.listdir(self.cern_folder) if file[-3:] == 'txt']
        
        assert len(all_files) != 0, "No text files identified."
        
        if debug:
            all_files = all_files[0:2]
        
        # Read all into Dask with error check.
        # TODO: to re-write this error function
        try:
            df = dd.read_csv(all_files, names=['metre_id', 'timecode', 'kwh'], sep=' ',
                    dtype={'metre_id': 'int64', 'timecode': 'object', 'kwh': 'float64'})
        except:
            raise ValueError("The file is not in the correct format expected. Please redownload the raw CERN data.")
                
        # Convert all df timecodes.
        df['timestamp'] = df.map_partitions(self._cern_time_conversion, 'timecode',
                                            meta=(None, 'datetime64[ns]'))
        
        # Push dask dataframe into memory and as pandas DF (as we can fit).
        df_memory = df.compute()
        
        ######################################################################################################
        # MISSING AND DUPLICATE DATA
        
        print("\n(2 / 5)\t\t Duplicate and missing data imputation.")
        print("-----------------------------------------------------------------------------------------------")
        
        # Force cast
        df_memory = df_memory.astype({'metre_id': 'object'})
        
        # Keep only meters with 10,462 consecutive rows
        consec_counts = df_memory.groupby('metre_id').apply(self._cern_calculate_consecutive_intervals)
        meters_to_keep = consec_counts[consec_counts['max_consec'] == 10462].index
        df_memory = df_memory[df_memory['metre_id'].isin(meters_to_keep)]
        
        # Fix duplicates using average
        df_memory = df_memory.groupby(['timestamp', 'metre_id']).mean().reset_index()
        
        # Fix missing dates
        df_memory = self._cern_fix_missing_dates(df_memory)
        
        ########################################################################################################
        # REMOVE DATA FOR WHICH EVEN FORWARD PROP COULD NOT FIX, AS MISSING DATA FORWARD PROP WAS TOO LARGE.
        
        print("\n(3 / 5)\t\t Remove meters with too much missing data.")
        print("-----------------------------------------------------------------------------------------------")
        
        # Obtain list of all meters with fully consecutive intervals
        final_consec_intervals = df_memory.groupby('metre_id').progress_apply(self._cern_calculate_consecutive_intervals)
        consec_meters_to_keep = final_consec_intervals[final_consec_intervals['total_count'] != final_consec_intervals['max_consec']]
        
        # Filter out meters with non-consecutive intervals
        df_memory = df_memory[~df_memory['metre_id'].isin(consec_meters_to_keep.index)]
        
        ########################################################################################################
        # ENSURE ALL METERS START AT 00:29 AND END AT 23:59. PROTRACT AND RETRACT UNTIL THIS IS TRUE.
        
        df_memory = df_memory.groupby('metre_id').progress_apply(self._cern_retractend_protractstart_midnight).reset_index(level = 1).drop(['level_1', 'metre_id'], axis=1).reset_index()
        
        #######################################################################################################
        # REMOVE METERS WITH TOO MANY ZEROES
        
        print("\n(4 / 5)\t\t Remove meters with too many zeroes.")
        print("-----------------------------------------------------------------------------------------------")
        
        # List of number of zeroes and percentage of zeroes
        df_zeroes = df_memory.groupby('metre_id').progress_apply(self._cern_count_zeroes)
        
        # List of all meters to be removed
        zero_meters_to_be_removed = df_zeroes[df_zeroes.pct > meter_pct_zeroes_remove].index
        
        # Filter
        df_memory = df_memory[~df_memory.metre_id.isin(zero_meters_to_be_removed)]
        
        #######################################################################################################
        # TRAIN AND TEST SPLIT
        
        print("\n(5 / 5)\t\t Train and test split.")
        print("-----------------------------------------------------------------------------------------------")
        
        # Work out date to split on
        # nb: 95% split refers to day split, not amount of records so won't result in exact 95% records.
        timespan = df_memory['timestamp'].max() - df_memory['timestamp'].min()
        days_to_split = (timespan * train_test_split).days
        split_date = df_memory['timestamp'].min() + pd.to_timedelta('{} days'.format(days_to_split))
        
        # Split
        df_train, df_test = self.tt_split_on_time(df_memory, split_date, 'timestamp')
        
        # Print final split percentage
        print("Train and test split:\t {:.2f} / {:.2f}".format(len(df_train) / len(df_memory), len(df_test) / len(df_memory)))
        
        #######################################################################################################
        # WRITE
        
        # CHECK: All meters have exactly 48 entries per day
        for df in [df_train, df_test]:
            assert sum(df.groupby('metre_id').apply(func=self._cern_check_mod_48)) == len(df.metre_id.unique()), "Non-48 day entry identified for meter"
        
        print("\n(6)\t\t Write files.")
        print("-----------------------------------------------------------------------------------------------")
        
        # Write
        for name, df in zip(['cern_train.csv', 'cern_test.csv'], [df_train, df_test]):
            
            chunks = np.array_split(df.index, 100) # https://stackoverflow.com/questions/64695352/pandas-to-csv-progress-bar-with-tqdm
            
            for chunck, subset in enumerate(tqdm(chunks)):
                if chunck == 0: # first row
                    df.loc[subset].to_csv(name, mode='w', index=True)
                else:
                    df.loc[subset].to_csv(name, header=None, mode='a', index=True)

        
    @staticmethod
    def tt_split_on_time(df, timestamp, timestamp_col = 'timestamp'):
    
        train_df = df[df[timestamp_col] < pd.to_datetime(timestamp)]
        test_df = df[df[timestamp_col] >= pd.to_datetime(timestamp)]
    
        return train_df, test_df
        
    @staticmethod
    def _cern_time_conversion(df, timecode='timecode'):
    
        """
        Auxiliary CERN: convert timecodes to date codes
        """
    
        # Implement the intial date (1/1/19)
        init_date = datetime.datetime(2009, 1, 1)
        
        # Split out the timecodes to day codes and minute codes
        day_code = df[timecode].str.slice(stop=3).astype('int64')
        minute_code = df[timecode].str.slice(start=3).astype('int64')
        
        # Note: dd.to_timedelta() does not seem to work with dd.map_partitions()
        # resorting to using pd.to_timedelta() though I believe this is slower
        
        # Add day code
        temp_date = init_date + pd.to_timedelta(day_code-1, unit='day')
        # Add minute code
        temp_date = temp_date + pd.to_timedelta(minute_code*30, unit='minute')
        # Reduce by one second to keep within same day
        temp_date = temp_date - pd.to_timedelta(1, unit='seconds')
        
        return temp_date
        

    @staticmethod
    def _cern_fix_missing_dates(group_df):
    
        """
        Auxiliary CERN: Inserts the additionally interpolated dates based on frequency using forward propogation for each meter.
        """
        
        def _cern_reindex_by_date(group_df):
    
            """
            Auxiliary CERN: Reindexes to every 30 minutes (fills empties) - function to be used
            within the groupby operation.
            """
            
            group_df = group_df.set_index('timestamp')
            
            # Interpolates the times based on frequency
            dates = pd.date_range(group_df.index.min(), group_df.index.max(), freq='30min')
            
            return group_df.reindex(dates, method='ffill', limit=48) # Limit 48 chosen to prevent to many forward propagation (i.e., max propagation of one day).
            
        group_df = group_df.groupby('metre_id').progress_apply(_cern_reindex_by_date).reset_index(level=1).reset_index(drop=True)
        
        group_df = group_df.rename({'level_1': 'timestamp'}, axis=1)
        
        return group_df
        
    @staticmethod
    def _cern_calculate_consecutive_intervals(group_df):
    
        """
        Auxiliary CERN: Calculate largest consecutive interval length, as well as start and end dates, for CERN dataset.
        """
        
        # Calculate consecutive counter
        group_df = group_df.sort_values(by='timestamp', ascending=True)
        group_df = group_df.reset_index(drop=True)
        group_df_a = (group_df.timestamp.shift(-1) - group_df.timestamp) == pd.to_timedelta(30, 'minutes')
        group_df_b = group_df_a.cumsum()
        group_df['cumcount'] = group_df_b.sub(group_df_b.mask(group_df_a).ffill().fillna(0).astype(int))
        
        # Calculate start and end date
        end = group_df.loc[group_df['cumcount'].idxmax(),'timestamp']
        start = group_df.loc[group_df['cumcount'].idxmax() - group_df['cumcount'].max() + 1,'timestamp']
        
        return pd.Series({'total_count': len(group_df), 'max_consec': group_df.cumcount.max() + 1, 'start': start,'end': end})
        
    @staticmethod
    def _cern_retractend_protractstart_midnight(group_df):
    
        """
        Auxiliary CERN: Retracts all meters to most recent midnights, or protracts them to closest midnights
        """
        
        # Check if we even need to make modifications
        if group_df.timestamp.max().time() == datetime.time(23, 59, 59) and group_df.timestamp.min().time() == datetime.time(0, 29, 59):
            return group_df
        
        #################################################################################################################################
        ##### FIX FIRST DAY
        
        if group_df.timestamp.min().time() != datetime.time(0, 29, 59):
        
            # Grab the latest day
            first_day = group_df.timestamp.min().date()

            # Evaluate from above, what to filter on (i.e., midnight before last day)
            filter_date_start = pd.to_datetime(first_day) + pd.to_timedelta('1 day') + pd.to_timedelta('0 hours 29 minutes 59 seconds')

            group_df = group_df[group_df['timestamp'] >= filter_date_start]
        
        #################################################################################################################################
        ##### FIX LAST DAY
        
        if group_df.timestamp.max().time() != datetime.time(23, 59, 59):
        
            # Grab the latest day
            last_day = group_df.timestamp.max().date()

            # Evaluate from above, what to filter on (i.e., midnight before last day)
            filter_date_end = pd.to_datetime(last_day) - pd.to_timedelta('1 day') + pd.to_timedelta('23 hours 59 minutes 59 seconds')

            group_df = group_df[group_df['timestamp'] <= filter_date_end]
        
        return group_df
        
    @staticmethod
    def _cern_check_mod_48(group_df):
    
        """
        AUXILIARY CERN: Checks to ensure that there are exactly 48 entries for all days between start and finish, for each meter. 
        """
    
        days = (group_df.timestamp.max().date() - group_df.timestamp.min().date()).days + 1 # Number of days between start and end for meter
        no_records = len(group_df) # Total number of records
        expected_records = days * 48 # Recalc of expected records
        
        return no_records == expected_records
        
    @staticmethod
    def _cern_count_zeroes(group_df):
        
        """
        AUXILIARY CERN: Counts zeroes per groupby subset
        """
        
        return pd.Series((sum(group_df.kwh == 0), len(group_df),
                            sum(group_df.kwh == 0) / len(group_df) * 100),
                            index = ['zeroes', 'records', 'pct'])
