from singleton_decorator import singleton
import os
import datetime
import shutil
import pandas as pd
from pd_explain.experimental.query_recommenders import consts as consts


@singleton
class QueryLogger:

    def __init__(self):
        """
        Initialize the QueryLogger with the log file location, and logging flag.
        """
        self._log_file_location = os.getenv(consts.DOT_ENV_PD_EXPLAIN_LOG_FILE_LOCATION)
        self._use_logging = os.getenv(consts.DOT_ENV_PD_EXPLAIN_LOG_QUERIES) == 'True'
        if not os.path.exists(self.log_file_location) and self._use_logging:
            with open(self.log_file_location, "w") as f:
                f.write("dataframe_name,query,interestingness_score,timestamp\n")
        if self._use_logging:
            self._log = pd.DataFrame(pd.read_csv(self.log_file_location, index_col=0))
        else:
            self._log = None
        self._write_index_flag = False
        self._set_use_logging_func = lambda use_logging: None



    @property
    def use_logging(self):
        """
        Get the use_logging flag.
        """
        return self._use_logging

    @use_logging.setter
    def use_logging(self, value: bool | str):
        """
        Set the use_logging flag.
        """
        self._use_logging = bool(value)
        self._set_use_logging_func(self._use_logging)

    @property
    def log_file_location(self):
        """
        Get the log file location.
        """
        return self._log_file_location

    @log_file_location.setter
    def log_file_location(self, value: str):
        """
        Set the log file location.
        """
        old_location = self._log_file_location
        self._log_file_location = value
        # Copy the existing log file to the new location if it exists
        if os.path.exists(self._log_file_location):
            shutil.copy(old_location, self._log_file_location)
            os.remove(old_location)
        # Create a new log file
        else:
            with open(self._log_file_location, "w") as f:
                f.write("dataframe_name,query,interestingness_score,timestamp\n")


    def log_query(self, dataframe_name: str, query: str, score: float):
        """
        Log the query to the log file.
        We store the dataframe name, query, score and timestamp.
        We store the logs as a pandas dataframe, where the index is the dataframe name, and the columns are the query, score and timestamp.
        """
        if self._use_logging:
            if not os.path.exists(self.log_file_location):
                with open(self.log_file_location, "w") as f:
                    f.write("dataframe_name,query,interestingness_score,timestamp\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = pd.DataFrame({
                'query': [query],
                'interestingness_score': [score],
                'timestamp': [timestamp]
            }, index=pd.Index([dataframe_name], name='dataframe_name'))
            # Append the log entry to the log dataframe
            self._log = pd.concat([self._log, log_entry])
            # Save the log to the log file
            log_entry.to_csv(self.log_file_location, mode='a', header=self._write_index_flag)
            # Set the write index flag to False after the first write
            self._write_index_flag = False


    def delete_log(self, data_only=True):
        """
        Delete the log file.
        """
        if os.path.exists(self.log_file_location):
            if data_only:
                with open(self.log_file_location, "w") as f:
                    f.write("dataframe_name,query,interestingness_score,timestamp\n")
            else:
                # Remove the log file
                os.remove(self.log_file_location)
                self._write_index_flag = True
            self._log = pd.DataFrame(columns=['query', 'interestingness_score', 'timestamp'], index=pd.Index([], name='dataframe_name'))


    def get_log(self, dataframe_name: str = None, k: int = 10):
        """
        Get the log dataframe.
        :param dataframe_name: The name of the dataframe to get the log for. If None, return the entire log.
        :param k: The number of most recent entries to return. If None or 0, return all entries.

        :return: The log dataframe.
        """
        if dataframe_name and self._use_logging:
            log = self._log[self._log.index == dataframe_name]
            if k and k > 0:
                # Sort the log by timestamp in descending order
                log = log.sort_values(by=['timestamp'], ascending=False)
                return log.tail(k)
            else:
                return log
        else:
            return self._log


