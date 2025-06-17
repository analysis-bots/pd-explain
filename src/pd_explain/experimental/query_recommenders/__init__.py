import os
import dotenv
import pd_explain.experimental.query_recommenders.consts as consts
from pd_explain.experimental.query_recommenders.query_logger import QueryLogger

# Determine the directory of the current file
package_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file within the package directory
env_path = os.path.join(package_dir, ".env")

if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LOG_QUERIES}=False\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LOG_FILE_LOCATION}={os.path.join(package_dir, consts.PD_EXPLAIN_DEFAULT_LOG_FILE_LOCATION)}\n")

# Load the environment variables from the .env file
dotenv.load_dotenv(env_path, override=False)

logger = QueryLogger()

# Methods to set up logging for the query recommender
class LoggingSetupFunctions:

    @staticmethod
    def set_use_logging(use_logging: bool):
        """
        Set the use_logging flag in the environment variables and .env file.
        This flag determines whether to log queries or not.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LOG_QUERIES] = str(use_logging)
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LOG_QUERIES, str(use_logging))
        logger.use_logging = use_logging

    @staticmethod
    def set_log_file_location(log_file_location: str):
        """
        Set the log file location in the environment variables and .env file.
        This location is where the logs will be stored.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LOG_FILE_LOCATION] = log_file_location
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LOG_FILE_LOCATION, log_file_location)
        # Update the logger's log file location
        logger.log_file_location = log_file_location

logger._set_use_logging_func = LoggingSetupFunctions.set_use_logging


if consts.DOT_ENV_PD_EXPLAIN_LOG_QUERIES not in os.environ:
    LoggingSetupFunctions.set_use_logging(False)
if consts.DOT_ENV_PD_EXPLAIN_LOG_FILE_LOCATION not in os.environ:
    LoggingSetupFunctions.set_log_file_location(os.path.join(package_dir, consts.PD_EXPLAIN_DEFAULT_LOG_FILE_LOCATION))



