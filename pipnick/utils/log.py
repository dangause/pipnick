import pkg_resources
import json
import logging
import numpy as np


def adjust_global_logger(log_level='INFO', name='all_others'):
    """
    Adjusts the global logging configuration to use a specific log level and output file.

    Parameters
    ----------
    log_level : int, str, optional
        String or integer representation of logging level to set for the console
        handler (default is 'INFO').  Integer mapping is: 1='CRITICAL',
        2='ERROR', 3='WARNING', 4='INFO', 5='DEBUG'.
    name : str, optional
        The base name for the log file (default is 'all_others').
    """
    if isinstance(log_level, str):
        _log_level = log_level 
    else:
        log_levels = {1:'CRITICAL', 2:'ERROR', 3:'WARNING', 4:'INFO', 5:'DEBUG'}
        _log_level = log_levels[np.clip(log_level, 1, 5)]

    # Load the JSON configuration for logging
    output_file = f"{name.split('.')[-1]}.log"
    
    # Adjust the configuration for the file name and log level
    with pkg_resources.resource_stream('pipnick.utils', 'logging_config.json') as f:
        config = json.load(f)
    config['handlers']['file']['filename'] = output_file
    config['handlers']['console']['level'] = _log_level

    # Configure logging with the loaded configuration
    logging.config.dictConfig(config)


def log_astropy_table(table):
    """
    Formats an Astropy Table into a string for logging purposes.

    Parameters
    ----------
    table : astropy.table.Table
        The Astropy Table to be formatted.

    Returns
    -------
    str
        The formatted string representation of the table.
    """
    # Convert each line of the table's formatted output into a string
    print_table = ""
    for line in table.pformat_all():
        print_table += f"{line}\n"
    return print_table
