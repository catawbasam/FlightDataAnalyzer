from hdfaccess.utils import concat_hdf
from flightdatautilities.filesystem_tools import copy_file

def join_files(first_part, second_part, copy=True):
    """
    Flight Joining of two parts. Expected for joining a START_ONLY and
    STOP_ONLY segment together.
    
    Q: Support deleting of first and second part?
    
    :param first_part: Path to first hdf file.
    :type first_part: String (path)
    :param second_part: Path to first hdf file.
    :type second_part: String (path)
    """
    if copy:
        dest = copy_file(first_part, postfix='_joined')
    else:
        dest = first_part
    hdf_path = concat_hdf([first_part, second_part], dest=dest)
    return hdf_path