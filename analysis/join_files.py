from hdfaccess.utils import concat_hdf

def join_files(first_part, second_part):
    """
    Flight Joining
    """
    hdf_path = concat_hdf([first_part, second_part], dest=first_part) 
    return hdf_path