import csv
import os
import itertools
import matplotlib.pyplot as plt
from analysis.node import DerivedParameterNode

from utilities.print_table import indent

from hdfaccess.file import hdf_file


def plot_parameter(array, show=True):
    """
    For quickly plotting a single parameter to see its shape.
    
    :param array: Numpy array
    :type array: np.array
    :param show: Whether to display the figure (and block)
    :type show: Boolean
    """
    plt.title("Length: %d | Min: %.2f | Max: %.2f" % (
               len(array), array.min(), array.max()))
    plt.plot(array)
    if show:
        plt.show()
    return
    

def plot_essential(hdf_path):
    """
    Plot the essential parameters for flight analysis.
    
    Assumes hdf_path file contains the parameter series:
    Frame Counter, Airspeed, Altitude STD, Head True
    
    show() is to be called elsewhere (from matplotlib.pyplot import show)
    
    :param hdf_path: Path to HDF file.
    :type hdf_path: string
    """
    fig = plt.figure() ##figsize=(10,8))
    plt.title(os.path.basename(hdf_path))
    
    with hdf_file(hdf_path) as hdf:
        ax1 = fig.add_subplot(4,1,1)
        #ax1.set_title('Frame Counter')
        ax1.plot(hdf['Frame Counter'].array, 'k--')
        ax2 = fig.add_subplot(4,1,2)
        ax2.plot(hdf['Airspeed'].array, 'r-')
        ax3 = fig.add_subplot(4,1,3,sharex=ax2)
        ax3.plot(hdf['Altitude STD'].array, 'g-')
        ax4 = fig.add_subplot(4,1,4,sharex=ax2)
        ax4.plot(hdf['Head True'].array, 'b-')    
   
   
def plot_flight(hdf_path, kti_list, kpv_list, phase_list):
    """
    """
    fig = plt.figure() ##figsize=(10,8))
    plt.title(os.path.basename(hdf_path))
    
    with hdf_file(hdf_path) as hdf:
        #---------- Axis 1 ----------
        ax1 = fig.add_subplot(4,1,1)
        alt_data = hdf['Altitude STD'].array
        alt = hdf.get('Altitude AAL For Flight Phases',hdf['Altitude STD']).array
        #frame = hdf['Frame Counter'].array
        frame = hdf.get('Frame Counter',hdf['Altitude STD']).array
        
        sections = []
        sections.append(alt_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Landing', 'Airborne', 'On Ground'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice])
            sections.append(alt[phase.slice])
            if phase.name == 'Takeoff':
                sections.append('r-')
            elif phase.name == 'Landing':
                sections.append('g-')
            elif phase.name == 'Airborne':
                sections.append('b-')
            elif phase.name == 'On Ground':
                sections.append('k-')
        ax1.plot(*sections)
        
        #---------- Axis 2 ----------
        ax2 = fig.add_subplot(4,1,2,sharex=ax1)
        roc = hdf.get('Rate Of Climb For Flight Phases', hdf['Altitude STD']).array
        roc_data = hdf.get('Rate Of Climb', hdf['Altitude STD']).array
        sections = []
        sections.append(roc_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Level Flight', 'Descending'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice]-frame[0])
            sections.append(roc[phase.slice])
            if phase.name == 'Takeoff':
                sections.append('g-')
            elif phase.name == 'Level Flight':
                sections.append('b-')
            elif phase.name == 'Descending':
                sections.append('c-')
        ax2.plot(*sections)
        
        #---------- Axis 3 ----------
        ax3 = fig.add_subplot(4,1,3,sharex=ax1)
        airspeed = hdf.get('Airspeed',hdf['Altitude STD']).array
        sections = []
        sections.append(airspeed)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Fast'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice]-frame[0])
            sections.append(airspeed[phase.slice])
            if phase.name == 'Fast':
                sections.append('r-')
        
        ax3.plot(*sections)
        
        #---------- Axis 4 ----------
        if 'Heading Continuous' in hdf:
            ax4 = fig.add_subplot(4,1,4,sharex=ax1)
            ax4.plot(hdf['Heading Continuous'].array, 'b-')  
    
    for kpv in kpv_list:
        label = '%s %s' % (kpv.name, kpv.value)
        ax1.annotate(label, xy=(kpv.index, alt[kpv.index]),
                     xytext=(-5, 100), 
                     textcoords='offset points',
                     #arrowprops=dict(arrowstyle="->"),
                     rotation='vertical'
                     )
    '''
    for kti in kti_list:
        label = '%s' % (kti.name)
        ax1.annotate(label, xy=(kti.index, alt[kti.index]),
                     xytext=(-5, 100), 
                     textcoords='offset points',
                     #arrowprops=dict(arrowstyle="->"),
                     rotation='vertical'
                     )
    '''
    plt.show()
    return


def _index_or_slice(x):
    try:
        return float(x.index)
    except (TypeError, AttributeError):
        return x.slice.start

def csv_flight_details(hdf_path, kti_list, kpv_list, phase_list, dest_path=None):
    """
    Currently writes to csv and prints to a table.
    
    :param dest_path: If None, writes to hdf_path.csv
    """
    ##iterable = itertools.chain(kti_list, kpv_list, phase_list)
    ##index_sorted_keys = sorted(iterable, key=_index_or_slice)

    rows = []
    params = ['Airspeed', 'Altitude STD', 'Pitch', 'Roll']
    attrs = ['value', 'slice', 'datetime'] # 'latitude', 'longitude'] 
    header = ['Type', 'Phase Start', 'Index', 'Phase End', 'Name'] + attrs + params

    def vals_for_iterable(iter_type, iterable):
        for value in iterable:
            # add required attributes
            index = _index_or_slice(value)
            if iter_type == 'Phase':
                
                # TACKY FIX FOR PHASE START AND STOP
                vals = [iter_type, value.name, value.slice.start, None, None]
                rows.append( vals )
                # EMBARRASING BUT WAITING FOR "at" METHOD ON DERIVED NODE
                
                vals = [iter_type, None, value.slice.stop, value.name, None]
            else:
                vals = [iter_type, None, index, None, value.name]
                
            # add optional attributes
            [vals.append(getattr(value, attr, None)) for attr in attrs]
            
            # add associated parameter information
            for param in params:
                try:
                    # Create DerivedParameterNode to utilise the .at() method
                    p = hdf[param]
                    dp = DerivedParameterNode(name=p.name, array=p.array, 
                                        frequency=p.frequency, offset=p.offset)
                    vals.append( dp.at(index) )
                except (KeyError, ValueError, IndexError):
                    vals.append(None)
            
            if iter_type == 'Phase':
                # Append the stop time for this phase.
                vals = [iter_type, None, value.slice.stop, value.name, None]
                # add optional attributes
                [vals.append(getattr(value, attr, None)) for attr in attrs]
                # add associated parameter information
                for param in params:
                    try:
                        p = hdf[param]
                        dp = DerivedParameterNode(name=p.name, array=p.array, 
                                        frequency=p.frequency, offset=p.offset)
                        vals.append( dp.at(index) )
                    except (KeyError, ValueError, IndexError):
                        vals.append(None)

            rows.append( vals )
            
    with hdf_file(hdf_path) as hdf:
        vals_for_iterable('Key Time Instance', kti_list)
        vals_for_iterable('Key Point Value', kpv_list)
        vals_for_iterable('Phase', phase_list)

    # sort rows
    rows = sorted(rows, key=lambda x: x[header.index('Index')])
    # print to CSV
    if not dest_path:
        dest_path = os.path.splitext(hdf_path)[0] + '_values_at_indexes.csv'
    with open(dest_path, 'wb') as dest:
        writer = csv.writer(dest)
        writer.writerow(header)
        writer.writerows(rows)
    
    # print to Debug I/O
    print indent([header] + rows, hasHeader=True, wrapfunc=lambda x:str(x))


if __name__ == '__main__':
    import sys
    hdf_path = sys.argv[1]
    plot_flight(hdf_path, [], [], [])