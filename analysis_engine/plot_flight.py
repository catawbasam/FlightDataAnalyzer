import csv
import os
import logging
import matplotlib.pyplot as plt
import simplekml

from analysis_engine.node import derived_param_from_hdf, Parameter
from settings import METRES_TO_FEET
from library import rms_noise, repair_mask
from utilities.print_table import indent
from hdfaccess.file import hdf_file

def add_track(kml, track_name, lat, lon, colour, alt_param=None):
    
    track_config = {'name': track_name}
    if alt_param:
        if alt_param.name in ['Altitude AAL', 'Altitude STD']:
            track_config['altitudemode'] = simplekml.constants.AltitudeMode.absolute
        elif alt_param.name in ['Altitude Radio', 'Altitude Radio']:
            track_config['altitudemode'] = simplekml.constants.AltitudeMode.relativetoground
        track_config['extrude'] = 1
        
    track_coords = []
    for i in range(len(lat.array)):
        if lat.array.mask[i] or lon.array.mask[i] or (alt_param and alt_param.array.mask[i]):
            pass  # Masked data not worth plotting
        else:
            if alt_param:
                track_coords.append((lon.at(i),lat.at(i), (alt_param.at(i)+241)/METRES_TO_FEET))
            else:
                track_coords.append((lon.at(i),lat.at(i)))
                
    track_config['coords'] = track_coords
    line = kml.newlinestring(**track_config)
    line.style.linestyle.color = colour
    line.style.polystyle.color = '66%s' % colour[2:] # set opacity of area fill to 40%
    return

def track_to_kml(hdf_path, kti_list, kpv_list, plot_altitude=None):
    hdf = hdf_file(hdf_path)
    #if 'Latitude Smoothed' not in hdf or 'Longitude Smoothed' not in hdf:
        ## unable to plot without these parameters
        #return
    kml = simplekml.Kml()
    if plot_altitude:
        alt = derived_param_from_hdf(hdf, plot_altitude)
        alt.array = repair_mask(alt.array, frequency=alt.frequency, repair_duration=None)
    else:
        alt = None
              
    smooth_lat = derived_param_from_hdf(hdf, 'Latitude Smoothed')
    smooth_lon = derived_param_from_hdf(hdf, 'Longitude Smoothed')
    #lat = derived_param_from_hdf(hdf, 'Latitude')
    #lon = derived_param_from_hdf(hdf, 'Longitude')
    
    #add_track(kml, 'Recorded', lat, lon, 'ff0000ff')
    add_track(kml, 'Smoothed', smooth_lat, smooth_lon, 'ff7fff7f', 
              alt_param=alt)

    for kti in kti_list:
        kti_point_values = {'name': kti.name}
        altitude = alt.at(kti.index) if plot_altitude else None
        if altitude:
            kti_point_values['coords'] = (
                (kti.longitude, kti.latitude, (altitude+241)/METRES_TO_FEET),) # TODO: AIRPORT OFFSET HACK REMOVE AFTER USE
            kti_point_values['altitudemode'] = simplekml.constants.AltitudeMode.absolute
        else:
            kti_point_values['coords'] = ((kti.longitude, kti.latitude,),)
        
        kml.newpoint(**kti_point_values)
        
    for kpv in kpv_list:
        kpv_point_values = {'name': '%s (%s)' % (kpv.name, kpv.value)}
        altitude = alt.at(kpv.index) if plot_altitude else None
        if altitude:
            kpv_point_values['coords'] = (
                (smooth_lon.at(kpv.index), smooth_lat.at(kpv.index), (altitude+241)/METRES_TO_FEET), # TODO: AIRPORT OFFSET HACK REMOVE AFTER USE
            )
            kpv_point_values['altitudemode'] = simplekml.constants.AltitudeMode.absolute
        else:
            kpv_point_values['coords'] = ((smooth_lon.at(kpv.index), smooth_lat.at(kpv.index)),)
        
        kml.newpoint(**kpv_point_values)
        

    kml.save(hdf_path+".kml")
    hdf.close()
    return

def plot_parameter(array, show=True, label=''):
    """
    For quickly plotting a single parameter to see its shape.
    
    :param array: Numpy array
    :type array: np.array
    :param show: Whether to display the figure (and block)
    :type show: Boolean
    """
    plt.title("Length: %d | Min: %.2f | Max: %.2f" % (
               len(array), array.min(), array.max()))
    plt.plot(array, label=label)
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
        frame = hdf['Time'].array
        #frame = hdf.get('Frame Counter',hdf['Altitude STD']).array
        
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
        ax2 = fig.add_subplot(4,1,2)
        #ax2 = fig.add_subplot(4,1,2,sharex=ax1)
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
        ax3 = fig.add_subplot(4,1,3)
        #ax3 = fig.add_subplot(4,1,3,sharex=ax1)
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
    rows = []
    params = ['Airspeed', 'Altitude AAL', 'Pitch', 'Roll']
    attrs = ['value', 'datetime', 'latitude', 'longitude'] 
    header = ['Type', 'Phase Start', 'Index', 'Phase End', 'Name'] + attrs + params

    with hdf_file(hdf_path) as hdf:
        for value in kti_list:
            vals = ['Key Time Instance', None, value.index, None, value.name, None,
                    value.datetime, value.latitude, value.longitude]
            rows.append( vals )

        for value in kpv_list:
            vals = ['Key Point Value', None, value.index, None, value.name, value.value,
                    value.datetime]+[None]*2
            rows.append( vals )

        for value in phase_list:
            vals = ['Phase', value.name, value.start_edge]+[None]*6
            rows.append( vals )
            vals = ['Phase', None, value.stop_edge, value.name]+[None]*5
            rows.append( vals )

        for param in params:
            # Create DerivedParameterNode to utilise the .at() method
            p = hdf[param]
            dp = Parameter(name=p.name, array=p.array, 
                                frequency=p.frequency, offset=p.offset)
            for row in rows:
                row.append(dp.at(row[2]))

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
    logging.info(indent([header] + rows, hasHeader=True, wrapfunc=lambda x:str(x)))


if __name__ == '__main__':
    import sys
    hdf_path = sys.argv[1]
    plot_flight(hdf_path, [], [], [])
