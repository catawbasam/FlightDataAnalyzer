import argparse
from copy import copy
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import simplekml

from analysis_engine.node import derived_param_from_hdf, Parameter
from hdfaccess.file import hdf_file
from library import bearing_and_distance, latitudes_and_longitudes, repair_mask
from settings import METRES_TO_FEET
from utilities.print_table import indent


logger = logging.getLogger(name=__name__)

class TypedWriter(object):
    """
    A CSV writer which will write rows to CSV file "f",
    which uses "fieldformats" to format fields.
    
    ref: http://stackoverflow.com/questions/2982642/specifying-formatting-for-csv-writer-in-python
    """

    def __init__(self, f, fieldnames, fieldformats, **kwds):
        self.writer = csv.DictWriter(f, fieldnames, **kwds)
        self.writer.writeheader()
        self.formats = fieldformats

    def _format(self, row):
        return dict((k, self.formats.get(k, '%s') % v if v or v == 0.0 else v) 
                    for k, v in row.iteritems())
    
    def writerow(self, row):
        self.writer.writerow(self._format(row))

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
            
    def rowlist(self, rows):
        "Return a list of formatted rows as ordered by fieldnames"
        res = []
        for row in rows:
            res.append(self.writer._dict_to_list(self._format(row)))
        return res
            
            
def add_track(kml, track_name, lat, lon, colour, alt_param=None):
    track_config = {'name': track_name}
    if alt_param:
        if alt_param.name in ['Altitude QNH', 'Altitude AAL', 'Altitude STD']:
            track_config['altitudemode'] = simplekml.constants.AltitudeMode.absolute
        elif alt_param.name in ['Altitude Radio']:
            track_config['altitudemode'] = simplekml.constants.AltitudeMode.relativetoground
        else:
            raise NotImplementedError("Altitude parameter '%s' not handled" % alt_param.name)
        track_config['extrude'] = 1
        
    track_coords = []
    scope_lon = np.ma.flatnotmasked_edges(lon.array)
    scope_lat = np.ma.flatnotmasked_edges(lat.array)
    begin = max(scope_lon[0], scope_lat[0])+1
    end = min(scope_lon[1], scope_lat[1])-1
    for i in range(begin, end):
        if lat.array.mask[i] or lon.array.mask[i] or (alt_param and alt_param.array.mask[i]):
            pass  # Masked data not worth plotting
        else:
            if alt_param:
                track_coords.append((lon.array[i],lat.array[i], alt_param.array[i]))
            else:
                track_coords.append((lon.array[i],lat.array[i]))
                
    track_config['coords'] = track_coords
    line = kml.newlinestring(**track_config)
    line.style.linestyle.color = colour
    line.style.polystyle.color = '66%s' % colour[2:] # set opacity of area fill to 40%
    return


def draw_centreline(kml, rwy):
    start_lat = rwy['start']['latitude']
    start_lon = rwy['start']['longitude']
    end_lat = rwy['end']['latitude']
    end_lon = rwy['end']['longitude']
    brg, dist = bearing_and_distance(end_lat, end_lon, start_lat, start_lon)
    brgs = np.ma.array([brg])
    dists = np.ma.array([30000])
    lat_30k, lon_30k = latitudes_and_longitudes(brgs, dists, rwy['start'])
    try:
        angle = np.deg2rad(rwy['glideslope']['angle'])
    except:
        angle = np.deg2rad(3.0)
    end_height = 30000 * np.tan(angle) * METRES_TO_FEET
    track_config = {'name': 'ILS'}
    track_coords = []
    track_coords.append((end_lon,end_lat))
    track_coords.append((lon_30k.data[0],lat_30k.data[0], end_height))
    track_config['coords'] = track_coords
    kml.newlinestring(**track_config)
    return


def track_to_kml(hdf_path, kti_list, kpv_list, flight_attrs,
                 plot_altitude=False, dest_path=None):
    '''
    Plot results of process_flight onto a KML track.
    
    :param flight_attrs: List of Flight Attributes
    :type flight_attrs: list
    :param plot_altitude: Name of Altitude parameter to use in KML
    :type plot_altitude: String
    '''
    #if 'Latitude Smoothed' not in hdf or 'Longitude Smoothed' not in hdf:
        ## unable to plot without these parameters
        #return
    kml = simplekml.Kml()
    with hdf_file(hdf_path) as hdf:
        if plot_altitude:
            alt = derived_param_from_hdf(hdf[plot_altitude])
            alt.array = repair_mask(alt.array, frequency=alt.frequency, repair_duration=None) / METRES_TO_FEET
        else:
            alt = None
                  
        smooth_lat = derived_param_from_hdf(hdf['Latitude Smoothed'])
        smooth_lon = derived_param_from_hdf(hdf['Longitude Smoothed'])
        add_track(kml, 'Smoothed', smooth_lat, smooth_lon, 'ff7fff7f', 
                  alt_param=alt)
        add_track(kml, 'Smoothed On Ground', smooth_lat, smooth_lon, 'ff7fff7f')        
    
        lat = derived_param_from_hdf(hdf['Latitude Prepared'])
        lon = derived_param_from_hdf(hdf['Longitude Prepared'])
        add_track(kml, 'Prepared', lat, lon, 'ff0000ff')
        
        lat_r = derived_param_from_hdf(hdf['Latitude'])
        lon_r = derived_param_from_hdf(hdf['Longitude'])
        
    add_track(kml, 'Recorded Track', lat_r, lon_r, 'ff0000ff')

    for kti in kti_list:
        kti_point_values = {'name': kti.name}
        if kti.name not in ['Transmit']:
            #and not kti.name.endswith('Descending')\
            #and not kti.name.endswith('Climbing')\
            altitude = alt.at(kti.index) if plot_altitude else None
            if altitude:
                kti_point_values['coords'] = ((kti.longitude, kti.latitude, altitude),)
                kti_point_values['altitudemode'] = simplekml.constants.AltitudeMode.relativetoground 
            else:
                kti_point_values['coords'] = ((kti.longitude, kti.latitude,),)
                kti_point_values['altitudemode'] = simplekml.constants.AltitudeMode.clamptoground 
        
            kml.newpoint(**kti_point_values)
        
    for kpv in kpv_list:

        # Trap kpvs with invalid latitude or longitude data (normally happens
        # at the start of the data where accelerometer offsets are declared,
        # and this avoids casting kpvs into the Atlantic.
        kpv_lat = smooth_lat.at(kpv.index)
        kpv_lon = smooth_lon.at(kpv.index)
        if kpv_lat == None or kpv_lon == None or \
           (kpv_lat == 0.0 and kpv_lon == 0.0):
            continue

        if kpv.name not in ['z']:
            style = simplekml.Style()
            style.iconstyle.color = simplekml.Color.red
            kpv_point_values = {'name': '%s (%s)' % (kpv.name, kpv.value)}
            altitude = alt.at(kpv.index) if plot_altitude else None
            if altitude:
                kpv_point_values['coords'] = (
                    (kpv_lon, kpv_lat, altitude,),)
                kpv_point_values['altitudemode'] = simplekml.constants.AltitudeMode.relativetoground 
            else:
                kpv_point_values['coords'] = (
                    (kpv_lon, kpv_lat,),)
                kpv_point_values['altitudemode'] = simplekml.constants.AltitudeMode.clamptoground 
            
            pnt = kml.newpoint(**kpv_point_values)
            pnt.style = style
            
    for attribute in flight_attrs:
        if attribute.name in ['FDR Approaches']:
            for app in attribute.value:
                try:
                    draw_centreline(kml, app['runway'])
                except:
                    pass

    if not dest_path:
        dest_path = hdf_path + ".kml"
    kml.save(dest_path)
    return


def plot_parameter(array, show=True, label=''):
    """
    For quickly plotting a single parameter to see its shape.
    
    :param array: Numpy array
    :type array: np.array
    :param show: Whether to display the figure (and block)
    :type show: Boolean
    """
    try:
        plt.title("Length: %d | Min: %.2f | Max: %.2f" % (
            len(array), array.min(), array.max()))
    except AttributeError:
        # if a non-np.array is passed in, make do
        plt.title("Length: %d | Min: %.2f | Max: %.2f" % (
            len(array), min(array), max(array)))
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


def plot_flight(hdf_path, kti_list, kpv_list, phase_list, aircraft_info):
    """
    """
    fig = plt.figure() ##figsize=(10,8))
    plt.title(os.path.basename(hdf_path))
    
    with hdf_file(hdf_path) as hdf:
        #---------- Axis 1 ----------
        ax1 = fig.add_subplot(4,1,1)
        alt_data = hdf['Altitude STD'].array
        alt = hdf.get('Altitude AAL For Flight Phases',
                      hdf['Altitude STD']).array
        frame = hdf['Time'].array
        #frame = hdf.get('Frame Counter',hdf['Altitude STD']).array
        
        sections = []
        sections.append(alt_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Landing', 'Airborne', 'Grounded'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice])
            sections.append(alt[phase.slice])
            if phase.name == 'Takeoff':
                sections.append('r-')
            elif phase.name == 'Landing':
                sections.append('g-')
            elif phase.name == 'Airborne':
                sections.append('b-')
            elif phase.name == 'Grounded':
                sections.append('k-')
        ax1.plot(*sections)
        
        #---------- Axis 2 ----------
        ax2 = fig.add_subplot(4,1,2)
        #ax2 = fig.add_subplot(4,1,2,sharex=ax1)
        vert_spd = hdf.get('Vertical Speed For Flight Phases', hdf['Altitude STD']).array
        vert_spd_data = hdf.get('Vertical Speed', hdf['Altitude STD']).array
        sections = []
        sections.append(vert_spd_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Level Flight', 'Descending'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice]-frame[0])
            sections.append(vert_spd[phase.slice])
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
    
    Phase types have a 'duration' column
    
    :param dest_path: Outputs CSV to dest_path (removing if exists). If None,
      collates results by appending to a single file: 'combined_test_output.csv'
    """
    rows = []
    params = ['Airspeed', 'Altitude AAL']
    attrs = ['value', 'datetime', 'latitude', 'longitude'] 
    header = ['path', 'type', 'index', 'duration', 'name'] + attrs + params
    if not dest_path:
        header.append('Path')
    formats = {'index': '%.3f',
               'value': '%.3f',
               'duration': '%.2f',
               'latitude': '%.4f',
               'longitude': '%.4f',
               'Airspeed': '%d kts',
               'Altitude AAL': '%d ft',
               }
    for value in kti_list:
        vals = value.todict()  # recordtype
        vals['path'] = hdf_path
        vals['type'] = 'Key Time Instance'
        rows.append( vals )

    for value in kpv_list:
        vals = value.todict()  # recordtype
        vals['path'] = hdf_path
        vals['type'] = 'Key Point Value'
        rows.append( vals )

    for value in phase_list:
        vals = value._asdict()  # namedtuple
        vals['name'] = value.name + ' [START]'
        vals['path'] = hdf_path
        vals['type'] = 'Phase'
        vals['index'] = value.start_edge
        vals['duration'] = value.stop_edge - value.start_edge  # (secs)
        rows.append(vals)
        # create another at the stop of the phase
        end = copy(vals)
        end['name'] = value.name + ' [END]'
        end['index'] = value.stop_edge
        rows.append(end)
    
    # Append values of useful parameters at this time
    with hdf_file(hdf_path) as hdf:
        for param in params:
            # Create DerivedParameterNode to utilise the .at() method
            p = hdf[param]
            dp = Parameter(name=p.name, array=p.array, 
                           frequency=p.frequency, offset=p.offset)
            for row in rows:
                row[param] = dp.at(row['index'])

    # sort rows
    rows = sorted(rows, key=lambda x: x['index'])
    
    # print to CSV
    if not dest_path:
        dest_path = 'combined_test_output.csv'
    elif os.path.isfile(dest_path):
        logger.info("Deleting existing copy of: %s", dest_path)
        os.remove(dest_path)
        
    with open(dest_path, 'ab') as dest:
        writer = TypedWriter(dest, fieldnames=header, fieldformats=formats,
                             extrasaction='ignore')
        writer.writerows(rows)
        # print to Debug I/O
        logger.info(indent([header] + writer.rowlist(rows), hasHeader=True, 
                           wrapfunc=lambda x:str(x)))
    return rows




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-tail', dest='tail_number', type=str, default='G-ABCD',
                        help='Aircraft Tail Number for processing.')
    parser.add_argument('-frame', dest='frame', type=str, default=None,
                        help='Data frame name.')
    args = parser.parse_args()    
       
    plot_flight(args.file, [], [], [], {
        'Tail Number': args.tail_number,
        'Precise Positioning': True,
        'Frame': args.frame})
