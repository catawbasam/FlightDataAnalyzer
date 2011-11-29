import os
import matplotlib.pyplot as plt

from analysis.hdf_access import hdf_file


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
   
   
def plot_flight (hdf_path, kti_list, kpv_list, phase_list):
    """
    """
    ax2.annotate(point, xy=(event[0]-first,
            fp.altitude_std.data[event[0]]),
            xytext=(-5, 100), 
            textcoords='offset points',
            #arrowprops=dict(arrowstyle="->"),
            rotation='vertical'
            )
    
    ax1.plot(ph['Takeoff'], 'r-', ph['Landing'], 'g-',
    ph['Air'], 'b-',
    ph['Ground'], 'k-')
                
    ##first, last, fp, kpt, kpv, ph, rejected_takeoff=False):
    """
    @param rejected_takeoff: Boolean
    """

    ##first, last = block.start, block.stop
    
    fig = figure(figsize=(10,8))
    
    if rejected_takeoff:
        ax1 = fig.add_subplot(2,1,1)
        ax1.autoscale(enable=False, axis='x')
        ax1.set_xbound(first, last)
        ax1.plot(fp.airspeed.data[first:last], 'r-')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.plot(fp.altitude_std.data[first:last], 'b-')
        ax2.grid(True) 
    else: #normal flight
        ax1 = fig.add_subplot(4,1,1)
        ax1.autoscale(enable=False, axis='x')
        ax1.set_xbound(first, last)
        ax1.plot(ph['Takeoff'], 'r-', ph['Landing'], 'g-',
                ph['Air'], 'b-',
                ph['Ground'], 'k-')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(4,1,2, sharex=ax1)
        # ax2.set_ybound(-1000,45000)
        ax2.plot(
                 # fp.altitude_std.data, 'y-',
                 # fp.altitude_aal_takeoff.data, 'r-',
                 ph['Climbing'], 'g-', 
                 ph['LevelFlight'], 'b-',
                 ph['Descending'], 'r-')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(4,1,3, sharex=ax1)
        ax3.plot(fp.rate_of_climb_smooth.data[first:last], 'k-')
        ax3.grid(True)
            
        ax4 = fig.add_subplot(4,1,4, sharex=ax1)
        ax4.plot(ph['Initial_Climb'], 'r-',
                 ph['Climb'], 'g-',
                 ph['Cruise'], 'b-',
                 ph['Descent'], 'g-',
                 ph['Approach'], 'r-',
                 ph['Ground'], 'k-')    
        ax4.grid(True)

        
    for point in kpv:
        for event in kpv[point]:
            ax2.annotate(point, xy=(event[0]-first,
                        fp.altitude_std.data[event[0]]),
                        xytext=(-5, 100), 
                        textcoords='offset points',
                        #arrowprops=dict(arrowstyle="->"),
                        rotation='vertical'
                        )
    for moment in kpt:
        for special_moment in kpt[moment]:
            ax1.annotate(moment, xy=(special_moment-first,1.0),
                        xytext=(-5, 100),
                        textcoords='offset points',
                        #arrowprops=dict(arrowstyle="->"),
                        rotation='vertical'
                        )
    show()