from matplotlib.pyplot import figure, show

def plot_flight (first, last, fp, kpt, kpv, ph, rejected_takeoff=False):
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