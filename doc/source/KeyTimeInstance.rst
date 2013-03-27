.. _KeyTimeInstance:

=======================
Key Time Instance (KTI)
=======================

-----------
Description
-----------

A Key Time Instance is a point of interest on a flight. A KTI allows one time
identification of an instance in a (flight eg. Touchdown) ensureing a
consistent index is used. The key attribute of a KTI is the **index** which
indicates how far into the flight in seconds the moment of interest occurred.



--------
Benefits
--------



Can be used in Key Point Value (KPV) to look up value at that point
Can be used to create phase eg. x -> touchdown

-------
Example
-------

A simple example of a Key Time Instance is LocalizerEstablishedStart, The index
where ILS Localizer is first Established.
    
    class LocalizerEstablishedStart(KeyTimeInstanceNode):
        '''
        The index where ILS Localizer is first Established.
        '''
        def derive(self, ilss=S('ILS Localizer Established')):
            for ils in ilss:
                self.create_kti(ils.slice.start)

----------------
Helper Functions
----------------

 * create_kti
 * create_ktis_on_state_change
 * create_ktis_at_edges