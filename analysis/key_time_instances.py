import numpy as np
from collections import namedtuple

KeyTimeInstance = namedtuple('KeyTimeInstance', 'index state')

'''
kpt['FlapDeployed'] = []
kpt['FlapRetracted'] = []
for flap_operated_period in np.ma.flatnotmasked_contiguous(np.ma.masked_equal(fp.flap.data[block],0.0)):
    kpt['FlapDeployed'].append(first+flap_operated_period.start)
    kpt['FlapRetracted'].append(first+flap_operated_period.stop)
'''
          

class FlapStateChanges(Derived):
    dependencies = [FLAP]
    
    def derive(self, ph, params):
        flap = params[FLAP].data        
        # Mark all flap changes, irrespective of the aircraft type :o)
        kti_list = []
        previous = None
        for index, value in enumerate(flap):
            if value == previous:
                continue
            else:
                # Flap moved from previous setting, so record this change:
                kti = KeyTimeInstance(index, 'Flap %d' % value)
                kti_list.append(kti)
        return kti_list