import logging
import numpy as np

from analysis.node import SectionNode, P, S


#Q: What about using a different letter than "P" for non-parameters

##class _500ftToTouchdown(SectionNode):
    ##def derive(self, alt=P('Altitude STD'), tdwn=P('Touchdown')):
        
        
class _500ftTo0ft(SectionNode):
    name = "500ft To 0ft"
    
    def derive(self, alt=P('Altitude AAL'), desc=S('Descending')):
        """
        Creates slices for each time we're descending between the specified 
        altitudes.
        """
        alt_500_0 = np.ma.masked_outside(alt, 500, 0)
        # mask where the slices are active
        mask = np.ones(len(alt_500_0))
        for d_slice in desc:
            mask[d_slice] = 0
        mask_indices = np.where(mask == 1)
        alt_500_0[mask_indices] = np.ma.masked
        
        alt_slices = np.ma.clump_unmasked(alt_500_0)
        self.create_sections(alt_slices)
        
