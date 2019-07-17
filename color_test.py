# Test the color creation function

import numpy as np
from matplotlib import pyplot as plt

def coeff_color(coeff):
    if np.abs(coeff)>1:
        raise Exception('The abs. value of a coefficient was over unity. Value given:{coeff}')
    if coeff>=0:
        return (1, 1-coeff, 1-coeff)
    else:
        return (1+coeff, 1+coeff, 1)
        
        
        
        
if __name__=='__main__':
    
    x = np.linspace(-1, 1, 50)
    
    cols = [coeff_color(xs) for xs in x]
    
    for i, xs in enumerate(x):
        plt.scatter(xs, 0, color=coeff_color(xs))
        
    plt.show()
