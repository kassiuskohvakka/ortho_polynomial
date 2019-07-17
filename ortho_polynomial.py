# A quick project to study polynomial expansion of arbitrary functions. Not very heavy dependence-wise
# but an anaconda stack is recommended.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections as matcoll
from matplotlib import animation
import scipy.integrate


######################################
### PARSE COMMAND LINE ARGUMENTS
######################################

# Used to parse command line arguments
import argparse

# Create parser
parser = argparse.ArgumentParser()

###############

# -showmaterr - Plot the B*B.T - I matrix, which SHOULD be zeros for orthonormal basis matrix B.
parser.add_argument('-showmaterr', type=int, default=0, help='Display orthogonality error matrix. Set -showmaterr=1 to view.')

# -v - Verbose. Print all kinds of crap. ALL. KINDS.
parser.add_argument('-v', type=int, default=0, help='Set -v=1 for verbose output.')

# -save - Saving the animation in a .gif file
parser.add_argument('-save', type=int, default=0, help='Set -save=1 to save the expansion animation in .gif format.')

# -fps - Framerate for the potentially saved animation
parser.add_argument('-fps', type=int, default=15, help='Set the framerate for the .gif animation. Defaults to 15.')

###############

# Parse the arguments
args = parser.parse_args()

SHOW_MAT_ERR = args.showmaterr
VERBOSE = args.v
SAVE = args.save
FPS = args.fps


#######################################
### DEFINE SOME FUNCTIONS
#######################################

# Used to normalize a given function arr. The parameter h gives the linear spacing between the sample points.
def normalize(arr, h):
    return arr/np.sqrt(scipy.integrate.simps(arr*arr, dx=h))#h*np.dot(arr, arr))

# A function for a nice coloring of the basis vectors based on the absolute values of the corresponding coefficients in the expansion. 
def coeff_color(coeff):
    if np.abs(coeff)>1:
        raise Exception(f'The abs. value of a coefficient was over unity. Value given:{coeff}')
    if coeff>=0:
        return (1, 1-coeff, 1-coeff)
    else:
        return (1+coeff, 1+coeff, 1)


########################################
### START THE SCRIPT
########################################

# Parameters for the discretization space
start = -1
end = 1
N = 1000
h = (end-start)/(N-1) # Grid spacing

# Define the x-axis for the desired interval
x = np.linspace(start, end, N)

# The dimension of the orthonormal basis
Nbasis = 80

# Begin with the two first basis functions

qs = np.ones(N) 
basis = normalize(qs, h)

qs = x - scipy.integrate.simps(x*basis, dx=h)
basis = np.vstack((basis, normalize(qs, h)))

# ...and do the rest
for i in range(2,Nbasis):
    
    # Initialize the next trial polynomial
    qs = x*basis[i-1,:]
    
    # Naive way
    #for j in range(i):
    #    qs = qs - scipy.integrate.simps(qs*basis[j], dx=h)*basis[j]#h*np.dot(qs, basis[j])*basis[j]
        
        
    # Recursion way. The naive way introduces WAY too much cumulative error to be usable.
    # Nevermind - it does not. However, using the NAIVE NATURAL POLYNOMIAL AS THE TRIAL DOES. Not sure why. Be that as it may, the recursion is 
    # still faster and accumulates less error, so that's what we should probably go with.
    
    qs = qs - scipy.integrate.simps(qs*basis[i-1], dx=h)*basis[i-1] - scipy.integrate.simps(qs*basis[i-2], dx=h)*basis[i-2]
    basis = np.vstack((basis, normalize(qs, h)))
    
if SHOW_MAT_ERR:
    plt.matshow(np.round(h*np.dot(basis, np.transpose(basis)), 2) - np.eye(Nbasis))
    plt.show()


####################################
### Create the signal to be expanded
####################################

# Random seed for reproducibility
np.random.seed(889)

# Parameter for signal complexity
Nsignal = 30

# Create random parameters for the sine wave signals to be superposed
A = np.random.uniform(5, 10, Nsignal)
f = np.random.uniform(0, 10, Nsignal)
p = np.random.uniform(0, 2*np.pi, Nsignal)

# Construct the final signal
signal = np.sum( A * np.sin( 2*np.pi*f * np.transpose( np.array([x for i in range(len(f))]) )+p), 1)


#####################################
### Do the coefficient expansion
#####################################

coeffs = scipy.integrate.simps(basis*signal, dx=h)

if VERBOSE:
    print(f'\n***\n\n Expansion coefficients : \n{coeffs}\n\n***\n')

# Do some fancy visualization. Color the basis function more red if it has a large positive coefficient,
# more blue if it has a large negative coefficient. The coloring is relative to the maximum absolute
# value of a coefficient

max_abs_coeff = np.max(np.abs(coeffs))

if VERBOSE:
    print(f'max. abs. coeff : {max_abs_coeff}\n\n***\n')

scaled_coeffs = coeffs/max_abs_coeff

# Start plotting the expansion visualizations
f3, [ax3, ax4] = plt.subplots(nrows=1, ncols=2, figsize=(10,5), dpi=80)

# Pick only the few most significant components to plot
nof_comps = 5 # nof_comps=few
comp_indices = np.argsort(np.abs(coeffs))

for i in comp_indices[-nof_comps:]:
    ax3.plot(basis[i,:], color=coeff_color(scaled_coeffs[i]))

coeff_x = [i for i in range(len(coeffs))]

pos_lines = []
neg_lines = []
for i in range(len(coeff_x)):
    if coeffs[i] >= 0:
        pair=[(coeff_x[i],0), (coeff_x[i], coeffs[i])]
        pos_lines.append(pair)
        ax4.scatter(coeff_x[i], coeffs[i], s=15, color='r')
    else:
        pair=[(coeff_x[i],0), (coeff_x[i], coeffs[i])]
        neg_lines.append(pair)
        ax4.scatter(coeff_x[i], coeffs[i], s=15, color='b')

pos_linecoll = matcoll.LineCollection(pos_lines, color='r')
neg_linecoll = matcoll.LineCollection(neg_lines, color='b')
ax4.add_collection(pos_linecoll)
ax4.add_collection(neg_linecoll)

ax4.plot([0, coeff_x[-1]], [0, 0], color='k', linewidth=2)

# plt.show()

f3.savefig('testi.pdf')

#######################################
### Plot the signal and the expansion
#######################################

# Compute the truncated expansions

expansions = np.zeros(np.shape(basis))

for i, coeff in enumerate(coeffs):
    if i==0:
        expansions[0,:] = coeffs[0]*basis[0,:]
    else:
        expansions[i,:] = expansions[i-1] + coeffs[i]*basis[i,:]    


# Start plottting the signal and the animated expansion.

f5, [ax5, ax6] = plt.subplots(ncols=2, figsize=(8,4), dpi=80)
ax5.plot(x, signal)


ax5.set_title("Signal and expansion")
ax4.set_xlabel("x")


ax6.set_title("Error in the expansion")
ax6.set_xlabel("Dimension of expansion basis")


# Side plot for the errors with a moving cursor
errors = []

for i in range(np.size(expansions, 0)):
    errors.append( scipy.integrate.simps( (signal - expansions[i,:])**2, dx=h) )
ax6.semilogy(coeff_x, errors)

line1, = ax5.plot([], [], lw=2, color='r')
line2 = ax6.axvline(x=1, color='r')#plot([], [], lw=2, color='r')

lines = [line1, line2]

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i, x, expansions):
    y = expansions[i,:]
    line1.set_data(x, y)
    
    line2.set_data([i,i], [0, 10000])
    return lines

anim = animation.FuncAnimation(f5, lambda i: animate(i, x, expansions), init_func=init,
                               frames=np.size(expansions,0), interval=200, blit=True)


if SAVE:
    writer = animation.PillowWriter(fps=FPS, bitrate=1800)
    print("Saving animation...\n")
    anim.save('image.gif', writer=writer)


plt.show(block=False)


# Exit in a controlled manner
input("Press enter to exit.")




























