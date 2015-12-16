
# coding: utf-8

#
# Viscoplastic thermal convection in a 2-D box
# =======
#
# Benchmarks from Tosi et al. 2015
# --------
#
#

# Load python functions needed for underworld. Some additional python functions from os, math and numpy used later on.

# In[1]:

import underworld as uw
import math
from underworld import function as fn
import glucifer
#import matplotlib.pyplot as pyplot
import time
import numpy as np
import os
import sys


# In[2]:

############
#Need to manually set these two
############
Model = "T"
ModNum = 5

if len(sys.argv) == 1:
    ModIt = "Base"
elif sys.argv[1] == '-f':
    ModIt = "Base"
else:
    ModIt = str(sys.argv[1])


# In[ ]:




# Set physical constants and parameters, including the Rayleigh number (*RA*).

# In[3]:

#Do you want to write hdf5 files - Temp, RMS, viscosity, stress?
writeFiles = True
loadTemp = False
refineMesh = False


# In[4]:

ETA_T = 1e5
newvisc= math.exp(math.log(ETA_T)*0.53)


# In[5]:

newvisc


# In[6]:

###########
#Constants
###########
RA  = 1e2*newvisc       # Rayleigh number
TS  = 0          # surface temperature
TB  = 1          # bottom boundary temperature (melting point)
ETA_T = 1e5
ETA_Y = 10
ETA0 = 1e-3*newvisc
RES = 40
YSTRESS = 1.*newvisc
D = 2890.

MINX = -1.
ALPHA = 11.

stickyAir = False


# In[7]:



if MINX == 0.:
    squareModel = True
else:
    squareModel = False


# In[8]:

##########
#model variables, these can be defined with STDIN,
##########
#The == '-f': check is just to check if we're running a notebook - careful, haven't tested

#Watch the type assignemnt on sys.argv[1]

DEFAULT = 96
ModIt   = str(DEFAULT)


if len(sys.argv) == 1:
    RES = DEFAULT
elif sys.argv[1] == '-f':
    RES = DEFAULT
else:
    RES = int(sys.argv[1])



# In[9]:

outputPath = str(Model) + "/" + str(ModNum) + "/"
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '_' + str(ModIt) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)


# In[10]:

dim = 2          # number of spatial dimensions


if MINX == 0.:
    Xres = RES
else:
    Xres = 2*RES

if stickyAir:
    Yres = RES + 8
    MAXY = float(Yres)/RES

else:
    Yres = RES
    MAXY = 1.


yelsize = MAXY/Yres
MAXY, yelsize, yelsize*D


# ## Create mesh objects. These store the indices and spatial coordiates of the grid points on the mesh.

# In[11]:

elementMesh = uw.mesh.FeMesh_Cartesian( elementType=("Q1/dQ0"),
                                         elementRes=(Xres, Yres),
                                           minCoord=(MINX,0.),
                                           maxCoord=(1.,MAXY), periodic=[True,False] )
linearMesh   = elementMesh
constantMesh = elementMesh.subMesh


# Create Finite Element (FE) variables for the velocity, pressure and temperature fields. The last two of these are scalar fields needing only one value at each mesh point, while the velocity field contains a vector of *dim* dimensions at each mesh point.

# In[12]:

velocityField    = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=dim )
pressureField    = uw.fevariable.FeVariable( feMesh=constantMesh, nodeDofCount=1 )
temperatureField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )


# Create some dummy fevariables for doing top and bottom boundary calculations.

# ##Refine mesh

# In[13]:

#THis one for the rectangular mesh

if refineMesh:
    alpha = ALPHA
    newys = []
    newxs = []
    for index, coord in enumerate(linearMesh.data):
        y0 = coord[1]
        x0 = abs(coord[0])
        if y0 >= 1.0:
            newy = y0
        else:
            newy = (math.log(alpha*y0 + math.e) - 1)*(1/(math.log(alpha + math.e) - 1))
        if coord[0] > 0:
            newx = (math.e**(x0*(math.log((alpha/2.) + math.e) - 1) + 1 ) - math.e)/(alpha/2.)
        else:
            newx = -1.*(math.e**(x0*(math.log((alpha/2.) + math.e) - 1) + 1 ) - math.e)/(alpha/2.)
        newys.append(newy)
        newxs.append(newx)
        #print y0,newy

    with linearMesh.deform_mesh():
            linearMesh.data[:,1] = newys
            linearMesh.data[:,0] = newxs


# #ICs and BCs

# In[14]:

# Initialise data.. Note that we are also setting boundary conditions here
velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:] = 0.

# Setup temperature initial condition via numpy arrays
A = 0.01
#Note that width = height = 1
tempNump = temperatureField.data
for index, coord in enumerate(linearMesh.data):
    pertCoeff = (1- coord[1]) + A*math.cos( math.pi * abs(1. - coord[0]) ) * math.sin( math.pi * coord[1] )
    tempNump[index] = pertCoeff;
    if coord[1] > 1:
        tempNump[index] = 0.


# In[15]:

#For notebook runs
ModIt = "96"


#

# In[16]:

#icfnam= "R(11)_" + "2_" + str(RES) + "_init_temp.hdf5"
#icpath = "temp_ics/" + icfnam
#icfnam= str(Model) + str(ModIt) + "_init_temp.hdf5"
#temperatureField.save(icpath)
#icfnam
#icpath


# In[17]:

# Get the actual sets
#
#  HJJJJJJH
#  I      I
#  I      I
#  I      I
#  HJJJJJJH
#
#  Note that H = I & J

# Note that we use operator overloading to combine sets
IWalls = linearMesh.specialSets["MinI_VertexSet"] + linearMesh.specialSets["MaxI_VertexSet"]
JWalls = linearMesh.specialSets["MinJ_VertexSet"] + linearMesh.specialSets["MaxJ_VertexSet"]
TWalls = linearMesh.specialSets["MaxJ_VertexSet"]
BWalls = linearMesh.specialSets["MinJ_VertexSet"]


# In[ ]:




# In[18]:

# Now setup the dirichlet boundary condition
# Note that through this object, we are flagging to the system
# that these nodes are to be considered as boundary conditions.
# Also note that we provide a tuple of sets.. One for the Vx, one for Vy.
freeslipBC = uw.conditions.DirichletCondition(     variable=velocityField,
                                              nodeIndexSets=(None, JWalls) )

# also set dirichlet for temp field
tempBC = uw.conditions.DirichletCondition(     variable=temperatureField,
                                              nodeIndexSets=(JWalls,) )


# ##Add Random 125 K temp perturbation
#

# In[19]:

tempNump = temperatureField.data
for index, coord in enumerate(linearMesh.data):
    pertCoeff = math.sin(math.pi*coord[1])*(0.05*np.random.rand(1)[0])
    ict = tempNump[index]
    tempNump[index] = ict + pertCoeff


# ##Reset bottom Dirichlet conds.

# In[20]:

# Set temp boundaries
# on the boundaries
for index in linearMesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = TB
for index in linearMesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = TS


# ##Material properties

# In[21]:

#Make variables required for plasticity

secinvCopy = fn.tensor.second_invariant(
                    fn.tensor.symmetric(
                        velocityField.gradientFn ))

coordinate = fn.input()

depth = 1. - coordinate[1]

depthField = uw.fevariable.FeVariable( feMesh=linearMesh,   nodeDofCount=1 )
depthField.data[:] = depth.evaluate(linearMesh)
depthField.data[np.where(depthField.data[:] < 0.)[0]] = 0.


# In[22]:

#Compositional Rayligh number of rock-water

g = 9.81
rho = 3300
a = 1.25*10**-5
kappa = 10**-6
dT = 2500
eta0 = rho*g*a*dT*((D*1e3)**3)/(RA*kappa)
#Composisitional Rayleigh number
Rc = (3300*g*(D*1000)**3)/(eta0*kappa)


# In[23]:


viscosityl2 = newvisc*fn.math.exp((math.log(ETA_T)*-1*temperatureField) + (depthField*math.log(ETA_Y)))
viscosityFn1 = viscosityl2 #This one always gets passed to the first velcotity solve
#Von Mises effective viscosity
viscosityp = ETA0 + YSTRESS/(secinvCopy/math.sqrt(0.5)) #extra factor to account for underworld second invariant form
viscosityFn2 = 2./(1./viscosityl2 + 1./viscosityp)


# In[24]:

CompRAfact = Rc/RA
airviscosity = 0.001
airdensity = RA*CompRAfact

##This block sets up rheolgoy for models with crust rheology;
viscreduct = 0.1
#Von Mises effective viscosity
crustviscosityp = viscreduct*ETA0 + ((viscreduct*YSTRESS)/(secinvCopy/math.sqrt(0.5))) #extra factor to account for underworld second invariant form
crustviscosityFn2 = 2./(1./viscosityl2 + 1./crustviscosityp)


# Set up simulation parameters and functions
# ====
#
# Here the functions for density, viscosity etc. are set. These functions and/or values are preserved for the entire simulation time.

# In[25]:


gravity = ( 0.0, 1.0 )

#buoyancyFn = gravity*densityMapFn


densityFn = RA * temperatureField
buoyancyFn = gravity*densityFn


# Build the Stokes system, solvers, advection-diffusion
# ------
#
# Setup linear Stokes system to get the initial velocity.

# In[26]:

#We first set up a l
stokesPIC = uw.systems.Stokes(velocityField=velocityField,
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
#                              viscosityFn=viscosityFn1,
                              viscosityFn=fn.exception.SafeMaths(viscosityFn1),
                              bodyForceFn=buoyancyFn)


# We do one solve with linear viscosity to get the initial strain rate invariant. This solve step also calculates a 'guess' of the the velocity field based on the linear system, which is used later in the non-linear solver.

# In[27]:

stokesPIC.solve()


# In[28]:

# Setup the Stokes system again, now with linear or nonlinear visocity viscosity.
stokesPIC2 = uw.systems.Stokes(velocityField=velocityField,
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              viscosityFn=fn.exception.SafeMaths(viscosityFn1),
                              bodyForceFn=buoyancyFn)


# In[29]:

solver = uw.systems.Solver(stokesPIC2) # altered from PIC2

#solver.options.main.Q22_pc_type='uwscale'  # also try 'gtkg', 'gkgdiag' and 'uwscale'
#solver.options.main.penalty = 1.0
#solver.options.A11.ksp_rtol=1e-6
#solver.options.scr.ksp_rtol=1e-5
#solver.options.scr.use_previous_guess = True
#solver.options.scr.ksp_set_min_it_converge = 1
#solver.options.scr.ksp_set_max_it = 100
#solver.options.mg.levels = 4
#solver.options.mg.mg_levels_ksp_type = 'chebyshev'
#solver.options.mg_accel.mg_accelerating_smoothing = True
#solver.options.mg_accel.mg_accelerating_smoothing_view = False
#solver.options.mg_accel.mg_smooths_to_start = 1


# Solve for initial pressure and velocity using a quick non-linear Picard iteration
#

# In[30]:

solver.solve(nonLinearIterate=True)


# Create an advective-diffusive system
# =====
#
# Setup the system in underworld by flagging the temperature and velocity field variables.

# In[31]:

#Create advdiff system
advDiff = uw.systems.AdvectionDiffusion( temperatureField, velocityField, diffusivity=1., conditions=[tempBC,] )

#advector = uw.systems.SwarmAdvector( swarm=gSwarm, velocityField=velocityField, order=1)


# Main simulation loop
# =======
#
# The main time stepping loop begins here. Before this the time and timestep are initialised to zero and the output statistics arrays are set up. Also the frequency of outputting basic statistics to the screen is set in steps_output.
#

# In[32]:

#pics = uw.swarm.PICIntegrationSwarm(gSwarm)


# In[33]:

realtime = 0.
step = 0
timevals = [0.]
steps_end = 5
steps_display_info = 20
swarm_update = 10
swarm_repop = 100
files_output = 400
gldbs_output = 1e6
checkpoint_every = 10000
metric_output = np.floor(10.*RES/64)


# In[34]:

# initialise timer for computation
start = time.clock()
# setup summary output file (name above)
f_o = open(outputPath+outputFile, 'w')
# Perform steps
#while realtime < 0.4:
while step < 5:
    #Enter non-linear loop
    print step
    solver.solve(nonLinearIterate=True)
    dt = advDiff.get_max_dt()
    if step == 0:
        dt = 0.
    advDiff.integrate(dt)
    # Advect swarm using this timestep size
    #advector.integrate(dt)
    # Increment
    realtime += dt
    step += 1
    timevals.append(realtime)
    for index, coord in enumerate(linearMesh.data):
        if coord[1] >= 1.:
            temperatureField.data[index] = 0.


f_o.close()
#checkpoint(step, checkpointPath)


# In[35]:

machine_time = (time.clock()-start)
print("total time is: " + str(machine_time))


# In[ ]:
