## Simulation of the column densities obtained as the compact object eclipses companion in a Binary System orbiting a common Center of Mass
## For a better idea of the System Geometry Please refer to the 3 Dimensional diagram alongside
## The Compact ofject and the Accretion disk is referred to as Star B
## The Companion Star is referred to as Star A
## All Distance units are in Solar Radii
## All Angle units are converted into radians for the ease of using Trigonometric Functions

## Main Module
rh <- c() 																  # Wind Density along the line of Sight
time <- c()
r = 0.001 #as.numeric(readline(prompt="Enter emitter radius(solar radii)= "))                 # INPUT 1 = Radius of smaller star B (The compact object)
R = 2 #as.numeric(readline(prompt="Enter companion radius(solar radii)= "))               # INPUT 2 = Radius of the larger star A (The Companion)
d1 = 11 #as.numeric(readline(prompt="Enter emitter separation from COM(solar radii)= "))	  # INPUT 3 = Radial distance of Star B from Center of Mass  
d2 = 8 #as.numeric(readline(prompt="Enter companion separation from COM(solar radii)= ")) # INPUT 4 = Radial distance of Star A from Center of Mass
d = d1 + d2                												  # Separation betweeen stars A and B along the line of sight (LOS)
gma0 = -90 #as.numeric(readline(prompt = "Enter starting phase angle(degrees)= "))  # INPUT 5 = initial phase angle in degrees
gma <- gma0*(pi/180) 													  # initial value for phase angle in radians
fl <- pi*(r^2)
flx <- fl/(pi*(r)^2)    												  # column integral (accn wind)
flx2 <- flx 															  # column integral (non-accn wind)
j = 1                   
a = 0
b = 0
q = 1
i0 = 26 #as.numeric(readline(prompt="Enter orbital inclination(degrees)= "))    # INPUT 6 = Inclination angle in Degrees
dth = as.numeric(readline(prompt="Enter orbital increment(degrees)= "))     # INPUT 7 = Orbital increment
i = i0*(pi/180)															  # Inclination angle in radians
h1 = d1*sin(gma)*sin(i)  # Projected Vertical distance of center of star B from the point of Center of Mass viewed along Line of Sight
h2 = d2*sin(gma)*sin(i)  # Projected Vertical distance of center of star A from the point of Center of Mass viewed along Line of Sight
L1 = d1*cos(gma)     	 # Initial value for Horizontal distance of Star B from the point of Center of Mass viewed along Line of Sight
L2 = d2*cos(gma)		 # Initial value for Horizontal distance of Star A form the point of Center of Mass viewed along Line of Sight
h = (h1 + h2)			 # Total Vertical distance between Star A and Star B
L = (L1 + L2) 			 # Total Horizontal distance between Star A and Star B
L3 <- c()
l3 <- c()
h3 <- c()
ph <- c()
phase <- c()
deg <- c()
icd <- c()
A2 <- c()
l1 = sqrt((h1^2)+(L1^2)) # initial value for separation along the viewing plane (perpendicular plane from LOS) 
l2 = sqrt((h2^2)+(L2^2))
l = (l1 + l2)
for (j in 1:(360/dth))
{
	h1 = d1*sin(gma)*sin(i)
	h2 = d2*sin(gma)*sin(i)
	L1 = d1*cos(gma) 
	L2 = d2*cos(gma)
	l1 = sqrt((h1^2)+(L1^2))
	l2 = sqrt((h2^2)+(L2^2))
	h = (h1 + h2)
	L = (L1 + L2)
	l = (l1 + l2)
	a <- ((l^2)+(r^2)-(R^2))   
	b <- (2*abs(l)*r)
	n <- (l/(R+r))
 	if ( gma <= pi )
 	{
  		if  ( n >= 1 )                 # The flat portion of the curve OR the non-eclipsed region of the curve
  		{ 
    		source("wind_los2.R")
    		source("density_fnc.R")
    		flx[j] <- (sum(lw)/sum(A)) #Atoms/(solar radius)^4
    		flx2[j] <- (sum(lw2)/sum(A))
    		icd[j] <- (sum(lw))        #Atoms/(solar radius)^2
    		A2[j] <- sum(A)
  		}
  		else
  		{
   			n2 <- a/b
   			n3 <- (l/(R-r))
   			if ( abs(n3) <= 1 )          # To avoid complex value of the angles 
   			{
  				flx[j] <- 0
  				flx2[j] <- 0
   			}
   			else
   			{
    			source("wind_los2.R")
    			source("density_fnc.R")
    			flx[j] <- (sum(lw)/sum(A)) #Atoms/(solar radius)^4
    			flx2[j] <- (sum(lw2)/sum(A))
    			icd[j] <- (sum(lw))        #Atoms/(solar radius)^2
    			A2[j] <- sum(A)
   			} 
  		}
 	}
 	else
 	{
   		source("wind_los2.R")
   		source("density_fnc.R")
   		flx[j] <- (sum(lw)/sum(A)) #Atoms/(solar radius)^4
   		flx2[j] <- (sum(lw2)/sum(A))
   		icd[j] <- (sum(lw))        #Atoms/(solar radius)^2
   		A2[j] <- sum(A)
 	}
  	ph[j] <- gma
  	deg[j] <- (gma*180/pi)
  	time[j] <- deg[j]*348.42
  	phase[j] <- ((gma-(gma0*(pi/180)))/(2*pi))
  	gma = gma + (dth*(pi/180))
  	l3[j] <- l
  	L3[j] <- L  
  	h3[j] <- h
  	print(deg[j])
  	assign(paste0("rho",j),rh)
}

period = time[length(time)] - time[1]
var <- data.frame(deg,ph,phase,A2,flx)
lam = (0.589537/mean(flx))
lam2 = (0.589537/mean(flx2))
fl <- flx*lam
fl2 <- flx2*lam2
#nfl <- (110.2067 / (11.3171 + fl))
nfl_hard_av <- 9.524*exp(-fl*0.057) # Scaled Flux Hard Band 2 -10 KeV Accelerated Wind
nfl_hard_cv <- 9.524*exp(-fl2*0.057) # Scaled Flux Hard Band Constant Velo Wind
nfl_soft_av <- 9.3923*exp(-fl*2.5062) # Scaled Flux Soft Band 0.3-2 KeV Accelerated Wind
nfl_soft_cv <- 9.3923*exp(-fl2*2.5062) # Scaled Flux Soft Band Constant Velo Wind
pho_count_hard_av <- 0.0001464*exp(-fl*0.1066818) # Scaled Photon Count Hard Band Accelerated Wind
pho_count_soft_av <- 0.0005275*exp(-fl*2.7556631) # Scaled Photon Count Soft Band Accelerated Wind
#nfl3 <- (8.647e+03) / ((37.940 + fl)^2)
#nfl3 <- (97.6480 / (16.1595 + fl))
#nfl3 <- (1.723006 / (0.169244 + fl))
#write.csv(var,file="output.csv")