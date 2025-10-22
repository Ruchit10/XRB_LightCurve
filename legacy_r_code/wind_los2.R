## This Module calculates the Column Integral along the Line of Sight of the emission region of Star B in presence of Stellar wind of Star A
## Column Integral Module
dz <- 0.1                       # Differential element along the line of sight
z3 <-  c()                      # To Record the column element length
con <- c()                      # To Record the Wind Density value at measured distance (accelerating wind)
con2 <- c()                     # To Record the Wind Density value at measured distance (non-accelerating wind)
lw  <- c()                      # Scaled Column Integral of all cells (accn wind)
lw2 <- c()                      # Scaled Col int of all cells (non-accn wind)
los <- c()                      # Column Integral of the Wind Density in a single cell (accn wind)
los2 <- c()                     # Col Int of the Wind density in a single cell (non-accn wind)
source("grid4.R")
if ( (length(av$av.db)) >= 1 )  # Excluding total eclipse (Shadow region)
{
  for (f in 1:length(av$av.db)) 
  { 
    p <- 1
    cl = av$av.db[f]
    dl = d
    z1 = d1*sin(gma)*cos(i)
    z2 = d2*sin(gma)*cos(i)
    z = (z1+z2)
    t <- sqrt(((2*d)^2) - (cl^2))
    while ( abs(z) <= t )
    {
      z3[p]  <-  z	 
      con[p] <- (dl^(-5/2))  #Atoms/(solar radius)^5
      con2[p] <- (dl^(-2))
      z  = z - 0.1
      dl <- sqrt(cl^2 + z^2)
      p <- p + 1
    }
    los[f] <- dz*sum(con)    #Atoms/(solar radius)^4
    los2[f] <- dz*sum(con2)
  }
  for (g in 1:length(los))
  {
    lw[g] <- (los[g])*(A[g]) # Atoms/(solar radius)^2
    lw2[g] <- los2[g]*(A[g])
  }
}