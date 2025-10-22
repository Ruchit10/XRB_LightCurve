## Creating Grid for wind integral of Eclipsing binaries
## Emitter Grid Module
d2h <- 6                                          # angle size for the polar grid cell
g1_r <- seq((r/10),r,length.out=10)               # polar variable r for A
g1_th <- seq(0,2*pi,length.out=((360/d2h)+1))     # polar variable theta for A

g1 <- expand.grid(x=g1_r,y=g1_th)       # polar Grid for A


thi <- g1$y[1]
 
g1_s_r <- c()
g1_s_th <- c()
A     <- c()
av.x  <- c()
av.th <- c()
av.db <- c()
k <- 1
v <- 1
#m <- 1
# Imposing conditions on which points to accept and which ones to reject
 
if ( gma < pi )
{
  for (m in 1:length(g1$x))
  {
    r1 <- g1$x[m]
    r2 <- g1$x[m+1]
    #th1 <- g1$y[m]
    #th2 <- g1$y[m+1]
    nn <- sqrt((r1^2) + (l^2) - 2*r1*l*cos(thi))
    if ( nn >= R )
    {
      g1_s_r[k] <- r1
      g1_s_th[k] <- thi
      k <- k + 1
    }
    else
    {
    }
    thi <- g1$y[m]
  }
}
if (gma > pi) 
{
  g1_s_r <- g1$x
  g1_s_th <- g1$y
}
g1_s <- data.frame(g1_s_r,g1_s_th)
 
# Making small area segments of the eclipsed grid
 
for (s in 1:(length(g1_s$g1_s_r) - 1))
{
  x1  <- g1_s$g1_s_r[s]
  x2  <- g1_s$g1_s_r[s+1]
  th1 <- g1_s$g1_s_th[s]
  th2 <- g1_s$g1_s_th[s] + (dth*(pi/180))
  dx <- (x2 - x1)
  if (dx > 0)
  {
    av.x[v]  <-  (x1 + x2)/2
    av.th[v] <- (th1 +th2)/2
    av.db[v] <- sqrt((av.x[v]^2) + (l^2) - 2*av.x[v]*l*cos(av.th[v]))
    A[v] <- ((0.5*d2h*(pi/180))*((x2^2)-(x1^2)))   # (solar radius)^2
    v <- v + 1
  }
  else
  {
  }
}
av <- data.frame(av.x,av.th,av.db)
#radial.plot(g1_s$g1_s_r,g1_s$g1_s_th,rp.type="s",radial.lim=c(0,1))
