## This Module is used for obtaining functional Relationship of the Wind density along the Line of sight
## Wind Density Module
w = 1
d4 <- d
colm <- c()
z5 <- c()
z4 = d*sin(gma)*cos(i)
t4 = sqrt(((2*d)^2) - (l^2))
while (abs(z4) <= t4)
{
	z5[w] <- z4
	colm[w] <- (d4^(-2))
	z4 = z4 - 0.1
	d4 = sqrt(l^2 + z4^2)
	w = w + 1
}
rh <- data.frame(z5,colm)
q = q + 1