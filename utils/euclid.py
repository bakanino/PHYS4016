def euclid(n,x,y):
    ans = 0.0
    for i in range(n):
       ans += (x[i]-y[i])**2

    return ans**(1/2)

a1 = euclid(3, [7.0, 3.2, 4.7], [4.9, 3.0, 1.4])
a2 = euclid(3, [7.0, 3.2, 4.7], [5.1, 3.5, 1.4])
print(a1,a2)
