import constants
import main
import matplotlib.pyplot as plt
"""The Vasicˇek Model model: Here the differential equation is given by dr(t)=(b+beta*r(t))dt+sigma dW^*(t).
    """




def short_rate_vasicek_model(b, beta, sigma, initial_r):
    Brownian_Motion = [i / (len(constants.gather_unit)) ** 0.5 for i in main.random_walk(prob=0.5, gather=constants.gather_unit)]

    short_rate = [initial_r]
    last = initial_r
    last_b = 0
    for i in Brownian_Motion:
        short_rate += [last + (b + beta * last)/constants.width + sigma * (i-last_b)]
        last = short_rate[-1]
        last_b = i

    short_rate.pop()
    return short_rate

# First plot: Sample of 3 OUP with same parameters
"""
beta = -0.86
b = 0.09 * abs(beta)
sigma = 0.0148
initial_r = 0.15

r_path3 = short_rate_vasicek_model(b, beta, sigma, .15)
r_path2 = short_rate_vasicek_model(b, beta, sigma, .15)
r_path = short_rate_vasicek_model(b, beta, .01, .15)

plt.plot(constants.gather_unit, r_path3)
plt.plot(constants.gather_unit, r_path2)
plt.plot(constants.gather_unit, r_path)
plt.show()
"""
# Second plot: Sample of 3 OUP with different initial r but the remaining parameters are the same
"""
beta = -1.86
b = 0.09 * abs(beta)
sigma = 0.0248

r_path4 = short_rate_vasicek_model(b, beta, sigma, .2)
r_path3 = short_rate_vasicek_model(b, beta, sigma, .0)
r_path2 = short_rate_vasicek_model(b, beta, sigma, .05)
r_path = short_rate_vasicek_model(b, beta, sigma, .01)

plt.plot(constants.gather_unit, r_path4)
plt.plot(constants.gather_unit, r_path3)
plt.plot(constants.gather_unit, r_path2)
plt.plot(constants.gather_unit, r_path)
plt.show()
"""
# Third plot: OUP with same mean reversion level but different Mean-Rev-Speed
# additionaly the OUPs initial rate is equal to the mean reversion level

beta = -1.86
b = 0.09 * abs(beta)
sigma = 0.0248
r_initial = b/abs(beta)

r_path4 = short_rate_vasicek_model(b, beta, 0, r_initial)
r_path3 = short_rate_vasicek_model(b, beta, .01, r_initial)
r_path2 = short_rate_vasicek_model(b, beta, .05, r_initial)
r_path = short_rate_vasicek_model(b, beta, .08, r_initial)

plt.plot(constants.gather_unit, r_path4)
plt.plot(constants.gather_unit, r_path3)
plt.plot(constants.gather_unit, r_path2)
plt.plot(constants.gather_unit, r_path)
plt.show()