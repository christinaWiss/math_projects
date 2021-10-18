import constants
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


# In this class we want to implement the basic mathematical objects in discrete time
def random_walk(prob=0.5, gather=constants.gather_unit):
    path = [0]
    cumsum = 0
    for i in gather:
        random = int(bernoulli.rvs(p=prob, size=1))
        if random:
            cumsum += random
            path += [cumsum]
        else:
            cumsum += -1
            path += [cumsum]
    path.pop()
    return path


def stochastic_exponential(path, width=constants.width):
    exponential_path = []
    cumprod = 1
    last_value = 0
    for value in path:
        cumprod *= 1+value/width-last_value
        exponential_path += [cumprod]
        last_value = value/width
    return exponential_path


def short_rates_delta(drift=[0]*(constants.gather_force-1), sigma=[0]*(constants.gather_force-1), delta=constants.width):
    r = []
    for drift_now, sigma_now in zip(drift, sigma):
        random = int(bernoulli.rvs(p=.5, size=1))
        if random:
            delta_random = 1
            r_new = drift_now * delta + sigma_now * delta_random * delta
        else:
            delta_random = -1
            r_new = drift_now * delta + sigma_now * delta_random * delta
        r += [r_new]

    return r


def short_rates(short_rate_delta_path, initial_r=.1):
    path = [initial_r]
    last = initial_r
    for current in short_rate_delta_path:
        path += [last + current]
        last += current
    return path


def bank_account(initial_money, short_rate, rate_type="annual", width=constants.width):
    path = [initial_money]
    current_money = initial_money

    time = 0
    for r_new in short_rate:
        if rate_type == "annual":
            current_money *= (1 + r_new)
            path += [current_money]
        elif rate_type == "continuous":

            current_money *= (1 + (r_new*time/len(short_rate))/width)**width
            path += [current_money]
            time += 1
    path.pop()
#    if rate_type == "annual":
#        path.pop()
    return path


""" In the first example we want to provide the case, where r is deterministic and a constant"""

######### (1.1) time-discrete case with a duration of 10 periods and r=0.05 for all time-points 1<=i<=10 and

initial_r = .05
initial_money = 0.613918
times = 10


delta_r = short_rates_delta([0]*times, [0]*times)
print("The path of the shortrate deltas:", delta_r)


r_path = short_rates(delta_r, initial_r)
print("The path of the shortrates:", r_path)


money_path = bank_account(initial_money, r_path, "annual", width=1)
print("The resulting risk-free money account:", money_path)

gather = list(range(times+1))
#plt.scatter(gather, money_path)
#plt.show()
"""

"""
########## (1.2) time-continuous case with a duration of one year and r=.05

initial_r = .05
initial_money = 0.64465


delta_r = short_rates_delta()
r_path = short_rates(delta_r, initial_r)
money_path = bank_account(initial_money, r_path, "continuous")
print("At time zero we invested", initial_money, ". After one year do possess", money_path[-1], "on our money account")



#plt.plot(constants.gather_unit, money_path)
#plt.show()
