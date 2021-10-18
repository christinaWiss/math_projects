import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

granularity = 1000
generations = 4

class BesselProcess:
    def __init__(self, y_start):
        self.x_start=0
        self.x_end=1
        self.y_start=y_start
        self.besselpath = self._get_bessel_path(y_start)

    def _get_brownian_motion_unit(self):
        plus_minus_func = np.vectorize(lambda number: 1 if number < 0.5 else -1)
        brownian_motion_unit_interval = (1/math.sqrt(granularity))*np.cumsum(plus_minus_func(np.random.uniform(0, 1, granularity)))
        return brownian_motion_unit_interval

    def _get_bessel_path(self, y_start):
        brown_3 = self._get_brownian_motion_unit()**2 + self._get_brownian_motion_unit()**2 \
                 + self._get_brownian_motion_unit()**2 + y_start
        bessel = np.array([math.sqrt(x) for x in brown_3])
        return bessel

    def plot_bessel_3(self):
        plt.plot(np.arange(0,1,1/1000), self.besselpath)
        plt.suptitle(r'Bessel process starting from x=' + str(self.y_start))
        plt.show()



class Individual:
    def __init__(self, birthplace=0, birthtime=0, generation="1", lambda_ = 1):
        lifetime = np.random.exponential(lambda_)
        lifepath, life_axis = self.brownian_motion_on_lifepath(lifetime, birthtime, birthplace)
        self.lifetime = lifetime
        self.lifepath = lifepath
        self.life_axis = life_axis
        self.death_place = lifepath[-1]
        self.birth_place = birthplace
        self.birthtime = birthtime
        self.generation = generation

    def brownian_motion_on_lifepath(self, lifetime, birthtime, birthplace):
        path = self.get_brownian_motion_unit()
        return lifetime*path + birthplace, np.linspace(0, 1, granularity)*lifetime + birthtime

    def get_brownian_motion_unit(self):
        plus_minus_func = np.vectorize(lambda number: 1 if number < 0.5 else -1)
        brownian_motion_unit_interval = (1/math.sqrt(granularity))*np.cumsum(plus_minus_func(np.random.uniform(0, 1, granularity)))
        return brownian_motion_unit_interval

class BranchingProcess:
    def __init__(self, start=0, end=1, granularity=1000):
        self.start = start
        self.end = end
        self.granularity = granularity
        self.anchestor = Individual()
        self.genealogy = self._get_genealogy()
        #self.all_alive = self.

    def _get_genealogy(self):
        result_dict = {"1": self.anchestor}
        last = self.anchestor.death_place
        last_generation = ["1"]

        max_stage = 1
        while last < generations:
            next_generation = []
            for key in last_generation:
                if len(key) == generations-1:
                    last = generations + 1
                key1 = key + "0"
                key2 = key + "1"

                result_dict[key1] = Individual(result_dict[key].death_place,#.lifepath[-1],
                                      result_dict[key].birthtime+result_dict[key].lifetime, key1)

                result_dict[key2] = Individual(result_dict[key].death_place,#lifepath[-1],
                                      result_dict[key].birthtime+result_dict[key].lifetime, key2)
                next_generation += [key1, key2]
                print(result_dict[key2].birthtime)
                print(key2)
            last_generation = next_generation
        return result_dict

    def get_minimum_from_last_generation(self):
        minimum = 0
        minimum_key = None
        for key in self.genealogy.keys():
            death = self.genealogy[key].birthtime + self.genealogy[key].lifetime
            if len(key) == generations and (minimum > death or minimum == 0):
                minimum = death
                minimum_key = key
        return minimum #, minimum_key

    def get_offspring(self, ancestor):
        birthtime = ancestor.birthtime + ancestor.lifetime
        return ancestor

    def create_plot(self, min_plot=True):
        for life in self.genealogy.keys():
            if not min_plot:
                plt.plot(self.genealogy[life].life_axis, self.genealogy[life].lifepath)
            else:
                which = np.where(self.genealogy[life].life_axis < self.get_minimum_from_last_generation())[0]
                axis = self.genealogy[life].life_axis[which]
                path = self.genealogy[life].lifepath[which]
                plt.plot(axis, path)
        plt.show()

    def creat_max_particle_plot(self, min_plot=True):
        max_val, last_key = -100, ""
        for life in self.genealogy.keys():
            if not min_plot:
                plt.plot(self.genealogy[life].life_axis, self.genealogy[life].lifepath)
            else:
                which = np.where(self.genealogy[life].life_axis < self.get_minimum_from_last_generation())[0]
                axis = self.genealogy[life].life_axis[which]
                path = self.genealogy[life].lifepath[which]
                plt.plot(axis, path, color='grey')
                if len(path) > 0:
                    last_key = life if path[-1] > max_val else last_key
                    max_val = path[-1] if path[-1] > max_val else max_val
        while last_key != "":
            which = np.where(self.genealogy[last_key].life_axis < self.get_minimum_from_last_generation())[0]
            axis = self.genealogy[last_key].life_axis[which]
            path = self.genealogy[last_key].lifepath[which]
            plt.plot(axis, path, color='red')
            last_key = last_key[:-1]
        plt.show()


def plot_kappa():
    t_values = [2, 4, 8, 3200]
    result_y = dict()
    figure, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle(r'$\kappa$ zu den Werten t=2, 4, 8 und 16')
    for count, t in enumerate(t_values):
        x_axis = np.arange(0, t, 0.01)
        result_y[t] = (x_axis+1)**(2/3)*(x_axis<=t/2) + (t-x_axis+1)**(2/3)*(x_axis>t/2)#
        x_1 = x_axis[np.where(x_axis<t/2)]
        y_1 = result_y[t][np.where(x_axis <t/2)]
        x_2 = x_axis[np.where(x_axis >t/2)]
        y_2 = result_y[t][np.where(x_axis >t/2)]
        point_l = [x_1[-1], y_1[-1]]
        if count==0:
            axes[0, 0].plot(x_1, y_1,color='#1f77b4')
            axes[0, 0].plot(x_2, y_2, color='#1f77b4')
            #axes[0, 0].plot(x_2[0], y_2[0], "o", markerfacecolor="white", color="black")
            #axes[0, 0].plot(x_1[-1], y_1[-1], 'ro', markersize=6, color='black')
        if count==1:
            axes[0, 1].plot(x_1, y_1,color='#1f77b4')
            axes[0, 1].plot(x_2, y_2, color='#1f77b4')
            #axes[0, 1].plot(x_2[0], y_2[0], "o", markerfacecolor="white", color="black")
            #axes[0, 1].plot(x_1[-1], y_1[-1], 'ro', markersize=6, color='black')
        if count==2:
            axes[1, 0].plot(x_1, y_1,color='#1f77b4')
            axes[1, 0].plot(x_2, y_2, color='#1f77b4')
            #axes[1, 0].plot(x_2[0], y_2[0], "o", markerfacecolor="white", color="black")
            #axes[1, 0].plot(x_1[-1], y_1[-1], 'ro', markersize=6, color='black')
        if count==3:
            axes[1, 1].plot(x_1, y_1,color='#1f77b4')
            axes[1, 1].plot(x_2, y_2, color='#1f77b4')
            #axes[1, 1].plot(x_2[0], y_2[0], "o", markerfacecolor="white", color="black")
            #axes[1, 1].plot(x_1[-1], y_1[-1], 'ro', markersize=6, color='black')
    plt.show()

def plot_l():
    t_values = [2, 4, 8, 16]
    result_y = dict()
    figure, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle(r'$l$ zu den Werten t=2, 4, 8 und 16')
    for count, t in enumerate(t_values):
        x_axis = np.arange(0, t, 0.01)
        result_y[t] = np.array([math.log(x+1)*(3/(2*math.sqrt(2))) for x in x_axis])*(x_axis <= t / 2) +\
                      np.array([math.log(t-x+1)*(3/(2*math.sqrt(2))) for x in x_axis])*(x_axis > t / 2)
        x_1 = x_axis[np.where(x_axis < t / 2)]
        y_1 = result_y[t][np.where(x_axis < t / 2)]
        x_2 = x_axis[np.where(x_axis > t / 2)]
        y_2 = result_y[t][np.where(x_axis > t / 2)]
        point_l = [x_1[-1], y_1[-1]]
        if count == 0:
            axes[0, 0].plot(x_1, y_1, color='#1f77b4')
            axes[0, 0].plot(x_2, y_2, color='#1f77b4')
        if count == 1:
            axes[0, 1].plot(x_1, y_1, color='#1f77b4')
            axes[0, 1].plot(x_2, y_2, color='#1f77b4')
        if count == 2:
            axes[1, 0].plot(x_1, y_1, color='#1f77b4')
            axes[1, 0].plot(x_2, y_2, color='#1f77b4')
        if count == 3:
            axes[1, 1].plot(x_1, y_1, color='#1f77b4')
            axes[1, 1].plot(x_2, y_2, color='#1f77b4')
    plt.show()

def smooth_l():
    t= 10
    x_axis = np.arange(0, t, 0.01)
    y = np.array([math.log(x + 1) * (3 / (2 * math.sqrt(2))) for x in x_axis]) * (x_axis <= t / 2) + \
    np.array([math.log(t - x + 1) * (3 / (2 * math.sqrt(2))) for x in x_axis]) * (x_axis > t / 2)
    plt.vlines(t/2-1, 0, max(y), colors='red')
    plt.vlines(t/2+1, 0, max(y), colors='red')
    plt.legend([r'$t/2\pm 1$'])
    plt.plot(x_axis, y)
    plt.show()


def lemma_3_plt():
    xz_p = [[5,10], [10,5], [20,20], [40, 40]]
    t_axis = np.arange(50, 100, 100/granularity)

    gamma = 2**(1/2)/(math.pi**(1/2))

    figure, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle(r'Illustration der Schranke für die Dichte des Bessel prozesses')

    for count, xz in enumerate(xz_p):
        lower = np.array([gamma*(xz[1]**2)/(t**(3/2))*math.e**(-xz[1]**2/(2*t)-xz[0]**2/(2*t)) for t in t_axis])
        actual = np.array([(xz[1]/(xz[0]*math.sqrt(2*math.pi*t)))*(math.e**(-(xz[1]-xz[0])**2/(2*t))
                                               -math.e**(-(xz[1]+xz[0])**2/(2*t))) for t in t_axis])
        upper = np.array([gamma*(xz[1]**2)/(t**(3/2)) for t in t_axis])
        print(count)
        print(xz)
        if count == 0:
            axes[0, 0].plot(t_axis, lower)
            axes[0, 0].plot(t_axis, actual)
            axes[0, 0].plot(t_axis, upper)
            axes[0, 0].set_title(r"Ungleichungen in $t$ für $x=$" + str(xz[0]) + " und " + "$z=$" + str(xz[1]),
                                 fontsize=8)
        if count == 1:
            axes[0, 1].plot(t_axis, lower)
            axes[0, 1].plot(t_axis, actual)
            axes[0, 1].plot(t_axis, upper)
            axes[0, 1].set_title(r"Ungleichungen in $t$ für $x=$" + str(xz[0]) + " und " + "$z=$" + str(xz[1]),
                                 fontsize=8)
        if count == 2:
            axes[1, 0].plot(t_axis, lower)
            axes[1, 0].plot(t_axis, actual)
            axes[1, 0].plot(t_axis, upper)
            axes[1, 0].set_title(r"Ungleichungen in $t$ für $x=$" + str(xz[0]) + " und " + "$z=$" + str(xz[1]),
                                 fontsize=8)
        if count == 3:
            axes[1, 1].plot(t_axis, lower)
            axes[1, 1].plot(t_axis, actual)
            axes[1, 1].plot(t_axis, upper)
            axes[1, 1].set_title(r"Ungleichungen in $t$ für $x=$" + str(xz[0]) + " und " + "$z=$" + str(xz[1]),
                                 fontsize=8)
    plt.show()

def plot_y1_y2_tau():
    tau = 0.5
    y1 = BesselProcess(1).besselpath
    x1 = BesselProcess(1).besselpath
    y2 = y1 * (np.arange(0, 1, 1 / 1000) < 0.5) + (x1 + y1[500] - x1[500]) * (np.arange(0, 1, 1 / 1000) >= 0.5)
    plt.plot(np.arange(0, 1, 1 / 1000), y1)
    plt.plot(np.arange(0, 1, 1 / 1000), y2 + 0.02)
    plt.vlines(tau, 0, max(max(y1), max(y2)), colors='red')
    plt.legend([r"$Y_1$" ,r"$Y_2$" ,r'$\tau$'])
    # ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    plt.plot(0.5, 0, marker="x", color='red')
    plt.show()

def plot_individual_element_H():
    ind_w_ancestor = Individual(0, 0, lambda_=3)
    t = ind_w_ancestor.lifetime
    axis = ind_w_ancestor.life_axis
    y = 0.2
    beta = math.sqrt(2) - 3 / (2 * math.sqrt(2)) * (math.log(t)) / (t) + y / t
    past_bound = axis * beta + 1
    plt.plot(axis, ind_w_ancestor.lifepath)
    plt.plot(axis, past_bound)
    # plt.hlines(beta*t -1,0, axis[-1], colors='red')
    # plt.hlines(beta*t,0, axis[-1], colors='red')
    plt.vlines(axis[-1], beta * t - 1, beta * t, colors='red')
    plt.legend(['Individuum', r'$\beta s +1$', r'$[\beta t-1,\beta t]$'])
    plt.show()

def plot_individual_element_gamma():
    ind_w_ancestor = Individual(0, 0, lambda_=3)
    t = ind_w_ancestor.lifetime
    axis = ind_w_ancestor.life_axis
    y = 100
    result_y = dict()
    beta = math.sqrt(2) - 3 / (2 * math.sqrt(2)) * (math.log(t)) / (t) + y / t

    x_axis = axis#np.arange(0, t, 0.01)
    result_y[t] = np.array([math.log(x + 1) * (3 / (2 * math.sqrt(2))) for x in x_axis]) * (x_axis <= t / 2) + \
                  np.array([math.log(t - x + 1) * (3 / (2 * math.sqrt(2))) for x in x_axis]) * (x_axis > t / 2)
    x_1 = x_axis[np.where(x_axis < t / 2)]
    y_1 = result_y[t][np.where(x_axis < t / 2)]
    x_2 = x_axis[np.where(x_axis > t / 2)]
    y_2 = result_y[t][np.where(x_axis > t / 2)]

    plt.plot(x_1, y_1 + beta * x_1 + y + 1, color='orange')
    plt.plot(x_2, y_2 + beta * x_2 + y + 1,  color='orange')
    plt.plot(axis, ind_w_ancestor.lifepath, color='#1f77b4')
    plt.vlines(axis[-1], beta * t - 1, beta * t + y, colors='red')
    #plt.legend(['Individuum', r'$\beta s +1$', r'$[\beta t-1,\beta t]$'])
    plt.show()
    return 0

def split_prob():
    ind_w_ancestor = Individual(0, 0, lambda_=10)
    brown_path = ind_w_ancestor.lifepath
    x_axis = ind_w_ancestor.life_axis
    t = ind_w_ancestor.lifetime
    k_mid = max(brown_path)
    plt.plot(x_axis,brown_path)
    plt.hlines(k_mid+1, 0, x_axis[-1], color='orange')
    plt.hlines(k_mid-1, 0, x_axis[-1], color='orange')
    plt.axis('off')
    #plt.show()

    plt.plot(x_axis,brown_path, color='#1f77b4')

    gather = range(round(t**2+.5))

    for j in gather:
        plt.vlines(j/t, min(brown_path), max(brown_path)+1, color='orange')

    plt.show()

    #uproundt = round(t+0.5)
    #for vline in range(uproundt+1):
    #    print(vline)
    return 0







##########################
instance = BranchingProcess()
split_prob()
x = pd.DataFrame([[2, 3], [2, 3], [2, 3]], columns=["Start", "End", "Start"])
x.rename(columns=lambda x: x if x!="Start2" else "Start")
instance.creat_max_particle_plot()




