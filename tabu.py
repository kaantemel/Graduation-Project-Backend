import sys
from itertools import combinations
import copy
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
import random

def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

service_time_in = []
pickup_delivery_time_in = []

def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4):
        next(file_it)

    pickup_delivery_time_in.append([0,0,0])
    nb_trucks = int(next(file_it))
    truck_capacity = int(next(file_it))

    for i in range(12):
        next(file_it)
    cust_no = []
    locations_x = []
    locations_y = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []


    while True:
        val = next(file_it, None)
        if val is None:
            break
        i = int(val)
        cust_no.append(i)
        locations_x.append(float(next(file_it)))
        locations_y.append(float(next(file_it)))
        demands.append(int(next(file_it)))
        ready = int(next(file_it))
        due = int(next(file_it))
        stime = int(next(file_it))
        earliest_start.append(ready)
        # in input files due date is meant as latest start time
        latest_end.append(due)
        service_time.append(stime)
        random_list = random.sample(range(1, 7), 3)  # Generate a list of 3 unique elements from 1 to 6
        pickup_delivery_time_in.append(random_list)

    pickup_delivery_time_in.append([0,0,0])
    nb_customers = i
    array = np.array([cust_no,locations_x,locations_y,demands,earliest_start,latest_end,service_time])
    
    array = array.transpose()

    

    for start, end in zip(earliest_start, latest_end):
      service_time_in.append([start, end])


    column_values = ["CUST_NO.","XCOORD.","YCOORD.","DEMAND","Et","Lt","St"]

    df = pd.DataFrame(data = array,
                  columns = column_values)


    return df,truck_capacity,nb_trucks,nb_customers

df, Q, no_of_vehicles, no_of_customers= read_input_cvrptw("output.txt")
df2 = pd.DataFrame({'CUST_NO.': [no_of_customers + 1],
                    'XCOORD.': [df["XCOORD."][0]],
                    'YCOORD.': [df["YCOORD."][0]],
                    'DEMAND': [df["DEMAND"][0]],
                    'Et': [df["Et"][0]],
                    'Lt': [df["Lt"][0]],
                    'St': [df["St"][0]]})
df = pd.concat([df, df2], ignore_index=True)
service_time_in.append(service_time_in[0])
Y, X = list(df["YCOORD."]), list(df["XCOORD."])

distance_mtrx = np.empty([no_of_customers+2, no_of_customers+2])
for i in range(len(X)):
    for j in range(len(Y)):
        distance_mtrx[i, j] = round(np.sqrt((X[i] - X[j]) **  2 + (Y[i] - Y[j]) **  2),4)


"""
service_time_in = [[0, 1236],
                   [912, 967],
                   [825, 870],
                   [416, 428],
                   [65, 146],
                   [727, 782],
                   [15, 67],
                   [621, 702],
                   [170, 225],
                   [255, 324],
                   [534, 605],
                   [357, 410],
                   [448, 505],
                   [652, 721],
                   [30, 92],
                   [567, 620],
                   [384, 429],
                   [475, 528],
                   [99, 148],
                   [179, 254],
                   [278, 345],
                   [10, 73],
                   [914, 965],
                   [812, 883],
                   [732, 777],
                   [65, 144],
                   [169, 224],
                   [622, 701],
                   [261, 316],
                   [546, 593],
                   [358, 405],
                   [449, 504],
                   [200, 237],
                   [31, 100],
                   [87, 158],
                   [751, 816],
                   [283, 344],
                   [665, 716],
                   [383, 434],
                   [479, 522],
                   [567, 624],
                   [264, 321],
                   [166, 235],
                   [68, 149],
                   [16, 80],
                   [359, 412],
                   [541, 600],
                   [448, 509],
                   [1054, 1127],
                   [632, 693],
                   [1001, 1066],
                   [815, 880],
                   [725, 786],
                   [912, 969],
                   [286, 347],
                   [186, 257],
                   [95, 158],
                   [385, 436],
                   [35, 87],
                   [471, 534],
                   [651, 740],
                   [562, 629],
                   [531, 610],
                   [262, 317],
                   [171, 218],
                   [632, 693],
                   [76, 129],
                   [826, 875],
                   [12, 77],
                   [734, 777],
                   [916, 969],
                   [387, 456],
                   [293, 360],
                   [450, 505],
                   [478, 551],
                   [353, 412],
                   [997, 1068],
                   [203, 260],
                   [574, 643],
                   [109, 170],
                   [668, 731],
                   [769, 820],
                   [47, 124],
                   [369, 420],
                   [265, 338],
                   [458, 523],
                   [555, 612],
                   [173, 238],
                   [85, 144],
                   [645, 708],
                   [737, 802],
                   [20, 84],
                   [836, 889],
                   [368, 441],
                   [475, 518],
                   [285, 336],
                   [196, 239],
                   [95, 156],
                   [561, 622],
                   [30, 84],
                   [743, 820],
                   [647, 726]]
"""
"""
pickup_delivery_time_in = [[0, 0, 0],
                           [2, 1, 1],
                           [2, 1, 1],
                           [2, 2, 2],
                           [4, 3, 3],
                           [2, 2, 2],
                           [3, 2, 2],
                           [4, 3, 3],
                           [4, 3, 3],
                           [3, 2, 2],
                           [3, 2, 2],
                           [3, 3, 3],
                           [4, 2, 2],
                           [3, 2, 2],
                           [2, 3, 3],
                           [3, 3, 3],
                           [3, 2, 2],
                           [2, 3, 3],
                           [4, 4, 4],
                           [3, 3, 3],
                           [3, 2, 2],
                           [2, 1, 1],
                           [4, 3, 3],
                           [5, 2, 2],
                           [3, 3, 3],
                           [0, 0, 0]]
"""

number_of_vehicle = 15
tabu_itrs = 40
aspiration = 100
sn = 0
retention = 7
en = len(distance_mtrx) - 1
unserviced = list(range(1, en + 1))
tabu_list = []
logging = False


def remove_us(c):
    if c != en and c in unserviced:
        unserviced.remove(c)


'''
following method calculates cost for travelling to one node to other node
Parameters :
1. prev     -- previous node number -- if travelling from node 2 to node 5 then prev = 2
2. c        -- next node(customer)  -- if travelling from node 2 to node 5 then c = 5
3. sst_prev -- service start time for previous node
4. isPdl    -- boolean -- True if there is delay reaching to previous customer false otherwise

returns:

1. d                        -- distance travelled
2. w                        -- waiting time
3. dl                       -- delay time
4. (w/4)+(dl/4)+(dl)        -- cost which needs to be minimized by greedy and tabu search
5. sst                      -- Service start time
6. isd                      -- is delayed 
'''


def get_cost(prev, c, sst_prev, ispdl):
    d = distance_mtrx[prev][c]
    pd = pickup_delivery_time_in[prev][0] + pickup_delivery_time_in[prev][1]+pickup_delivery_time_in[prev][2] if not ispdl else 0
    dl = sst_prev + pd + d - service_time_in[c][1] if sst_prev + pd + d - service_time_in[c][1] > 0 else 0
    w = service_time_in[c][0] - sst_prev - pd - d if service_time_in[c][0] - sst_prev - pd - d > 0 else 0
    sst = sst_prev + pd + d + w
    isd = True if dl > 0 else False
    return d, w, dl, d, sst, isd


'''
Greedy algorithm to find initial solution

NO parameters passed
returns: solution with route in 2D array like solution : [[route-1][route-2].....] 
'''


def get_initial_solution():
    rs = []
    k = 1
    sst_prev = 0
    ispdl = False
    bn = 0
    bc = 0
    while k in range(1, number_of_vehicle + 1):
        prev = 0
        rt = [0]

        while not prev == en:
            minim = 99999
            for c in unserviced:
                if prev == en:
                    break
                if prev == 0 and c == en:
                    continue
                if k == number_of_vehicle and len(unserviced) > 1 and c == en:
                    continue
                d, w, dl, cost, sst, isd = get_cost(prev, c, sst_prev, ispdl)
                if cost < minim:
                    bn = c
                    bc = cost
                    minim = cost
                    sst_prev = sst
                    ispdl = isd
            rt.append(bn)
            prev = bn
            remove_us(bn)
        rs.append(rt)
        k += 1
    return rs


'''
Following function takes solution as input and returns the neighbouring solution by exchanging one node from a solution 
to one node from other solution

returns the all neighbouring solution sorted with cost in ascending order
'''


def get_exchange_neighbour(soln):
    neighbours = []
    for combo in list(combinations(soln, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:
                if i == 0 or j == 0:
                    continue
                _tmp = copy.deepcopy(soln)
                _c0 = copy.deepcopy(combo[0])
                _c1 = copy.deepcopy(combo[1])
                idx1 = _tmp.index(_c0)
                idx2 = _tmp.index(_c1)
                _c1.insert(_c1.index(j), i)
                _c0.insert(_c0.index(i), j)
                _c1.remove(j)
                _c0.remove(i)
                _tmp[idx1] = _c0
                _tmp[idx2] = _c1
                if is_move_allowed((j, i, idx1, idx2), soln, _tmp, 3):
                    neighbours.append((_tmp, get_solution_actual_cost(_tmp), (3, j, i, idx2, idx1, retention)))

        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if i == 0 or j == 0:
                    continue
                _tmp = copy.deepcopy(soln)
                _c0 = copy.deepcopy(combo[1])
                _c1 = copy.deepcopy(combo[0])
                idx1 = _tmp.index(_c0)
                idx2 = _tmp.index(_c1)
                _c1.insert(_c1.index(j), i)
                _c0.insert(_c0.index(i), j)
                _c1.remove(j)
                _c0.remove(i)
                _tmp[idx1] = _c0
                _tmp[idx2] = _c1
                if is_move_allowed((j, i, idx1, idx2), soln, _tmp, 3):
                    neighbours.append((_tmp, get_solution_actual_cost(_tmp), (3, j, i, idx2, idx1, retention)))

    # print("{0} number of Neighbours after Exchange {1}".format(len(neighbours), neighbours))
    neighbours.sort(key=lambda x: x[1][-1])
    return neighbours[0] if len(neighbours) > 0 else -1;


'''
Following function takes solution as input and returns the neighbouring solution by relocating one node from a solution 
in to other solution

returns the all neighbouring solution sorted with cost in ascending order
'''


def get_relocate_neighbour(soln):
    neighbours = []
    for combo in list(combinations(soln, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:
                if j == 0:
                    continue
                _tmp = copy.deepcopy(soln)
                _c0 = copy.deepcopy(combo[0])
                _c1 = copy.deepcopy(combo[1])
                idx1 = _tmp.index(_c0)
                idx2 = _tmp.index(_c1)
                _c1.remove(j)
                _c0.insert(_c0.index(i) + 1, j)
                _tmp[idx1] = _c0
                _tmp[idx2] = _c1
                if is_move_allowed((j, i, idx1, idx2), soln, _tmp, 1):
                    neighbours.append((_tmp, get_solution_actual_cost(_tmp), (1, j, i, idx2, idx1, retention)))

        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if j == 0:
                    continue
                _tmp = copy.deepcopy(soln)
                _c0 = copy.deepcopy(combo[1])
                _c1 = copy.deepcopy(combo[0])
                idx1 = _tmp.index(_c0)
                idx2 = _tmp.index(_c1)
                _c1.remove(j)
                _c0.insert(_c0.index(i) + 1, j)
                _tmp[idx1] = _c0
                _tmp[idx2] = _c1
                if is_move_allowed((j, i, idx1, idx2), soln, _tmp, 1):
                    neighbours.append((_tmp, get_solution_actual_cost(_tmp), (1, j, i, idx2, idx1, retention)))
               

    # print("{0} number of Neighbours after relocation {1}".format(len(neighbours), neighbours))
    neighbours.sort(key=lambda x: x[1][-1])
    return neighbours[0]


'''
Following function takes solution as input and returns the neighbouring solution by shuffling nodes within a route with each other


returns the all neighbouring solution sorted with cost in ascending order
'''


def get_shuffle_neighbours(soln):
    neighbours = []
    for r in soln:
        for i in r[1:-1]:
            for j in r[1:-1]:
                _tmp = copy.deepcopy(soln)
                _r = copy.deepcopy(r)
                idx = _tmp.index(r)
                if i == j:
                    continue
                tmp = j
                idxi = r.index(i)
                _r[r.index(j)] = i
                _r[idxi] = j
                _tmp[idx] = _r
                if is_move_allowed((j, i, idx, idx), soln, _tmp, 2):
                    neighbours.append((_tmp, get_solution_actual_cost(_tmp), (2, j, i, idx, idx, retention)))
    neighbours.sort(key=lambda x: x[1][-1])
    return neighbours[0] if len(neighbours) > 0 else -1


'''
wrapper for above three functions
'''


def get_neighbours(op, soln):
    if op == 1:
        return get_relocate_neighbour(soln)
    elif op == 2:
        return get_shuffle_neighbours(soln)
    elif op == 3:
        return get_exchange_neighbour(soln)


'''
following function calculate the cost for solution it uses the  get_cost function internally

parameters :
1. soln -- solution

Returns :
1. distance     -- distance
2. delay        -- delay time
3. wait         -- wait time
4. cost         -- cost 
5. serviced     -- number of customers (nodes) services successfully 
6. unserviced   -- number of customers (nodes) not serviced due to delay 
7. details      -- route details for route [(wait time,delay time,service start time)] each () have details for each 
                   customer and each row represents a route 
'''


def get_solution_cost(soln: list):
    cost = 0
    wait = 0
    delay = 0
    serviced = 0
    unserviced = 0
    distance = 0
    details = []
    for route in soln:
        prev = 0
        prev_sst = 0
        details_tmp = []
        is_delayed = False
        for customer in route[1:]:
            d, w, dl, c, sst, isd = get_cost(prev, customer, prev_sst, is_delayed)
            prev_sst = sst
            is_delayed = isd
            prev = customer
            if isd:
                unserviced += 1
            else:
                serviced += 1
            distance += d
            delay += dl
            wait += w
            cost += c
            details_tmp.append((w, dl, sst))
        details.append(details_tmp)

    return distance, delay, wait, cost, serviced, unserviced, details


'''
following function calculate the cost for solution it uses the  get_cost function internally its same as above but 
returns less parameters

parameters :
1. soln -- solution

Returns :
1. distance     -- distance
2. delay        -- delay time
3. wait         -- wait time
4. cost         -- cost 
'''


def get_solution_actual_cost(soln: list):
    cost = 0
    wait = 0
    delay = 0
    serviced = 0
    unserviced = 0
    distance = 0
    details = []
    for route in soln:
        prev = 0
        prev_sst = 0
        details_tmp = []
        is_delayed = False
        for customer in route[1:]:
            d, w, dl, c, sst, isd = get_cost(prev, customer, prev_sst, is_delayed)
            # c = c + (dl * 39)
            prev_sst = sst
            is_delayed = isd
            prev = customer
            if isd:
                unserviced += 1
            else:
                serviced += 1
            distance += d
            delay += dl
            wait += w
            cost += c
            details_tmp.append((w, dl, sst))
        details.append(details_tmp)

    return distance, delay, wait, cost


'''
Function is to find total distance for solution without pickup/delivery time
'''


def get_distance_for_solution(soln: list):
    d = 0
    distance = []
    for route in soln:
        prev = 0
        for customer in route[1:]:
            d += get_distance(prev, customer)
            prev = customer
        distance.append(d)
        d = 0
    return distance


'''
Tabu search driver method
'''


def tabu_search(routes: list, itrations):
    best_solution_ever = routes
    best_cost_ever = get_solution_actual_cost(routes)
    best_solution_ever_not_chaned_itr_count = 0
    best_soln = routes
    best_cost = ()
    tmp12 = []
    global tabu_list
    for i in range(itrations - 1):
        tmp12 = []
        if best_solution_ever_not_chaned_itr_count > 7:
            break
        tmp12.append(get_neighbours(1, best_soln))
        tmp12.append(get_neighbours(3, best_soln))
        tmp11 = get_neighbours(2, best_soln)
        if not tmp11 == -1:
            tmp12.append(tmp11)
        tmp12.sort(key=lambda x: x[1][-1])
        if tmp12[1] == -1 or tmp12[0] == -1:
            break
        best_soln = tmp12[0][0]
        best_cost = tmp12[0][1]
        tabu_list.append(TabuListClass(tmp12[0][2][0], tmp12[0][2][1:-1], tmp12[0][2][-1]))

        if best_cost_ever[-1] > best_cost[-1]:
            best_cost_ever = best_cost
            best_solution_ever = best_soln
        else:
            best_solution_ever_not_chaned_itr_count += 1
        iteration_update_tabu_list()

    return best_solution_ever, best_cost_ever


# ------------- input provider methods-----------------------------------------------------------------
def get_distance(src, dest):
    return distance_mtrx[src][dest]


def get_pickup_time(cust):
    return pickup_delivery_time_in[cust][0]


def get_latest_service_time(cust):
    return service_time_in[cust][1]


def is_empty_route(route: list):
    if len(route) == 2 and 0 in route and len(distance_mtrx) - 1 in route:
        return True
    return False


def contains(list, filter):
    for x in list:
        if filter(x):
            return True
    return False


def read_input_file(filename):
    with open(filename) as f:
        lines = list(f)
        i = 0
        for line in lines[:-3]:
            set_input(i, parse_line_to_list(line))
            i += 1
        set_input(i, int(lines[-3]))
        set_input(i + 1, int(lines[-2]))
        set_input(i + 1, int(lines[-1]))


def parse_line_to_list(line):
    ret = []
    str = line[2:-3]
    # print(str)
    for in1 in str.split(']['):
        tmp = []
        for i in in1.split(','):
            if i.isnumeric():
                tmp.append(int(i))
        ret.append(tmp)
    return ret


def set_input(inp, value):
    if inp == 0:
        global distance_mtrx
        distance_mtrx = value
    elif inp == 1:
        global service_time_in
        service_time_in = value
    elif inp == 2:
        global pickup_delivery_time_in
        pickup_delivery_time_in = value
    elif inp == 3:
        global number_of_vehicle
        number_of_vehicle = value
    elif inp == 4:
        global tabu_itrs
        tabu_itrs = value
    elif inp == 5:
        global aspiration
        aspiration = value


# -----------------end ---------input provider methods---------------------------------------------------------

class TabuListClass:
    def __init__(self, op, move, valid_for):
        self.op = op
        self.move = move
        self.valid_for = valid_for

    def checked(self):
        if self.valid_for > 0:
            self.valid_for -= 1
            return self.valid_for
        else:
            return -1

    def find(self, move, aspired, op):
        if self.op == op and self.move == move and self.valid_for > 0 and not aspired:
            return True
        return False


'''
to check current move against tabu list if not available in tabu list then move is allowed otherwise not allowed
function also check for aspiration criteria
'''


def is_move_allowed(move, soln_prev, soln_curr, op):
    if len(tabu_list) < 1:
        return True
    cost_prev = get_solution_actual_cost(soln_prev)[-1]
    cost_curr = get_solution_actual_cost(soln_curr)[-1]
    if cost_prev - cost_curr > aspiration:
        return not contains(tabu_list, lambda x: x.find(move, True, op))
    else:
        return not contains(tabu_list, lambda x: x.find(move, False, op))


'''
to update tabu list iteration wise
'''


def iteration_update_tabu_list():
    for i in tabu_list:
        if i.checked() < 0:
            tabu_list.remove(i)


# utility function to print 2d array linewise rows
def print2D(arr):
    for row in arr:
        print(row)


# log utility method
def print_log(log):
    if logging:
        print(log)


# log = open("myprog.log", "a")
# sys.stdout = log

# read_input_file("vrptw_test_4_nodes.txt")
routes = get_initial_solution()
# routes.remove([])
best_soln, best_cost = tabu_search(routes, tabu_itrs)
best_cost = get_solution_actual_cost(best_soln)
index1 = 0
for route in best_soln:
    index1 += 1


distance, delay, wait, cost, serviced, unserviced, details = get_solution_cost(best_soln)

def node_exist(i,j):
  res = False
  for tour in best_soln:
    for k in range(1,len(tour)):
      if i == tour[k -1] and j ==tour[k]:
        res = True
  return res
      



coordinates = []

for route in best_soln:
    route_coords = []
    if(len(route)!=2):
        for customer in route:
            x_coord = df.loc[df['CUST_NO.'] == customer, 'XCOORD.'].values[0]
            y_coord = df.loc[df['CUST_NO.'] == customer, 'YCOORD.'].values[0]
            route_coords.append((x_coord, y_coord))
        coordinates.append(route_coords)
n = len(X)
X= np.empty([n, n])
for i in range(n):
  for j in range(n):
        if(node_exist(i,j) == True):
          X[i,j] = 1
        else:
          X[i,j] = 0



def plot_tours(solution_x):
        tours = [[i, j] for i in range(solution_x.shape[0]) for j in range(solution_x.shape[1]) if solution_x[i, j] ==1]
        for t, tour in enumerate(tours):
            plt.plot(df["XCOORD."][tour], df["YCOORD."][tour], color = "black", linewidth=0.5)
        plt.scatter(df["XCOORD."][1:], df["YCOORD."][1:], marker = 'x', color = 'g', label = "customers")
        plt.scatter(df["XCOORD."][0], df["YCOORD."][0], marker = "o", color = 'b', label = "depot")
        plt.xlabel("X"), plt.ylabel("Y"), plt.title("Tours"), plt.legend(loc = 4)
        plt.show()

print(coordinates)
print(best_soln)