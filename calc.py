import pandas as pd
import numpy as np
from re import X
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gurobipy import *
from copy import copy

params = {
"WLSACCESSID": '904ebd4a-0b3d-419e-a1eb-a1787e2c134e',
"WLSSECRET": 'd2b6f163-1998-4709-9739-307fd52392a4',
"LICENSEID": 962239,
}

def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def read_input_cvrptw(filename):
    file_it = iter(read_elem(filename))

    for i in range(4):
        next(file_it)

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
        print(i)
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

    nb_customers = i
    print(nb_customers)
    array = np.array([cust_no,locations_x,locations_y,demands,earliest_start,latest_end,service_time])

    array = array.transpose()

    print(latest_end)

    column_values = ["CUST_NO.","XCOORD.","YCOORD.","DEMAND","Et","Lt","St"]

    df = pd.DataFrame(data = array,
                  columns = column_values)
    

    return df,truck_capacity,nb_trucks,nb_customers



env = Env(params=params)

results = []

''' we can also check different vehicle capacities for optimal routes'''

df, Q, no_of_vehicles, no_of_customers= read_input_cvrptw("output.txt")
df = df.iloc[0:no_of_customers+1]

Y, X = list(df["YCOORD."]), list(df["XCOORD."])
coordinates = np.column_stack((X, Y))

et, lt, st = list(df["Et"]), list(df["Lt"]), list(df["St"])  # needed for time windows


Demand = list(df["DEMAND"])
n = len(coordinates)
print(coordinates)
depot, customers = coordinates[0, :], coordinates[1:, :]
M = 10**10  # big number
k = no_of_vehicles
if no_of_customers > 0 and no_of_customers <=25:
  increment = 1
elif no_of_customers > 25 and no_of_customers <=50:
  increment = 2
else:
  increment = 3
for i in range(1,k + 1,increment):
  print("I am here with number", i)
  no_of_vehicles = i
  print(no_of_vehicles)
  m = Model(env=env)
  x, y, z = {}, {}, {}  #intialize the decision variables

  '''distance matrix (32*32 array)'''
  dist_matrix = np.empty([n, n])
  for i in range(len(X)):
      for j in range(len(Y)):
          '''variable_1: X[i,j] =(0,1), i,j = Nodes'''
          x[i, j] = m.addVar(vtype=GRB.BINARY, name="x%d,%d" % (i, j))
          dist_matrix[i, j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
          if i == j:
              dist_matrix[i, j] = M  ## big 'M'
          continue
  m.update()

  '''variable_2: y[j] = cumulative demand covered'''
  for j in range(n):
      y[j] = m.addVar(vtype=GRB.INTEGER, name="y%d" % (j))   # cumulative demand satisfied variable
      z[j] = m.addVar(vtype=GRB.INTEGER, name="z%d" % (j))   # cumulative time variable
  m.update()



  '''constraint_1: sum(x[i,j]) = 1, for i = 1,2,...,32'''  # vehicles leaving each customer node
  for i in range(n - 1):
      m.addConstr(quicksum(x[(i + 1, j)] for j in range(n)) == 1)
  m.update()

  ''' constraint_2: sum(x[i,j] =1 for j = 1,2,.....,32)'''  # vehicles arriving to each customer node
  for j in range(n - 1):
      m.addConstr(quicksum(x[(i, j + 1)] for i in range(n)) == 1)
  m.update()

  '''constraint_3: sum(x[0,j]) = 5'''  # vehicles leaving depot
  m.addConstr(quicksum(x[(0, j)] for j in range(n)) == no_of_vehicles)
  m.update()

  '''constraint_4: sum(x[i,0]) = 5'''  # vehicles arriving to depot
  m.addConstr(quicksum(x[(i, 0)] for i in range(n)) == no_of_vehicles)
  m.update()

  ''' Either of the constraint_5 or the constrain_6 can eliminate sub-tours independently'''
  print(dist_matrix)
  print(st)
  print(st[1] + dist_matrix[1, 2]/100)

  for i in range(n - 1):
      # assumption: service starts at 9:00 AM, 9 == 0 minutes, each hour after 9 is 60 minutes plus previous hours
      m.addConstr(z[i + 1] >= (et[i + 1]))  # service should start after the earliest service start time
      m.addConstr(z[i + 1] <= (lt[i + 1]))  # service can't be started after the latest service start time
      for j in range(n - 1):
          # taking the linear distance from one node to other as travelling time in minutes between those nodes
          m.addConstr(z[i + 1] >= z[j + 1] + (st[j + 1] + dist_matrix[j + 1, i + 1]/100) * x[j + 1, i + 1] - lt[0]*2*(1-(x[j+1, i+1])))
  '''
  for i in range(n - 1):
      m.addConstr(z[i + 1] <= (st[0] + dist_matrix[0, i + 1]/100) * x[0, i + 1] + lt[0]*2*(1-(x[0, i+1])))
  '''
  '''constraint_5: capacity of vehicle and also eliminating sub-tours'''
  for j in range(n - 1):
      m.addConstr(y[j + 1] <= Q)
      m.addConstr(y[j + 1] >= Demand[j + 1])
      for i in range(n - 1):
          constraint =m.addConstr(y[j + 1] >= y[i + 1] + Demand[j + 1] * (x[i + 1, j + 1]) - Q * (1 - (x[i + 1, j + 1])))
          m.addConstr(y[j + 1] <= y[i + 1] + Demand[j + 1] * (x[i+1, j + 1]) + Q * (1 - (x[i + 1, j + 1])))
  for i in range(n - 1):
    m.addConstr(y[i + 1] <= Demand[i + 1] * (x[0, i + 1]) + Q * (1 - (x[0, i + 1])))



  m.update()

  print(Demand)

  '''
    Vehicle v should arrive at customer c between ready time and due time. 
    Vehicle v should spend service_time before leaving customer c.

  '''
  '''constraint_6: time-windows and also eliminating sub-tours'''



  print("asd")
  print(dist_matrix[0,1])
  '''objective function'''
  m.setObjective(quicksum(quicksum(x[(i, j)]*dist_matrix[(i, j)] for j in range(n)) for i in range(n)),GRB.MINIMIZE)
  m.update()

  m.write("file_name.lp")    

  '''optimize'''
  '''retrieve the solution'''

 
  m.update()
  if no_of_customers <= 50:
    softlimit = 5
  else:
    softlimit = 180

  m._overtime = False
  def softtime(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        final_runtime = model.cbGet(GRB.Callback.RUNTIME)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        no_of_solutions = model.cbGet(GRB.Callback.MIP_SOLCNT)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs((objbst - objbnd) / objbst)
        final_gap = abs((objbst - objbnd) / objbst)


        if runtime > softlimit and no_of_solutions == 0:
            model.terminate()
        if no_of_customers <=50:
          if runtime > 120 and gap > 0.1:
            model._overtime = True
            model.terminate()
        else:
          if runtime > 300 and gap > 0.4:
            model._overtime = True
            model.terminate()

        
  print("Hey I am here at status",m.Status)
  m.Params.TimeLimit = 600
  m.optimize(softtime)
  print(m._overtime)
  try:
    if m._overtime:
      raise Exception("Sorry, try at next round")
    sol_y, sol_x, sol_z = m.getAttr('x', y), m.getAttr('x', x), m.getAttr('x', z)
    X, Y, Z = np.empty([n, n]), np.empty([n]), np.empty([n])
    for i in range(n):
        Y[i] = sol_y[i]
        Z[i] = sol_z[i]
        for j in range(n):
            X[i, j] = int(sol_x[i, j])
    print('\nObjective is:', m.objVal)
    print('\nDecision variable X (binary decision of travelling from one node to another):\n', X.astype('int32'))
    print('\nDecision variable z:(service start time of every customers in minutes)\n', Z.astype('int32')[1:])
    print('\nDecision variable y (cumulative demand collected at every customer node):\n', Y.astype('int32')[1:])
    print("Total number of vehicles: ", no_of_vehicles)

    listofroutes=[]
    routetoappend=[]

    Copy_X = copy(X)

    def route_finder(inp_matrix, row, routelist):
      if(row==0):
        listofroutes.append(routelist)
        routelist=[]
      for i in range(len(inp_matrix[0])):
        if inp_matrix[row][i] == 1:
          routelist.append(row)
          inp_matrix[row][i] = 0
          route_finder(inp_matrix,i,routelist)

    route_finder(Copy_X,0,routetoappend)

    print(listofroutes)

    copyZ = copy(Z)
    copyX_2 = copy(X)
    print(copyZ[2] >= copyZ[1] + (st[1] + dist_matrix[1, 2]/100) * copyX_2[1,2] - M*(1-copyX_2[1, 2]))



    print(copy)
    print(z[2])
    print(z[1])
    print(st[1])
    print(dist_matrix[1,2]/100)




    def plot_tours(solution_x):
        tours = [[i, j] for i in range(solution_x.shape[0]) for j in range(solution_x.shape[1]) if solution_x[i, j] ==1]
        for t, tour in enumerate(tours):
            plt.plot(df["XCOORD."][tour], df["YCOORD."][tour], color = "black", linewidth=0.5)
        plt.scatter(df["XCOORD."][1:], df["YCOORD."][1:], marker = 'x', color = 'g', label = "customers")
        plt.scatter(df["XCOORD."][0], df["YCOORD."][0], marker = "o", color = 'b', label = "depot")
        plt.xlabel("X"), plt.ylabel("Y"), plt.title("Tours"), plt.legend(loc = 4)
        plt.show()

    break
  except:
    print("Errorrr")
    continue

routes_with_coords = []

for route in listofroutes:
    route_with_coords = [coordinates[i].tolist() for i in route]  # converts numpy arrays to lists
    route_with_coords.append(coordinates[0].tolist())
    routes_with_coords.append(route_with_coords)

print(routes_with_coords[1:])

