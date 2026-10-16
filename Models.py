from scipy.spatial import cKDTree
import math
import osmnx as ox
from shapely.geometry import Point
import random
import networkx as nx
import numpy as np


'''
2. MODELS
This code defines the two agent-based models tested. Both are calibrated to OSM data
using the OSMNX package. 

'''

#The following function uses OSMNX (openstreetmaps package) to extract 
#a 2D grid in meters from the geodata of the location string. In this case,
#the Copenhagen University Lyngby campus. This data is rich and includes buildings, 
#walkways, etc. In simulation_lean(), we use it just to set the right X and Y
#for an empty grid of realistic proportions in meters. In the below simulation with
#scheduled mobility, we also use the building data to set home and work anchors.

def hour_of_day(t):
    return (t % 288) // 12

def extract_grid(location):
    buildings = ox.features_from_place(location, tags={'building': True})
    # Project buildings GeoDataFrame to UTM: conversion system of lat/long to meters.
    utm_crs = buildings.estimate_utm_crs() #estimates the timezone CRS code for UTM conversion.
    utm = buildings.to_crs(utm_crs)
    minx, miny, maxx, maxy = utm.total_bounds
    grid = round(maxx - minx),round(maxy - miny)
    return grid

#The random movement model:

class Simulation_lean:
  def __init__(self,N,T,s,r,cumulative=True,location="Technical University of Denmark, Lyngby, Denmark"):
    self.N = N
    self.T = T
    self.r = r
    self.s = s
    self.location = location
    self.X, self.Y = extract_grid(self.location)
    self.cumulative = cumulative
    #If cumulative is False, this generates a single unweighted snapshot graph.
    #However, to avoid the completely random initial positions, we still run 
    #the simulation for T timesteps first.
    self.G = self.initialize_graph()

  def initialize_graph(self):
    G = nx.Graph()
    G.add_nodes_from(range(self.N))
    if self.cumulative == True:
      nx.set_edge_attributes(G, 0, 'weight')
    initpos = {}
    occupied = set()
    while len(initpos) < self.N:
      x = random.randrange(self.X)
      y = random.randrange(self.Y)
      if (x,y) in occupied:
        continue
      i = len(initpos)
      initpos[i] = (x,y)
      occupied.add((x,y))
    nx.set_node_attributes(G, initpos, 'position')
    self.occupied = occupied
    return G

  def update_positions(self):
    orderlist = np.array(random.sample(list(self.G.nodes),len(self.G.nodes)))
    for agent in orderlist:
      step_size = self.s
      x,y = self.G.nodes[agent]["position"]
      angle = random.uniform(0, 2*math.pi)
      dx = step_size*math.cos(angle)
      dy = step_size*math.sin(angle)
      new_x = int(round(x + dx))
      new_y = int(round(y + dy))
      if 0 <= new_x < self.X and 0 <= new_y < self.Y and (new_x, new_y) not in self.occupied:
        self.G.nodes[agent]["position"] = (new_x, new_y)
        self.occupied.discard((x, y))
        self.occupied.add((new_x, new_y))
    return self.G

  def update_edges_2(self):
    nodes = list(self.G.nodes)
    pos_arr = np.array([self.G.nodes[n]['position'] for n in nodes])
    tree = cKDTree(pos_arr)
    # find all pairs within radius self.r
    pairs = tree.query_pairs(self.r)
    for i_idx, j_idx in pairs:
      i = nodes[i_idx]; j = nodes[j_idx]
      if self.cumulative == True:
        if not self.G.has_edge(i, j):
          self.G.add_edge(i, j, weight=1)
        else:
          self.G.edges[i,j]["weight"]+=1
      else:
        if not self.G.has_edge(i,j):
          self.G.add_edge(i, j)

  def run(self):
    if self.cumulative == True:
      for t in range(self.T):
        self.update_positions()
        self.update_edges_2()
    else:
      #For non-cumulative runs, just save the last-t network:
      for t in range(self.T):
        self.update_positions()
      self.update_edges_2()


#The following functions will be used by the scheduled mobility model class to 
#define and assign home and work anchors.

def random_edge_point(X, Y):
    side = random.choice(["top", "bottom", "left", "right"])
    if side == 'left':
        return (0, random.randint(0, Y-1))
    if side == 'right':
        return (X-1, random.randint(0, Y-1))
    if side == 'bottom':
        return (random.randint(0, X-1), 0)
    if side == 'top':
        return (random.randint(0, X-1), Y-1)

PRECOMPUTED = None

def precompute_environment(location, frac_homes, frac_work):
    """
    Run once:
    - fetch OSM buildings
    - rasterize polygons
    - choose fixed home/work building pools
    """
    global PRECOMPUTED
    if PRECOMPUTED is not None:
        return PRECOMPUTED

    buildings = ox.features_from_place(location, tags={'building': True})
    utm_crs = buildings.estimate_utm_crs()
    utm = buildings.to_crs(utm_crs)

    building_polygons = utm['geometry'].tolist()
    minx, miny, maxx, maxy = utm.total_bounds
    X = int(round(maxx - minx))
    Y = int(round(maxy - miny))

    def poly_to_coords(poly):
        coords = []
        minx_p, miny_p, maxx_p, maxy_p = poly.bounds
        for x in range(int(minx_p), int(maxx_p)+1):
            for y in range(int(miny_p), int(maxy_p)+1):
                if poly.contains(Point(x, y)):
                    gx = int(round(x - minx))
                    gy = int(round(maxy - y))
                    if 0 <= gx < X and 0 <= gy < Y:
                        coords.append((gx, gy))
        return coords

    building_coords = [poly_to_coords(p) for p in building_polygons]

    n_total = len(building_polygons)
    n_homes = max(1, int(n_total * frac_homes))
    n_work  = max(1, int(n_total * frac_work))

    idx = list(range(n_total))
    random.shuffle(idx)

    home_idx = idx[:n_homes]
    work_idx = idx[n_homes:n_homes + n_work]

    home_coords = [building_coords[i] for i in home_idx]
    work_coords = [building_coords[i] for i in work_idx]

    PRECOMPUTED = {
        "X": X,
        "Y": Y,
        "home_coords": home_coords,
        "work_coords": work_coords,
        "building_coords": building_coords,
    }
    return PRECOMPUTED

class Simulation_scheduledmobility:
  def __init__(self,N,T,s,r,PW,WM,
               frac_homes=0.3,
               frac_ext=0.3,
               location="Technical University of Denmark, Lyngby, Denmark"):
    '''
    N: number of agents
    T: number of timesteps
    r: range of interaction
    s: step distance per 5 minute bin
    PW: probability of waiting
    WM: mean wait time (exponential)
    frac_homes: fraction of homes
    frac_ext: fraction of homes off-campus
    '''
    self.N = N
    self.T = T
    self.r = r
    self.s = s
    self.PW = PW
    self.WM = WM
    self.frac_homes = frac_homes
    self.frac_work = 1-self.frac_homes
    self.frac_ext = frac_ext

    data = precompute_environment(location,self.frac_homes,self.frac_work)
    self.X = data["X"]
    self.Y = data["Y"]
    self.home_coords = data["home_coords"]
    self.work_coords = data["work_coords"]
    self.building_coords = data["building_coords"]

    self.G = self.initialize_graph()

  def initialize_graph(self):
    G = nx.Graph()
    G.add_nodes_from(range(self.N))
    nx.set_edge_attributes(G, 0, 'weight')
    home_assignments = {}
    initpos = {}
    external_flags = {}
    orderlist = np.array(random.sample(list(G.nodes),len(G.nodes)))
    for i in orderlist:
      if random.random() < self.frac_ext:
        h = random_edge_point(self.X, self.Y)
        home_assignments[i] = h
        initpos[i] = h
        external_flags[i] = True
      else:
        h = random.randrange(len(self.home_coords))
        home_assignments[i] = h
        initpos[i] = random.choice(self.home_coords[h])
        external_flags[i] = False
    work_assignments = {i: random.randrange(len(self.work_coords)) for i in range(self.N)}
    nx.set_node_attributes(G,external_flags,'external')
    nx.set_node_attributes(G,home_assignments, 'home')
    nx.set_node_attributes(G,work_assignments, 'work')
    nx.set_node_attributes(G, initpos, 'position')
    nx.set_node_attributes(G, 0, 'inactivity timer')
    return G

  def update_positions(self, t):
    current_hour = hour_of_day(t)
    orderlist = random.sample(list(self.G.nodes), len(self.G.nodes))

    for agent in orderlist:
        node = self.G.nodes[agent]
        x, y = node["position"]
        target = None

        if 8 <= current_hour < 17:
            # Start each workday by travelling to the primary workplace.
            if node.get("work destination") is None:
                node["work destination"] = node["work"]
                node["work target"] = random.choice(
                    self.work_coords[node["work"]]
                )

            target = node["work target"]

            # Finish the current journey before making another decision.
            if target is not None and node["position"] == target:
                node["work target"] = None
                target = None

            if target is None:
                # An existing inactivity period cannot be interrupted.
                if node["inactivity timer"] > 0:
                    node["inactivity timer"] -= 1
                    continue

                # Begin a new inactivity period.
                if random.random() < self.PW:
                    #Duration: random draw from an exponential distribution with mean WM.
                    #The results is rounded up to the nearest integer, and at least 1 timestep.
                    duration = max(
                        1, int(np.ceil(np.random.exponential(self.WM)))
                    )
                    # This timestep counts as the first inactive step.
                    node["inactivity timer"] = duration - 1
                    continue

                # Otherwise choose and retain a new workplace target.
                building = random.randrange(len(self.work_coords))
                node["work destination"] = building
                target = random.choice(self.work_coords[building])
                node["work target"] = target

        else:
            # End work activity and reset for the next workday.
            node["work destination"] = None
            node["work target"] = None
            node["inactivity timer"] = 0

            home = node["home"]
            if isinstance(home, tuple):
                if node["position"] == home:
                    continue
                target = home
            else:
                if node["position"] in self.home_coords[home]:
                    node["home target"] = None
                    continue
                if node.get("home target") is None:
                    node["home target"] = random.choice(self.home_coords[home])
                target = node["home target"]

        # Existing directed movement rule.
        tx, ty = target
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        if dist <= self.s:
          #To prevent constantly overshooting and zooming around the 
          #target, if close to it, travel the rest of the distance.
            node["position"] = target
        else:
            angle = math.atan2(dy, dx) + random.gauss(0, 0.2)
            new_x = int(round(x + self.s * math.cos(angle)))
            new_y = int(round(y + self.s * math.sin(angle)))

            if 0 <= new_x < self.X and 0 <= new_y < self.Y:
                node["position"] = (new_x, new_y)

    return self.G

  def update_edges(self):
    """Record proximity except for external agents currently at their home, 
    who are supposed to be off-campus somewhere.
    """
    contact_nodes = [
        n for n in self.G.nodes
        if not (
            self.G.nodes[n]["external"]
            and self.G.nodes[n]["position"] == self.G.nodes[n]["home"]
        )
    ]
    # cKDTree needs a two-dimensional coordinate array; handle empty/singleton
    # contact populations explicitly (e.g. before the morning commute).
    if len(contact_nodes) < 2:
        return

    pos_arr = np.array([self.G.nodes[n]["position"] for n in contact_nodes])
    tree = cKDTree(pos_arr)
    for i_idx, j_idx in tree.query_pairs(self.r):
        i = contact_nodes[i_idx]
        j = contact_nodes[j_idx]
        if not self.G.has_edge(i, j):
            self.G.add_edge(i, j, weight=1)
        else:
            self.G.edges[i, j]["weight"] += 1

  def step(self,t):
        self.update_positions(t)
        self.update_edges()

  def run(self):
    for t in range(self.T):
      self.update_positions(t)
      self.update_edges()

