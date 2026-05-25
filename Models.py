import scipy
from scipy.spatial import cKDTree
from scipy import stats
import math
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point
import rasterio
from rasterio.features import rasterize
import random
from itertools import combinations
import matplotlib.pyplot as plt
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
    #initpos = {i: pos for i, pos in enumerate(random.sample(list(self.coordinate_space), self.N))}
    initpos = {}
    occupied = set()
    #Initial positions without building the whole coordinate space:
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
    #self.coordinate_space -= self.occupied
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

  def visualize(self):
    plt.figure(figsize=(6, 6))
    if self.cumulative == True:
      self.weights = [self.G.edges[edge]["weight"] for edge in self.G.edges]
      edge_thickness = [0.1*(weight / max(self.weights)) for weight in self.weights]
    layout = {i: self.G.nodes[i]['position'] for i in self.G.nodes}
    if self.cumulative == True:
      nx.draw_networkx(self.G, node_size=2, node_color='blue', pos=layout, width=edge_thickness, with_labels=False)
    else:
      nx.draw_networkx(self.G, node_size=2, node_color='blue', pos=layout, with_labels=False)
    plt.legend()
    plt.title(f'Range={self.r}, N={self.N}, grid={self.X}x{self.Y}, T={self.T}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

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

  def update_positions(self,t):
    current_hour = hour_of_day(t)
    orderlist = np.array(random.sample(list(self.G.nodes),len(self.G.nodes)))
    for agent in orderlist:
      x,y = self.G.nodes[agent]["position"]
      step_size = self.s
      target = None
      #Daily schedule step 1: from home to work around 8.
      if 8 <= current_hour <= 17:
        workplace = self.G.nodes[agent]["work"]
        if self.G.nodes[agent]["position"] not in self.work_coords[workplace]:
          target = random.choice(self.work_coords[workplace])
        else:
          if random.random() < self.PW:  # chance to be inactive
            if self.G.nodes[agent]["inactivity timer"] == 0:
              waiting_period = np.random.exponential(self.WM)
              self.G.nodes[agent]["inactivity timer"] = waiting_period
            else:
              self.G.nodes[agent]["inactivity timer"] -= 1
              if self.G.nodes[agent]["inactivity timer"] > 0:
                continue  # still inactive
          else:
            building = random.randrange(len(self.work_coords))
            target = random.choice(self.work_coords[building])

      else:
        home = self.G.nodes[agent]["home"]
        if isinstance(home, tuple):
        # external home: fixed coordinate
          target = home
        else:
        # internal home: building index
          target = random.choice(self.home_coords[home])

      if target:
        tx,ty = target
        dx,dy = tx-x, ty-y
        dist = math.hypot(dx,dy)
        if dist <= step_size:
          nx_, ny_ = tx, ty
          self.G.nodes[agent]["position"] = (nx_, ny_)
        else:
          tx,ty = target
          dx = tx-x
          dy = ty-y
          #Perfect angle in target direction set by tangent line:
          angle = math.atan2(dy,dx)
          #Set some noise:
          angle += random.gauss(0,0.2)
          dx = step_size*math.cos(angle)
          dy = step_size*math.sin(angle)
          new_x = int(round(x + dx))
          new_y = int(round(y + dy))
          if 0 <= new_x < self.X and 0 <= new_y < self.Y:
            self.G.nodes[agent]["position"] = (new_x, new_y)
    return self.G

  def update_edges(self):
    internal_nodes = [n for n in self.G.nodes if not self.G.nodes[n]["external"]]
    pos_arr = np.array([self.G.nodes[n]['position'] for n in internal_nodes])
    tree = cKDTree(pos_arr)
    # find all pairs within radius self.r
    pairs = tree.query_pairs(self.r)
    for i_idx, j_idx in pairs:
      i = internal_nodes[i_idx]; j = internal_nodes[j_idx]
      if not self.G.has_edge(i, j):
        self.G.add_edge(i, j, weight=1)
      else:
        self.G.edges[i,j]["weight"]+=1

  def step(self,t):
        self.update_positions(t)
        self.update_edges()

  def run(self):
    for t in range(self.T):
      self.update_positions(t)
      self.update_edges()

  def visualize(self):
    plt.figure(figsize=(6, 6))
    self.weights = [self.G.edges[edge]["weight"] for edge in self.G.edges]
    edge_thickness = [0.1*(weight / max(self.weights)) for weight in self.weights]
    layout = {i: self.G.nodes[i]['position'] for i in self.G.nodes}
    nx.draw_networkx(self.G, node_size=2, node_color='blue', pos=layout, width=edge_thickness, with_labels=False)
    plt.legend()
    plt.title(f'Range={self.r}, N={self.N}, grid={self.X}x{self.Y}, T={self.T}')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


