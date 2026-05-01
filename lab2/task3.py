#!/usr/bin/env python
# coding: utf-8

# # Nadokreślony układ równań- zadanie 3

# ### Importy i rysowanie

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from networkx import conductance

plt.style.use('default')
import networkx as nx
import random
random.seed(123)


# In[48]:


def read_graph(file_path: str) -> nx.Graph:
    try:
        G = nx.read_edgelist(file_path, nodetype=int)
    except:
        G = nx.read_edgelist(file_path, nodetype=eval)

    return G


# In[49]:


def draw_undirected_graph(G : nx.Graph) -> None:
    plt.figure()
    # Rysowanie grafu
    pos = nx.shell_layout(G)  # Wyliczenie pozycji węzłów
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_weight='bold')
    # Rysowanie labeli
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()


# In[50]:


def draw_solution(G : nx.DiGraph, title: str = None, draw_function = nx.shell_layout) -> None:
    """
    :param G:
    Rysuje graf skierowany, gdzie wagi krawędzi określają kolor gradientu
    """
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_facecolor('white')

    pos = draw_function(G)  # Wyliczenie pozycji węzłów
    edges = G.edges()
    weights = [G[u][v]['intensity'] for u, v in edges]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax)

    # Rysowanie krawędzi
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=weights,
        edge_cmap=plt.cm.plasma,
        width=2.5,
        arrowsize=20,
        ax=ax
    )

    # Dodanie etykiet na krawędziach
    edge_labels = {(u, v): f"{w:.4f}A" for (u, v), w in zip(edges, weights)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        ax=ax
    )

    # Stworzenie paska kolorów
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.plasma,
        norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
    )
    sm.set_array([]) # Wymagane przez Matplotlib do wygenerowania paska

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Natężenie [A]')

    ax.set_title(title)
    ax.axis('off')
    plt.show()


# ### Metoda I i II prawa Kirchoffa

# In[51]:


def solve_Kirchoff(G : nx.Graph, s : int, t : int, E : float) -> nx.DiGraph:
    """
    Oblicza natężenia w układzie korzystając z I i II prawa Kirchoffa do wyznaczenia układu równań
    :param G: graf ważony oporników
    :param s, t: para wierzchołków pomiędzy którymi przykładamy siłę elektromotoryczną
    :param E: wartość przyłożonej siły
    :return: nx.DiGraph- graf skierowany obliczonych natężeń układu
    """
    G_temp = G.copy()
    if G_temp.has_edge(s, t):     # istnieje taka krawedz
        G_temp[s][t]['sem'] = E
    else:
        # Tworzymy wirtualny drut bez oporu
        G_temp.add_edge(s, t, weight=1e-12, sem=E)

    edges_list = list(G_temp.edges())
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges_list)}
    m = len(edges_list)

    A = [] # macierz współczynników
    b = [] # wektor wyrazów wolnych

    # II prawo Kirchoffa
    # Używamy bazy cykli układu -> m-n+1 równań
    for cycle in nx.cycle_basis(G_temp):
        n = len(cycle)
        new_row = [0.0] * m
        b_value = 0.0

        for i in range(n):
            u, v = cycle[i], cycle[(i+1) % n]
            R = G_temp[u][v]['weight']

            # Sprawdzamy kierunek: jeśli krawędź u->v jest w słowniku, idziemy zgodnie z założonym kierunkiem prądu
            if (u, v) in edge_to_idx:
                idx = edge_to_idx[(u, v)]
                sign = 1.0
            else:
                idx = edge_to_idx[(v, u)]
                sign = -1.0

            new_row[idx] = sign * R

            if {u, v} == {s, t}:
                # Ustalamy znak siły elektromotorycznej w zależności od kierunku przejścia
                if u == s:
                    b_value += E
                else:
                    b_value -= E
        A.append(new_row)
        b.append(b_value)

    # I prawo Kirchoffa
    # równanie dla każdego węzła poza jednym -> n-1 równań
    nodes = list(G_temp.nodes())
    for node in nodes[:-1]:
        new_row = [0.0] * m
        for neighbor in G_temp.neighbors(node):
            if (node, neighbor) in edge_to_idx:
                new_row[edge_to_idx[(node, neighbor)]] = -1.0 # Prąd wypływa z węzła
            else:
                new_row[edge_to_idx[(neighbor, node)]] = 1.0 # Prąd wpływa do węzła

        A.append(new_row)
        b.append(0.0)

    sol = np.linalg.solve(A, b)

    # stworzenie nowego grafu skierowanego układu z natężeniami
    DiG = nx.DiGraph()
    DiG.add_nodes_from(G.nodes(data=True))

    # wyciągamy krawędzie i wagi z G_temp
    DiG.add_edges_from(
        (u, v, {**data, 'intensity': val}) if val >= 0 else (v, u, {**data, 'intensity': -val})
        for (u, v, data), val in zip(G_temp.edges(data=True), sol)
    )

    return DiG


# ## Metoda potencjałów węzłowych

# In[52]:


def solve_nodal_analysis(G : nx.Graph, s : int, t : int, E : float) -> nx.DiGraph:
    G_temp = G.copy()
    if G_temp.has_edge(s, t):
        G_temp[s][t]['sem'] = E
    else:
        G_temp.add_edge(s, t, weight=1e-8, sem=E)

    nodes = list(G_temp.nodes())
    other_nodes = [node for node in nodes if node != t]       # uziemiamy węzeł t
    node_to_idx = {n: i for i, n in enumerate(other_nodes)}

    n = len(other_nodes)
    A = np.zeros((n, n))
    b = np.zeros(n)

    # liczymy sumy konduktancji węzłów
    for i, node in enumerate(other_nodes):
        conductance_sum = 0
        for neighbor in G_temp.neighbors(node):
            R = G_temp[node][neighbor]['weight']
            conductance = 1.0 / R

            if neighbor != t:
                j = node_to_idx[neighbor]
                A[i, j] -= conductance

            conductance_sum += conductance

            if {node, neighbor} == {s, t} and node == s:
                    b[i] -= conductance * E

        A[i, i] = conductance_sum

    V_other = np.linalg.solve(A, b)     # pozostałe potencjały

    voltages = {t: 0.0}
    for i, node in enumerate(other_nodes):
        voltages[node] = V_other[i]

    # prawo Ohma i budowanie grafu skierowanego
    DiG = nx.DiGraph()
    DiG.add_nodes_from(G.nodes(data=True))

    for u, v, data in G_temp.edges(data=True):
        R = data['weight']
        U = voltages[u] - voltages[v]

        # Na krawędzi (s, t) do spadku napięcia na samym rezystorze dodajemy siłę baterii
        if {u, v} == {s, t}:
            if u == s and v == t:
                U += E
            elif u == t and v == s:
                U -= E

        I = U / R
        edge_attrs = {**data}

        # Ustalamy kierunek na podstawie znaku prądu
        if I >= 0:
            edge_attrs['intensity'] = I
            DiG.add_edge(u, v, **edge_attrs)
        else:
            edge_attrs['intensity'] = -I
            DiG.add_edge(v, u, **edge_attrs)

    return DiG


# In[53]:


G = read_graph('graphs/P5.txt')
draw_undirected_graph(G)


# In[54]:


sol = solve_Kirchoff(G, 0, 1, 3)
draw_solution(sol)


# In[55]:


draw_solution(solve_nodal_analysis(G, 0, 1, 3))


# ### Generowanie grafów:
# - Spójny graf losowy (Erdos-Renyi)
# - Graf 3-regularny (kubiczny)
# - Graf złożony z dwóch grafów losowych połaczonych mostkiem
# - Graf siatka 2D
# - Graf typu small-world

# In[56]:


from enum import Enum

def generate_graph(generator_enum, *args, file_path: str = None, **kwargs) -> nx.Graph:
    G= generator_enum(*args, **kwargs)

    for _, _, data in G.edges(data=True):
        data['weight'] = random.uniform(0, 100)

    if file_path is not None:
        nx.write_edgelist(G, file_path)

    return G

def random_graphs_with_bridge(n_1, p_1, n_2, p_2, file_path=None) -> nx.Graph:
    G1 = nx.erdos_renyi_graph(n_1, p_1)
    G2 = nx.erdos_renyi_graph(n_2, p_2)

    G = nx.disjoint_union(G1, G2)

    node_from_G1 = random.choice(range(n_1))
    node_from_G2 = random.choice(range(n_1, n_1 + n_2))

    G.add_edge(node_from_G1, node_from_G2)

    if file_path is not None:
        nx.write_edgelist(G, file_path)

    return G

class GraphGenerator(Enum):
    ERDOS_RENYI = nx.erdos_renyi_graph              # (n-liczba wierz., p- gęstość)
    CUBIC = nx.random_regular_graph                        # ()
    GRID_2D = nx.grid_2d_graph                      # (m, n - wymiary)
    SMALL_WORLD = nx.watts_strogatz_graph           # (n - liczba wierz., k- liczba sąsiadów jednego wierz., p- gęstość)
    RANDOM_2_BRIDGE = random_graphs_with_bridge     # (n1, n2, p1, p2 - analogicznie jak przy Erdos-Renyi)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


# In[57]:


# ================= DATA ======================
generate_graph(GraphGenerator.ERDOS_RENYI, 25, random.uniform(0.4, 0.8), file_path='graphs/erdos_renyi.txt')

generate_graph(GraphGenerator.CUBIC, 3, 16, file_path='graphs/cubic.txt')

generate_graph(GraphGenerator.GRID_2D, 8, 6, file_path='graphs/grid_2d.txt')

generate_graph(GraphGenerator.SMALL_WORLD, 40, 6, random.uniform(0.2, 0.5), file_path='graphs/small_world.txt')

generate_graph(GraphGenerator.RANDOM_2_BRIDGE, 10, random.uniform(0.4, 0.6), 10, random.uniform(0.4, 0.6), file_path='graphs/random_2_bridge.txt')


# ### Testy poprawności

# In[58]:


def Kirchoff_check(DiG: nx.DiGraph) -> bool:
    for u in DiG.nodes():
        in_sum = DiG.in_degree(u, weight='intensity')
        out_sum = DiG.out_degree(u, weight='intensity')

        if not np.isclose(in_sum, out_sum, atol=1e-8):
            return False

    return True

def grid_2d_layout(G):
    # Wierzchołki siatki 2D są krotkami współrzędnych (x, y), używamy ich jako pozycji
    try:
        return {n: (n[0], n[1]) for n in G.nodes()}
    except Exception:
        return nx.spring_layout(G)

DRAWERS = {
    "ERDOS_RENYI": nx.spring_layout,
    "CUBIC": nx.spring_layout,
    "GRID_2D": grid_2d_layout,
    "SMALL_WORLD": nx.circular_layout,
    "RANDOM_2_BRIDGE": nx.spring_layout,
    "P5": nx.circular_layout,
}

from pathlib import Path

for file in Path('graphs').glob('*.txt'):
    name = file.stem.upper()
    print(name)
    G = read_graph(file)
    E = random.uniform(50, 1000)
    nodes = list(G.nodes())
    s, t = nodes[0], nodes[1]

    sol_Kirchoff = solve_Kirchoff(G, s, t, E)
    sol_nodal = solve_nodal_analysis(G, s, t, E)

    draw_fun = DRAWERS[name]
    draw_solution(sol_Kirchoff, title=name, draw_function=draw_fun)
    draw_solution(sol_nodal, title=name, draw_function=draw_fun)

    print(f'Kirchoff: {'OK' if Kirchoff_check(sol_Kirchoff) else 'ERR'}')
    print(f'Kirchoff: {'OK' if Kirchoff_check(sol_Kirchoff) else 'ERR'}')


# # TODO:
# ##   1. opisy
# ##   2. testerka
# ##   3. prettify
