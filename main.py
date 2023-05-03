import numpy as np
import pandas as pd
import json
import math


class Node:
    def __init__(self, name):
        self.name = name
        self.value = [0, 0]
        self.next = []
        self.edge_functions = []

    def __repr__(self):
        return f"name: {self.name}\nvalue: {self.value}\n"

    def get_name(self) -> str:
        return self.name

    def add_a_neighbor(self, node, edge_function):
        self.next.append(node)
        self.edge_functions.append(edge_function)

    def get_neighbors(self):
        return self.next

    def get_edge_relation_function(self):
        return self.edge_functions

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def sample_value(self):
        return np.random.normal(self.value[0], self.value[1])


def build_graph() -> [Node]:
    #read model
    with open('model.json') as f:
        model = json.load(f)

    # collect all nodes
    nodes = dict()
    for variable in model['variables']:
        _node = Node(name=variable)
        nodes[variable] = _node
    # setup edge relations
    for relation in model['relations']:
        [from_node, to_node] = relation.split(" -> ")
        f_node = nodes[from_node]
        t_node = nodes[to_node]
        f_node.add_a_neighbor(node=t_node, edge_function=model['relations'][relation])
    # setup node values
    for _node in model['values']:
        node = nodes[_node]
        node.set_value(model['values'][_node])

    starts = [nodes[_node] for _node in model['start']]
    return starts


def parse_graph(node, depth=0):
    print(f"{''.join([' '] * depth)}visiting {node.get_name()}:{node.sample_value()}")
    x = node.sample_value()
    eqns = node.get_edge_relation_function()
    for i, neighbor in enumerate(node.get_neighbors()):
        eqn = eqns[i]
        res = 0
        for exp in eqn[::-1]:
            res += int(exp[0]) * math.pow(x, int(exp[2]))
        parse_graph(neighbor, depth+1)


def parse_and_update_graph(node, depth=0):
    #print(f"{''.join([' '] * depth)}visiting {node.get_name()}:{node.sample_value()}")
    x = node.sample_value()
    eqns = node.get_edge_relation_function()
    for i, neighbor in enumerate(node.get_neighbors()):
        eqn = eqns[i]
        res = 0
        for exp in eqn[::-1]:
            res += float(exp[0]) * math.pow(x, int(exp[2]))
        old_value = neighbor.get_value()
        new_value = [old_value[0] + res, old_value[1]]
        neighbor.set_value(new_value)
        parse_and_update_graph(neighbor, depth+1)


def run_a_simulation(nodes):
    for node in nodes:
        parse_and_update_graph(node)


def make_observation(node, record, depth=0):
    # print(f"{''.join([' '] * depth)}visiting {node.get_name()}:{node.sample_value()}")
    x = node.sample_value()
    record[node.get_name()] = x
    for i, neighbor in enumerate(node.get_neighbors()):
        make_observation(neighbor, record, depth+1)


def get_variables():
    with open('model.json') as f:
        model = json.load(f)
    return model['variables']

def main():
    # read model -> build graph -> run simulation -> make observation
    variables = get_variables()
    dataframe = pd.DataFrame(dict(zip(variables, [[]] * len(variables))))

    for _ in range(0, 10):
        nodes = build_graph()
        run_a_simulation(nodes)
        record = dict()
        for node in nodes:
            make_observation(node, record)
        dataframe.loc[len(dataframe)] = record

    print(dataframe)
    dataframe.to_csv('observations.csv', index=False)

if __name__ == '__main__':
    main()

