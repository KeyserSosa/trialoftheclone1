import pydot


def parse_act(actfile):
    transitions = []

    graph = pydot.Dot(graph_type='graph')

    def add_edge(*bits):
        kw = {"dir": "forward"}
        if len(bits) > 3:
            kw['label'] = bits[3]
            if bits[3].startswith("win"):
                kw['color'] = "green"
            elif bits[3].startswith("lose"):
                kw['color'] = "red"
        elif bits[2].upper() == "LOSE":
            kw['color'] = "red"
        edge = pydot.Edge(bits[1], bits[2], **kw)
        graph.add_edge(edge)

    battle = False
    graph.add_node(pydot.Node("LOSE", fillcolor = "red",
                              style = "filled", shape = "rectangle"))
    graph.add_node(pydot.Node("ACT2", fillcolor = "red",
                              style = "filled", shape = "rectangle"))
    with open(actfile) as handle:
        lines = list(handle)

        for line in lines:
            bits = [x.strip() for x in line.split(",")]
            if battle and bits[1] != battle:
                # implicit LOSE
                add_edge(bits[0], battle, "LOSE")
                battle = None
            if bits[2].lower().startswith("battle"):
                battle = bits[1]
                graph.add_node(pydot.Node(bits[1], fillcolor = "cyan",
                                          style = "filled", shape = "rectangle"))
            elif bits[2].lower().startswith("minigame"):
                graph.add_node(pydot.Node(bits[1], fillcolor = "green",
                                          style = "filled"))
            else:
                transitions.append((bits[1], bits[2]))
                add_edge(*bits)
                if (len(bits) > 3 and battle == bits[1] and
                    bits[3].lower().startswith("lose")):
                    battle = None
    graph.write_png('/tmp/act1.png')

