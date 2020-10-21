"""Temporal graph
"""
import pydot
import re


class TemporalGraph(object):

    @staticmethod
    def parse_dot_str(dot_str: str,  attempt_fix: bool = True):
        """Parses a temporal graph represented as a dot string into a 
        Temporal graph.
        Args:
            dot_str (str): The dot graph
            tag (str): a string identifier for the graph
            attempt_fix (bool, optional): If true, the method would attempt to fix some common
                                          formatting issues with the graph. Defaults to True.

        Returns:
            A temporal graph
        """
        if attempt_fix:
            if dot_str[0] == "{":  # fix header
                dot_str = "strict digraph " + dot_str
            dot_str = re.sub(' -- ', ' -> ', dot_str)   # fix arrows
        dot_str = re.sub('rel=', 'label=', dot_str)
        dot_graph = pydot.graph_from_dot_data(dot_str)[0]
        return dot_graph


if __name__ == "__main__":
    import sys
    import json
    with open(sys.argv[1], "r") as f:
        dot_graph_str = json.load(f)["temporal_graph"]

    dot_graph = TemporalGraph.parse_dot_str(dot_graph_str)
    with open(sys.argv[2], "wb") as fout:
        fout.write(dot_graph.create_png())
