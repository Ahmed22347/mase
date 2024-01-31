import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from ...utils import get_mase_op, get_mase_type, get_node_actual_target, get_node_by_name


logger = logging.getLogger(__name__)


def graph_iterator_compare_nodes(
    ori_graph, graph, save_path=None, silent=False
) -> pd.DataFrame:
    """List all nodes in the graph and compare the original and quantized nodes."""

    def get_type_str(node):
        if node.op == "call_module":
            return type(get_node_actual_target(node)).__name__
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
            "patched_func",
        ]:
            return get_node_actual_target(node).__name__
        elif get_mase_type(node) in ["implicit_func"]:
            actual_target = get_node_actual_target(node)
            if isinstance(actual_target, str):
                return actual_target
            else:
                return actual_target.__name__
        else:
            return node.target

    headers = [
        "Ori name",
        "New name",
        "MASE_TYPE",
        "Mase_OP",
        "Original type",
        "Quantized type",
        "Changed",
    ]
    rows = []
    for ori_n, n in zip(ori_graph.fx_graph.nodes, graph.fx_graph.nodes):
        rows.append(
            [
                ori_n.name,
                n.name,
                get_mase_type(n),
                get_mase_op(n),
                get_type_str(ori_n),
                get_type_str(n),
                type(get_node_actual_target(n)) != type(get_node_actual_target(ori_n)),
            ]
        )
    if not silent:
        logger.debug("Compare nodes:")
        logger.debug("\n" + tabulate(rows, headers=headers, tablefmt="orgtbl"))
    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(tabulate(rows, headers=headers))

    df = pd.DataFrame(rows, columns=headers)
    if save_path is not None:
        df.to_csv(save_path)

    return df


def graph_iterator_node_histogram(ori_graph, graph, save_path: str = None):
    """Group nodes by their types and count the number of nodes in each group."""
    df = graph_iterator_compare_nodes(ori_graph, graph, save_path=None, silent=True)
    histogram_df = df.groupby(["Original type"]).agg(
        OP=pd.NamedAgg(column="Mase_OP", aggfunc="first"),
        Total=pd.NamedAgg(column="Changed", aggfunc="count"),
        Changed=pd.NamedAgg(column="Changed", aggfunc=lambda x: np.sum(x)),
        Unchanged=pd.NamedAgg(
            column="Changed", aggfunc=lambda x: np.sum(1 - np.array(x))
        ),
    )
    logger.info("Quantized graph histogram:")
    logger.info("\n" + tabulate(histogram_df, headers="keys", tablefmt="orgtbl"))
    if save_path is not None:
        histogram_df.to_csv(save_path)


# def graph_iterator_compare_nodes(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


# def graph_iterator_node_histogram(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


def summarize_quantization_analysis_pass(
    ori_graph, graph, save_dir: str = None
) -> None:
    """
    Summarizes the quantization analysis pass.

    Args:
        ori_graph: The original graph.
        graph: The modified graph.
        save_dir (optional): The directory to save the summary files. Defaults to None.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    table_path = os.path.join(save_dir, "quantize_table.csv") if save_dir else None
    histogram_path = (
        os.path.join(save_dir, "quantize_histogram.csv") if save_dir else None
    )
    graph_iterator_compare_nodes(ori_graph, graph, save_path=table_path, silent=False)
    graph_iterator_node_histogram(ori_graph, graph, save_path=histogram_path)


def list_changes_pass(ori_graph, graph, print_true)-> None:
    # Compare nodes between original and quantized graphs
    comparison_df = graph_iterator_compare_nodes(ori_graph, graph, silent=True)

    # Number of nodes in the original and quantized graphs
    num_nodes_ori = len(list(ori_graph.fx_graph.nodes))
    num_nodes_quantized = len(list(graph.fx_graph.nodes))

    # Nodes that have changed
    changed_nodes = comparison_df[comparison_df['Changed'] == True]
    num_changed_nodes = len(changed_nodes)
    

    # Display the results
    if (print_true):
        print(f"Number of nodes in the original graph: {num_nodes_ori}")
        print(f"Number of nodes in the quantized graph: {num_nodes_quantized}")
        print(f"Number of changed nodes: {num_changed_nodes}")
        if num_changed_nodes > 0:
            print("Changed nodes:")
            print(changed_nodes[["Ori name", 'New name', 'Original type', 'Quantized type']])
        else:
            print("No nodes have changed.")

    return changed_nodes["Ori name"]



def verify_quantisation_pass(ori_graph, quantized_graph):
    # Get the list of changed node names and convert it to a set for efficient lookup
    changed_node_names = set(list_changes_pass(ori_graph, quantized_graph, 0))
    print("/////////////////////////////////////////////////////////////////////")
    # Iterate through the nodes in ori_graph.fx_graph
    for node in ori_graph.fx_graph.nodes:
        if node.op == 'call_module' and node.name in changed_node_names:
            
            quantised_node = get_node_by_name(quantized_graph.fx_graph, node.name)
            
            print(f'MASE OP: {node.meta["mase"].parameters["common"]["mase_op"]}\n')
            print("Data In:")
            print(
                f"Precision->   Original: {node.meta['mase'].parameters['common']['args']['data_in_0']['precision']} "
                f"Quantised: {quantised_node.meta['mase'].parameters['common']['args']['data_in_0']['precision']}\n"
            )
            print("Original weights:")
            print(
                f"Precision->   Original: {node.meta['mase'].parameters['common']['args']['weight']['precision']} "
                f"Quantised: {quantised_node.meta['mase'].parameters['common']['args']['weight']['precision']}\n"
            )
            print("Original Bias:")
            print(
                f"Precision->   Original: {node.meta['mase'].parameters['common']['args']['bias']['precision']} "
                f"Quantised: {quantised_node.meta['mase'].parameters['common']['args']['bias']['precision']}\n"
            )
           