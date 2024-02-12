# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from chop.passes.graph.analysis.report.report_graph import report_graph_analysis_pass
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type, get_parent_name
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

DEFAULT_QUANTIZATION_CONFIG = {
    "config": {
        "name": "integer",
        "bypass": True,
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }
}


class GraphSearchSpaceMixedPrecisionPTQ(SearchSpaceBase):
    """
    Post-Training quantization search space for mase graph.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_QUANTIZATION_CONFIG

        # quantize the model by type or name
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph
        
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg, _ = init_metadata_analysis_pass(mg, None)
            mg, _ = add_common_metadata_analysis_pass(
                mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
            )
            self.mg = mg
        if sampled_config is not None:
            mg, _ = quantize_transform_pass(self.mg, sampled_config)
        mg.model.to(self.accelerator)
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """
        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        match self.config["setup"]["by"]:
            case "name":
                # iterate through all the quantizeable nodes in the graph
                # if the node_name is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    if n_info["mase_op"] in QUANTIZEABLE_OP:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case "type":
                # iterate through all the quantizeable nodes in the graph
                # if the node mase_op is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    n_op = n_info["mase_op"]
                    if n_op in QUANTIZEABLE_OP:
                        if n_op in seed:
                            choices[n_name] = deepcopy(seed[n_op])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }
        print(f"search_space.choice_length:{self.choice_lengths_flattened}")

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]
        #for k, v in indexes.items():
         #   safe_index = max(0, min(v, len(self.choices_flattened[k]) - 1))
          #  flattened_config[k] = self.choices_flattened[k][safe_index]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        #print(f"config{config}")
        return config

    def to_dict(self):
        # Implement logic to convert your custom search space into a dictionary format
        # Example (you will need to adapt this to your actual structure):
        search_space_dict = {}
        for key, value in self.choices_flattened.items():
            search_space_dict[key] = value
        return search_space_dict
    
 

class RedefineLinearSearchSpace(SearchSpaceBase):
    def _post_init_setup(self):
        #super()._post_init_setup()
        # Initialize the default configuration
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self.default_config = {
        "by": "name",  # Example configuration parameter
        "default": {"config": {"name": None}},  # Default configuration for unspecified layers
        }
    
        # Define the range of multipliers for layer adjustments
        self.multipliers = list(range(2, 8))  # Multipliers from 2 to 7 inclusive
        
        # Specify target layers for adjustments
        self.target_layers = ['seq_blocks_2', 'seq_blocks_4', 'seq_blocks_6']
    
        

    def instantiate_linear(self, in_features, out_features, bias):
        return nn.Linear(in_features=in_features, out_features=out_features, bias=bias is not None)

 

    def redefine_linear_transform_pass(self, mg, pass_args=None):
        """
        Adjusts linear layers within a MaseGraph based on the provided scaling configurations.

        :param mg: The MaseGraph instance to be transformed.
        :param pass_args: Dictionary containing scaling configurations and default settings.
        """
        if pass_args is None:
            pass_args = {}

        main_config = pass_args.get('by', {})
        default_config = pass_args.get('default', {'config': {'name': None}})
        
        # Iterating over nodes to transform linear layers
        i=0
        for node in mg.fx_graph.nodes:
            # Ensure we are only transforming linear layers
            if mg.modules.get(node.target) and isinstance(mg.modules[node.target], nn.Linear):
                module = mg.modules[node.target]
                
                # Fetching the specific configuration for this node, if exists; else, use default
                # node_specific_key = next((key for key in pass_args if key.startswith('seq_blocks') and key.endswith(node.name)), None)
                # node_config = pass_args[node_specific_key] if node_specific_key else default_config
                node_config = pass_args.get(node.name, pass_args.get('default'))
                config = node_config.get('config', {})
                name = config.get("name", None)

                print("ooga booga", node_config)
                # Apply transformations based on the specified configuration
                if name in ["output_only", "both", "input_only"]:
                    in_features = module.in_features
                    out_features = module.out_features
                    bias = module.bias is not None
                    if (i==1):
                        multiplier = config.get("channel_multiplier", 1)
                    
                    if name == "output_only" or name == "both":
                        out_features = int(out_features * multiplier)
                    if name == "input_only" or name == "both":
                        in_features = int(in_features * multiplier)
                    print(f"Multiplier{multiplier}")
                    #print("in", in_features)
                    #print("out", out_features)
                    # Reconstruct the linear layer with new parameter
                    new_module = nn.Linear(in_features, out_features, bias)
                    parent_name, name_ = get_parent_name(node.target)
                    setattr(mg.modules[parent_name], name_, new_module)  # Replace the module in the original model
                    #print(f"Transformed {node.name} with {name} configuration.")
                    i+=1
        # Optionally, update the MaseGraph if needed
        # This step depends on whether the MaseGraph requires explicit updates after modifications
        return mg

    def build_search_space(self):
        """
        Build a unified search space where the multiplier combinations are applied all at once,
        maintaining specific configurations for each of the sequence blocks.
        """
        # Define the range of multipliers to be used across all configurations
        multipliers = [2, 3, 4, 5, 6, 7]

        
        config = {
            "by": "name",
            "default": {"config": {"name": [None]}},
            "seq_blocks_2": {
                "config": {
                    "name": ["output_only"],
                    # weight
                    "channel_multiplier": [2, 3, 4, 5, 6, 7],
                    }
                },
            "seq_blocks_4": {
                "config": {
                    "name": ["both"],
                    "channel_multiplier": [2, 3, 4, 5, 6, 7],
                    }
                },
            "seq_blocks_6": {
                "config": {
                    "name": ["input_only"],
                    "channel_multiplier": [2, 3, 4, 5, 6, 7],
                    }
                },
        }

        self.config = config

        choices = deepcopy(config)
        choices.pop("by")
        choices.pop("default")
        self.default_config = {"config": {"name": None}}

        # Here, we're simplifying the search space to a single dimension: the choice of multiplier
        # self.choices_flattened = {
        #     "multiplier": multipliers
        # }

        # # The length reflects the number of different multipliers available
        # self.choice_lengths_flattened = {
        #     "multiplier": len(multipliers)
        # }

        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }
        #print(f"search_space.choice_length:{self.choice_lengths_flattened}")



    # def flattened_indexes_to_config(self, indexes: dict[str, int]):
    #     """
    #     Convert flattened indexes back into a nested configuration dictionary.
    #     This version is tailored for a search space where a single multiplier choice
    #     is applied across multiple blocks, maintaining specific roles for each.
    #     """
    #     config = {
    #         "by": "name",  # Assuming 'by' and 'default' are globally applicable and do not change
    #         "default": {"config": {"name": None}},
    #     }

    #     # Assuming 'indexes' contains a single key for the multiplier choice
    #     # and its value corresponds to the index in the list of multipliers
    #     if 'multiplier' in indexes:
    #         multiplier_index = indexes['multiplier']
    #         # Assuming self.choices_flattened["multiplier"] is the list of multiplier options
    #         selected_multiplier = self.choices_flattened["multiplier"][multiplier_index]

    #         # Define the specific configurations for each block using the selected multiplier
    #         config["seq_blocks_2"] = {
    #             "config": {
    #                 "name": "output_only",
    #                 "channel_multiplier": selected_multiplier,
    #             }
    #         }
    #         config["seq_blocks_4"] = {
    #             "config": {
    #                 "name": "both",
    #                 "channel_multiplier": selected_multiplier,
    #             }
    #         }
    #         config["seq_blocks_6"] = {
    #             "config": {
    #                 "name": "input_only",
    #                 "channel_multiplier": selected_multiplier,
    #             }
    #         }

    #     return config

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]
        #for k, v in indexes.items():
         #   safe_index = max(0, min(v, len(self.choices_flattened[k]) - 1))
          #  flattened_config[k] = self.choices_flattened[k][safe_index]
        print(f"indexes.items():{indexes.items()}")

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["by"]
        #print(f"config{config}")
        return config

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        #"""
        #Adjust the model based on the sampled configuration, then rebuild it as a MaseGraph.
        #This includes setting the correct mode, applying any necessary transformations,
        #and ensuring the model is in the appropriate state before conversion.
        #"""
        # Set the model to the correct mode
        self.model.to(self.accelerator)
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        # Initialize MaseGraph if not already done
        # if self.mg is None:
        assert self.model_info.is_fx_traceable, "Model must be FX traceable"
        self.mg = MaseGraph(self.model)
        # Perform initial passes if required
        self.mg, _ = init_metadata_analysis_pass(self.mg, None)
        self.mg, _ = add_common_metadata_analysis_pass(
            self.mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
        )
        
        # Apply configurations to adjust linear layers before creating the MaseGraph

        print("b4")
        report_graph_analysis_pass(self.mg, {})
        if sampled_config:
            # Call the transformation pass function, passing the MaseGraph and the sampled configuration
            self.mg = self.redefine_linear_transform_pass(self.mg, sampled_config)
        print(f"sampled_config{sampled_config}")
        report_graph_analysis_pass(self.mg, {})

        # Ensure the MaseGraph's model is placed on the correct device after adjustments
        self.mg.model.to(self.accelerator)

        return self.mg
