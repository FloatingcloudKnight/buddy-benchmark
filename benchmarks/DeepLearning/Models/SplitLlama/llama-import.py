# ===- llama-import.py --------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the test of llama2 model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import torch
import torch._dynamo as dynamo
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp
# from torchviz import make_dot
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse, apply_classic_fusion

class SubModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 提取前置模块
        self.embed_tokens = original_model.model.embed_tokens
        self.layers = torch.nn.ModuleList([original_model.model.layers[0]])  # 仅第一个 Transformer 层
        self.norm = original_model.model.norm  # 如果需要包含最后的 LayerNorm

    def forward(self, input_ids, attention_mask, position_ids):
        # Embedding 层
        hidden_states = self.embed_tokens(input_ids)
        
        # 第一个 Transformer 层
        hidden_states = self.layers[0](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )[0]
        
        return hidden_states

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="LLaMA2 model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files."
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Retrieve the LLaMA model path from environment variables.
model_path = os.environ.get("LLAMA_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LLAMA_MODEL_PATH' is not set or is invalid."
    )

# Initialize the tokenizer and model from the specified model path.
# LlamaTokenizer用于对文本进行分词。分词器将文本转换为模型可以处理的输入格式（通常是标记或ID序列）
# from_pretrained用于从预训练模型的路径加载分词器配置和词汇表。
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)

# LlamaForCausalLM用于加载和使用LLaMA模型进行因果语言建模任务。
# from_pretrained用于从预训练模型的路径加载模型权重和配置。
model = LlamaForCausalLM.from_pretrained(model_path, torchscript=True)
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = torch.tensor([[1 for i in range(40)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]

driver = GraphDriver(graphs[0], 2)
for i in range(len(driver.subgraphs)):
    driver.subgraphs[i].lower_to_top_level_ir()

driver.construct_main_graph(True)
# Save the generated files to the specified output directory.
for i in range(len(driver.subgraphs)): 
    with open(os.path.join(output_dir, f"subgraph{i}.mlir"), "w") as module_file:
        print(driver.subgraphs[i]._imported_module, file=module_file)
    with open(os.path.join(output_dir, f"forward{i}.mlir"), "w") as module_file:
        print(driver.modules[i], file=module_file)

for entry in driver._subgraph_param_info.items():
    driver.construct_sub_params(params, entry, output_dir)
