//===- llama-main.cpp -----------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "llamaInput.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  /// Print the title of this example.
  const std::string title = "LLaMA 2 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;

  /// Initialize data containers
  //  - Input container.
  //  - Result container
  //  - Parameters container.
  MemRef<float, 3> myMemRef1({1, MaxTokenLength, HiddenSize});
  MemRef<float, 2> myMemRef2({MaxTokenLength, HiddenSize1});
  MemRef<float, 3> myMemRef3({1, MaxTokenLength, HiddenSize0});
  MemRef<float, 3> myMemRef4({1, MaxTokenLength, HiddenSize0});
  MemRefContainer resultContainer(myMemRef1, myMemRef2, myMemRef3, myMemRef4);
  MemRefContainer *resultContainerPtr = &resultContainer;

  /// Fill data into containers
  //  - Input: register vocabulary and tokenize the input string.
  //  - Output: register vocabulary.
  //  - Parameters: load parameters from the `arg0` file into the container.

  // SharedQueue shared_queue;
  InputQueue shared_queue;
  InputMess inputMess(shared_queue, resultContainerPtr);
  Comp comp(shared_queue, resultContainerPtr);

  std::thread input_thread([&inputMess] { inputMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  input_thread.join();
  comp_thread.join();

  // std::cout << "\n\033[33;1m[Input]\033[0m " << inputStr << std::endl;

  return 0;
}
