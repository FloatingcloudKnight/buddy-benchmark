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
#include "llamaOutput.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  OutputQueue shared_queue;
  std::string addr0 = "ws://[" + std::string(DEVICE8) + "]:" + std::to_string(PORT12);
  std::string addr1 = "ws://[" + std::string(DEVICE9) + "]:" + std::to_string(PORT13);
  std::string addr2 = "ws://[" + std::string(DEVICE1) + "]:" + std::to_string(PORT1);


  OutputMess outputMess(shared_queue, addr0, addr1, addr2);
  // OutputMess outputMess(shared_queue, "ws://[240e:404:1930:9f06:6496:1921:e1fc:ec8b]:9012", "ws://[240e:404:1930:9f06:fc12:3639:c84a:a867]:9013", "ws://[240e:404:1930:9f06:f5da:f1fd:3607:ece6]:9001");
  //  OutputMess outputMess(shared_queue, "ws://localhost:9112", "ws://localhost:9113", "ws://localhost:9101");
  Comp comp(shared_queue);

  std::thread output_thread([&outputMess] { outputMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  output_thread.join();
  comp_thread.join();

  return 0;
}
