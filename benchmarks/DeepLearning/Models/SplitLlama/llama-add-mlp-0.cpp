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
#include "llamaAdd.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  std::string addr0 = "ws://localhost:" + std::to_string(PORT9);
  std::string addr1 = "ws://localhost:" + std::to_string(PORT11);
  std::array<std::string, 4> devicesForAddr2 = {
      DEVICE2, DEVICE4, DEVICE6, DEVICE8};
  if (PART_ID < 0 || PART_ID >= static_cast<int>(devicesForAddr2.size())) {
    std::cerr << "Invalid PART_ID: " << PART_ID << std::endl;
    return -1;
  }
  std::string addr2 = "ws://[" + devicesForAddr2[PART_ID] + "]:" + std::to_string(PORT10);
  std::string addr3 = "ws://localhost:" + std::to_string(PORT3);

  
  AddQueue shared_queue;
    AddMess addMess("AddMess1", 1, shared_queue, PORT13, addr0, addr2, addr1, addr3);
  // AddMess addMess("AddMess1", 1, shared_queue, 9013, "ws://localhost:9009",
  //                 "ws://[240e:404:1930:9f06:6496:1921:e1fc:ec8b]:9010", "ws://localhost:9011", "ws://localhost:9003","ws://localhost:9003");
    // AddMess addMess("AddMess1", 1, shared_queue, 9113, "ws://localhost:9109",
                  // "ws://localhost:9110", "ws://localhost:9111", "ws://localhost:9103");
  Comp comp(shared_queue);

  std::thread add_thread([&addMess] { addMess.run(); });
  std::thread comp_thread([&comp] { comp.run(); });

  add_thread.join();
  comp_thread.join();

  return 0;
}
