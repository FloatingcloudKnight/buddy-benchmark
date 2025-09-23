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
  std::string addr0 = "ws://localhost:" + std::to_string(PORT2);
  std::string addr1 = "ws://localhost:" + std::to_string(PORT4);

  std::array<std::string, 4> devicesForAddr2 = {
      DEVICE3, DEVICE5, DEVICE7, DEVICE9};
  if (PART_ID < 0 || PART_ID >= static_cast<int>(devicesForAddr2.size())) {
    std::cerr << "Invalid PART_ID: " << PART_ID << std::endl;
    return -1;
  }
  std::string addr2 = "ws://[" + devicesForAddr2[PART_ID] + "]:" + std::to_string(PORT5);

  AddQueue shared_queue;
  AddMess addMess("AddMess0", 0, shared_queue, PORT6, addr0, addr1, addr2, addr0);
  // AddMess addMess("AddMess0", 0, shared_queue, 9006, "ws://localhost:9002",
                  // "ws://localhost:9004", "ws://[240e:404:1930:9f06:fc12:3639:c84a:a867]:9005", "ws://localhost:9002","ws://localhost:9002");
    // AddMess addMess("AddMess0", 0, shared_queue, 9106, "ws://localhost:9102",
                  // "ws://localhost:9104", "ws://localhost:9105", "ws://localhost:9102");
  Comp comp(shared_queue);

  std::thread add_thread([&addMess] { addMess.run(); });
  std::thread comp_thread([&comp] { comp.run(); });

  add_thread.join();
  comp_thread.join();

  return 0;
}
