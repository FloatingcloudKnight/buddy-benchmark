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
#include "llamaMHA.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  std::string addr0 = "ws://[" + std::string(DEVICE1) + "]:" + std::to_string(PORT1);
  std::string addr1 = "ws://localhost:" + std::to_string(PORT2);

  std::array<std::string, 4> devicesForAddr2 = {
      DEVICE3, DEVICE5, DEVICE7, DEVICE9};
  if (PART_ID < 0 || PART_ID >= static_cast<int>(devicesForAddr2.size())) {
    std::cerr << "Invalid PART_ID: " << PART_ID << std::endl;
    return -1;
  }

  std::string addr2 = "ws://[" + devicesForAddr2[PART_ID] + "]:" + std::to_string(PORT3);

  MHAQueue shared_queue;
  MHAMess mahMess("MHAMess0", shared_queue, PORT4, addr0, addr1, addr2);
    // MHAMess mahMess("MHAMess0", shared_queue, 9104, "ws://localhost:9101", "ws://localhost:9102", "ws://localhost:9103");
  Comp comp(shared_queue, "0");

  std::thread mha_thread([&mahMess] { mahMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  mha_thread.join();
  comp_thread.join();

  return 0;
}
