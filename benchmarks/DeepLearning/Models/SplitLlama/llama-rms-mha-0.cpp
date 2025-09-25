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
#include "llamaRMS.h"
// -----------------------------------------------------------------------------
// LLaMA Inference Main Entry
// -----------------------------------------------------------------------------

int main() {
  std::string addr0 = "ws://[" + std::string(DEVICE1) + "]:" + std::to_string(PORT1);
  std::string addr1 = "ws://[" + std::string(DEVICE3) + "]:" + std::to_string(PORT13);
  std::string addr2 = "ws://[" + std::string(DEVICE5) + "]:" + std::to_string(PORT13);
  // std::string addr3 = "ws://[" + std::string(DEVICE7) + "]:" + std::to_string(PORT13);
  // std::string addr1 = "ws://[" + std::string(DEVICE3) + "]:" + std::to_string(PORT1);
  // std::string addr2 = "ws://[" + std::string(DEVICE5) + "]:" + std::to_string(PORT1);
  // std::string addr3 = "ws://[" + std::string(DEVICE7) + "]:" + std::to_string(PORT1);
  std::string addr3 = "ws://[" + std::string(DEVICE3) + "]:" + std::to_string(PORT13);

  std::array<std::string, 4> addrs = {addr0, addr1, addr2, addr3};
  std::string targetAddr;
  if (PART_ID >= 0 && PART_ID < static_cast<int>(addrs.size())) {
    targetAddr = addrs[PART_ID];
  } else {
    std::cerr << "Invalid PART_ID: " << PART_ID << std::endl;
    return -1;
  }

  RMSQueue shared_queue;
  RMSMess rmsMess("RMSMess1", shared_queue, PORT3, targetAddr);
  // RMSMess rmsMess("RMSMess1", shared_queue, 9103, "ws://localhost:9101");
  Comp comp(shared_queue, 1);

  std::thread rms_thread([&rmsMess] { rmsMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  rms_thread.join();
  comp_thread.join();

  return 0;
}
