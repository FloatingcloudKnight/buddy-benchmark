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
  
  std::string addr1 = "ws://localhost:" + std::to_string(PORT6);
  RMSQueue shared_queue;
  RMSMess rmsMess("RMSMess0", shared_queue, PORT8, addr1);
  // RMSMess rmsMess("RMSMess0", shared_queue, 9008, "ws://localhost:9006");
  // RMSMess rmsMess("RMSMess0", shared_queue, 9108, "ws://localhost:9106");
  Comp comp(shared_queue, 0);

  std::thread rms_thread([&rmsMess] { rmsMess.run(); });
  std::this_thread::sleep_for(std::chrono::seconds(1));
  comp.init();
  std::thread comp_thread([&comp] { comp.run(); });

  rms_thread.join();
  comp_thread.join();

  return 0;
}
