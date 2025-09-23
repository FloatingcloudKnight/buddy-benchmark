#ifndef LLAMAMLP_H // 作用：防止llamaMLP.h被重复引用
#define LLAMAMLP_H
#include "BaseDisModel.h"
#include "SharedQueueTemp.h"
#include <any>
#include <boost/asio.hpp>
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <variant>
#include <vector>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

using namespace buddy;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

typedef websocketpp::server<websocketpp::config::asio> server;
typedef websocketpp::client<websocketpp::config::asio> client;

constexpr size_t MaxVocabSize = 32000;
constexpr size_t MaxTokenLength = 40;
constexpr size_t SubMaxTokenLength = 20;
constexpr size_t HiddenSize = 4096;
constexpr size_t HiddenSize0 = 128;
constexpr size_t HiddenSize1 = 41;

/// Declare LLaMA forward function.
extern "C" void _mlir_ciface_forward5(MemRef<float, 2> *, MemRef<float, 1> *,
                                      MemRef<float, 3> *);

//--------------------- MLPMess (主线程) ---------------------

class MLPQueue : public SharedQueueTemp {
public:
  MLPQueue() : SharedQueueTemp({"input0", "input1", "output"}) {}
};

class MLPMess {
public:
  MLPMess(const std::string name, MLPQueue &queue, const uint16_t &port,
          const std::string &uri0, const std::string &uri1)
      : mlpServer(), name(name), sharedQueue(queue), hdlsSymbol(),
        resultContainer(MemRef<float, 2>({MaxTokenLength, HiddenSize})),
        dataId(0) {
    /// 服务器初始化
    mlpServer.set_access_channels(websocketpp::log::alevel::all);
    mlpServer.clear_access_channels(websocketpp::log::alevel::all);
    mlpServer.init_asio();

    mlpServer.set_close_handler([this](websocketpp::connection_hdl hdl) {
      std::lock_guard<std::mutex> lock(symbolMutex);
      auto it = connections.find(hdl);
      if (it != connections.end()) {
        std::string user_id = it->second;
        hdlsSymbol.erase(user_id);
        connections.erase(hdl);
        std::cout << user_id << " 已断开连接" << std::endl;
      }
    });

    mlpServer.set_message_handler(
        bind(&MLPMess::on_server_message, this, _1, _2));
    mlpServer.listen(port);
    mlpServer.set_reuse_addr(true);
    mlpServer.start_accept();

    /// 客户端初始化
    // 禁用客户端日志
    rmsClient.set_access_channels(websocketpp::log::alevel::all);
    rmsClient.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    rmsClient.init_asio();
    // 设置服务器的消息回调
    rmsClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      rmsClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    rmsClient.set_message_handler(
        bind(&MLPMess::on_rmsClient_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code rmsec;
    auto rmscon = rmsClient.get_connection(uri0, rmsec);
    if (rmsec) {
      std::cerr << "连接RMSClient错误: " << rmsec.message() << std::endl;
      // 处理错误，例如重试或退出
    } else {
      rmsClient.connect(rmscon);
    }

    // 禁用客户端日志
    rmsClient0.set_access_channels(websocketpp::log::alevel::all);
    rmsClient0.clear_access_channels(websocketpp::log::alevel::all);
    // 初始化服务器并绑定ioService
    rmsClient0.init_asio();
    // 设置服务器的消息回调
    rmsClient0.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      rmsClient0.send(hdl, name, websocketpp::frame::opcode::text);
    });
    rmsClient0.set_message_handler(
        bind(&MLPMess::on_rmsClient0_message, this, _1, _2));
    // 启动连接
    websocketpp::lib::error_code rmsec0;
    auto rmscon0 = rmsClient0.get_connection(uri1, rmsec0);
    if (rmsec0) {
      std::cerr << "连接RMSClient0错误: " << rmsec0.message() << std::endl;
    } else {
      rmsClient0.connect(rmscon0);
    }
    rmsClient.set_open_handshake_timeout(5000); // 5秒超时
    rmsClient0.set_open_handshake_timeout(5000);
  }

  void run() {
    std::thread rmsClient_thread([this]() { rmsClient.run(); });
    std::thread rmsClient0_thread([this]() { rmsClient0.run(); });
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { mlpServer.run(); });

    // 新增：启动输出监听线程，向AddMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainer = sharedQueue.pop<MemRef<float, 2>>("output");
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        MemRef<float, 2> subResultContainer0({SubMaxTokenLength, HiddenSize});
        MemRef<float, 2> subResultContainer1({SubMaxTokenLength, HiddenSize});
        resultContainer.splitMemRef(std::move(resultContainer),
                                    subResultContainer0, subResultContainer1, 0,
                                    20);
        std::map<std::string, std::vector<std::vector<float>>> sendMap = {
            {"AddMess0", {subResultContainer0.getDataVector()}},
          {"AddMess1", {subResultContainer1.getDataVector()}}};
        BaseDisModel::sendToClient(sendMap, hdlsSymbol, dataId, mlpServer);
      }
    });
    rmsClient_thread.join();
    rmsClient0_thread.join();
    server_thread.join();
    output_thread.join();
  }

private:
  server mlpServer;
  client rmsClient;
  client rmsClient0;
  const std::string name;
  MLPQueue &sharedQueue;
  std::map<std::string, websocketpp::connection_hdl> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  MemRef<float, 2> resultContainer;
  //  确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId;
  std::mutex dataMutex;

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
    std::string payload = msg->get_payload();
    if (payload.find("AddMess") != std::string::npos) {
      std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
      hdlsSymbol[payload] = hdl;
      connections[hdl] = payload;
      std::cout << payload << " 已连接" << std::endl;
      return;
    }
  }

  std::vector<float> getFloatData(client::message_ptr msg) {
    if (msg->get_opcode() != websocketpp::frame::opcode::binary) {
      std::cout << "忽略非二进制消息" << std::endl;
      return {};
    }

    const std::string &payload = msg->get_payload();
    if (payload.size() < 10) {
      std::cerr << "错误: 协议头不完整(需要至少10字节)" << std::endl;
      return {};
    }

    // 解析协议头
    uint32_t batch_id;
    uint8_t totalChunks, seqChunk;
    uint32_t num_elements;

    memcpy(&batch_id, payload.data(), 4);
    totalChunks = payload[4];
    seqChunk = payload[5];
    memcpy(&num_elements, payload.data() + 6, 4);

    // 验证分块序号有效性
    if (seqChunk >= totalChunks) {
      std::cerr << "错误：非法分块序号 " << (int)seqChunk
                << " (总块数=" << (int)totalChunks << ")" << std::endl;
      return {};
    }

    // 验证数据长度
    const size_t expectedSize = 10 + num_elements * sizeof(float);
    if (payload.size() != expectedSize) {
      std::cerr << "错误：数据长度不匹配(预期=" << expectedSize
                << " 实际=" << payload.size() << ")" << std::endl;
      return {};
    }

    // 提取浮点数据
    const float *float_data =
        reinterpret_cast<const float *>(payload.data() + 10);
    std::vector<float> chunk(float_data, float_data + num_elements);
    return chunk;
  }

  void on_rmsClient_message(websocketpp::connection_hdl,
                            client::message_ptr msg) {
    std::lock_guard<std::mutex> lock(dataMutex);
    auto chunk = getFloatData(msg);
    intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
    MemRef<float, 3> subResultContainer(chunk.data(), sizes);
    sharedQueue.push("input0", subResultContainer);
    std::cout << "接收到RMSMess0数据." << std::endl;
  }

  void on_rmsClient0_message(websocketpp::connection_hdl,
                             client::message_ptr msg) {
    std::lock_guard<std::mutex> lock(dataMutex);
    auto chunk = getFloatData(msg);
    intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
    MemRef<float, 3> subResultContainer(chunk.data(), sizes);
    sharedQueue.push("input1", subResultContainer);
    std::cout << "接收到RMSMess1数据." << std::endl;
  }
};

//--------------------- Comp (子线程) ------------------------------------------
//  -splitNum: 标志当前MLP模块是第几个子模块
//------------------------------------------------------------------------------
class Comp {
public:
  Comp(MLPQueue &queue, const std::string splitNum = "0")
      : sharedQueue(queue), splitNum(splitNum) {}

  void init() { loadAllParameters(); }

  void run() {
    while (true) {
      MemRef<float, 3> rmsInput0 = sharedQueue.pop<MemRef<float, 3>>("input0");
      MemRef<float, 3> rmsInput1 = sharedQueue.pop<MemRef<float, 3>>("input1");
      MemRef<float, 3> input0({1, MaxTokenLength, HiddenSize});
      MemRef<float, 2> resultContainer({MaxTokenLength, HiddenSize});
      input0.concatenateMemRefs(rmsInput0, rmsInput1, input0, 1);
      _mlir_ciface_forward5(&resultContainer, &paramsContainers[index],
                            &input0);
      std::cout << "第" << index << "次forward5 computed." << std::endl;
      sharedQueue.push("output", resultContainer);
      index = (index + 1) % paramsContainers.size();
    }
  }

private:
  MLPQueue &sharedQueue;
  std::vector<MemRef<float, 1>> paramsContainers;
  uint32_t index = 0;
  const std::string splitNum;

  void loadAllParameters() {
    constexpr size_t paramSize_group[] = {
        131072064, 4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 4096,     33554432, 0, 4096, 67633152,
        0,         4096, 33554432, 0, 4096,     67633152, 0, 4096, 33554432,
        0,         4096, 67633152, 0, 131076096};
    size_t group_len = sizeof(paramSize_group) / sizeof(paramSize_group[0]);
    BaseDisModel::getParameters(paramSize_group, group_len, 67633152, splitNum,
                                paramsContainers);
    //   /// Define directories of vacabulary and parameter file.
    //   std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;

    //   for (int i = 0; i < 194; i++) { // N 为需要生成的数量
    //     if (paramSize_group[i] == 67633152) {
    //       std::string paramsDir = llamaBuildDir + "/subgraph" +
    //                               std::to_string(i) + "_arg" + splitNum +
    //                               ".data";
    //       MemRef<float, 1> paramsContainer({paramSize_group[i]});
    //       loadParameters(paramsDir, paramsContainer);
    //       paramsContainers.push_back(std::move(paramsContainer));
    //     }
    //   }
  }
};

#endif // LLAMAMLP_H
