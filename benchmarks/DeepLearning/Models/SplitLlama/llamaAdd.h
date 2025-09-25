#ifndef LLAMAAdd_H // 作用：防止llamaAdd.h被重复引用
#define LLAMAAdd_H
#include "SharedQueueTemp.h"
#include "BaseDisModel.h"
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
extern "C" void _mlir_ciface_forward3(MemRef<float, 3> *, MemRef<float, 2> *,
                                      MemRef<float, 3> *);

//--------------------- AddMess (主线程) ---------------------
class AddQueue : public SharedQueueTemp {
public:
  AddQueue() : SharedQueueTemp({"input", "input0", "input1", "output"}) {}
};

class AddMess {
public:
  AddMess(const std::string name, bool isLast, AddQueue &queue,
          const uint16_t &port, const std::string &uri0,
          const std::string &uri1, const std::string &uri2,
          const std::string &uri3)
      : addServer(), name(name), sharedQueue(queue), hdlsSymbol(),
        isLast(isLast),
        resultContainer(MemRef<float, 3>({1, SubMaxTokenLength, HiddenSize})) {
    /// 服务器初始化
    addServer.set_access_channels(websocketpp::log::alevel::none);
    addServer.clear_access_channels(websocketpp::log::alevel::all);
    addServer.init_asio();

    // addServer.set_close_handler([this](websocketpp::connection_hdl hdl) {
    //   std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
    //   auto it = connections.find(hdl);
    //   if (it != connections.end()) {
    //     std::string user_id = it->second;
    //     hdlsSymbol.erase(user_id);
    //     connections.erase(hdl);
    //   }
    // });

    addServer.set_message_handler(
        bind(&AddMess::on_server_message, this, _1, _2));
    addServer.listen(port);
    addServer.set_reuse_addr(true);
    addServer.start_accept();

    /// 客户端初始化
    rmsClient.set_access_channels(websocketpp::log::alevel::none);
    rmsClient.clear_access_channels(websocketpp::log::alevel::all);
    rmsClient.init_asio();
    rmsClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      mhaClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    rmsClient.set_message_handler(
        bind(&AddMess::on_rmsClient_message, this, _1, _2));
    websocketpp::lib::error_code rmsec;
    auto rmscon = rmsClient.get_connection(uri0, rmsec);
    rmsClient.connect(rmscon);

    mhaClient.set_access_channels(websocketpp::log::alevel::none);
    mhaClient.clear_access_channels(websocketpp::log::alevel::all);
    mhaClient.init_asio();
    mhaClient.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      mhaClient.send(hdl, name, websocketpp::frame::opcode::text);
    });
    mhaClient.set_message_handler(
        bind(&AddMess::on_mhaClient_message, this, _1, _2));
    websocketpp::lib::error_code mhaec;
    auto mhacon = mhaClient.get_connection(uri1, mhaec);
    mhaClient.connect(mhacon);

    mhaClient0.set_access_channels(websocketpp::log::alevel::none);
    mhaClient0.clear_access_channels(websocketpp::log::alevel::all);
    mhaClient0.init_asio();
    mhaClient0.set_open_handler([this, name](websocketpp::connection_hdl hdl) {
      mhaClient0.send(hdl, name, websocketpp::frame::opcode::text);
    });
    mhaClient0.set_message_handler(
        bind(&AddMess::on_mhaClient0_message, this, _1, _2));
    websocketpp::lib::error_code mhaec0;
    auto mhacon0 = mhaClient0.get_connection(uri2, mhaec0);
    mhaClient0.connect(mhacon0);

    if (isLast) {
      rmsClient0.set_access_channels(websocketpp::log::alevel::none);
      rmsClient0.clear_access_channels(websocketpp::log::alevel::all);
      rmsClient0.init_asio();
      rmsClient0.set_open_handler([this](websocketpp::connection_hdl hdl) {
        rmsClient0.send(hdl, "LastAdd", websocketpp::frame::opcode::text);
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        hdlsSymbol["FirstRMS"] = hdl;
        connections[hdl] = "FirstRMS";
        std::cout << "已连接到RMSServer" << std::endl;
      });
      websocketpp::lib::error_code rmsec0;
      auto rmscon0 = rmsClient0.get_connection(uri3, rmsec0);
      rmsClient0.connect(rmscon0);
    }
  }

  void run() {
    std::thread rmsClient_thread([this]() { rmsClient.run(); });
    std::thread mhaClient_thread([this]() { mhaClient.run(); });
    std::thread mhaClient0_thread([this]() { mhaClient0.run(); });
    // 启动 WebSocket 服务器线程
    std::thread server_thread([this]() { addServer.run(); });
    std::thread rmsClient0_thread;

    // 新增：启动输出监听线程，向OutputMess和RMSMess发送数据
    std::thread output_thread([this]() {
      while (true) {
        resultContainer = sharedQueue.pop<MemRef<float, 3>>("output");
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        if (isLast) {
          if (tfCount == 31) { 
            // printMemRef(resultContainer);
            std::map<std::string, std::vector<std::vector<float>>> sendMap = {
                {"OutputMess", {resultContainer.getDataVector()}}};
            BaseDisModel::sendToClient(sendMap, hdlsSymbol, dataId, addServer);
            
            tfCount = PART_ID * layerNum;

          } else if(tfCount < PART_ID * layerNum + layerNum-1){
            std::map<std::string, std::vector<std::vector<float>>> sendMap = {
                {"FirstRMS", {resultContainer.getDataVector()}}};
            BaseDisModel::sendToClient(sendMap, hdlsSymbol, dataId, addServer);
            tfCount++;
          }else if (tfCount == PART_ID * layerNum + layerNum-1) {
            // printMemRef(resultContainer);
            std::map<std::string, std::vector<std::vector<float>>> sendMap = {
                {"RMSMess", {resultContainer.getDataVector()}}};
            BaseDisModel::sendToClient(sendMap, hdlsSymbol, dataId, addServer);
            tfCount = PART_ID * layerNum;
            } else {
              std::cout << "transformer层推理次数过多" << std::endl;
            }
          } else { 
            std::map<std::string, std::vector<std::vector<float>>> sendMap = {
                {"RMSMess", {resultContainer.getDataVector()}}};
            BaseDisModel::sendToClient(sendMap, hdlsSymbol, dataId, addServer);
          }
        }
      });
      if (isLast) {
        rmsClient0_thread = std::thread([this]() { rmsClient0.run(); });
      }
      rmsClient_thread.join();
      mhaClient_thread.join();
      mhaClient0_thread.join();
      server_thread.join();
      output_thread.join();
      rmsClient0_thread.join();
  }

private:
  server addServer;
  client rmsClient;
  client rmsClient0;
  client mhaClient;
  client mhaClient0;
  const std::string name;
  AddQueue &sharedQueue;
  std::map<std::string, websocketpp::connection_hdl> hdlsSymbol;
  std::map<websocketpp::connection_hdl, std::string,
           std::owner_less<websocketpp::connection_hdl>>
      connections;
  websocketpp::connection_hdl firstRMSHdl;
  std::mutex symbolMutex; // 保护 hdlsSymbol 的互斥锁
  MemRef<float, 3> resultContainer;
  // 确保对dataId的操作是​​原子​​的
  std::atomic<uint32_t> dataId = 0;
  std::mutex dataMutex;
  // 记录已经进行过的Transformer层计算次数
  // uint32_t tfCount = PART_ID * 8;
  uint32_t layerNum=32/PART_NUMS;
  uint32_t tfCount = PART_ID * layerNum;
  // 是否是最后一个add模块
  bool isLast;

  void printMemRef(MemRef<float, 3> &mem) {
    auto dataVec = mem.getDataVector();
    auto sizes = mem.getSizes(); // [batch, token, hidden]

    std::cout << "MemRef<float,3> 内容: shape = ["
              << sizes[0] << ", " << sizes[1] << ", " << sizes[2] << "]\n";

    for (size_t b = 0; b < sizes[0]; ++b) {
      std::cout << "Batch " << b << ":\n";
      for (size_t t = 0; t < sizes[1]; ++t) {
        std::cout << "  Token " << t << ": ";
        for (size_t h = 0; h < std::min<size_t>(sizes[2], 16); ++h) {
          std::cout << dataVec[b * sizes[1] * sizes[2] + t * sizes[2] + h]
                    << " ";
        }
        if (sizes[2] > 16) std::cout << "...";
        std::cout << "\n";
      }
    }
  }

  void on_server_message(websocketpp::connection_hdl hdl,
                         server::message_ptr msg) {
      std::string payload = msg->get_payload();
      if (payload.find("RMSMess") != std::string::npos) {
        std::lock_guard<std::mutex> lock(symbolMutex); // 加锁保护符号表
        hdlsSymbol["RMSMess"] = hdl;
        connections[hdl] = payload;
        std::cout << payload << " 已连接" << std::endl;
        return;
      } else if (payload.find("OutputMess") != std::string::npos) {
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

  void on_mhaClient_message(websocketpp::connection_hdl hdl,
                            client::message_ptr msg) {
      std::lock_guard<std::mutex> lock(dataMutex);
      auto chunk = getFloatData(msg);
      intptr_t sizes[2] = {SubMaxTokenLength, HiddenSize};
      MemRef<float, 2> subResultContainer(chunk.data(), sizes);
      sharedQueue.push("input0", subResultContainer);
      std::cout << "接收到MHAMess0数据" << std::endl;
  }

  void on_mhaClient0_message(websocketpp::connection_hdl hdl,
                             client::message_ptr msg) {
      std::lock_guard<std::mutex> lock(dataMutex);
      auto chunk = getFloatData(msg);
      intptr_t sizes[2] = {SubMaxTokenLength, HiddenSize};
      MemRef<float, 2> subResultContainer(chunk.data(), sizes);
      sharedQueue.push("input1", subResultContainer);
      std::cout << "接收到MHAMess1数据" << std::endl;
  }

  void on_rmsClient_message(websocketpp::connection_hdl hdl,
                            client::message_ptr msg) {
      std::lock_guard<std::mutex> lock(dataMutex);
      auto chunk = getFloatData(msg);
      intptr_t sizes[3] = {1, SubMaxTokenLength, HiddenSize};
      MemRef<float, 3> subResultContainer(chunk.data(), sizes);
      sharedQueue.push<MemRef<float, 3>>("input", subResultContainer);
      std::cout << "接收到RMSMess数据" << std::endl;
  }
  };

  //--------------------- Comp (子线程) ---------------------
  class Comp {
  public:
    Comp(AddQueue &queue) : sharedQueue(queue) {}

    void run() {
      while (true) {
        auto input1 = sharedQueue.pop<MemRef<float, 2>>("input1");
        auto input2 = sharedQueue.pop<MemRef<float, 3>>("input");
        auto input0 = sharedQueue.pop<MemRef<float, 2>>("input0");
        input0.addMemRef(input0, input1);
        MemRef<float, 3> resultContainer({1, SubMaxTokenLength, HiddenSize});
        _mlir_ciface_forward3(&resultContainer, &input0, &input2);
        std::cout << "forward3 computed." << std::endl;
        sharedQueue.push("output", resultContainer);
      }
    }

  private:
    AddQueue &sharedQueue;
  };

#endif // LLAMAAdd_H
