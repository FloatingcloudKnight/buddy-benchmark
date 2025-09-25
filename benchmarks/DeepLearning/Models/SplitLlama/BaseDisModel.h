#ifndef BASEDISMODEL_H // 作用：防止BaseDisModel.h被重复引用
#define BASEDISMODEL_H
#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

using namespace buddy;
class BaseDisModel {
public:
  /// Print [Log] label in bold blue format.
  static void printLogLabel() { std::cout << "\033[34;1m[Log] \033[0m"; }

  /// Load parameters into data container.
  static void loadParameters(const std::string &paramFilePath,
                             MemRef<float, 1> &params) {
    const auto loadStart = std::chrono::high_resolution_clock::now();
    std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
    if (!paramFile.is_open()) {
      std::cout << paramFilePath << std::endl;
      throw std::runtime_error("[Error] Failed to open params file!");
    }
    printLogLabel();
    std::cout << "Loading params..." << std::endl;
    printLogLabel();
    std::cout << "Params file: " << std::filesystem::canonical(paramFilePath)
              << std::endl;
    paramFile.read(reinterpret_cast<char *>(params.getData()),
                   sizeof(float) * (params.getSize()));
    if (paramFile.fail()) {
      throw std::runtime_error("Error occurred while reading params file!");
    }
    paramFile.close();
    const auto loadEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> loadTime =
        loadEnd - loadStart;
    printLogLabel();
    std::cout << "Params load time: " << (double)(loadTime.count()) / 1000
              << "s\n"
              << std::endl;
  }

  static void getParameters(const size_t *paramSize_group, size_t group_len,
                            int size, const std::string &splitNum,
                            std::vector<MemRef<float, 1>> &paramsContainers) {

    std::string llamaBuildDir = LLAMA_EXAMPLE_BUILD_PATH;

    size_t partSize = group_len / 2;

    size_t start=0;
    size_t end=0;
    if (PART_ID==0){
      start = 0;
      end   = partSize;
    }else if(PART_ID==3){
      start = partSize;
      end   = group_len;
    }else{
      start = PART_ID * partSize; 
      end   = start + partSize;
    }


    // size_t start = PART_ID * partSize;
    // size_t end   = (PART_ID == 3) ? group_len : start + partSize; 

    for (size_t i = start; i < end; i++) {
      if (paramSize_group[i] == size) {
        std::string paramsDir = llamaBuildDir + "/subgraph" +
                                std::to_string(i) + "_arg" + splitNum + ".data";
        MemRef<float, 1> paramsContainer({paramSize_group[i]});

        BaseDisModel::loadParameters(paramsDir, paramsContainer);
        paramsContainers.push_back(std::move(paramsContainer));
      }
    }
  }

  //  Tokenize input data in the container.
  static void tokenizeInput(const std::string &vocabFile,
                            Text<size_t, 2> &inputContainer,
                            const size_t MaxTokenLength) {
    printLogLabel();
    std::cout << "Vocab file: " << std::filesystem::canonical(vocabFile)
              << std::endl;
    const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
    inputContainer.tokenizeLlama(vocabFile, MaxTokenLength);
    const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> buddyTokenizeTime =
        buddyTokenizeEnd - buddyTokenizeStart;
    printLogLabel();
    std::cout << "Tokenize time: " << buddyTokenizeTime.count() << "ms"
              << std::endl;
  }

  // Add the inference Token to the input sequence , and translate the result of
  // that Token inputContainer: Current input container, token will be appended
  // msg: Token to be appended
  static void appendToken(Text<size_t, 2> &inputContainer, std::string &msg) {
    int maxIndex = std::stoi(msg);
    // Determine the generated token.
    int tokenIndex = inputContainer.getTokenCnt() - 1;
    std::string tok = inputContainer.getStr(maxIndex);
    // printIterInfo
    std::cout << "\033[32;1m[Iteration " << tokenIndex << "] \033[0m";
    std::cout << "Token: " << tok << std::endl;

    // Append the generated token into the input and output container.
    inputContainer.appendTokenIdx(maxIndex);
  }

  // Generate a Token based on the inference result and add the token to the
  // output sequence Return value: the index of the token generated; if the
  // generation is finished (the terminator is encountered), -1 is returned.
  //  resultContainer: container for the result of one inference of the model
  //  outputContainer: store the output sequence of tokens
  //  currentToken: the number of tokens currently generated
  //  tokenCnt: the number of tokens in the original input
  //  MaxVocabSize:The maximum number of tokens allowed in the model's
  //  vocabulary. separatorTokenIndex: The vocabulary index of the
  //  end-of-inference token.
  static int generatedToken(MemRef<float, 3> &resultContainer,
                            Text<size_t, 2> &outputContainer,
                            uint32_t &currentToken, uint32_t tokenCnt,
                            const size_t MaxVocabSize,
                            const size_t separatorTokenIndex) {
    int tokenIndex = currentToken + tokenCnt - 1;
    currentToken++;
    // Determine the generated token.
    const float *startPtr =
        resultContainer.getData() + tokenIndex * MaxVocabSize;
    const float *endPtr = startPtr + MaxVocabSize;
    // int maxIndex = findMaxIndex(startPtr, endPtr);
    int maxIndex = std::distance(startPtr, std::max_element(startPtr, endPtr));

    // Stop if a separator token or line break token (13<0x0A>) is generated.
    if (maxIndex == separatorTokenIndex)
      return -1;

    outputContainer.appendTokenIdx(maxIndex);

    return maxIndex;
  }

  static void send_data(websocketpp::connection_hdl hdl, uint32_t dataId,
            const std::vector<std::vector<float>> &data,
            websocketpp::server<websocketpp::config::asio> &server) {
    const uint8_t total = data.size();

    auto con = server.get_con_from_hdl(hdl);
    if (!con || con->get_state() != websocketpp::session::state::open) {
      std::cerr << "连接已关闭，无法发送数据。" << std::endl;
      return;
    }

    for (uint8_t i = 0; i < total; ++i) {
      const auto &subdata = data[i];

      // 构造协议头
      std::vector<uint8_t> packet(10); // 4+1+1+2=8字节头
      memcpy(packet.data(), &dataId, 4);
      packet[4] = total;
      packet[5] = i;
      uint32_t num = subdata.size();
      memcpy(packet.data() + 6, &num, 4);

      // 添加浮点数据
      const uint8_t *binaryData =
          reinterpret_cast<const uint8_t *>(subdata.data());
      packet.insert(packet.end(), binaryData,
                    binaryData + subdata.size() * sizeof(float));

      server.send(hdl, packet.data(), packet.size(),
                  websocketpp::frame::opcode::binary);
    }
  }

  static void sendToClient(const std::map<std::string, std::vector<std::vector<float>>> &nameToDataVecs,
    const std::map<std::string, websocketpp::connection_hdl> &hdlMap,
                uint32_t dataId,
               websocketpp::server<websocketpp::config::asio> &server) {
    for (const auto &[name, dataVecs] : nameToDataVecs) {
      auto it = hdlMap.find(name);
      if (it == hdlMap.end()) {
        std::cerr << "[Warning] 未找到连接: " << name
                  << std::endl;
        continue;
      }
      websocketpp::connection_hdl hdl = it->second;

      // // 调用已有的 send_data 封装
      // send_data(hdl, dataId, data, server);
      try {
      send_data(hdl, dataId++, dataVecs, server);
      std::cout << "成功向"<< name <<"发送数据" << std::endl;

    } catch (const websocketpp::exception &e) {
      std::cerr << "[Error] 向 " << name << " 发送失败: " << e.what() << std::endl;
    }
    }
  }
};

#endif // BASEDISMODEL_H
