#ifndef SHAREDQUEUETEMP_H
#define SHAREDQUEUETEMP_H

#include <any>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>

/// Generic shared memory classes: for inter-thread communication

class SharedQueueTemp {
public:
  /// Constructor: pass the name of the queue you wish to support, e.g.
  /// {"input", "input0", "input1", "output"}
  SharedQueueTemp(const std::vector<std::string> &queueNames) {
    for (const auto &name : queueNames) {
      queues[name] = std::queue<std::any>();
      mutexes[name] = std::make_shared<std::mutex>();
      cvs[name] = std::make_shared<std::condition_variable>();
    }
  }

  template <typename T> void push(const std::string &queueName, const T &data) {
    checkQueueExists(queueName);
    {
      std::lock_guard<std::mutex> lock(*mutexes[queueName]);
      queues[queueName].push(data);
    }
    cvs[queueName]->notify_one();
  }

  template <typename T> T pop(const std::string &queueName) {
    checkQueueExists(queueName);
    std::unique_lock<std::mutex> lock(*mutexes[queueName]);
    cvs[queueName]->wait(
        lock, [this, &queueName] { return !queues[queueName].empty(); });
    std::any data = queues[queueName].front();
    queues[queueName].pop();
    try {
      return std::any_cast<T>(data);
    } catch (const std::bad_any_cast &e) {
      throw std::runtime_error("Type mismatch in queue '" + queueName +
                               "': " + e.what());
    }
  }

private:
  std::map<std::string, std::queue<std::any>> queues;
  std::map<std::string, std::shared_ptr<std::mutex>> mutexes;
  std::map<std::string, std::shared_ptr<std::condition_variable>> cvs;
  void checkQueueExists(const std::string &queueName) const {
    if (queues.find(queueName) == queues.end()) {
      throw std::invalid_argument("Queue '" + queueName + "' not found.");
    }
  }
};

#endif // SHARED_QUEUE_H