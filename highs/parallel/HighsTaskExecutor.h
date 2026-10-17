/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_TASKEXECUTOR_H_
#define HIGHS_TASKEXECUTOR_H_

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsSchedulerConstants.h"
#include "parallel/HighsSplitDeque.h"
#include "util/HighsInt.h"
#include "util/HighsRandom.h"

class HighsTaskExecutor {
 public:
  using cache_aligned = highs::cache_aligned;
  // Injectable launcher for deterministic startup-failure tests, not a solver option.
  using WorkerFactory = std::thread (*)(void (*)(int, HighsTaskExecutor*), int,
                                       HighsTaskExecutor*);
  struct ExecutorHandle {
    HighsTaskExecutor* ptr{nullptr};
    bool isMain{false};

    void dispose();
    ~ExecutorHandle() { dispose(); }
  };

 private:
#ifdef _WIN32
  static HighsSplitDeque*& threadLocalWorkerDeque();
  static ExecutorHandle& threadLocalExecutorHandle();
#else
  static thread_local HighsSplitDeque* threadLocalWorkerDequePtr;
  static thread_local ExecutorHandle globalExecutorHandle;

  static HighsSplitDeque*& threadLocalWorkerDeque() {
    return threadLocalWorkerDequePtr;
  }

  static ExecutorHandle& threadLocalExecutorHandle() {
    return globalExecutorHandle;
  }
#endif

  std::atomic<int> referenceCount{1};
  std::atomic<bool> hasStopped{false};
  std::atomic<int> startupReady{0};

  cache_aligned::shared_ptr<HighsSplitDeque::WorkerBunk> workerBunk;
  std::vector<cache_aligned::unique_ptr<HighsSplitDeque>> workerDeques;
  std::vector<std::thread> workerThreads;

  HighsTask* random_steal_loop(HighsSplitDeque* localDeque) {
    const int numWorkers = static_cast<int>(workerDeques.size());

    int numTries = HighsSchedulerConstants::initialSpinTries(numWorkers);

    auto tStart = HighsSchedulerConstants::Clock::now();

    while (true) {
      for (int s = 0; s < numTries; ++s) {
        HighsTask* task = localDeque->randomSteal();
        if (task) return task;
      }

      if (!workerBunk->haveJobs.load(std::memory_order_relaxed)) break;

      auto numMicroSecs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              HighsSchedulerConstants::Clock::now() - tStart)
              .count();

      if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeGlobalSync)
        numTries = HighsSchedulerConstants::nextSpinTries(numTries);
      else
        break;
    }

    return nullptr;
  }

  static void run_worker(int workerId, HighsTaskExecutor* ptr) {
    auto& executorHandle = threadLocalExecutorHandle();
    executorHandle.ptr = ptr;

    if (!ptr->hasStopped) {
      HighsSplitDeque* localDeque = ptr->workerDeques[workerId].get();
      threadLocalWorkerDeque() = localDeque;

      HighsTask* currentTask = ptr->workerBunk->waitForNewTask(localDeque, &ptr->startupReady);
      while (currentTask != nullptr) {
        localDeque->runStolenTask(currentTask);

        currentTask = ptr->random_steal_loop(localDeque);
        if (currentTask != nullptr) continue;

        currentTask = ptr->workerBunk->waitForNewTask(localDeque);
      }
    }

    executorHandle.dispose();
  }

  void startWorkers(WorkerFactory factory) {
    for (int i = 1; i < static_cast<int>(workerDeques.size()); ++i) {
      // Reserve a handle before the new worker can finish. The owner's handle
      // keeps the executor alive throughout partial-startup cleanup.
      ++referenceCount;
      try {
        std::thread worker = factory ? factory(&run_worker, i, this)
                                    : std::thread(&run_worker, i, this);
        if (!worker.joinable())
          throw std::system_error(
              std::make_error_code(std::errc::resource_unavailable_try_again));
        workerThreads.emplace_back(std::move(worker));
      } catch (...) {
        --referenceCount;
        stopWorkerThreads(true);
        throw;
      }
    }
    // Publishing the first task before sleepers register can miss its wakeup.
    // Wait only at startup; steady-state scheduling and stealing are unchanged.
    while (startupReady.load(std::memory_order_acquire) !=
           static_cast<int>(workerThreads.size()))
      std::this_thread::yield();
  }

 public:
  HighsTaskExecutor(int numThreads) {
    if (numThreads <= 0)
      throw std::invalid_argument("scheduler worker count must be positive");

    workerDeques.resize(numThreads);
    workerBunk = cache_aligned::make_shared<HighsSplitDeque::WorkerBunk>();
    for (int i = 0; i < numThreads; ++i)
      workerDeques[i] = cache_aligned::make_unique<HighsSplitDeque>(
          workerBunk, workerDeques.data(), i, numThreads);

    workerThreads.reserve(numThreads - 1);
  }

  void stopWorkerThreads(bool blocking = false) {
    // Check if stop has been called already.
    auto& executorHandle = threadLocalExecutorHandle();
    if (executorHandle.ptr == nullptr || hasStopped.exchange(true)) return;

    // now inject the null task as termination signal to every worker
    for (auto& workerDeque : workerDeques) {
      workerDeque->injectTaskAndNotify(nullptr);
    }

    // only block if called on main thread, otherwise deadlock may occur
    if (blocking && executorHandle.isMain) {
      for (auto& workerThread : workerThreads) {
        workerThread.join();
      }
    } else {
      for (auto& workerThread : workerThreads) {
        workerThread.detach();
      }
    }
  }

  static HighsSplitDeque* getThisWorkerDeque() {
    return threadLocalWorkerDeque();
  }

  static int getNumWorkerThreads() {
    return threadLocalWorkerDeque()->getNumWorkers();
  }

  static void initialize(int numThreads, WorkerFactory factory = nullptr) {
    auto& executorHandle = threadLocalExecutorHandle();
    if (executorHandle.ptr == nullptr) {
      auto executor = cache_aligned::make_unique<HighsTaskExecutor>(numThreads);
      executorHandle.isMain = true;
      executorHandle.ptr = executor.release();
      threadLocalWorkerDeque() = executorHandle.ptr->workerDeques[0].get();
      try {
        executorHandle.ptr->startWorkers(factory);
      } catch (...) {
        executorHandle.dispose();
        throw;
      }
    }
  }

  // can be called on main or worker threads
  // blocking ignored unless called on main thread
  static void shutdown(bool blocking = false) {
    auto& executorHandle = threadLocalExecutorHandle();

    if (executorHandle.ptr != nullptr) {
      executorHandle.ptr->stopWorkerThreads(blocking);
      executorHandle.dispose();
    }
  }

  static void sync_stolen_task(HighsSplitDeque* localDeque,
                               HighsTask* stolenTask) {
    HighsSplitDeque* stealer;
    if (!localDeque->leapfrogStolenTask(stolenTask, stealer)) {
      const int numWorkers = localDeque->getNumWorkers();
      int numTries = HighsSchedulerConstants::initialSpinTries(numWorkers);

      auto tStart = HighsSchedulerConstants::Clock::now();

      while (true) {
        for (int s = 0; s < numTries; ++s) {
          if (stolenTask->isFinished()) {
            localDeque->popStolen();
            return;
          }
          localDeque->yield();
        }

        auto numMicroSecs =
            std::chrono::duration_cast<std::chrono::microseconds>(
                HighsSchedulerConstants::Clock::now() - tStart)
                .count();

        if (numMicroSecs < HighsSchedulerConstants::kMicroSecsBeforeSleep)
          numTries = HighsSchedulerConstants::nextSpinTries(numTries);
        else
          break;
      }

      if (!stolenTask->isFinished())
        localDeque->waitForTaskToFinish(stolenTask, stealer);
    }

    localDeque->popStolen();
  }
};

#endif
