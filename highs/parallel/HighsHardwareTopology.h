/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_HARDWARE_TOPOLOGY_H_
#define HIGHS_HARDWARE_TOPOLOGY_H_

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lp_data/HConst.h"

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

struct HighsHardwareTopology {
  HighsInt logical_cpus = 0;
  HighsInt physical_cores = 0;
  HighsInt performance_cores = 0;
  HighsInt numa_nodes = 0;
  HighsInt last_level_cache_domains = 0;
  HighsInt heterogeneous_core_classes = 0;
  bool cpu_binding_available = false;
  bool memory_binding_available = false;
  bool complete = false;
};

#if defined(__APPLE__)
inline HighsInt highsReadSysctlInteger(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || value <= 0)
    return 0;
  return static_cast<HighsInt>(value);
}
#endif

inline HighsHardwareTopology highsDetectHardwareTopology() {
  HighsHardwareTopology topology;

#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  std::vector<HighsInt> cpus;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
    topology.cpu_binding_available = true;
    for (HighsInt cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
  }
  topology.logical_cpus = static_cast<HighsInt>(cpus.size());

  std::set<std::pair<HighsInt, HighsInt>> physical_cores;
  std::set<HighsInt> cache_domains;
  std::set<HighsInt> numa_nodes;
  for (HighsInt cpu : cpus) {
    const std::string prefix =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    std::ifstream package_input(prefix + "/topology/physical_package_id");
    std::ifstream core_input(prefix + "/topology/core_id");
    HighsInt package_id = -1;
    HighsInt core_id = -1;
    if ((package_input >> package_id) && (core_input >> core_id))
      physical_cores.emplace(package_id, core_id);

    std::ifstream cache_input(prefix + "/cache/index3/id");
    HighsInt cache_id = -1;
    if (cache_input >> cache_id) cache_domains.insert(cache_id);

    DIR* cpu_directory = opendir(prefix.c_str());
    if (cpu_directory != nullptr) {
      while (dirent* entry = readdir(cpu_directory)) {
        const std::string name(entry->d_name);
        if (name.compare(0, 4, "node") != 0 || name.size() == 4) continue;
        HighsInt node = 0;
        bool numeric = true;
        for (size_t i = 4; i != name.size(); ++i) {
          if (name[i] < '0' || name[i] > '9') {
            numeric = false;
            break;
          }
          node = 10 * node + static_cast<HighsInt>(name[i] - '0');
        }
        if (numeric) numa_nodes.insert(node);
      }
      closedir(cpu_directory);
    }
  }
  topology.physical_cores = static_cast<HighsInt>(physical_cores.size());
  topology.last_level_cache_domains =
      static_cast<HighsInt>(cache_domains.size());
  topology.numa_nodes = static_cast<HighsInt>(numa_nodes.size());
  topology.memory_binding_available = topology.numa_nodes > 0;
  topology.complete = topology.logical_cpus > 0 && topology.physical_cores > 0 &&
                      topology.numa_nodes > 0 &&
                      topology.last_level_cache_domains > 0;
#elif defined(__APPLE__)
  topology.logical_cpus = highsReadSysctlInteger("hw.logicalcpu");
  topology.physical_cores = highsReadSysctlInteger("hw.physicalcpu");
  topology.performance_cores =
      highsReadSysctlInteger("hw.perflevel0.physicalcpu");
  topology.heterogeneous_core_classes = highsReadSysctlInteger("hw.nperflevels");
  topology.numa_nodes = 1;
  topology.last_level_cache_domains = 1;
  topology.complete =
      topology.logical_cpus > 0 && topology.physical_cores > 0;
#else
  topology.logical_cpus =
      static_cast<HighsInt>(std::thread::hardware_concurrency());
#endif

  if (topology.logical_cpus <= 0) {
    topology.logical_cpus =
        static_cast<HighsInt>(std::thread::hardware_concurrency());
  }
  return topology;
}

inline HighsInt highsTopologyAwareWorkerCount(
    HighsInt configured_threads, const HighsHardwareTopology& topology) {
  if (configured_threads <= 1) return 1;
  HighsInt workers = configured_threads;
  if (topology.logical_cpus > 0)
    workers = std::min(workers, topology.logical_cpus);
  if (topology.heterogeneous_core_classes > 1 &&
      topology.performance_cores > 0) {
    workers = std::min(workers, topology.performance_cores);
  } else if (topology.physical_cores > 0) {
    workers = std::min(workers, topology.physical_cores);
  } else if (!topology.complete) {
    workers = 1;
  }
  return std::max(HighsInt{1}, workers);
}

inline HighsInt highsTopologyAwareWorkerCount(HighsInt configured_threads) {
  static const HighsHardwareTopology process_topology =
      highsDetectHardwareTopology();
  return highsTopologyAwareWorkerCount(configured_threads, process_topology);
}

#endif
