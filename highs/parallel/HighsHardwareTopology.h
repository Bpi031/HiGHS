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
#include <limits>
#include <memory>
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

inline bool highsTopologySuffix(const std::string& name,
                               const std::string& prefix, HighsInt& value) {
  if (name.compare(0, prefix.size(), prefix) != 0 || name.size() == prefix.size())
    return false;
  HighsInt parsed = 0;
  for (size_t i = prefix.size(); i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') return false;
    const HighsInt digit = name[i] - '0';
    if (parsed > (std::numeric_limits<HighsInt>::max() - digit) / 10) return false;
    parsed = 10 * parsed + digit;
  }
  value = parsed;
  return true;
}

struct HighsCacheTopologyEntry {
  HighsInt level = 0;
  std::string type;
  std::string shared_cpus;
  HighsInt id = -1;
};

inline std::pair<HighsInt, std::string> highsLastLevelDataCache(
    const std::vector<HighsCacheTopologyEntry>& entries) {
  HighsInt level = 0;
  for (const auto& entry : entries) {
    if (entry.type == "Data" || entry.type == "Unified")
      level = std::max(level, entry.level);
  }
  for (const auto& entry : entries) {
    if (entry.level != level || level <= 0 ||
        (entry.type != "Data" && entry.type != "Unified"))
      continue;
    if (!entry.shared_cpus.empty()) return {level, "cpus:" + entry.shared_cpus};
    if (entry.id >= 0)
      return {level, entry.type + ":id:" + std::to_string(entry.id)};
  }
  // Do not substitute a lower-level cache when the LLC identity is missing.
  return {0, ""};
}

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
    for (HighsInt cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
  }
  topology.logical_cpus = static_cast<HighsInt>(cpus.size());

  std::set<std::pair<HighsInt, HighsInt>> physical_cores;
  std::set<std::pair<HighsInt, std::string>> cache_domains;
  std::set<HighsInt> numa_nodes;
  bool all_cpu_metadata = !cpus.empty();
  for (HighsInt cpu : cpus) {
    const std::string prefix =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    std::ifstream package_input(prefix + "/topology/physical_package_id");
    std::ifstream core_input(prefix + "/topology/core_id");
    HighsInt package_id = -1;
    HighsInt core_id = -1;
    if ((package_input >> package_id) && (core_input >> core_id) &&
        package_id >= 0 && core_id >= 0)
      physical_cores.emplace(package_id, core_id);
    else
      all_cpu_metadata = false;

    std::vector<HighsCacheTopologyEntry> caches;
    std::unique_ptr<DIR, decltype(&closedir)> cache_directory(
        opendir((prefix + "/cache").c_str()), &closedir);
    if (cache_directory != nullptr) {
      while (dirent* entry = readdir(cache_directory.get())) {
        const std::string name(entry->d_name);
        HighsInt index;
        if (!highsTopologySuffix(name, "index", index)) continue;
        const std::string cache_prefix = prefix + "/cache/" + name;
        HighsCacheTopologyEntry cache;
        std::ifstream(cache_prefix + "/level") >> cache.level;
        std::ifstream(cache_prefix + "/type") >> cache.type;
        std::ifstream(cache_prefix + "/shared_cpu_list") >> cache.shared_cpus;
        std::ifstream(cache_prefix + "/id") >> cache.id;
        caches.push_back(std::move(cache));
      }
    }
    const auto cache = highsLastLevelDataCache(caches);
    if (cache.first > 0)
      cache_domains.insert(cache);
    else
      all_cpu_metadata = false;

    bool cpu_numa_known = false;
    std::unique_ptr<DIR, decltype(&closedir)> cpu_directory(
        opendir(prefix.c_str()), &closedir);
    if (cpu_directory != nullptr) {
      while (dirent* entry = readdir(cpu_directory.get())) {
        const std::string name(entry->d_name);
        HighsInt node;
        if (highsTopologySuffix(name, "node", node)) {
          numa_nodes.insert(node);
          cpu_numa_known = true;
        }
      }
    }
    all_cpu_metadata = all_cpu_metadata && cpu_numa_known;
  }
  topology.physical_cores = static_cast<HighsInt>(physical_cores.size());
  topology.last_level_cache_domains =
      static_cast<HighsInt>(cache_domains.size());
  topology.numa_nodes = static_cast<HighsInt>(numa_nodes.size());
  // Discovering nodes is not evidence that a memory-binding operation works.
  topology.complete = all_cpu_metadata && topology.logical_cpus > 0 &&
                      topology.physical_cores > 0 &&
                      topology.numa_nodes > 0 &&
                      topology.last_level_cache_domains > 0;
#elif defined(__APPLE__)
  topology.logical_cpus = highsReadSysctlInteger("hw.logicalcpu");
  topology.physical_cores = highsReadSysctlInteger("hw.physicalcpu");
  topology.performance_cores =
      highsReadSysctlInteger("hw.perflevel0.physicalcpu");
  topology.heterogeneous_core_classes = highsReadSysctlInteger("hw.nperflevels");
  // sysctl CPU counts do not establish NUMA or shared-cache domain counts.
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
  // Core class and cache discovery are observations, not allocation authority.
  // Missing metadata must not serialize an explicitly configured thread pool.
  return std::max(HighsInt{1}, workers);
}

inline HighsInt highsTopologyAwareWorkerCount(HighsInt configured_threads) {
  static const HighsHardwareTopology process_topology =
      highsDetectHardwareTopology();
  return highsTopologyAwareWorkerCount(configured_threads, process_topology);
}

#endif
