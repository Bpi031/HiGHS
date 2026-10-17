#ifndef HIGHS_SEARCH_BATCH_AUDIT_H_
#define HIGHS_SEARCH_BATCH_AUDIT_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Joined observations only. No field is an input to scheduling or proof.
struct HighsSearchBatchRecord {
  int64_t restart = 0, epoch = 0;
  int64_t threads = 0, capacity = 0, allocated = 0, issued = 0;
  int64_t ready_before_install = 0, ready_after_install = 0;
  int64_t queued_after_join = 0, tasks_observed = 0;
  bool root = false, valid = true;
  double started_s = -1, joined_s = -1, reconciled_s = -1;
  double task_interval_s = 0, last_return_s = -1, penultimate_return_s = -1;
  double first_start_s = (std::numeric_limits<double>::max)();

  void observeTask(double start, double finish) {
    if (tasks_observed == (std::numeric_limits<int64_t>::max)()) {
      valid = false;
      return;
    }
    ++tasks_observed;
    if (!std::isfinite(start) || !std::isfinite(finish) ||
        start < started_s || finish < start || finish > joined_s) {
      valid = false;
      return;
    }
    first_start_s = std::min(first_start_s, start);
    task_interval_s += finish - start;
    if (finish >= last_return_s) {
      penultimate_return_s = last_return_s;
      last_return_s = finish;
    } else {
      penultimate_return_s = std::max(penultimate_return_s, finish);
    }
  }

  bool complete() const {
    return valid && restart >= 0 && epoch >= 0 && threads > 0 && capacity > 0 &&
        allocated > 0 && allocated <= capacity && issued > 0 &&
        issued <= allocated && tasks_observed == issued &&
        ready_before_install >= 0 && ready_after_install >= 0 &&
        queued_after_join >= 0 && std::isfinite(started_s) && started_s >= 0 &&
        std::isfinite(joined_s) && joined_s >= started_s &&
        std::isfinite(reconciled_s) && reconciled_s >= joined_s &&
        std::isfinite(first_start_s) && first_start_s >= started_s &&
        last_return_s >= first_start_s && last_return_s <= joined_s &&
        (issued == 1 ? penultimate_return_s == -1 :
             penultimate_return_s >= first_start_s && penultimate_return_s <= last_return_s) &&
        std::isfinite(task_interval_s) && task_interval_s >= 0;
  }
  double spanSeconds() const { return joined_s - started_s; }
  double tailSeconds() const {
    return issued > 1 ? last_return_s - penultimate_return_s : 0;
  }
  double reconciliationSeconds() const { return reconciled_s - joined_s; }
};

class HighsSearchBatchAudit {
 public:
  static const size_t kRetained = 128;
  void record(const HighsSearchBatchRecord& record) {
    increment(batches_);
    if (!record.complete()) {
      increment(invalid_);
      return;
    }
    increment(valid_);
    add(span_s_, record.spanSeconds());
    add(task_interval_s_, record.task_interval_s);
    add(tail_s_, record.tailSeconds());
    add(reconciliation_s_, record.reconciliationSeconds());
    max_tail_s_ = std::max(max_tail_s_, record.tailSeconds());
    if (retained_ < kRetained) {
      records_[retained_++] = record;
    } else {
      // Preserve the first 64 and the largest later one-task tails.
      auto begin = records_.begin() + kRetained / 2;
      auto smallest = std::min_element(begin, records_.end(),
          [](const HighsSearchBatchRecord& a, const HighsSearchBatchRecord& b) {
            return a.tailSeconds() < b.tailSeconds();
          });
      if (record.tailSeconds() > smallest->tailSeconds()) *smallest = record;
      increment(dropped_);
    }
  }
  size_t retained() const { return retained_; }
  const HighsSearchBatchRecord& at(size_t i) const {
    if (i >= retained_) throw std::out_of_range("unretained search batch");
    return records_[i];
  }
  int64_t batches() const { return batches_; }
  int64_t valid() const { return valid_; }
  int64_t invalid() const { return invalid_; }
  int64_t dropped() const { return dropped_; }
  double spanSeconds() const { return span_s_; }
  double taskIntervalSeconds() const { return task_interval_s_; }
  double tailSeconds() const { return tail_s_; }
  double maxTailSeconds() const { return max_tail_s_; }
  double reconciliationSeconds() const { return reconciliation_s_; }

 private:
  static void increment(int64_t& value) {
    if (value != (std::numeric_limits<int64_t>::max)()) ++value;
  }
  static void add(double& value, double amount) {
    value += std::min(amount, (std::numeric_limits<double>::max)() - value);
  }
  std::array<HighsSearchBatchRecord, kRetained> records_{};
  size_t retained_ = 0;
  int64_t batches_ = 0, valid_ = 0, invalid_ = 0, dropped_ = 0;
  double span_s_ = 0, task_interval_s_ = 0, tail_s_ = 0;
  double max_tail_s_ = 0, reconciliation_s_ = 0;
};

#endif
