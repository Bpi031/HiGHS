#ifndef HIGHS_ASSIGNMENT_SUPPORT_H_
#define HIGHS_ASSIGNMENT_SUPPORT_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <numeric>
#include <vector>
#include "lp_data/HighsLp.h"

struct HighsAssignmentSupportLimits {
  HighsInt max_free_columns = 4096;
  HighsInt max_row_degree = 128;
  HighsInt expansion_rounds = 3;
  int64_t max_edge_visits = 10000000;
  size_t max_index_bytes = 64 * 1024 * 1024;
};

struct HighsAssignmentSlots {
  // Original model coordinates; -1 denotes a non-assignment column.
  std::vector<HighsInt> column_to_slot;
  HighsInt count = 0;
  HighsAssignmentSupportLimits limits;
};

struct HighsAssignmentSupportResult {
  std::vector<HighsInt> support, free_columns;
  HighsInt free_slots = 0;
  int64_t visited_rows = 0, visited_edges = 0, skipped_dense_rows = 0;
  bool support_cap_reached = false, work_cap_reached = false;
};

// Candidate-only selection. No bounds, rows or proof state are changed here.
class HighsAssignmentSupport {
 public:
  static bool read(std::istream& input, const HighsLp& model,
                   HighsAssignmentSlots& output) {
    std::string version;
    HighsInt cols = 0, count = 0, support = 0;
    uint64_t bytes = 0;
    HighsAssignmentSlots parsed;
    auto& limits = parsed.limits;
    if (!(input >> version >> cols >> count >> support >> limits.max_free_columns
                >> limits.max_row_degree >> limits.expansion_rounds
                >> limits.max_edge_visits >> bytes) ||
        version != "ROST_ASSIGNMENT_SLOTS_V1" || cols != model.num_col_ ||
        count <= 1 || count > cols || support < count || support > cols ||
        limits.max_free_columns <= 0 || limits.max_free_columns >= support ||
        limits.max_row_degree <= 0 || limits.max_row_degree > 512 ||
        limits.expansion_rounds <= 0 || limits.expansion_rounds > 16 ||
        limits.max_edge_visits <= 0 || limits.max_edge_visits > 100000000 ||
        bytes == 0 || bytes > 256 * 1024 * 1024 ||
        uint64_t(cols) * (sizeof(HighsInt) + 1) + uint64_t(count) * sizeof(HighsInt) > bytes ||
        model.sense_ != ObjSense::kMinimize ||
        model.col_names_.size() != size_t(cols) ||
        model.integrality_.size() != size_t(cols) ||
        model.col_lower_.size() != size_t(cols) || model.col_upper_.size() != size_t(cols))
      return false;
    parsed.count = count;
    limits.max_index_bytes = size_t(bytes);
    parsed.column_to_slot.assign(cols, -1);
    std::vector<HighsInt> sizes(count, 0);
    for (HighsInt record = 0; record < support; ++record) {
      HighsInt col, slot;
      std::string name;
      if (!(input >> col >> slot >> name) || col < 0 || col >= cols ||
          slot < 0 || slot >= count || parsed.column_to_slot[col] != -1 ||
          model.col_names_[col] != name || model.integrality_[col] != HighsVarType::kInteger ||
          !(0 <= model.col_lower_[col] && model.col_lower_[col] <= model.col_upper_[col] &&
            model.col_upper_[col] <= 1)) return false;
      parsed.column_to_slot[col] = slot;
      ++sizes[slot];
    }
    std::string extra;
    if (input >> extra || !input.eof()) return false;
    for (HighsInt size : sizes)
      if (!size || size > limits.max_free_columns) return false;
    output = std::move(parsed);
    return true;
  }

  static bool select(const HighsLp& model, const std::vector<double>& seed,
                     const HighsAssignmentSlots& slots, int64_t ordinal,
                     HighsAssignmentSupportResult& output) {
    const auto& limits = slots.limits;
    const size_t n = size_t(std::max(HighsInt{0}, model.num_col_));
    if (model.num_col_ < 0 || model.num_row_ < 0 || ordinal < 0 || slots.count <= 1 ||
        slots.count > model.num_col_ || seed.size() != n || slots.column_to_slot.size() != n ||
        model.col_cost_.size() != n || model.integrality_.size() != n ||
        model.col_lower_.size() != n || model.col_upper_.size() != n ||
        limits.max_free_columns <= 0 || limits.max_row_degree <= 0 ||
        limits.max_row_degree > 512 || limits.expansion_rounds <= 0 ||
        limits.expansion_rounds > 16 || limits.max_edge_visits <= 0) return false;
    const auto& matrix = model.a_matrix_;
    if ((matrix.format_ != MatrixFormat::kColwise && matrix.format_ != MatrixFormat::kRowwise) ||
        matrix.num_col_ != model.num_col_ || matrix.num_row_ != model.num_row_ ||
        matrix.index_.size() != matrix.value_.size()) return false;
    const size_t major = matrix.format_ == MatrixFormat::kColwise ? n : size_t(model.num_row_);
    const HighsInt minor = matrix.format_ == MatrixFormat::kColwise ? model.num_row_ : model.num_col_;
    if (matrix.start_.size() != major + 1 || matrix.start_.front() != 0 ||
        matrix.start_.back() < 0 || size_t(matrix.start_.back()) != matrix.index_.size()) return false;
    for (size_t i = 1; i < matrix.start_.size(); ++i)
      if (matrix.start_[i] < matrix.start_[i - 1]) return false;
    for (size_t k = 0; k < matrix.index_.size(); ++k)
      if (matrix.index_[k] < 0 || matrix.index_[k] >= minor || !std::isfinite(matrix.value_[k])) return false;
    // Conservative bound includes transpose, CSR incidence and ranking scratch.
    const uint64_t estimate = uint64_t(matrix.index_.size()) * 64 +
        uint64_t(n + size_t(model.num_row_) + size_t(slots.count) + 1) * 128;
    if (estimate > limits.max_index_bytes ||
        matrix.index_.size() > size_t(limits.max_edge_visits)) return false;
    HighsAssignmentSupportResult result;
    std::vector<HighsInt> sizes(slots.count, 0);
    for (HighsInt col = 0; col < model.num_col_; ++col) {
      const HighsInt slot = slots.column_to_slot[col];
      if (slot < -1 || slot >= slots.count || !std::isfinite(seed[col]) ||
          !std::isfinite(model.col_cost_[col])) return false;
      if (slot < 0) continue;
      if (model.integrality_[col] != HighsVarType::kInteger ||
          !(0 <= model.col_lower_[col] && model.col_lower_[col] <= model.col_upper_[col] &&
            model.col_upper_[col] <= 1)) return false;
      result.support.push_back(col);
      ++sizes[slot];
    }
    if (limits.max_free_columns >= HighsInt(result.support.size())) return false;
    for (HighsInt size : sizes) if (!size || size > limits.max_free_columns) return false;
    auto rows = matrix;
    rows.ensureRowwise();
    std::vector<HighsInt> starts{0}, columns;
    std::vector<double> scores;
    std::vector<HighsInt> row_columns;
    for (HighsInt row = 0; row < model.num_row_; ++row) {
      row_columns.clear();
      long double score = 0;
      for (HighsInt k = rows.start_[row]; k < rows.start_[row + 1]; ++k) {
        ++result.visited_edges;
        if (!rows.value_[k]) continue;
        const HighsInt col = rows.index_[k];
        if (slots.column_to_slot[col] >= 0) row_columns.push_back(col);
        else if (model.col_cost_[col] > 0 && seed[col] > 0)
          score += static_cast<long double>(model.col_cost_[col]) * seed[col];
      }
      std::sort(row_columns.begin(), row_columns.end());
      row_columns.erase(std::unique(row_columns.begin(), row_columns.end()), row_columns.end());
      if (row_columns.empty()) continue;
      if (row_columns.size() > size_t(limits.max_row_degree)) {
        ++result.skipped_dense_rows;
        continue;
      }
      if (!std::isfinite(double(score))) return false;
      columns.insert(columns.end(), row_columns.begin(), row_columns.end());
      starts.push_back(HighsInt(columns.size()));
      scores.push_back(double(score));
    }
    if (scores.empty()) return false;
    auto row_less = [&](HighsInt a, HighsInt b) {
      if (scores[a] != scores[b]) return scores[a] > scores[b];
      return std::lexicographical_compare(columns.begin() + starts[a], columns.begin() + starts[a + 1],
                                          columns.begin() + starts[b], columns.begin() + starts[b + 1]);
    };
    std::vector<HighsInt> anchors;
    for (HighsInt row = 0; row < HighsInt(scores.size()); ++row)
      if (scores[row] > 0) anchors.push_back(row);
    if (anchors.empty()) return false;
    std::stable_sort(anchors.begin(), anchors.end(), row_less);
    // The owner advances selection attempts independently of child reservations.
    const HighsInt anchor = anchors[size_t(ordinal) % anchors.size()];
    // Incidence belongs to whole slots, including alternative duties that do
    // not occur in the anchor row. Column-only traversal misses exchange chains.
    std::vector<HighsInt> adj_start(size_t(slots.count) + 1, 0);
    for (HighsInt col : columns) ++adj_start[slots.column_to_slot[col] + 1];
    for (HighsInt slot = 0; slot < slots.count; ++slot) adj_start[slot + 1] += adj_start[slot];
    auto cursor = adj_start;
    std::vector<HighsInt> adjacency(columns.size());
    for (HighsInt row = 0; row < HighsInt(scores.size()); ++row)
      for (HighsInt k = starts[row]; k < starts[row + 1]; ++k)
        adjacency[cursor[slots.column_to_slot[columns[k]]]++] = row;
    std::vector<unsigned char> selected(slots.count, 0), seen(scores.size(), 0);
    std::vector<HighsInt> frontier{anchor}, following, additions;
    HighsInt free_count = 0;
    for (HighsInt round = 0; round < limits.expansion_rounds && !frontier.empty(); ++round) {
      std::stable_sort(frontier.begin(), frontier.end(), row_less);
      following.clear();
      for (HighsInt row : frontier) {
        if (seen[row]) continue;
        seen[row] = 1;
        ++result.visited_rows;
        additions.clear();
        for (HighsInt k = starts[row]; k < starts[row + 1]; ++k)
          if (!selected[slots.column_to_slot[columns[k]]]) additions.push_back(slots.column_to_slot[columns[k]]);
        std::sort(additions.begin(), additions.end());
        additions.erase(std::unique(additions.begin(), additions.end()), additions.end());
        HighsInt needed = 0;
        for (HighsInt slot : additions) needed += sizes[slot];
        if (needed > limits.max_free_columns - free_count) {
          result.support_cap_reached = true;
          continue;
        }
        for (HighsInt slot : additions) { selected[slot] = 1; ++result.free_slots; }
        free_count += needed;
        for (HighsInt slot : additions) {
          for (HighsInt j = adj_start[slot]; j < adj_start[slot + 1]; ++j) {
            if (result.visited_edges >= limits.max_edge_visits) {
              result.work_cap_reached = true;
              break;
            }
            ++result.visited_edges;
            if (!seen[adjacency[j]]) following.push_back(adjacency[j]);
          }
          if (result.work_cap_reached) break;
        }
        if (result.work_cap_reached) break;
      }
      if (result.work_cap_reached) break;
      std::sort(following.begin(), following.end());
      following.erase(std::unique(following.begin(), following.end()), following.end());
      frontier.swap(following);
    }
    for (HighsInt col : result.support)
      if (selected[slots.column_to_slot[col]]) result.free_columns.push_back(col);
    if (result.free_columns.empty()) return false;
    output = std::move(result);
    return true;
  }
};
#endif
